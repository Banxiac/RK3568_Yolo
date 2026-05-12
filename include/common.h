#pragma once

#include <cstdint>
#include <cstring>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <linux/videodev2.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <errno.h>

#define SHM_NAME "/edge_ai_frame"

#define MAX_DETECTS 64

#define MJPEG_MAX_BYTES (512 * 1024)
constexpr int CAP_REF_QUEUE_SIZE = 4;
constexpr int INFER_RESULT_QUEUE_SIZE = 4;

// SharedDetect: 存储检测结果，供 inference_thread 填入 shm
struct SharedDetect {
    char name[16];
    float prop;// 置信度，0.0~1.0
    int left;// 左边界
    int top;// 上边界
    int right;// 右边界
    int bottom;// 下边界
};
// SharedFrameSlot: shm 中的一个 slot，包含一帧图像数据（JPEG）和对应的检测结果，由 inference_thread 填入，web_thread 画框，web_thread 和 web_viewer 共享。
struct SharedFrameSlot {
    std::atomic<uint64_t> lock_seq{0};

    uint64_t frame_seq{0};

    int width{0};
    int height{0};

    uint8_t jpeg[MJPEG_MAX_BYTES];
    uint32_t jpeg_size{0};

    SharedDetect detects[MAX_DETECTS];
    int detect_count{0};
};

struct SharedFrame {
    std::atomic<uint64_t> global_seq{0};// 全局序列号，每次更新都增加，供 web_thread 和 web_viewer 判断是否有新帧
    std::atomic<int> active_index{0};// 当前活跃的 slot 索引，0 或 1，供 inference_thread 切换使用
    /*
        双缓冲的两个 slot，inference_thread 填入一个 slot 后切换 active_index，
        web_thread 根据 active_index 读取最新的 slot 进行画框，web_viewer 根据 global_seq 判断是否有新帧需要显示
    */ 
    SharedFrameSlot slots[2];
    // 构造函数，初始化原子变量和 slot，确保共享内存中的数据结构正确初始化
    SharedFrame()
    {
        global_seq.store(0, std::memory_order_relaxed);
        active_index.store(0, std::memory_order_relaxed);

        for (int i = 0; i < 2; ++i) {
            slots[i].lock_seq.store(0, std::memory_order_relaxed);
            slots[i].frame_seq = 0;
            slots[i].width = 0;
            slots[i].height = 0;
            slots[i].jpeg_size = 0;
            slots[i].detect_count = 0;
        }
    }
};



// struct RawFrame {
//     uint8_t data[MJPEG_MAX_BYTES];
//     uint32_t size = 0;
//     int width = 1920;
//     int height = 1080;
//     uint64_t seq = 0;
// };
// CaptureFrameRef: capture 线程放入队列的帧引用，包含 V4L2 buffer 的 index、dma_fd、mmap 地址和 JPEG 信息，供 inference_thread 使用零拷贝方式解码。
struct CaptureFrameRef {
    int index = -1;// V4L2 buffer index，供 capture 线程归还 buffer 使用。
    int dma_fd = -1;// 如果支持 dma_fd 则 inference_thread 可以使用零拷贝方式解码，否则只能通过 data 指向的 mmap 地址解码。
    void* data = nullptr;// mmap 地址，供 inference_thread 直接解码使用。
    uint32_t bytesused = 0;// V4L2 buffer 的 bytesused，表示有效数据长度。
    uint32_t jpg_size = 0;// 实际 JPEG 大小，可能小于 bytesused，因为部分摄像头在 MJPEG buffer 中可能存在 padding。
    size_t jpg_offset = 0;// JPEG 数据在 buffer 中的偏移，通常为 0，但部分设备可能会在 buffer 前面添加一些 header 导致 JPEG 数据不从头开始。
    int width = 0;
    int height = 0;
    uint64_t seq = 0;// 帧序列号，供统计和调试使用。
};

// V4L2Shared: capture 线程和 inference 线程共享的 V4L2 相关资源和统计信息，包含 fd、ioctl 互斥锁和一些原子计数器。
struct V4L2Shared {
    int fd = -1;// V4L2 device fd，供 capture 线程和 inference 线程共享，capture 线程负责打开和关闭，inference 线程只负责使用，不关闭。
    pthread_mutex_t ioctl_mutex = PTHREAD_MUTEX_INITIALIZER;// ioctl 互斥锁，确保 capture 线程中的 DQBUF 和 inference 线程中的 QBUF 不会并发调用 ioctl 导致冲突。

    std::atomic<uint64_t> qbuf_count{0};// 成功 QBUF 的计数，供统计使用。
    std::atomic<uint64_t> ref_queue_full{0};// 参考队列满的计数，供统计使用。
    std::atomic<uint64_t> invalid_jpeg{0};// 无效 JPEG 的计数，供统计使用。
    std::atomic<uint64_t> dq_count{0};// 成功 DQBUF 的计数，供统计使用。
    // 归还 buffer 给 capture 线程，成功返回 true，失败返回 false。
    bool qbuf_return(int index)
    {
        if (fd < 0 || index < 0) return false;

        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = (uint32_t)index;

        pthread_mutex_lock(&ioctl_mutex);
        int ret;
        do {
            ret = ioctl(fd, VIDIOC_QBUF, &buf);
        } while (ret < 0 && errno == EINTR);
        pthread_mutex_unlock(&ioctl_mutex);

        if (ret < 0) {
            return false;
        }

        qbuf_count.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
};

struct DetectResult {
    char name[16];
    float prop;
    int left;
    int top;
    int right;
    int bottom;
};

struct InferResult {
    DetectResult results[MAX_DETECTS];
    int count = 0;
    uint64_t seq = 0;
};

template<typename T, int N>
class RingQueue {
public:
    // Thread-safe bounded queue.
    // 容量为 N；push() 在满时返回 false，不修改队列。
    bool push(const T& item)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        if (count_ >= N) {
            drop_count_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        buf_[write_] = item;
        // write_ = (write_ + 1) % N;
        write_ = (write_ + 1) & (N - 1);
        count_++;
        return true;
    }

    // 满时丢弃最旧元素，并在锁外调用 release_old()。
    // 这用于 CaptureFrameRef 队列：被丢弃的旧帧必须归还 V4L2 buffer。
    template<typename Fn>
    bool push_drop_old(const T& item, Fn release_old)
    {
        T old{};
        bool has_old = false;

        {
            std::lock_guard<std::mutex> lk(mtx_);

            if (count_ >= N) {
                old = buf_[read_];
                // read_ = (read_ + 1) % N;
                read_ = (read_ + 1) & (N - 1);
                count_--;
                drop_count_.fetch_add(1, std::memory_order_relaxed);
                has_old = true;
            }

            buf_[write_] = item;
            // write_ = (write_ + 1) % N;
            write_ = (write_ + 1) & (N - 1);
            count_++;
        }

        // 注意：不要在队列锁内做 ioctl(QBUF)，否则可能拖慢 capture 线程。
        if (has_old) {
            release_old(old);
        }

        return true;
    }

    bool pop(T& item)
    {
        std::lock_guard<std::mutex> lk(mtx_);

        if (count_ == 0) {
            return false;
        }

        item = buf_[read_];
        // read_ = (read_ + 1) % N;
        read_ = (read_ + 1) & (N - 1);
        count_--;
        return true;
    }

    template<typename Fn>
    bool pop_latest(T& latest, Fn release_old)
    {
        T item{};
        bool got = false;

        while (pop(item)) {
            if (got) {
                release_old(latest);
            }

            latest = item;
            got = true;
        }

        return got;
    }

    uint64_t drop_count() const
    {
        return drop_count_.load(std::memory_order_relaxed);
    }

private:
    T buf_[N]{};
    mutable std::mutex mtx_;
    int read_{0};
    int write_{0};
    int count_{0};
    std::atomic<uint64_t> drop_count_{0};
};
