#include "capture.h"

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <linux/videodev2.h>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>

static int xioctl(int fd, unsigned long req, void* arg)
{
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static void bind_current_thread_to_cpu(int cpu, const char* name)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);

    int ret = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (ret == 0) {
        fprintf(stderr, "[affinity] bind %s to CPU%d ok\n", name, cpu);
    } else {
        fprintf(stderr, "[affinity] bind %s to CPU%d failed ret=%d\n", name, cpu, ret);
    }

#if defined(__linux__)
    if (name) {
        pthread_setname_np(pthread_self(), name);
    }
#endif
}

static const char* fourcc_to_str(uint32_t fmt, char out[5])
{
    out[0] = fmt & 0xff;
    out[1] = (fmt >> 8) & 0xff;
    out[2] = (fmt >> 16) & 0xff;
    out[3] = (fmt >> 24) & 0xff;
    out[4] = '\0';
    return out;
}


struct JpegRange {
    size_t offset = 0;
    size_t size = 0;
    bool has_eoi = false;
};

// 更宽松的 MJPEG 判断：
// 1. 查找 SOI: ff d8
// 2. 如果能找到 EOI: ff d9，就裁剪到 EOI
// 3. 如果找不到 EOI，但找到 SOI，则直接使用 bytesused - soi
//
// 原因：部分 UVC 摄像头在 MJPEG buffer 中可能存在 padding，或者尾部不稳定。
// 如果 capture 线程强制要求 EOI，容易误杀可由 MPP 解码的帧。
static JpegRange find_jpeg_range_relaxed(const uint8_t* data, size_t size)
{
    JpegRange r{};

    if (!data || size < 4) return r;

    size_t soi = size;

    for (size_t i = 0; i + 1 < size; ++i) {
        if (data[i] == 0xff && data[i + 1] == 0xd8) {
            soi = i;
            break;
        }
    }

    if (soi == size) {
        return r;
    }

    // 优先寻找 EOI。如果找不到，不认为无效。
    for (size_t i = size - 2; i > soi; --i) {
        if (data[i] == 0xff && data[i + 1] == 0xd9) {
            r.offset = soi;
            r.size = i + 2 - soi;
            r.has_eoi = true;
            return r;
        }
    }

    r.offset = soi;
    r.size = size - soi;
    r.has_eoi = false;
    return r;
}


// V4L2Buffer: 存储 capture 线程的 V4L2 buffer 信息，包括 mmap 地址和 dma_fd（如果支持）。
struct V4L2Buffer {
    void* start = nullptr;
    size_t len = 0;
    int dma_fd = -1;
};

void capture_thread(RingQueue<CaptureFrameRef, CAP_REF_QUEUE_SIZE>& queue,
                    V4L2Shared& v4l2_shared,
                    std::atomic<bool>& running,
                    const AppConfig& cfg)
{
    bind_current_thread_to_cpu(0, "capture");// 绑定到 CPU0，避免与 inference 线程争抢 CPU。

    const char* dev = cfg.video_device;
    const int W = cfg.capture_width;
    const int H = cfg.capture_height;
    const int NBUF = 8;// 请求 8 个 V4L2 buffer，实际使用时如果设备分配的不足 8 个，也能正常工作。

    V4L2Buffer bufs[NBUF]{};// 存储 V4L2 buffer 的 mmap 地址和 dma_fd。

    int fd = open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        perror("[capture] open");
        return;
    }

    v4l2_shared.fd = fd;// 将 V4L2 设备 fd 存入共享结构，供 inference 线程归还 buffer 时使用。

    v4l2_capability cap{};
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
        fprintf(stderr,
                "[capture] driver=%s card=%s bus=%s\n",
                cap.driver,
                cap.card,
                cap.bus_info);
    }

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = W;
    fmt.fmt.pix.height = H;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("[capture] VIDIOC_S_FMT MJPEG");
        close(fd);
        v4l2_shared.fd = -1;
        return;
    }

    char fourcc[5];// 将像素格式转换为字符串，便于日志输出。
    fprintf(stderr,
            "[capture] actual format=%s size=%ux%u sizeimage=%u\n",
            fourcc_to_str(fmt.fmt.pix.pixelformat, fourcc),
            fmt.fmt.pix.width,
            fmt.fmt.pix.height,
            fmt.fmt.pix.sizeimage);

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
        fprintf(stderr, "[capture] device does not support MJPEG\n");
        close(fd);
        v4l2_shared.fd = -1;
        return;
    }

    int cap_w = (int)fmt.fmt.pix.width;
    int cap_h = (int)fmt.fmt.pix.height;

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;// 设置帧率为 30fps，部分设备需要明确设置帧率才能正常工作。
    parm.parm.capture.timeperframe.denominator = 30;
    xioctl(fd, VIDIOC_S_PARM, &parm);

    v4l2_requestbuffers req{};
    req.count = NBUF;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("[capture] VIDIOC_REQBUFS");
        close(fd);
        v4l2_shared.fd = -1;
        return;
    }

    int nbuf = req.count < NBUF ? req.count : NBUF;
    fprintf(stderr, "[capture] requested buffers=%d actual=%d\n", NBUF, nbuf);

    if (nbuf <= 0) {
        fprintf(stderr, "[capture] no V4L2 buffers allocated\n");
        close(fd);
        v4l2_shared.fd = -1;
        return;
    }

    bool stream_on = false;
    // 查询并 mmap 每个 buffer，准备好后放入队列。
    for (int i = 0; i < nbuf; i++) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("[capture] VIDIOC_QUERYBUF");
            running.store(false);
            goto cleanup;
        }

        bufs[i].len = buf.length;
        bufs[i].start = mmap(nullptr,
                             buf.length,
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED,
                             fd,
                             buf.m.offset);

        if (bufs[i].start == MAP_FAILED) {
            perror("[capture] mmap");
            bufs[i].start = nullptr;
            running.store(false);
            goto cleanup;
        }

        v4l2_exportbuffer expbuf{};// 查询 dma_fd，供 inference 线程使用零拷贝方式解码。
        expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        expbuf.index = i;
        expbuf.plane = 0;// 对于单平面格式，plane 固定为 0。
        expbuf.flags = O_CLOEXEC;// 设置 O_CLOEXEC，避免子进程继承 fd 导致资源泄漏。

        if (xioctl(fd, VIDIOC_EXPBUF, &expbuf) == 0) {
            bufs[i].dma_fd = expbuf.fd;
        } else {
            bufs[i].dma_fd = -1;
            perror("[capture] VIDIOC_EXPBUF");
        }

        fprintf(stderr,
                "[capture] buffer %d mmap=%p len=%zu dma_fd=%d\n",
                i,
                bufs[i].start,
                bufs[i].len,
                bufs[i].dma_fd);

        if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            perror("[capture] VIDIOC_QBUF init");
            running.store(false);
            goto cleanup;
        }
    }

    {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) {
            perror("[capture] VIDIOC_STREAMON");
            running.store(false);
            goto cleanup;
        }
        stream_on = true;
    }

    fprintf(stderr, "[capture] started: DQBUF -> CaptureFrameRef queue, no MJPEG memcpy\n");

    {
        uint64_t seq = 0;
        uint64_t timeout_count = 0;
        uint64_t no_eoi_count = 0;

        while (running.load()) {
            fd_set fds;
            FD_ZERO(&fds);// 将 V4L2 设备 fd 加入 select 的 fd_set，等待 buffer 可用。
            FD_SET(fd, &fds);

            timeval tv{};
            tv.tv_sec = 2;
            tv.tv_usec = 0;

            int sel = select(fd + 1, &fds, nullptr, nullptr, &tv);// 等待 V4L2 buffer 可用，超时则打印统计信息。
            if (sel < 0) {
                if (errno == EINTR) continue;
                perror("[capture] select");
                break;
            }

            if (sel == 0) {
                timeout_count++;

                uint64_t dq = v4l2_shared.dq_count.load(std::memory_order_relaxed);
                uint64_t qbuf = v4l2_shared.qbuf_count.load(std::memory_order_relaxed);
                uint64_t invalid = v4l2_shared.invalid_jpeg.load(std::memory_order_relaxed);
                uint64_t ref_full = v4l2_shared.ref_queue_full.load(std::memory_order_relaxed);

                fprintf(stderr,
                        "[capture] select timeout count=%lu dq=%lu qbuf=%lu diff=%ld invalid=%lu ref_full=%lu\n",
                        timeout_count,
                        dq,
                        qbuf,
                        (long)dq - (long)qbuf,
                        invalid,
                        ref_full);
                continue;
            }

            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            pthread_mutex_lock(&v4l2_shared.ioctl_mutex);// DQBUF 需要加锁，确保与 inference 线程中的 QBUF 不会并发调用 ioctl 导致冲突。
            int ret = xioctl(fd, VIDIOC_DQBUF, &buf);// DQBUF 可能会返回 EAGAIN，表示没有可用的 buffer，这时不认为是错误，直接继续等待。
            pthread_mutex_unlock(&v4l2_shared.ioctl_mutex);//

            if (ret < 0) {
                if (errno == EAGAIN) continue;
                perror("[capture] VIDIOC_DQBUF");
                break;
            }

            v4l2_shared.dq_count.fetch_add(1, std::memory_order_relaxed);// 成功 DQBUF 后增加计数，供统计使用。

            if (buf.index >= (uint32_t)nbuf) {
                fprintf(stderr, "[capture] invalid buffer index=%u\n", buf.index);// 归还 buffer 后继续循环，避免丢帧。
                v4l2_shared.qbuf_return((int)buf.index);
                continue;
            }

            // 通过 buf.index 获取 buffer 的 mmap 地址和 bytesused，进行基本的有效性检查，确保 JPEG 数据合法，避免将无效数据传递给 inference 线程导致解码失败。
            uint8_t* base = (uint8_t*)bufs[buf.index].start;// buffer 的 mmap 地址，供 inference 线程使用零拷贝方式解码。
            uint32_t used = buf.bytesused;

            if (!base || used < 4 || used > bufs[buf.index].len) {
                v4l2_shared.invalid_jpeg.fetch_add(1, std::memory_order_relaxed);// 无效 buffer 计数增加，供统计使用。

                uint64_t invalid = v4l2_shared.invalid_jpeg.load(std::memory_order_relaxed);
                if (invalid <= 10) {
                    fprintf(stderr,
                            "[capture] invalid buffer idx=%u used=%u len=%zu base=%p\n",
                            buf.index,
                            used,
                            bufs[buf.index].len,
                            base);
                }

                v4l2_shared.qbuf_return((int)buf.index);// 归还 buffer 后继续循环，避免丢帧。
                continue;
            }

            JpegRange jpg = find_jpeg_range_relaxed(base, used);
            if (jpg.size == 0) {
                v4l2_shared.invalid_jpeg.fetch_add(1, std::memory_order_relaxed);

                uint64_t invalid = v4l2_shared.invalid_jpeg.load(std::memory_order_relaxed);
                if (invalid <= 20) {
                    fprintf(stderr,
                            "[capture] invalid jpeg idx=%u used=%u first=%02x %02x %02x %02x last=%02x %02x\n",
                            buf.index,
                            used,
                            used > 0 ? base[0] : 0,
                            used > 1 ? base[1] : 0,
                            used > 2 ? base[2] : 0,
                            used > 3 ? base[3] : 0,
                            used > 1 ? base[used - 2] : 0,
                            used > 0 ? base[used - 1] : 0);
                }

                v4l2_shared.qbuf_return((int)buf.index);
                continue;
            }

            if (!jpg.has_eoi) {
                no_eoi_count++;
                if (no_eoi_count <= 10) {
                    fprintf(stderr,
                            "[capture] jpeg without EOI idx=%u used=%u offset=%zu size=%zu first=%02x %02x\n",
                            buf.index,
                            used,
                            jpg.offset,
                            jpg.size,
                            base[jpg.offset],
                            base[jpg.offset + 1]);
                }
            }

            CaptureFrameRef ref{};// 构造 CaptureFrameRef，包含 buffer index、dma_fd、mmap 地址和 JPEG 信息，供 inference_thread 使用零拷贝方式解码。
            ref.index = (int)buf.index;
            ref.dma_fd = bufs[buf.index].dma_fd;
            ref.data = base + jpg.offset;// JPEG 数据的 mmap 地址，供 inference_thread 直接解码使用。
            ref.bytesused = used;
            ref.jpg_size = (uint32_t)jpg.size;
            ref.jpg_offset = jpg.offset;
            ref.width = cap_w;
            ref.height = cap_h;
            ref.seq = seq++;

            // 队列满时在 RingQueue 内部原子化丢弃最旧帧，避免 capture 线程直接 pop()
            // 与 inference 线程的 pop_latest() 并发修改队列读指针。
            queue.push_drop_old(ref, [&](const CaptureFrameRef& old_ref) {
                v4l2_shared.ref_queue_full.fetch_add(1, std::memory_order_relaxed);
                v4l2_shared.qbuf_return(old_ref.index);
            });
        }
    }

cleanup:
    fprintf(stderr, "[capture] cleanup\n");

    if (stream_on) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd, VIDIOC_STREAMOFF, &type);
    }

    for (int i = 0; i < nbuf; i++) {
        if (bufs[i].start) {
            munmap(bufs[i].start, bufs[i].len);
            bufs[i].start = nullptr;
        }

        if (bufs[i].dma_fd >= 0) {
            close(bufs[i].dma_fd);
            bufs[i].dma_fd = -1;
        }
    }

    close(fd);
    v4l2_shared.fd = -1;

    running.store(false);

    fprintf(stderr,
            "[capture] exit dq=%lu qbuf=%lu diff=%ld ref_full=%lu invalid_jpeg=%lu\n",
            v4l2_shared.dq_count.load(std::memory_order_relaxed),
            v4l2_shared.qbuf_count.load(std::memory_order_relaxed),
            (long)v4l2_shared.dq_count.load(std::memory_order_relaxed) -
                (long)v4l2_shared.qbuf_count.load(std::memory_order_relaxed),
            v4l2_shared.ref_queue_full.load(std::memory_order_relaxed),
            v4l2_shared.invalid_jpeg.load(std::memory_order_relaxed));
}
