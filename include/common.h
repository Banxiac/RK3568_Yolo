#pragma once
#include <cstdint>
#include <cstring>
#include <atomic>
#include <mutex>
#include <condition_variable>

// 共享内存名称与结构（主进程写，网页进程读）
#define SHM_NAME "/edge_ai_frame"
#define SHM_WIDTH  640
#define SHM_HEIGHT 480

struct SharedDetect {
    char name[16];
    float prop;
    int left, top, right, bottom;
};

struct SharedFrame {
    std::atomic<uint64_t> seq{0};       // 写入序号，读者用于检测新帧
    uint8_t yuyv[SHM_WIDTH * SHM_HEIGHT * 2];
    // 最新推理结果（供web_viewer画框）
    SharedDetect detects[64];
    int detect_count{0};
};

// 采集帧：YUYV原始数据
struct RawFrame {
    uint8_t data[640 * 480 * 2];  // YUYV
    int width = 640;
    int height = 480;
    uint64_t seq = 0;
};

// 推理结果帧
struct DetectResult {
    char name[16];
    float prop;
    int left, top, right, bottom;
};

struct InferResult {
    DetectResult results[64];
    int count = 0;
    uint64_t seq = 0;
};

// 固定容量环形队列（单生产者单消费者，无锁）
template<typename T, int N>
class RingQueue {
public:
    bool push(const T& item) {
        int next = (write_ + 1) % N;
        if (next == read_.load(std::memory_order_acquire)) return false; // full, drop
        buf_[write_] = item;
        write_.store(next, std::memory_order_release);
        return true;
    }
    bool pop(T& item) {
        int r = read_.load(std::memory_order_relaxed);
        if (r == write_.load(std::memory_order_acquire)) return false; // empty
        item = buf_[r];
        read_.store((r + 1) % N, std::memory_order_release);
        return true;
    }
private:
    T buf_[N];
    std::atomic<int> read_{0}, write_{0};
};
