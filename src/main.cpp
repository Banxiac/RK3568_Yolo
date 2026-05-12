#include "capture.h"
#include "inference.h"
#include "alarm.h"
#include "config.h"

#include <thread>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <memory>

static std::atomic<bool> running{true};
static void sig_handler(int) { running = false; }

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.rknn> [-c config.ini]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];

    AppConfig cfg;
    for (int i = 2; i < argc - 1; i++) {
        if (!strcmp(argv[i], "-c")) {
            if (!load_config(argv[i + 1], cfg)) {
                fprintf(stderr,
                        "Warning: cannot open config '%s', using defaults\n",
                        argv[i + 1]);
            }
            i++;
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    auto cap_queue = std::make_unique<RingQueue<CaptureFrameRef, CAP_REF_QUEUE_SIZE>>();// 生产者线程直接放入 CaptureFrameRef，避免冗余的 MJPEG memcpy。
    auto infer_queue = std::make_unique<RingQueue<InferResult, INFER_RESULT_QUEUE_SIZE>>();// 推理线程直接输出 InferResult，避免冗余的 DetectResult memcpy。

    V4L2Shared v4l2_shared;

    std::thread t_cap([&] {
        capture_thread(*cap_queue, v4l2_shared, running, cfg);
    });

    std::thread t_inf([&] {
        inference_thread(*cap_queue, v4l2_shared, *infer_queue, running, model_path);
    });

    std::thread t_alm([&] {
        alarm_thread(*infer_queue,
                     running,
                     cfg.mqtt_host,
                     cfg.mqtt_port,
                     cfg.mqtt_topic);
    });

    t_cap.join();

    running.store(false);

    t_inf.join();
    t_alm.join();

    // 归还 capture 线程可能遗留在队列中的 CaptureFrameRef。
    CaptureFrameRef ref{};
    while (cap_queue->pop(ref)) {
        v4l2_shared.qbuf_return(ref.index);
    }

    return 0;
}
