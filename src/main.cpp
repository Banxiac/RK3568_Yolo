#include "capture.h"
#include "inference.h"
#include "alarm.h"
#include <thread>
#include <atomic>
#include <csignal>
#include <cstdio>

static std::atomic<bool> running{true};

static void sig_handler(int) { running = false; }

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.rknn> [mqtt_host] [mqtt_port] [topic]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* mqtt_host  = argc > 2 ? argv[2] : "127.0.0.1";
    int         mqtt_port  = argc > 3 ? atoi(argv[3]) : 1883;
    const char* topic      = argc > 4 ? argv[4] : "edge/detect";

    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    RingQueue<RawFrame,    4> cap_queue;
    RingQueue<InferResult, 4> infer_queue;

    std::thread t_cap([&]{ capture_thread(cap_queue, running); });
    std::thread t_inf([&]{ inference_thread(cap_queue, infer_queue, running, model_path); });
    std::thread t_alm([&]{ alarm_thread(infer_queue, running, mqtt_host, mqtt_port, topic); });

    t_cap.join();
    t_inf.join();
    t_alm.join();
    return 0;
}
