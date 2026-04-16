#pragma once
#include "common.h"
#include <atomic>

void inference_thread(RingQueue<RawFrame, 4>& in_queue,
                      RingQueue<InferResult, 4>& out_queue,
                      std::atomic<bool>& running,
                      const char* model_path);
