#pragma once

#include "common.h"
#include <atomic>

void inference_thread(RingQueue<CaptureFrameRef, 4>& in_queue,
                      V4L2Shared& v4l2_shared,
                      RingQueue<InferResult, 4>& out_queue,
                      std::atomic<bool>& running,
                      const char* model_path);
