#pragma once

#include "common.h"
#include "config.h"
#include <atomic>

void capture_thread(RingQueue<CaptureFrameRef, CAP_REF_QUEUE_SIZE>& queue,
                    V4L2Shared& v4l2_shared,
                    std::atomic<bool>& running,
                    const AppConfig& cfg);
