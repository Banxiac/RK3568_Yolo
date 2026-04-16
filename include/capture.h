#pragma once
#include "common.h"
#include <atomic>

void capture_thread(RingQueue<RawFrame, 4>& queue, std::atomic<bool>& running);
