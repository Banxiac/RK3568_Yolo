# edge_ai_pipeline

基于 Rockchip RK356X SoC（如 LubanCat 开发板）的实时边缘 AI 目标检测流水线。

从 V4L2 摄像头采集视频，在 Rockchip NPU 上运行 YOLOv5 推理，通过 MQTT 发布检测结果，并通过 HTTP 将标注视频流推送到浏览器。

## 架构

两个可执行文件通过 POSIX 共享内存通信：

```
摄像头 (V4L2)
    │
    ▼
capture_thread ──[RingQueue<RawFrame>]──▶ inference_thread ──[RingQueue<InferResult>]──▶ alarm_thread
                                                │                                              │
                                         POSIX 共享内存                                  MQTT 发布
                                         /edge_ai_frame                               edge/detect
                                                │
                                                ▼
                                          web_viewer
                                        （独立进程）
                                     HTTP :8080 / MJPEG / SSE
```

### `edge_ai_pipeline`（主进程）

| 线程 | 功能 |
|---|---|
| `capture_thread` | 通过 V4L2 从 `/dev/video9` 采集 640×480 YUYV 帧 |
| `inference_thread` | RGA 硬件加速 YUYV→RGB888 转换，RKNN YOLOv5 推理，结果写入共享内存 |
| `alarm_thread` | 将检测结果序列化为 JSON，发布到 MQTT |

### `web_viewer`（独立进程）

读取共享内存，订阅 MQTT，提供以下接口：

- `GET /` — 浏览器查看界面
- `GET /stream` — MJPEG 视频流（添加 `?boxes=1` 显示检测框）
- `GET /events` — SSE 实时推送检测 JSON

## 依赖

| 库 | 用途 |
|---|---|
| RKNN 运行时 (`librknnrt.so`) | 通过 rknpu2 驱动 NPU 推理 |
| RGA (`librga.so`) | 硬件图像格式转换与缩放 |
| OpenCV | MJPEG 流的图像编码 |
| libmosquitto | MQTT 客户端 |
| V4L2 | Linux 摄像头 API |

## 编译

交叉编译到 `aarch64`（需要 `aarch64-linux-gnu-g++` 和 rknpu2 SDK）：

```bash
./build.sh
```

编译产物（二进制文件及 `.so` 依赖）输出到 `install/` 目录。

## 部署

将 `install/` 目录复制到目标开发板，然后：

```bash
# 启动推理流水线
./edge_ai_pipeline <model.rknn>

# 启动 Web 查看器（另开终端或后台运行）
./web_viewer

# 浏览器访问
http://<开发板IP>:8080
```

## 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| 摄像头设备 | `/dev/video9` | V4L2 设备节点 |
| 帧分辨率 | 640×480 | YUYV 采集分辨率 |
| MQTT 主题 | `edge/detect` | 检测结果发布主题 |
| HTTP 端口 | `8080` | Web 查看器端口 |
| 共享内存名 | `/edge_ai_frame` | 进程间通信段名称 |

## 设计说明

- 队列满时**丢弃**帧而非阻塞，优先保证数据新鲜度。
- `web_viewer` 与推理流水线完全解耦，可独立启停。
- RingQueue 为无锁实现（单生产者/单消费者）。

---

## 性能分析

### 可观测指标

| 指标 | 位置 |
|---|---|
| 推理帧率（FPS） | web_viewer 前端 JS 实时计算 |
| 端到端延迟 | 摄像头出帧 → MQTT 发布 |
| 丢帧率 | `RingQueue::push` 返回 false 的比例（当前无计数） |
| RGA 转换耗时 | `improcess` 单次耗时 |
| NPU 利用率 | `/sys/kernel/debug/rknpu/load` |

### 当前方案优势

- **架构简单**：三线程两队列，数据流向单一，易于理解和维护。
- **丢帧不阻塞**：队列满时静默丢帧，推理线程不会被采集或发布侧反压，系统自然降级。
- **进程隔离**：`web_viewer` 崩溃不影响推理主进程；共享内存只读挂载，无法破坏推理侧数据。
- **RGA 硬件加速**：YUYV→RGB888 + resize 走 RGA IP，不占 A55 核心。

### 当前方案劣势

- **每帧 3 次大块 memcpy**：614KB × 3 ≈ 1.8MB/帧，30fps 下约 54MB/s 纯 CPU 搬运，在内存带宽受限的 RK356X 上挤占 NPU/RGA 配额。
- **丢帧无可观测性**：`RingQueue` 满时静默丢弃，无计数无日志，无法判断瓶颈在采集侧还是推理侧。
- **共享内存竞态窗口**：`detect_count`/`detects[]` 写入与 `seq` 递增之间无原子保护，`web_viewer` 可能读到帧序号与检测框数量不一致的中间状态。
- **MJPEG 重复编码**：每个 `/stream` 连接独立调用 `cv::imencode`，多客户端时 CPU 开销线性增长。
- **alarm 线程轮询**：空队列时每 5ms 轮询一次，引入最多 5ms 延迟抖动，并浪费 CPU 时间片。

### 优化方向与取舍

| 优先级 | 方向 | 收益 | 代价 |
|---|---|---|---|
| 1 | **seqlock 修复共享内存竞态** | 消除检测框读写竞态，正确性保证 | 改动约 10 行，几乎无代价 |
| 2 | **丢帧计数可观测性** | 不改架构，先摸清瓶颈在哪 | 极小 |
| 3 | **减少 memcpy（V4L2 缓冲区直传 RGA）** | 消除最大一次拷贝（614KB/帧） | 破坏"无锁值传递"简洁性，需引用计数或双缓冲 |
| 4 | **RKNN 异步流水线** | NPU 利用率从 ~50% 提升到接近 100% | 缓冲区生命周期复杂，多 context 占用双倍模型内存；应在确认 NPU 是瓶颈后再做 |
| 5 | **RGA DMA-buf 零拷贝** | V4L2→RGA 完全绕过 CPU | 依赖驱动版本，BSP 内核不一定支持，调试难度大 |
| 6 | **MJPEG 共享编码** | 多客户端时编码开销从 O(n) 降为 O(1) | 单客户端无收益；需扩展共享内存结构 |
| 7 | **条件变量替换 alarm 轮询** | 消除 5ms 延迟抖动 | 需给 `RingQueue` 加通知机制，破坏无锁设计 |

> 方向 3/4/5 应按顺序推进：先减少内存带宽压力，再评估 NPU 是否仍是瓶颈，最后考虑 DMA-buf。跳过前序直接做流水线可能掩盖真正瓶颈。


