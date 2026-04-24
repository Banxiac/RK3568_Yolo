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


