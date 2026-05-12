#include "inference.h"
#include "common.h"
#include "postprocess.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"
#include "rknn_api.h"

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_buffer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#define STREAM_WIDTH  640
#define STREAM_HEIGHT 480

#ifndef MAX_DETECTS
#define MAX_DETECTS 64
#endif

#define WEB_FRAME_INTERVAL 2
#define STATS_INTERVAL_US 5000000LL

static inline int64_t now_us()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static inline int align16(int x)
{
    return (x + 15) & ~15;
}

static void bind_current_thread_to_cpu(int cpu_id, const char* name)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "[affinity] bind %s to CPU%d failed ret=%d\n", name, cpu_id, ret);
    } else {
        fprintf(stderr, "[affinity] bind %s to CPU%d ok\n", name, cpu_id);
    }

#if defined(__linux__)
    if (name) {
        pthread_setname_np(pthread_self(), name);
    }
#endif
}

static void bind_current_thread_to_cpu_mask(int cpu0, int cpu1, const char* name)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu0, &cpuset);
    CPU_SET(cpu1, &cpuset);

    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "[affinity] bind %s to CPU%d,%d failed ret=%d\n", name, cpu0, cpu1, ret);
    } else {
        fprintf(stderr, "[affinity] bind %s to CPU%d,%d ok\n", name, cpu0, cpu1);
    }

#if defined(__linux__)
    if (name) {
        pthread_setname_np(pthread_self(), name);
    }
#endif
}

static unsigned char* load_model(const char* path, int* size)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Open model %s failed\n", path);
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    long fsz = ftell(fp);
    rewind(fp);

    if (fsz <= 0) {
        fclose(fp);
        return nullptr;
    }

    unsigned char* buf = (unsigned char*)malloc((size_t)fsz);
    if (!buf) {
        fclose(fp);
        return nullptr;
    }

    if (fread(buf, 1, (size_t)fsz, fp) != (size_t)fsz) {
        free(buf);
        fclose(fp);
        return nullptr;
    }

    fclose(fp);
    *size = (int)fsz;
    return buf;
}

static void dump_tensor_attr(const char* tag, const rknn_tensor_attr& attr)
{
    fprintf(stderr,
            "%s index=%d name=%s n_dims=%d dims=[%d,%d,%d,%d] n_elems=%d size=%d "
            "fmt=%d type=%d qnt_type=%d zp=%d scale=%f\n",
            tag,
            attr.index,
            attr.name,
            attr.n_dims,
            attr.dims[0],
            attr.dims[1],
            attr.dims[2],
            attr.dims[3],
            attr.n_elems,
            attr.size,
            attr.fmt,
            attr.type,
            attr.qnt_type,
            attr.zp,
            attr.scale);
}

static int get_output_grid_area(const rknn_tensor_attr& attr)
{
    int area = 1;
    int used = 0;

    for (int i = 0; i < attr.n_dims; ++i) {
        int d = attr.dims[i];
        if (d > 1 && d <= 256) {
            area *= d;
            if (++used == 2) break;
        }
    }

    return area;
}

static const char* mpp_fmt_name(MppFrameFormat fmt)
{
    switch (fmt) {
    case MPP_FMT_YUV420SP:
        return "MPP_FMT_YUV420SP/NV12";
    case MPP_FMT_YUV420SP_VU:
        return "MPP_FMT_YUV420SP_VU/NV21";
    case MPP_FMT_YUV422SP:
        return "MPP_FMT_YUV422SP/NV16";
    case MPP_FMT_YUV422SP_VU:
        return "MPP_FMT_YUV422SP_VU/NV61";
    default:
        return "UNKNOWN";
    }
}

struct MppDecCtx {
    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    MppDecCfg cfg = nullptr;
    MppBufferGroup grp = nullptr;
    MppBuffer out_buf = nullptr;
    MppFrame out_frame = nullptr;

    int width = 0;
    int height = 0;
    int hor_stride = 0;
    int ver_stride = 0;

    uint64_t pkt_copy_count = 0;
    uint64_t pkt_copy_bytes = 0;
    int64_t pkt_copy_us = 0;

    bool init(int w, int h)
    {
        width = w;
        height = h;
        hor_stride = align16(w);
        ver_stride = align16(h);

        size_t buf_sz = (size_t)hor_stride * ver_stride * 4;

        MPP_RET ret = mpp_create(&ctx, &mpi);
        if (ret != MPP_OK) {
            fprintf(stderr, "[mpp] mpp_create failed ret=%d\n", ret);
            return false;
        }

        ret = mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG);
        if (ret != MPP_OK) {
            fprintf(stderr, "[mpp] mpp_init MJPEG failed ret=%d\n", ret);
            return false;
        }

        ret = mpp_dec_cfg_init(&cfg);
        if (ret != MPP_OK) {
            fprintf(stderr, "[mpp] mpp_dec_cfg_init failed ret=%d\n", ret);
            return false;
        }

        ret = mpi->control(ctx, MPP_DEC_GET_CFG, cfg);
        if (ret != MPP_OK) {
            fprintf(stderr, "[mpp] MPP_DEC_GET_CFG failed ret=%d\n", ret);
            return false;
        }

        RK_U32 need_split = 1;
        ret = mpp_dec_cfg_set_u32(cfg, "base:split_parse", need_split);
        if (ret != MPP_OK) {
            fprintf(stderr, "[mpp] set base:split_parse failed ret=%d\n", ret);
            return false;
        }

        ret = mpi->control(ctx, MPP_DEC_SET_CFG, cfg);
        if (ret != MPP_OK) {
            fprintf(stderr, "[mpp] MPP_DEC_SET_CFG failed ret=%d\n", ret);
            return false;
        }

        ret = mpp_buffer_group_get_internal(&grp, MPP_BUFFER_TYPE_DRM);
        if (ret != MPP_OK || !grp) {
            fprintf(stderr, "[mpp] mpp_buffer_group_get_internal failed ret=%d\n", ret);
            return false;
        }

        ret = mpp_buffer_get(grp, &out_buf, buf_sz);
        if (ret != MPP_OK || !out_buf) {
            fprintf(stderr, "[mpp] mpp_buffer_get output failed ret=%d size=%zu\n", ret, buf_sz);
            return false;
        }

        ret = mpp_frame_init(&out_frame);
        if (ret != MPP_OK || !out_frame) {
            fprintf(stderr, "[mpp] mpp_frame_init output failed ret=%d\n", ret);
            return false;
        }

        mpp_frame_set_buffer(out_frame, out_buf);

        fprintf(stderr,
                "[mpp] decoder ready output buffer=%zu stride=%dx%d\n",
                buf_sz,
                hor_stride,
                ver_stride);

        return true;
    }

    bool decode(const uint8_t* jpg, uint32_t jpg_size)
    {
        if (!jpg || jpg_size <= 4) return false;

        MppBuffer pkt_buf = nullptr;
        MPP_RET ret = mpp_buffer_get(nullptr, &pkt_buf, jpg_size);
        if (ret != MPP_OK || !pkt_buf) {
            fprintf(stderr, "[mpp] mpp_buffer_get packet failed ret=%d size=%u\n", ret, jpg_size);
            return false;
        }

        void* pkt_ptr = mpp_buffer_get_ptr(pkt_buf);
        if (!pkt_ptr) {
            fprintf(stderr, "[mpp] packet buffer ptr is null\n");
            mpp_buffer_put(pkt_buf);
            return false;
        }

        int64_t t_copy0 = now_us();
        memcpy(pkt_ptr, jpg, jpg_size);
        int64_t t_copy1 = now_us();

        pkt_copy_count++;
        pkt_copy_bytes += jpg_size;
        pkt_copy_us += t_copy1 - t_copy0;

        MppPacket packet = nullptr;
        ret = mpp_packet_init_with_buffer(&packet, pkt_buf);
        if (ret != MPP_OK || !packet) {
            fprintf(stderr, "[mpp] mpp_packet_init_with_buffer failed ret=%d\n", ret);
            mpp_buffer_put(pkt_buf);
            return false;
        }

        mpp_packet_set_pos(packet, pkt_ptr);
        mpp_packet_set_length(packet, jpg_size);

        MppMeta meta = mpp_packet_get_meta(packet);
        if (!meta) {
            fprintf(stderr, "[mpp] mpp_packet_get_meta failed\n");
            mpp_packet_deinit(&packet);
            mpp_buffer_put(pkt_buf);
            return false;
        }

        ret = mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, out_frame);
        if (ret != MPP_OK) {
            fprintf(stderr, "[mpp] mpp_meta_set_frame KEY_OUTPUT_FRAME failed ret=%d\n", ret);
            mpp_packet_deinit(&packet);
            mpp_buffer_put(pkt_buf);
            return false;
        }

        bool ok = false;

        ret = mpi->decode_put_packet(ctx, packet);
        if (ret == MPP_OK) {
            MppFrame frame_ret = nullptr;
            ret = mpi->decode_get_frame(ctx, &frame_ret);

            if (ret == MPP_OK && frame_ret) {
                int err = mpp_frame_get_errinfo(frame_ret);
                int discard = mpp_frame_get_discard(frame_ret);

                if (!err && !discard) {
                    ok = true;
                } else {
                    fprintf(stderr, "[mpp] bad frame err=%d discard=%d\n", err, discard);
                }

                if (frame_ret != out_frame) {
                    fprintf(stderr,
                            "[mpp] warning returned frame %p != out_frame %p\n",
                            frame_ret,
                            out_frame);
                    mpp_frame_deinit(&frame_ret);
                }
            } else {
                fprintf(stderr, "[mpp] decode_get_frame failed ret=%d frame=%p\n", ret, frame_ret);
            }
        } else {
            fprintf(stderr, "[mpp] decode_put_packet failed ret=%d jpg_size=%u\n", ret, jpg_size);
        }

        mpp_packet_deinit(&packet);
        mpp_buffer_put(pkt_buf);

        return ok;
    }

    void deinit()
    {
        if (out_frame) {
            mpp_frame_deinit(&out_frame);
            out_frame = nullptr;
        }

        if (out_buf) {
            mpp_buffer_put(out_buf);
            out_buf = nullptr;
        }

        if (grp) {
            mpp_buffer_group_put(grp);
            grp = nullptr;
        }

        if (cfg) {
            mpp_dec_cfg_deinit(cfg);
            cfg = nullptr;
        }

        if (ctx) {
            if (mpi) mpi->reset(ctx);
            mpp_destroy(ctx);
            ctx = nullptr;
            mpi = nullptr;
        }
    }
};

enum SlotState {
    SLOT_FREE = 0,
    SLOT_FILLING = 1,
    SLOT_READY = 2,
    SLOT_INFER = 3
};

struct PrepSlot {
    SlotState state = SLOT_FREE;
    uint64_t seq = 0;

    int dec_w = 0;
    int dec_h = 0;
    int hor_stride = 0;
    int ver_stride = 0;
    MppFrameFormat mpp_fmt = MPP_FMT_BUTT;

    void* rgb = nullptr;

    int64_t decode_us = 0;
    int64_t rga_us = 0;
    int64_t total_us = 0;
};

struct PrepBuffer {
    std::mutex mtx;// 保护 slots 的互斥锁，prep_thread 填充时锁定，infer_thread 读取时锁定
    std::condition_variable cv;// prep_thread 填充完一个 slot 后通知 infer_thread 可以读取
    PrepSlot slots[2];

    uint64_t produced = 0;
    uint64_t dropped_ready = 0;

    int choose_fill_slot_locked()
    {
        for (int i = 0; i < 2; ++i) {
            if (slots[i].state == SLOT_FREE) {
                return i;
            }
        }

        int oldest = -1;
        for (int i = 0; i < 2; ++i) {
            if (slots[i].state == SLOT_READY) {
                if (oldest < 0 || slots[i].seq < slots[oldest].seq) {
                    oldest = i;
                }
            }
        }

        if (oldest >= 0) {
            dropped_ready++;
            return oldest;
        }

        return -1;
    }

    int take_latest_ready_locked()
    {
        int latest = -1;
        for (int i = 0; i < 2; ++i) {
            if (slots[i].state == SLOT_READY) {
                if (latest < 0 || slots[i].seq > slots[latest].seq) {
                    latest = i;
                }
            }
        }
        return latest;
    }
};

// WebFrameSlot: 存原始 MJPEG 数据，供 web_thread 直接写入 shm
struct WebFrameSlot {
    std::vector<uint8_t> jpeg;
    int width{0};
    int height{0};
    uint64_t seq{0};
    bool valid{false};
    // 检测框由 infer_thread 填入
    SharedDetect detects[MAX_DETECTS];
    int detect_count{0};
    bool has_detects{false};
};

struct WebFrameBuf {
    std::mutex mtx;
    std::condition_variable cv;
    WebFrameSlot slot;
    bool updated{false};
};

static void publish_jpeg_to_shm(SharedFrame* shm,
                                 uint64_t frame_seq,
                                 int width,
                                 int height,
                                 const uint8_t* jpeg_data,
                                 uint32_t jpeg_size,
                                 const SharedDetect* detects,
                                 int detect_count)
{
    if (!shm || !jpeg_data || jpeg_size == 0) return;

    int cur = shm->active_index.load(std::memory_order_acquire);// 当前活跃的 slot 索引，0 或 1
    if (cur < 0 || cur > 1) cur = 0;
    int next = 1 - cur;

    SharedFrameSlot* next_slot = &shm->slots[next];// 下一个 slot，准备填入新数据
    /*
         读取当前 lock_seq，准备更新数据前先将 lock_seq 加 1 表示正在填充，其他线程看到 lock_seq 是奇数就知道这个 slot 正在被填充不应该读取
    */
    uint64_t s = next_slot->lock_seq.load(std::memory_order_relaxed);
    next_slot->lock_seq.store(s + 1, std::memory_order_release);

    next_slot->frame_seq = frame_seq;
    next_slot->width = width;
    next_slot->height = height;

    if (jpeg_size > MJPEG_MAX_BYTES) jpeg_size = MJPEG_MAX_BYTES;
    memcpy(next_slot->jpeg, jpeg_data, jpeg_size);
    next_slot->jpeg_size = jpeg_size;

    int cnt = detect_count;
    if (cnt < 0) cnt = 0;
    if (cnt > MAX_DETECTS) cnt = MAX_DETECTS;
    next_slot->detect_count = cnt;
    if (cnt > 0)
        memcpy(next_slot->detects, detects, cnt * sizeof(SharedDetect));

    next_slot->lock_seq.store(s + 2, std::memory_order_release);
    shm->active_index.store(next, std::memory_order_release);
    shm->global_seq.fetch_add(1, std::memory_order_release);
}

// 只更新检测框，不触碰图像数据；由 infer_thread 调用
static void publish_detections_to_shm(SharedFrame* shm,
                                      uint64_t frame_seq,
                                      int width,
                                      int height,
                                      const detect_result_group_t& grp)
{
    if (!shm) return;

    int cur = shm->active_index.load(std::memory_order_acquire);
    if (cur < 0 || cur > 1) cur = 0;
    int next = 1 - cur;

    SharedFrameSlot* next_slot = &shm->slots[next];

    // 读取当前 slot 的图像数据（JPEG）以便携带到 next slot
    SharedFrameSlot* cur_slot = &shm->slots[cur];

    uint64_t s = next_slot->lock_seq.load(std::memory_order_relaxed);
    next_slot->lock_seq.store(s + 1, std::memory_order_release);

    next_slot->frame_seq = frame_seq;
    next_slot->width = width > 0 ? width : cur_slot->width;
    next_slot->height = height > 0 ? height : cur_slot->height;

    // 携带当前 JPEG 图像
    uint32_t jsz = cur_slot->jpeg_size;
    if (jsz > MJPEG_MAX_BYTES) jsz = MJPEG_MAX_BYTES;
    next_slot->jpeg_size = jsz;
    if (jsz > 0)
        memcpy(next_slot->jpeg, cur_slot->jpeg, jsz);

    int count = grp.count;
    if (count < 0) count = 0;
    if (count > MAX_DETECTS) count = MAX_DETECTS;
    next_slot->detect_count = count;
    for (int i = 0; i < count; i++) {
        strncpy(next_slot->detects[i].name, grp.results[i].name, 15);
        next_slot->detects[i].name[15] = '\0';
        next_slot->detects[i].prop = grp.results[i].prop;
        next_slot->detects[i].left = grp.results[i].box.left;
        next_slot->detects[i].top = grp.results[i].box.top;
        next_slot->detects[i].right = grp.results[i].box.right;
        next_slot->detects[i].bottom = grp.results[i].box.bottom;
    }

    next_slot->lock_seq.store(s + 2, std::memory_order_release);
    shm->active_index.store(next, std::memory_order_release);
    shm->global_seq.fetch_add(1, std::memory_order_release);
}

void inference_thread(RingQueue<CaptureFrameRef, CAP_REF_QUEUE_SIZE>& in_queue,
                      V4L2Shared& v4l2_shared,
                      RingQueue<InferResult, INFER_RESULT_QUEUE_SIZE>& out_queue,
                      std::atomic<bool>& running,
                      const char* model_path)
{
    bind_current_thread_to_cpu_mask(2, 3, "infer");

    const int CTX_NUM = 2;

    int model_size = 0;
    unsigned char* model_data = load_model(model_path, &model_size);
    if (!model_data) return;

    rknn_context ctx[CTX_NUM]{};

    int ret = rknn_init(&ctx[0], model_data, model_size, 0, NULL);
    if (ret < 0) {
        fprintf(stderr, "[inference] rknn_init ctx0 failed ret=%d\n", ret);
        free(model_data);
        return;
    }

    ret = rknn_dup_context(&ctx[0], &ctx[1]);
    if (ret < 0) {
        fprintf(stderr, "[inference] rknn_dup_context ctx1 failed ret=%d\n", ret);
        free(model_data);
        rknn_destroy(ctx[0]);
        return;
    }

    free(model_data);

    fprintf(stderr,
            "[inference] v4l2-ref pipeline enabled, RKNN ctx0=%lu ctx1=%lu\n",
            (unsigned long)ctx[0],
            (unsigned long)ctx[1]);

    rknn_sdk_version sdk_ver{};
    if (rknn_query(ctx[0], RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver)) == RKNN_SUCC) {
        fprintf(stderr,
                "[inference] sdk version: %s driver version: %s\n",
                sdk_ver.api_version,
                sdk_ver.drv_version);
    }

    rknn_input_output_num io_num{};
    ret = rknn_query(ctx[0], RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0 || io_num.n_input < 1 || io_num.n_output < 3) {
        fprintf(stderr,
                "[inference] invalid io num ret=%d input=%d output=%d\n",
                ret,
                io_num.n_input,
                io_num.n_output);
        rknn_destroy(ctx[0]);
        rknn_destroy(ctx[1]);
        return;
    }

    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    for (int i = 0; i < (int)io_num.n_input; ++i) {
        input_attrs[i].index = i;
        rknn_query(ctx[0], RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        dump_tensor_attr("[inference] input", input_attrs[i]);
    }

    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    for (int i = 0; i < (int)io_num.n_output; ++i) {
        output_attrs[i].index = i;
        rknn_query(ctx[0], RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        dump_tensor_attr("[inference] output", output_attrs[i]);
    }

    int model_c = 3;
    int model_w = 0;
    int model_h = 0;

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        model_c = input_attrs[0].dims[1];
        model_h = input_attrs[0].dims[2];
        model_w = input_attrs[0].dims[3];
    } else {
        model_h = input_attrs[0].dims[1];
        model_w = input_attrs[0].dims[2];
        model_c = input_attrs[0].dims[3];
    }

    if (model_w <= 0 || model_h <= 0 || model_c != 3) {
        fprintf(stderr,
                "[inference] invalid input shape w=%d h=%d c=%d\n",
                model_w,
                model_h,
                model_c);
        rknn_destroy(ctx[0]);
        rknn_destroy(ctx[1]);
        return;
    }

    fprintf(stderr,
            "[inference] model input=%dx%dx%d outputs=%d\n",
            model_w,
            model_h,
            model_c,
            io_num.n_output);

    PrepBuffer prep_buf;// 2 slots for double buffering, prep_thread 写入，infer_thread 读取
    // 预分配输入张量大小的 RGB 内存，供 prep_thread 转换图像后直接写入，infer_thread 直接读取进行推理，避免每帧都 malloc/free 导致性能波动和内存碎片。
    for (int i = 0; i < 2; ++i) {
        prep_buf.slots[i].rgb = malloc((size_t)model_w * model_h * 3);
        if (!prep_buf.slots[i].rgb) {
            fprintf(stderr, "[inference] malloc prep rgb slot %d failed\n", i);
            for (int j = 0; j < i; ++j) free(prep_buf.slots[j].rgb);
            rknn_destroy(ctx[0]);
            rknn_destroy(ctx[1]);
            return;
        }
        fprintf(stderr, "[inference] prep slot%d rgb=%p\n", i, prep_buf.slots[i].rgb);
    }

    std::vector<rknn_output> outputs[CTX_NUM];
    for (int i = 0; i < CTX_NUM; ++i) {
        outputs[i].resize(io_num.n_output);
        memset(outputs[i].data(), 0, sizeof(rknn_output) * outputs[i].size());

        for (uint32_t j = 0; j < io_num.n_output; ++j) {
            outputs[i][j].want_float = 0;
        }
    }

    struct OutputSlot {
        int index;
        int area;
    };

    std::vector<OutputSlot> output_order;
    for (int i = 0; i < (int)io_num.n_output; ++i) {
        output_order.push_back({i, get_output_grid_area(output_attrs[i])});
    }

    std::sort(output_order.begin(), output_order.end(),
              [](const OutputSlot& a, const OutputSlot& b) {
                  return a.area > b.area;
              });

    std::vector<int32_t> out_zps;
    std::vector<float> out_scales;

    for (int i = 0; i < 3; ++i) {
        int idx = output_order[i].index;
        out_zps.push_back(output_attrs[idx].zp);
        out_scales.push_back(output_attrs[idx].scale);

        fprintf(stderr,
                "[inference] post output%d uses rknn output%d area=%d zp=%d scale=%f\n",
                i,
                idx,
                output_order[i].area,
                output_attrs[idx].zp,
                output_attrs[idx].scale);
    }

    shm_unlink(SHM_NAME);

    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        perror("[inference] shm_open");
        for (int i = 0; i < 2; ++i) free(prep_buf.slots[i].rgb);
        rknn_destroy(ctx[0]);
        rknn_destroy(ctx[1]);
        return;
    }
    // 设置共享内存大小为 SharedFrame 结构体的大小，供 web_thread 映射使用
    if (ftruncate(shm_fd, sizeof(SharedFrame)) != 0) {
        perror("[inference] ftruncate");
        close(shm_fd);
        shm_unlink(SHM_NAME);
        for (int i = 0; i < 2; ++i) free(prep_buf.slots[i].rgb);
        rknn_destroy(ctx[0]);
        rknn_destroy(ctx[1]);
        return;
    }

    SharedFrame* shm = (SharedFrame*)mmap(nullptr,
                                          sizeof(SharedFrame),
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED,
                                          shm_fd,
                                          0);

    if (shm == MAP_FAILED) {
        perror("[inference] mmap");
        close(shm_fd);
        shm_unlink(SHM_NAME);
        for (int i = 0; i < 2; ++i) free(prep_buf.slots[i].rgb);
        rknn_destroy(ctx[0]);
        rknn_destroy(ctx[1]);
        return;
    }

    new (shm) SharedFrame();// 在共享内存上构造 SharedFrame 对象，初始化原子变量等
    
    // web_thread 用于接收 web 线程写入的 MJPEG 数据，infer_thread 填充检测结果后供 web_thread 发送到 shm
    WebFrameBuf web_frame_buf;
    std::atomic<bool> prep_done{false};

    std::thread web_thread([&] {
        const int jpeg_quality = 70;// MJPEG 图像质量，范围 1-100，数值越大质量越好但文件越大，过低可能导致解码失败
        fprintf(stderr, "[web] thread started, async MJPEG->shm\n");

        while (running.load() || !prep_done.load()) {
        // 从 web_frame_buf 获取最新的 MJPEG 数据和检测结果，等待时间不超过 20ms，过短可能导致频繁轮询过多 CPU 占用，过长可能导致响应变慢
            WebFrameSlot snap;
            {
                std::unique_lock<std::mutex> lk(web_frame_buf.mtx);// 锁定 web_frame_buf 以读取最新数据，prep_thread 填充完数据后会通知 web_thread
                web_frame_buf.cv.wait_for(lk, std::chrono::milliseconds(20), [&] {
                    return web_frame_buf.updated || prep_done.load() || !running.load();
                });//等待 prep_thread 填充完数据后通知 web_thread，或者 prep_done 标志被设置，或者 running 被设置为 false 以退出线程
                if (!web_frame_buf.updated) continue;
                snap = std::move(web_frame_buf.slot);// 复制当前 slot 的数据到 snap，准备处理和发送到 shm
                web_frame_buf.updated = false;// 标记数据已被 web_thread 处理，等待 prep_thread 填充新数据
            }

            if (!snap.valid || snap.jpeg.empty()) continue;

            cv::Mat bgr = cv::imdecode(snap.jpeg, cv::IMREAD_COLOR);// 将 MJPEG 数据解码为 BGR 图像
            if (bgr.empty()) continue;

            cv::resize(bgr, bgr, cv::Size(STREAM_WIDTH, STREAM_HEIGHT));
            // 根据检测结果在图像上绘制检测框和标签，坐标需要根据原始图像尺寸和 STREAM_WIDTH/HEIGHT 进行缩放
            if (snap.has_detects && snap.detect_count > 0) {
                float sx = snap.width  > 0 ? (float)STREAM_WIDTH  / snap.width  : 1.0f;
                float sy = snap.height > 0 ? (float)STREAM_HEIGHT / snap.height : 1.0f;
                for (int i = 0; i < snap.detect_count; i++) {
                    const SharedDetect& d = snap.detects[i];
                    cv::rectangle(bgr,
                        cv::Point((int)(d.left*sx), (int)(d.top*sy)),
                        cv::Point((int)(d.right*sx), (int)(d.bottom*sy)),
                        cv::Scalar(0, 255, 0), 2);
                    char label[32];
                    snprintf(label, sizeof(label), "%s %.0f%%", d.name, d.prop * 100);
                    int ly = (int)(d.top * sy) - 4;
                    if (ly < 12) ly = 12;
                    cv::putText(bgr, label, cv::Point((int)(d.left*sx), ly),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                }
            }
            // 将处理后的 BGR 图像重新编码为 MJPEG 数据，并发送到 shm
            std::vector<uchar> out_jpeg;
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
            cv::imencode(".jpg", bgr, out_jpeg, params);

            publish_jpeg_to_shm(shm, snap.seq, STREAM_WIDTH, STREAM_HEIGHT,
                                out_jpeg.data(), (uint32_t)out_jpeg.size(),
                                snap.detects, snap.detect_count);
        }
        fprintf(stderr, "[web] thread exit\n");
    });

    std::thread prep_thread([&] {
        bind_current_thread_to_cpu(1, "prep");

        MppDecCtx mpp;
        bool mpp_ready = false;
        int last_w = 0;
        int last_h = 0;

        uint64_t popped = 0;
        uint64_t prepared = 0;
        uint64_t last_prepared = 0;
        uint64_t prep_fail = 0;
        uint64_t qbuf_after_decode = 0;

        int64_t sum_dec_us = 0;
        int64_t sum_rga_us = 0;
        int64_t sum_shm_us = 0;
        int64_t sum_total_us = 0;
        int64_t last_stats_time = now_us();

        CaptureFrameRef ref{};// 用于从 in_queue 获取 V4L2 捕获的 MJPEG 数据的引用，包含数据指针、大小、分辨率等信息

        fprintf(stderr, "[prep] thread started, consume CaptureFrameRef, no V4L2->RawFrame memcpy\n");

        while (running.load()) {
            /*
                从 in_queue 获取最新的 CaptureFrameRef，
                pop_latest 内部会自动丢弃旧的未处理的 ref，并将对应的 V4L2 buffer 返还给内核
            */ 
            bool got = in_queue.pop_latest(ref, [&](const CaptureFrameRef& old_ref) {
                v4l2_shared.qbuf_return(old_ref.index);
            });

            if (!got) {
                usleep(1000);
                continue;
            }

            popped++;// 统计从队列成功获取到的 CaptureFrameRef 数量
            bool ref_returned = false;
            // 用于返还 V4L2 buffer，确保无论解码成功与否都能正确返还，避免内核 buffer 泄漏
            auto return_ref = [&]() {
                if (!ref_returned) {
                    if (v4l2_shared.qbuf_return(ref.index)) {
                    qbuf_after_decode++;
                    }
                    ref_returned = true;
                }
            };
            if (!mpp_ready || ref.width != last_w || ref.height != last_h) {
                if (mpp_ready) mpp.deinit();

                if (!mpp.init(ref.width, ref.height)) {
                    fprintf(stderr, "[prep] MPP init failed for %dx%d\n", ref.width, ref.height);
                    // v4l2_shared.qbuf_return(ref.index);
                    return_ref();
                    prep_fail++;
                    continue;
                }

                mpp_ready = true;
                last_w = ref.width;
                last_h = ref.height;

                fprintf(stderr,
                        "[prep] MPP decoder ready for %dx%d\n",
                        last_w,
                        last_h);
            }

            int fill_slot = -1;
            /*
                选择一个 prep_buf slot 来填充新的解码数据，
                如果两个 slot 都在使用中（FILLING/READY/INFER），则选择一个 READY 状态的 slot 来覆盖掉
            */ 
            {
                std::unique_lock<std::mutex> lk(prep_buf.mtx);
                fill_slot = prep_buf.choose_fill_slot_locked();
                if (fill_slot < 0) {
                    // v4l2_shared.qbuf_return(ref.index);
                    return_ref();
                    prep_fail++;
                    continue;
                }
                prep_buf.slots[fill_slot].state = SLOT_FILLING;
            }

            int64_t t_total0 = now_us();
            int64_t t_dec0 = now_us();
            //进行MPP解码，输入 MJPEG 数据，输出 YUV buffer
            if (!mpp.decode((const uint8_t*)ref.data, ref.jpg_size)) {
                fprintf(stderr,
                        "[prep] MPP decode failed seq=%lu size=%u offset=%zu index=%d\n",
                        ref.seq,
                        ref.jpg_size,
                        ref.jpg_offset,
                        ref.index);

                // v4l2_shared.qbuf_return(ref.index);
                return_ref();

                {
                    std::unique_lock<std::mutex> lk(prep_buf.mtx);
                    prep_buf.slots[fill_slot].state = SLOT_FREE;
                }

                prep_fail++;
                continue;
            }
            int64_t t_dec1 = now_us();           
            // 获取解码后的 YUV buffer
            MppBuffer yuv_buf = mpp_frame_get_buffer(mpp.out_frame);
            void* yuv_ptr = yuv_buf ? mpp_buffer_get_ptr(yuv_buf) : nullptr;
            int dec_w = mpp_frame_get_width(mpp.out_frame);
            int dec_h = mpp_frame_get_height(mpp.out_frame);
            int hor_stride = mpp_frame_get_hor_stride(mpp.out_frame);
            int ver_stride = mpp_frame_get_ver_stride(mpp.out_frame);
            MppFrameFormat mpp_fmt = mpp_frame_get_fmt(mpp.out_frame);
            if (!yuv_ptr || dec_w <= 0 || dec_h <= 0 || hor_stride <= 0 || ver_stride <= 0) {
                fprintf(stderr,
                        "[prep] invalid decoded frame yuv=%p w=%d h=%d stride=%dx%d fmt=%d\n",
                        yuv_ptr,
                        dec_w,
                        dec_h,
                        hor_stride,
                        ver_stride,
                        mpp_fmt);
                return_ref();

                {
                    std::unique_lock<std::mutex> lk(prep_buf.mtx);
                    prep_buf.slots[fill_slot].state = SLOT_FREE;
                }

                prep_fail++;
                continue;
            }
            if (prepared == 0) {
                fprintf(stderr,
                        "[prep] decoded first frame %dx%d stride=%dx%d fmt=%d %s\n",
                        dec_w,
                        dec_h,
                        hor_stride,
                        ver_stride,
                        mpp_fmt,
                        mpp_fmt_name(mpp_fmt));
            }

            //RGA格式转换，根据 MPP 输出的 YUV 格式选择对应的 RGA 输入格式，通常是 YUV422SP 或 YUV420SP
            RgaSURF_FORMAT rga_src_fmt;// RGA 输入格式
            // 目前 MPP 输出的格式通常是 YUV422SP 或 YUV420SP，RGA 需要对应的输入格式来正确处理
            if (mpp_fmt == MPP_FMT_YUV422SP) {
                rga_src_fmt = RK_FORMAT_YCbCr_422_SP;
            } else if (mpp_fmt == MPP_FMT_YUV420SP) {
                rga_src_fmt = RK_FORMAT_YCbCr_420_SP;
            } else {
                fprintf(stderr, "[prep] unsupported MPP fmt=%d\n", mpp_fmt);
                return_ref();
                {
                    std::unique_lock<std::mutex> lk(prep_buf.mtx);
                    prep_buf.slots[fill_slot].state = SLOT_FREE;
                }

                prep_fail++;
                continue;
            }

            int64_t t_shm0 = now_us();
            // 发布原始 MJPEG 数据到 shm，供 web_thread 消费
            {
                std::unique_lock<std::mutex> lk(web_frame_buf.mtx);
                WebFrameSlot& ws = web_frame_buf.slot;
                ws.jpeg.resize(ref.jpg_size);
                memcpy(ws.jpeg.data(), ref.data, ref.jpg_size);
                ws.width = dec_w;
                ws.height = dec_h;
                ws.seq = ref.seq;
                ws.valid = true;
                web_frame_buf.updated = true;
                web_frame_buf.cv.notify_one();// 唤醒 web_thread 来消费新的 MJPEG 数据
            }
            return_ref();
            int64_t t_shm1 = now_us();

            // RGA 进行 YUV420SP/YUV422SP -> RGB888 转换，结果放在 prep_buf 的 rgb 内存中
            rga_buffer_t src = wrapbuffer_virtualaddr_t(yuv_ptr,
                                                        dec_w,
                                                        dec_h,
                                                        hor_stride,
                                                        ver_stride,
                                                        rga_src_fmt);

            rga_buffer_t dst = wrapbuffer_virtualaddr(prep_buf.slots[fill_slot].rgb,
                                                      model_w,
                                                      model_h,
                                                      RK_FORMAT_RGB_888);

            im_rect src_rect = {0, 0, dec_w, dec_h};
            im_rect dst_rect = {0, 0, model_w, model_h};

            int64_t t_rga0 = now_us();

            IM_STATUS rga_ret = improcess(src,
                                          dst,
                                          {},
                                          src_rect,
                                          dst_rect,
                                          {},
                                          IM_SYNC);

            int64_t t_rga1 = now_us();

            if (rga_ret != IM_STATUS_SUCCESS) {
                fprintf(stderr, "[prep] RGA YUV->RGB failed: %s\n", imStrError(rga_ret));

                {
                    std::unique_lock<std::mutex> lk(prep_buf.mtx);
                    prep_buf.slots[fill_slot].state = SLOT_FREE;
                }

                prep_fail++;
                continue;
            }

            int64_t t_total1 = now_us();
            // 生产 ready 的 slot，供 infer_thread 消费
            {
                std::unique_lock<std::mutex> lk(prep_buf.mtx);

                prep_buf.slots[fill_slot].seq = ref.seq;
                prep_buf.slots[fill_slot].dec_w = dec_w;
                prep_buf.slots[fill_slot].dec_h = dec_h;
                prep_buf.slots[fill_slot].hor_stride = hor_stride;
                prep_buf.slots[fill_slot].ver_stride = ver_stride;
                prep_buf.slots[fill_slot].mpp_fmt = mpp_fmt;
                prep_buf.slots[fill_slot].decode_us = t_dec1 - t_dec0;
                prep_buf.slots[fill_slot].rga_us = t_rga1 - t_rga0;
                prep_buf.slots[fill_slot].total_us = t_total1 - t_total0;
                prep_buf.slots[fill_slot].state = SLOT_READY;
                prep_buf.produced++;

                prep_buf.cv.notify_one();// 唤醒 infer_thread 来消费 ready 的 slot
            }

            prepared++;

            sum_dec_us += t_dec1 - t_dec0;
            sum_rga_us += t_rga1 - t_rga0;
            sum_shm_us += t_shm1 - t_shm0;
            sum_total_us += t_total1 - t_total0;

            int64_t now = now_us();
            if (now - last_stats_time >= STATS_INTERVAL_US) {
                float elapsed_s = (now - last_stats_time) / 1e6f;
                if (elapsed_s <= 0.0f) elapsed_s = 1e-6f;

                uint64_t frame_delta = prepared - last_prepared;
                if (frame_delta == 0) frame_delta = 1;

                uint64_t dropped_ready = 0;
                {
                    std::unique_lock<std::mutex> lk(prep_buf.mtx);
                    dropped_ready = prep_buf.dropped_ready;
                }

                double pkt_copy_ms = 0.0;
                double pkt_copy_kb = 0.0;
                if (mpp.pkt_copy_count) {
                    pkt_copy_ms = mpp.pkt_copy_us / (double)mpp.pkt_copy_count / 1000.0;
                    pkt_copy_kb = mpp.pkt_copy_bytes / (double)mpp.pkt_copy_count / 1024.0;
                }

                fprintf(stderr,
                        "[prep-stats] prepared=%lu popped=%lu fps=%.2f "
                        "dec=%.2fms rga=%.2fms shm=%.2fms prep_total=%.2fms "
                        "pkt_copy=%.3fms %.1fKB "
                        "dq=%lu qbuf=%lu qbuf_after_dec=%lu invalid=%lu ref_drop=%lu ready_drop=%lu prep_fail=%lu\n",
                        prepared,
                        popped,
                        frame_delta / elapsed_s,
                        sum_dec_us / (double)frame_delta / 1000.0,
                        sum_rga_us / (double)frame_delta / 1000.0,
                        sum_shm_us / (double)frame_delta / 1000.0,
                        sum_total_us / (double)frame_delta / 1000.0,
                        pkt_copy_ms,
                        pkt_copy_kb,
                        v4l2_shared.dq_count.load(std::memory_order_relaxed),
                        v4l2_shared.qbuf_count.load(std::memory_order_relaxed),
                        qbuf_after_decode,
                        v4l2_shared.invalid_jpeg.load(std::memory_order_relaxed),
                        in_queue.drop_count(),
                        dropped_ready,
                        prep_fail);

                sum_dec_us = 0;
                sum_rga_us = 0;
                sum_shm_us = 0;
                sum_total_us = 0;
                last_stats_time = now;
                last_prepared = prepared;
            }
        }
        

        if (mpp_ready) mpp.deinit();
        prep_done.store(true);
        prep_buf.cv.notify_all();

        fprintf(stderr,
                "[prep] exit popped=%lu prepared=%lu fail=%lu\n",
                popped,
                prepared,
                prep_fail);
    });

    int cur_ctx = 0;
    uint64_t infer_count = 0;
    uint64_t last_infer_count = 0;
    uint64_t ctx_use_count[CTX_NUM] = {0, 0};

    int64_t sum_input_us = 0;
    int64_t sum_run_us = 0;
    int64_t sum_output_us = 0;
    int64_t sum_post_us = 0;
    int64_t sum_release_us = 0;
    int64_t sum_total_us = 0;
    int64_t last_infer_stats_time = now_us();

    fprintf(stderr, "[infer] thread started\n");

    while (running.load() || !prep_done.load()) {
        int slot_idx = -1;

        {
            // 这里等待 prep_thread 生产 ready 的 slot，或者 prep_thread 退出的信号
            std::unique_lock<std::mutex> lk(prep_buf.mtx);

            prep_buf.cv.wait_for(lk, std::chrono::milliseconds(20), [&] {
                return prep_buf.take_latest_ready_locked() >= 0 ||
                       prep_done.load() ||
                       !running.load();
            });

            slot_idx = prep_buf.take_latest_ready_locked();// 获取最新的 ready slot 的索引，
            if (slot_idx < 0) {
                if (!running.load() && prep_done.load()) break;
                continue;
            }

            prep_buf.slots[slot_idx].state = SLOT_INFER;
        }

        PrepSlot& slot = prep_buf.slots[slot_idx];// 获取到一个 ready 的 slot，准备进行推理

        int use_ctx = cur_ctx;
        ctx_use_count[use_ctx]++;

        int64_t t_total0 = now_us();

        rknn_input input{};
        input.index = 0;// 目前模型只有一个输入，如果有多个输入需要根据 input_attrs 来设置对应的 input.index 和 input.buf
        input.type = RKNN_TENSOR_UINT8;
        input.size = (uint32_t)((size_t)model_w * model_h * 3);
        input.fmt = RKNN_TENSOR_NHWC;
        input.buf = slot.rgb;// 直接使用 prep_thread 转换好的 RGB 数据进行推理，避免 memcpy 导致的性能问题
        input.pass_through = 0;

        int64_t t_input0 = now_us();

        ret = rknn_inputs_set(ctx[use_ctx], io_num.n_input, &input);

        int64_t t_input1 = now_us();

        if (ret < 0) {
            fprintf(stderr, "[infer] rknn_inputs_set failed ctx=%d ret=%d\n", use_ctx, ret);
            {
                std::unique_lock<std::mutex> lk(prep_buf.mtx);
                slot.state = SLOT_FREE;
            }
            cur_ctx = 1 - cur_ctx;
            continue;
        }

        int64_t t_run0 = now_us();

        ret = rknn_run(ctx[use_ctx], NULL);// 同步接口，直到推理完成才返回

        int64_t t_run1 = now_us();

        if (ret < 0) {
            fprintf(stderr, "[infer] rknn_run failed ctx=%d ret=%d\n", use_ctx, ret);
            {
                std::unique_lock<std::mutex> lk(prep_buf.mtx);
                slot.state = SLOT_FREE;
            }
            cur_ctx = 1 - cur_ctx;
            continue;
        }

        int64_t t_out0 = now_us();

        ret = rknn_outputs_get(ctx[use_ctx],
                               io_num.n_output,
                               outputs[use_ctx].data(),
                               NULL);

        int64_t t_out1 = now_us();

        if (ret < 0) {
            fprintf(stderr, "[infer] rknn_outputs_get failed ctx=%d ret=%d\n", use_ctx, ret);
            {
                std::unique_lock<std::mutex> lk(prep_buf.mtx);
                slot.state = SLOT_FREE;
            }
            cur_ctx = 1 - cur_ctx;
            continue;
        }

        int dec_w = slot.dec_w;
        int dec_h = slot.dec_h;

        float scale_w = (float)model_w / (float)dec_w;
        float scale_h = (float)model_h / (float)dec_h;

        int o0 = output_order[0].index;
        int o1 = output_order[1].index;
        int o2 = output_order[2].index;

        detect_result_group_t grp{};

        int64_t t_post0 = now_us();
        // 后处理
        post_process((int8_t*)outputs[use_ctx][o0].buf,
                     (int8_t*)outputs[use_ctx][o1].buf,
                     (int8_t*)outputs[use_ctx][o2].buf,
                     model_h,
                     model_w,
                     BOX_THRESH,
                     NMS_THRESH,
                     scale_w,
                     scale_h,
                     out_zps,
                     out_scales,
                     &grp);
        
        for (int i = 0; i < grp.count; ++i) {
            auto& b = grp.results[i].box;
            if (b.left < 0) b.left = 0;
            if (b.top < 0) b.top = 0;
            if (b.right > dec_w) b.right = dec_w;
            if (b.bottom > dec_h) b.bottom = dec_h;
        }

        publish_detections_to_shm(shm,
                                  slot.seq,
                                  dec_w,
                                  dec_h,
                                  grp);

        // 把检测框写入 WebFrameBuf，供 web_thread 画框
        {
            std::unique_lock<std::mutex> lk(web_frame_buf.mtx);
            WebFrameSlot& ws = web_frame_buf.slot;
            int cnt = grp.count;
            if (cnt < 0) cnt = 0;
            if (cnt > MAX_DETECTS) cnt = MAX_DETECTS;
            ws.detect_count = cnt;
            for (int i = 0; i < cnt; i++) {
                strncpy(ws.detects[i].name, grp.results[i].name, 15);
                ws.detects[i].name[15] = '\0';
                ws.detects[i].prop   = grp.results[i].prop;
                ws.detects[i].left   = grp.results[i].box.left;
                ws.detects[i].top    = grp.results[i].box.top;
                ws.detects[i].right  = grp.results[i].box.right;
                ws.detects[i].bottom = grp.results[i].box.bottom;
            }
            ws.has_detects = true;// 标记当前帧有检测结果，web_thread 收到这个标记后会在图像上画框
        }

        InferResult result{};// 用于存储推理结果的结构体，包含序列号、检测框数量和检测结果数组，供 infer_thread 填充后发送到 out_queue
        result.seq = slot.seq;
        result.count = grp.count;
        if (result.count > MAX_DETECTS) result.count = MAX_DETECTS;

        for (int i = 0; i < result.count; i++) {
            strncpy(result.results[i].name, grp.results[i].name, 15);
            result.results[i].name[15] = '\0';
            result.results[i].prop = grp.results[i].prop;
            result.results[i].left = grp.results[i].box.left;
            result.results[i].top = grp.results[i].box.top;
            result.results[i].right = grp.results[i].box.right;
            result.results[i].bottom = grp.results[i].box.bottom;
        }

        out_queue.push(result);// 将推理结果发送到 out_queue，供其他线程消费，例如发送到网络或者保存到文件等

        int64_t t_rel0 = now_us();

        rknn_outputs_release(ctx[use_ctx],
                             io_num.n_output,
                             outputs[use_ctx].data());

        int64_t t_rel1 = now_us();

        int64_t t_post1 = now_us();
        int64_t t_total1 = now_us();

        {
            std::unique_lock<std::mutex> lk(prep_buf.mtx);
            slot.state = SLOT_FREE;
        }

        infer_count++;

        sum_input_us += t_input1 - t_input0;
        sum_run_us += t_run1 - t_run0;
        sum_output_us += t_out1 - t_out0;
        sum_release_us += t_rel1 - t_rel0;
        sum_post_us += t_post1 - t_post0;
        sum_total_us += t_total1 - t_total0;

        cur_ctx = 1 - cur_ctx;

        int64_t now = now_us();
        if (now - last_infer_stats_time >= STATS_INTERVAL_US) {
            float elapsed_s = (now - last_infer_stats_time) / 1e6f;
            if (elapsed_s <= 0.0f) elapsed_s = 1e-6f;

            uint64_t infer_delta = infer_count - last_infer_count;
            if (infer_delta == 0) infer_delta = 1;

            uint64_t produced = 0;
            uint64_t ready_drop = 0;
            {
                std::unique_lock<std::mutex> lk(prep_buf.mtx);
                produced = prep_buf.produced;
                ready_drop = prep_buf.dropped_ready;
            }

            fprintf(stderr,
                    "[infer-stats] infer=%lu produced=%lu fps=%.2f "
                    "input=%.2fms run=%.2fms output=%.2fms release=%.2fms post=%.2fms infer_total=%.2fms "
                    "ctx0=%lu ctx1=%lu ref_drop=%lu ready_drop=%lu infer_drop=%lu\n",
                    infer_count,
                    produced,
                    infer_delta / elapsed_s,
                    sum_input_us / (double)infer_delta / 1000.0,
                    sum_run_us / (double)infer_delta / 1000.0,
                    sum_output_us / (double)infer_delta / 1000.0,
                    sum_release_us / (double)infer_delta / 1000.0,
                    sum_post_us / (double)infer_delta / 1000.0,
                    sum_total_us / (double)infer_delta / 1000.0,
                    ctx_use_count[0],
                    ctx_use_count[1],
                    in_queue.drop_count(),
                    ready_drop,
                    out_queue.drop_count());

            sum_input_us = 0;
            sum_run_us = 0;
            sum_output_us = 0;
            sum_release_us = 0;
            sum_post_us = 0;
            sum_total_us = 0;

            last_infer_stats_time = now;
            last_infer_count = infer_count;
        }
    }

    prep_buf.cv.notify_all();
    web_frame_buf.cv.notify_all();

    if (prep_thread.joinable()) {
        prep_thread.join();
    }

    if (web_thread.joinable()) {
        web_thread.join();
    }

    CaptureFrameRef leftover{};
    while (in_queue.pop(leftover)) {
        v4l2_shared.qbuf_return(leftover.index);
    }

    for (int i = 0; i < 2; ++i) {
        if (prep_buf.slots[i].rgb) {
            free(prep_buf.slots[i].rgb);
            prep_buf.slots[i].rgb = nullptr;
        }
    }

    for (int i = 0; i < CTX_NUM; ++i) {
        if (ctx[i]) {
            rknn_destroy(ctx[i]);
            ctx[i] = 0;
        }
    }

    deinitPostProcess();

    munmap(shm, sizeof(SharedFrame));
    close(shm_fd);
    shm_unlink(SHM_NAME);

    fprintf(stderr,
            "[inference] v4l2-ref pipeline exit infer=%lu ctx0=%lu ctx1=%lu dq=%lu qbuf=%lu invalid=%lu\n",
            infer_count,
            ctx_use_count[0],
            ctx_use_count[1],
            v4l2_shared.dq_count.load(std::memory_order_relaxed),
            v4l2_shared.qbuf_count.load(std::memory_order_relaxed),
            v4l2_shared.invalid_jpeg.load(std::memory_order_relaxed));
}
