#include "inference.h"
#include "common.h"
#include "postprocess.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"
#include "rknn_api.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

// 加载模型文件到内存
static unsigned char* load_model(const char* path, int* size)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Open model %s failed\n", path);
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= 0) {
        fprintf(stderr, "Invalid model size: %ld\n", file_size);
        fclose(fp);
        return nullptr;
    }

    unsigned char* buf = (unsigned char*)malloc((size_t)file_size);
    if (!buf) {
        fprintf(stderr, "malloc model buffer failed\n");
        fclose(fp);
        return nullptr;
    }

    size_t read_size = fread(buf, 1, (size_t)file_size, fp);
    fclose(fp);

    if (read_size != (size_t)file_size) {
        fprintf(stderr, "Read model failed: %zu/%ld\n", read_size, file_size);
        free(buf);
        return nullptr;
    }

    *size = (int)file_size;
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

// YOLOv5 post_process 固定要求输入顺序为 stride 8/16/32。
// 这里根据输出特征图面积从大到小排序，避免模型导出后 RKNN 输出顺序变化导致类别/框错误。
static int get_output_grid_area(const rknn_tensor_attr& attr)
{
    int area = 1;
    int used = 0;

    for (int i = 0; i < attr.n_dims; ++i) {
        int d = attr.dims[i];
        // YOLO 输出常见形状: [1,255,80,80] / [1,80,80,255]。
        // 排除 batch=1 和 channel=PROP_BOX_SIZE*3 等大通道维，只取网格维度。
        if (d > 1 && d <= 256) {
            area *= d;
            used++;
            if (used == 2) break;
        }
    }

    return area;
}

// 推理线程函数
void inference_thread(RingQueue<RawFrame, 4>& in_queue,
                      RingQueue<InferResult, 4>& out_queue,
                      std::atomic<bool>& running,
                      const char* model_path)
{
    int model_size = 0;
    unsigned char* model_data = load_model(model_path, &model_size);
    if (!model_data) {
        fprintf(stderr, "[inference] load model failed\n");
        return;
    }

    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        fprintf(stderr, "[inference] rknn_init failed, ret=%d\n", ret);
        return;
    }

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0 || io_num.n_input < 1 || io_num.n_output < 3) {
        fprintf(stderr, "[inference] invalid io num, ret=%d input=%d output=%d\n",
                ret, io_num.n_input, io_num.n_output);
        rknn_destroy(ctx);
        return;
    }

    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    for (int i = 0; i < (int)io_num.n_input; ++i) {
        memset(&input_attrs[i], 0, sizeof(rknn_tensor_attr));
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "[inference] query input attr %d failed, ret=%d\n", i, ret);
            rknn_destroy(ctx);
            return;
        }
        dump_tensor_attr("[inference] input", input_attrs[i]);
    }

    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    for (int i = 0; i < (int)io_num.n_output; ++i) {
        memset(&output_attrs[i], 0, sizeof(rknn_tensor_attr));
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            fprintf(stderr, "[inference] query output attr %d failed, ret=%d\n", i, ret);
            rknn_destroy(ctx);
            return;
        }
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
        fprintf(stderr, "[inference] invalid input shape: w=%d h=%d c=%d\n", model_w, model_h, model_c);
        rknn_destroy(ctx);
        return;
    }

    void* rgb_buf = malloc((size_t)model_w * model_h * model_c);
    if (!rgb_buf) {
        fprintf(stderr, "[inference] malloc rgb_buf failed\n");
        rknn_destroy(ctx);
        return;
    }

    // 输出按 grid area 从大到小排序，映射到 post_process 的 stride 8/16/32。
    struct OutputSlot {
        int index;
        int area;
    };
    std::vector<OutputSlot> output_order;
    for (int i = 0; i < (int)io_num.n_output; ++i) {
        OutputSlot s;
        s.index = i;
        s.area = get_output_grid_area(output_attrs[i]);
        output_order.push_back(s);
    }

    std::sort(output_order.begin(), output_order.end(),
              [](const OutputSlot& a, const OutputSlot& b) {
                  return a.area > b.area;
              });

    std::vector<int32_t> out_zps;
    std::vector<float> out_scales;
    for (int i = 0; i < 3; ++i) {
        int out_idx = output_order[i].index;
        out_zps.push_back(output_attrs[out_idx].zp);
        out_scales.push_back(output_attrs[out_idx].scale);
        fprintf(stderr, "[inference] post output%d uses rknn output%d, area=%d, zp=%d, scale=%f\n",
                i, out_idx, output_order[i].area,
                output_attrs[out_idx].zp, output_attrs[out_idx].scale);
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = (uint32_t)((size_t)model_w * model_h * model_c);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = rgb_buf;

    std::vector<rknn_output> outputs(io_num.n_output);
    memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
    for (int i = 0; i < (int)io_num.n_output; ++i) {
        outputs[i].want_float = 0;
    }

    // 初始化共享内存
    shm_unlink(SHM_NAME);
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        perror("[inference] shm_open");
        free(rgb_buf);
        rknn_destroy(ctx);
        return;
    }

    if (ftruncate(shm_fd, sizeof(SharedFrame)) != 0) {
        perror("[inference] ftruncate");
        close(shm_fd);
        shm_unlink(SHM_NAME);
        free(rgb_buf);
        rknn_destroy(ctx);
        return;
    }

    SharedFrame* shm = (SharedFrame*)mmap(nullptr, sizeof(SharedFrame),
                                          PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm == MAP_FAILED) {
        perror("[inference] mmap");
        close(shm_fd);
        shm_unlink(SHM_NAME);
        free(rgb_buf);
        rknn_destroy(ctx);
        return;
    }
    new (shm) SharedFrame();

    RawFrame frame;
    while (running.load()) {
        if (!in_queue.pop(frame)) {
            usleep(1000);
            continue;
        }

        // 先把原始帧发布到共享内存。
        // web_viewer.cpp 的 /stream 只依赖 shm->yuyv 和 shm->seq；
        // 如果下面 RGA/RKNN 出错也不能阻塞画面刷新。
        memcpy(shm->yuyv, frame.data, sizeof(shm->yuyv));
        shm->detect_count = 0;
        shm->seq.fetch_add(1, std::memory_order_release);

        // 关键修正：
        // 参考 main.cc 是“直接 resize 到模型输入尺寸”，不是 letterbox。
        // postprocess.cc 也只支持 scale_w/scale_h 形式的坐标还原。
        // 因此这里必须和参考程序保持一致：YUYV422 -> RGB888，并拉伸到 model_w x model_h。
        memset(rgb_buf, 0, (size_t)model_w * model_h * model_c);

        rga_buffer_t src = wrapbuffer_virtualaddr(frame.data,
                                                  frame.width,
                                                  frame.height,
                                                  RK_FORMAT_YUYV_422);
        rga_buffer_t dst = wrapbuffer_virtualaddr(rgb_buf,
                                                  model_w,
                                                  model_h,
                                                  RK_FORMAT_RGB_888);

        im_rect src_rect = {0, 0, frame.width, frame.height};
        im_rect dst_rect = {0, 0, model_w, model_h};

        IM_STATUS rga_ret = improcess(src, dst, {}, src_rect, dst_rect, {}, IM_SYNC);
        if (rga_ret != IM_STATUS_SUCCESS) {
            fprintf(stderr, "[inference] RGA YUYV->RGB resize failed: %s\n", imStrError(rga_ret));
            // 不再 continue 前阻断网页画面；原始帧已经发布，下一帧继续尝试推理。
            continue;
        }

        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if (ret < 0) {
            fprintf(stderr, "[inference] rknn_inputs_set failed, ret=%d\n", ret);
            continue;
        }

        ret = rknn_run(ctx, NULL);
        if (ret < 0) {
            fprintf(stderr, "[inference] rknn_run failed, ret=%d\n", ret);
            continue;
        }

        ret = rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL);
        if (ret < 0) {
            fprintf(stderr, "[inference] rknn_outputs_get failed, ret=%d\n", ret);
            continue;
        }

        float scale_w = (float)model_w / (float)frame.width;
        float scale_h = (float)model_h / (float)frame.height;

        int o0 = output_order[0].index;
        int o1 = output_order[1].index;
        int o2 = output_order[2].index;

        detect_result_group_t grp;
        memset(&grp, 0, sizeof(grp));
        post_process((int8_t*)outputs[o0].buf,
                     (int8_t*)outputs[o1].buf,
                     (int8_t*)outputs[o2].buf,
                     model_h,
                     model_w,
                     BOX_THRESH,
                     NMS_THRESH,
                     scale_w,
                     scale_h,
                     out_zps,
                     out_scales,
                     &grp);

        // 坐标边界保护，避免网页绘制或下游使用时越界。
        for (int i = 0; i < grp.count; ++i) {
            if (grp.results[i].box.left < 0) grp.results[i].box.left = 0;
            if (grp.results[i].box.top < 0) grp.results[i].box.top = 0;
            if (grp.results[i].box.right > frame.width) grp.results[i].box.right = frame.width;
            if (grp.results[i].box.bottom > frame.height) grp.results[i].box.bottom = frame.height;
        }

        fprintf(stderr, "[inference] detect_count=%d scale_w=%.3f scale_h=%.3f\n",
                grp.count, scale_w, scale_h);
        for (int i = 0; i < grp.count; i++) {
            fprintf(stderr, "  [%d] %s %.1f%% (%d,%d,%d,%d)\n",
                    i,
                    grp.results[i].name,
                    grp.results[i].prop * 100,
                    grp.results[i].box.left,
                    grp.results[i].box.top,
                    grp.results[i].box.right,
                    grp.results[i].box.bottom);
        }

        // 更新检测结果。原始图像已经在本轮开始时写入 shm。
        // 这里再次更新 seq，让网页端可立即拿到带框结果。
        shm->detect_count = grp.count;
        for (int i = 0; i < grp.count; i++) {
            strncpy(shm->detects[i].name, grp.results[i].name, 15);
            shm->detects[i].name[15] = '\0';
            shm->detects[i].prop = grp.results[i].prop;
            shm->detects[i].left = grp.results[i].box.left;
            shm->detects[i].top = grp.results[i].box.top;
            shm->detects[i].right = grp.results[i].box.right;
            shm->detects[i].bottom = grp.results[i].box.bottom;
        }
        shm->seq.fetch_add(1, std::memory_order_release);

        InferResult result;
        memset(&result, 0, sizeof(result));
        result.seq = frame.seq;
        result.count = grp.count;
        for (int i = 0; i < grp.count; i++) {
            strncpy(result.results[i].name, grp.results[i].name, 15);
            result.results[i].name[15] = '\0';
            result.results[i].prop = grp.results[i].prop;
            result.results[i].left = grp.results[i].box.left;
            result.results[i].top = grp.results[i].box.top;
            result.results[i].right = grp.results[i].box.right;
            result.results[i].bottom = grp.results[i].box.bottom;
        }
        out_queue.push(result);

        rknn_outputs_release(ctx, io_num.n_output, outputs.data());
    }

    free(rgb_buf);
    deinitPostProcess();

    munmap(shm, sizeof(SharedFrame));
    close(shm_fd);
    shm_unlink(SHM_NAME);

    rknn_destroy(ctx);
}
