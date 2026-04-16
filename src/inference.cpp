#include "inference.h"
#include "common.h"
#include "postprocess.h"
#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"
#include "rknn_api.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
// 加载模型文件到内存
static unsigned char* load_model(const char* path, int* size) {
    FILE* fp = fopen(path, "rb");// 以二进制方式打开文件
    if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END);// 将文件指针移动到文件末尾
    *size = ftell(fp);// 获取文件大小
    fseek(fp, 0, SEEK_SET);// 将文件指针重新移动到文件开头
    // 分配内存并读取文件内容
    unsigned char* buf = (unsigned char*)malloc(*size);
    fread(buf, 1, *size, fp);
    // 关闭文件
    fclose(fp);
    // 返回模型数据和大小
    return buf;
}
// 推理线程函数
void inference_thread(RingQueue<RawFrame, 4>& in_queue,
                      RingQueue<InferResult, 4>& out_queue,
                      std::atomic<bool>& running,
                      const char* model_path)
{
    // 加载模型
    int model_size = 0;// 模型文件大小
    // 将模型文件加载到内存中，返回指向模型数据的指针
    unsigned char* model_data = load_model(model_path, &model_size);
    if (!model_data) { fprintf(stderr, "load model failed\n"); return; }
    // 初始化RKNN上下文，传入模型数据和大小，如果初始化失败则打印错误信息并释放模型数据内存
    rknn_context ctx;
    if (rknn_init(&ctx, model_data, model_size, 0, NULL) < 0) {
        fprintf(stderr, "rknn_init failed\n"); free(model_data); return;
    }
    // 模型数据已经被RKNN上下文使用，可以释放内存
    free(model_data);
    // 查询模型输入输出信息，获取输入输出的数量和属性
    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    // 输入属性数组，存储每个输入的属性信息，首先将数组清零，然后查询每个输入的属性并存储到数组中
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    input_attrs[0].index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[0], sizeof(rknn_tensor_attr));
    // 输出属性数组，存储每个输出的属性信息，首先将数组清零，然后查询每个输出的属性并存储到数组中
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < (int)io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    }
    // 获取模型输入的宽高，根据输入格式（NCHW或NHWC）从属性中提取宽度和高度信息
    int model_w = (input_attrs[0].fmt == RKNN_TENSOR_NCHW) ? input_attrs[0].dims[3] : input_attrs[0].dims[2];
    int model_h = (input_attrs[0].fmt == RKNN_TENSOR_NCHW) ? input_attrs[0].dims[2] : input_attrs[0].dims[1];
    // 初始化后处理模块，传入模型输入的宽高信息
    // RGA resize buffer: YUYV->RGB + resize
    void* rgb_buf = malloc(model_w * model_h * 3);
    // 输出的量化参数，分别存储每个输出的零点和缩放因子，首先创建两个空的向量，然后从输出属性中提取零点和缩放因子并存储到向量中
    std::vector<int32_t> out_zps;
    std::vector<float>   out_scales;
    for (int i = 0; i < (int)io_num.n_output; i++) {
        out_zps.push_back(output_attrs[i].zp);
        out_scales.push_back(output_attrs[i].scale);
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].size  = model_w * model_h * 3;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].buf   = rgb_buf;

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));

    // 初始化共享内存
    shm_unlink(SHM_NAME);
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, sizeof(SharedFrame));
    SharedFrame* shm = (SharedFrame*)mmap(nullptr, sizeof(SharedFrame),
                                          PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    new (shm) SharedFrame();  // placement new 初始化atomic

    RawFrame frame;
    while (running.load()) {
        if (!in_queue.pop(frame)) { usleep(1000); continue; }

        // RGA: YUYV -> RGB888 + resize to model input size
        rga_buffer_t src = wrapbuffer_virtualaddr(frame.data, frame.width, frame.height, RK_FORMAT_YCbCr_422_P);
        rga_buffer_t dst = wrapbuffer_virtualaddr(rgb_buf, model_w, model_h, RK_FORMAT_RGB_888);
        imresize(src, dst);

        rknn_inputs_set(ctx, io_num.n_input, inputs);
        rknn_run(ctx, NULL);
        rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

        float scale_w = (float)model_w / frame.width;
        float scale_h = (float)model_h / frame.height;
        detect_result_group_t grp;
        post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf,
                     model_h, model_w, BOX_THRESH, NMS_THRESH,
                     scale_w, scale_h, out_zps, out_scales, &grp);

        // 写入共享内存供网页进程读取（先写数据再更新seq）
        memcpy(shm->yuyv, frame.data, sizeof(shm->yuyv));
        shm->detect_count = grp.count;
        for (int i = 0; i < grp.count; i++) {
            strncpy(shm->detects[i].name, grp.results[i].name, 15);
            shm->detects[i].prop   = grp.results[i].prop;
            shm->detects[i].left   = grp.results[i].box.left;
            shm->detects[i].top    = grp.results[i].box.top;
            shm->detects[i].right  = grp.results[i].box.right;
            shm->detects[i].bottom = grp.results[i].box.bottom;
        }
        shm->seq.fetch_add(1, std::memory_order_release);

        InferResult result;
        result.seq   = frame.seq;
        result.count = grp.count;
        for (int i = 0; i < grp.count; i++) {
            strncpy(result.results[i].name, grp.results[i].name, 15);
            result.results[i].prop   = grp.results[i].prop;
            result.results[i].left   = grp.results[i].box.left;
            result.results[i].top    = grp.results[i].box.top;
            result.results[i].right  = grp.results[i].box.right;
            result.results[i].bottom = grp.results[i].box.bottom;
        }
        out_queue.push(result);

        rknn_outputs_release(ctx, io_num.n_output, outputs);
    }

    free(rgb_buf);
    rknn_destroy(ctx);
    deinitPostProcess();
    munmap(shm, sizeof(SharedFrame));
    close(shm_fd);
    shm_unlink(SHM_NAME);
}
