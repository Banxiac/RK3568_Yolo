#include "common.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <pthread.h>
#include <mosquitto.h>
#include <vector>
#include <mutex>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

// ---- SSE clients ----
static std::mutex sse_mtx;
static std::vector<int> sse_clients;

static void sse_broadcast(const char* data) {
    char buf[4096];
    int n = snprintf(buf, sizeof(buf), "data: %s\n\n", data);
    std::lock_guard<std::mutex> lk(sse_mtx);
    for (int i = (int)sse_clients.size()-1; i >= 0; i--) {
        if (write(sse_clients[i], buf, n) < 0) {
            close(sse_clients[i]);
            sse_clients.erase(sse_clients.begin()+i);
        }
    }
}

// ---- MQTT ----
static void on_message(struct mosquitto*, void*, const struct mosquitto_message* msg) {
    if (msg->payloadlen > 0) sse_broadcast((char*)msg->payload);
}

static void* mqtt_thread(void* arg) {
    const char** args = (const char**)arg;
    mosquitto_lib_init();
    struct mosquitto* mosq = mosquitto_new(nullptr, true, nullptr);
    mosquitto_message_callback_set(mosq, on_message);
    if (mosquitto_connect(mosq, args[0], atoi(args[1]), 60) == MOSQ_ERR_SUCCESS)
        mosquitto_subscribe(mosq, nullptr, args[2], 0);
    mosquitto_loop_forever(mosq, -1, 1);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return nullptr;
}

// ---- HTML ----
static const char INDEX_HTML[] =
"HTTP/1.0 200 OK\r\nContent-Type: text/html\r\n\r\n"
"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Edge AI Viewer</title>"
"<style>"
"body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0;padding:10px}"
"img{max-width:100%;border:2px solid #444}"
"#bar{max-width:640px;margin:8px auto;display:flex;justify-content:space-between;align-items:center}"
"#info{font-size:13px;color:#aaa}"
"button{background:#333;color:#eee;border:1px solid #555;padding:5px 14px;border-radius:4px;cursor:pointer}"
"button.on{background:#2a6;border-color:#2a6}"
"#results{text-align:left;max-width:640px;margin:6px auto;font-size:13px}"
".obj{background:#222;margin:3px;padding:4px 8px;border-radius:4px;display:inline-block}"
"</style></head><body>"
"<h2 style='margin:8px 0'>Edge AI Live Viewer</h2>"
"<img id='stream' src='/stream?boxes=0' />"
"<div id='bar'>"
"  <span id='info'>FPS: -- | --:--:--</span>"
"  <button id='btn' onclick='toggleBoxes()'>Show Boxes: OFF</button>"
"</div>"
"<div id='results'></div>"
"<script>"
"let boxes=0,lastSeq=-1,lastTime=0,fps=0;"
"function toggleBoxes(){"
"  boxes=boxes?0:1;"
"  const btn=document.getElementById('btn');"
"  btn.textContent='Show Boxes: '+(boxes?'ON':'OFF');"
"  btn.className=boxes?'on':'';"
"  document.getElementById('stream').src='/stream?boxes='+boxes+'&t='+Date.now();"
"}"
"const es=new EventSource('/events');"
"es.onmessage=e=>{"
"  const d=JSON.parse(e.data),now=Date.now();"
"  if(lastSeq>=0&&now>lastTime){fps=(0.8*fps+0.2*1000/(now-lastTime)).toFixed(1);}"
"  lastSeq=d.seq;lastTime=now;"
"  const t=new Date();const ts=t.toTimeString().slice(0,8);"
"  document.getElementById('info').textContent='FPS: '+fps+' | '+ts;"
"  document.getElementById('results').innerHTML="
"    '<b>Frame '+d.seq+'</b> &mdash; '+d.objects.length+' object(s)<br>'+"
"    d.objects.map(o=>`<span class=obj>${o.class} ${(o.conf*100).toFixed(1)}% [${o.box}]</span>`).join('');"
"};"
"</script></body></html>";

// ---- MJPEG ----
struct MjpegArg { int fd; SharedFrame* shm; bool show_boxes; };

static void serve_mjpeg(int fd, SharedFrame* shm, bool show_boxes) {
    const char* hdr =
        "HTTP/1.0 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace;boundary=frame\r\n\r\n";
    write(fd, hdr, strlen(hdr));

    uint64_t last_seq = UINT64_MAX;
    while (true) {
        uint64_t seq = shm->seq.load(std::memory_order_acquire);
        if (seq == last_seq) { usleep(10000); continue; }
        last_seq = seq;

        cv::Mat yuyv(SHM_HEIGHT, SHM_WIDTH, CV_8UC2, (void*)shm->yuyv);
        cv::Mat bgr;
        cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);

        // 画识别框
        if (show_boxes) {
            int cnt = shm->detect_count;
            for (int i = 0; i < cnt; i++) {
                const SharedDetect& d = shm->detects[i];
                cv::rectangle(bgr, cv::Point(d.left, d.top), cv::Point(d.right, d.bottom),
                              cv::Scalar(0, 255, 0), 2);
                char label[32];
                snprintf(label, sizeof(label), "%s %.0f%%", d.name, d.prop*100);
                cv::putText(bgr, label, cv::Point(d.left, d.top-4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
            }
        }

        std::vector<uchar> jpeg;
        cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, 70});

        char part_hdr[128];
        int n = snprintf(part_hdr, sizeof(part_hdr),
            "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n", jpeg.size());
        if (write(fd, part_hdr, n) < 0) break;
        if (write(fd, jpeg.data(), jpeg.size()) < 0) break;
        if (write(fd, "\r\n", 2) < 0) break;
    }
    close(fd);
}

static void* mjpeg_thread(void* arg) {
    MjpegArg* a = (MjpegArg*)arg;
    serve_mjpeg(a->fd, a->shm, a->show_boxes);
    delete a;
    return nullptr;
}

// ---- HTTP dispatch ----
static void handle_client(int fd, SharedFrame* shm) {
    char buf[512] = {};
    read(fd, buf, sizeof(buf)-1);

    if (strncmp(buf, "GET /stream", 11) == 0) {
        bool boxes = strstr(buf, "boxes=1") != nullptr;
        pthread_t t;
        MjpegArg* a = new MjpegArg{fd, shm, boxes};
        pthread_create(&t, nullptr, mjpeg_thread, a);
        pthread_detach(t);
    } else if (strncmp(buf, "GET /events", 11) == 0) {
        const char* hdr =
            "HTTP/1.0 200 OK\r\n"
            "Content-Type: text/event-stream\r\nCache-Control: no-cache\r\n\r\n";
        write(fd, hdr, strlen(hdr));
        std::lock_guard<std::mutex> lk(sse_mtx);
        sse_clients.push_back(fd);
    } else {
        write(fd, INDEX_HTML, strlen(INDEX_HTML));
        close(fd);
    }
}

int main(int argc, char** argv) {
    const char* mqtt_host  = argc > 1 ? argv[1] : "127.0.0.1";
    const char* mqtt_port  = argc > 2 ? argv[2] : "1883";
    const char* mqtt_topic = argc > 3 ? argv[3] : "edge/detect";
    int http_port          = argc > 4 ? atoi(argv[4]) : 8080;

    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (shm_fd < 0) { perror("shm_open"); return 1; }
    SharedFrame* shm = (SharedFrame*)mmap(nullptr, sizeof(SharedFrame),
                                          PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm == MAP_FAILED) { perror("mmap"); return 1; }

    const char* mqtt_args[] = {mqtt_host, mqtt_port, mqtt_topic};
    pthread_t mt;
    pthread_create(&mt, nullptr, mqtt_thread, (void*)mqtt_args);
    pthread_detach(mt);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(http_port);
    bind(srv, (sockaddr*)&addr, sizeof(addr));
    listen(srv, 8);
    printf("Web viewer: http://0.0.0.0:%d\n", http_port);

    while (true) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        handle_client(fd, shm);
    }
    return 0;
}
