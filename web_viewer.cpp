#include "common.h"
#include "config.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cstdint>
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

#define WEB_STREAM_INTERVAL_US 66000

static std::mutex sse_mtx;
static std::vector<int> sse_clients;

static void sse_broadcast(const char* data)
{
    char buf[4096];
    int n = snprintf(buf, sizeof(buf), "data: %s\n\n", data);

    std::lock_guard<std::mutex> lk(sse_mtx);

    for (int i = (int)sse_clients.size() - 1; i >= 0; i--) {
        if (write(sse_clients[i], buf, n) < 0) {
            close(sse_clients[i]);
            sse_clients.erase(sse_clients.begin() + i);
        }
    }
}

static void on_message(struct mosquitto*, void*, const struct mosquitto_message* msg)
{
    if (msg->payloadlen > 0) {
        sse_broadcast((char*)msg->payload);
    }
}

static void* mqtt_thread(void* arg)
{
    const char** args = (const char**)arg;

    mosquitto_lib_init();

    struct mosquitto* mosq = mosquitto_new(nullptr, true, nullptr);
    mosquitto_message_callback_set(mosq, on_message);

    if (mosquitto_connect(mosq, args[0], atoi(args[1]), 60) == MOSQ_ERR_SUCCESS) {
        mosquitto_subscribe(mosq, nullptr, args[2], 0);
    }

    mosquitto_loop_forever(mosq, -1, 1);

    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();

    return nullptr;
}

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

struct MjpegArg {
    int fd;
    SharedFrame* shm;
};

static bool read_jpeg_snap(SharedFrame* shm, uint64_t& last_seq, std::vector<uint8_t>& out)
{
    for (int retry = 0; retry < 5; ++retry) {
        int active = shm->active_index.load(std::memory_order_acquire);
        if (active < 0 || active > 1) { usleep(1000); continue; }

        SharedFrameSlot* slot = &shm->slots[active];
        uint64_t s1 = slot->lock_seq.load(std::memory_order_acquire);
        if (s1 & 1) { usleep(1000); continue; }

        uint64_t gseq = shm->global_seq.load(std::memory_order_acquire);
        if (gseq == last_seq) return false;

        uint32_t jsz = slot->jpeg_size;
        if (jsz == 0 || jsz > MJPEG_MAX_BYTES) return false;

        out.resize(jsz);
        memcpy(out.data(), slot->jpeg, jsz);

        uint64_t s2 = slot->lock_seq.load(std::memory_order_acquire);
        if (s1 != s2 || (s2 & 1)) continue;

        last_seq = gseq;
        return true;
    }
    return false;
}

static void serve_mjpeg(int fd, SharedFrame* shm)
{
    const char* hdr =
        "HTTP/1.0 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace;boundary=frame\r\n\r\n";
    write(fd, hdr, strlen(hdr));

    uint64_t last_seq = UINT64_MAX;
    std::vector<uint8_t> jpeg;

    while (true) {
        if (!read_jpeg_snap(shm, last_seq, jpeg)) {
            usleep(10000);
            continue;
        }

        char part_hdr[128];
        int n = snprintf(part_hdr, sizeof(part_hdr),
                         "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                         jpeg.size());
        if (write(fd, part_hdr, n) < 0) break;
        if (write(fd, jpeg.data(), jpeg.size()) < 0) break;
        if (write(fd, "\r\n", 2) < 0) break;

        usleep(WEB_STREAM_INTERVAL_US);
    }

    close(fd);
}

static void* mjpeg_thread(void* arg)
{
    MjpegArg* a = (MjpegArg*)arg;
    serve_mjpeg(a->fd, a->shm);
    delete a;
    return nullptr;
}

static void handle_client(int fd, SharedFrame* shm)
{
    char buf[512] = {};
    read(fd, buf, sizeof(buf) - 1);

    if (strncmp(buf, "GET /stream", 11) == 0) {
        pthread_t t;
        MjpegArg* a = new MjpegArg{fd, shm};
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

int main(int argc, char** argv)
{
    AppConfig cfg;

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "-c")) {
            if (!load_config(argv[i + 1], cfg)) {
                fprintf(stderr,
                        "Warning: cannot open config file '%s', using defaults\n",
                        argv[i + 1]);
            }
            i++;
        }
    }

    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (shm_fd < 0) {
        perror("shm_open");
        return 1;
    }

    SharedFrame* shm = (SharedFrame*)mmap(nullptr,
                                          sizeof(SharedFrame),
                                          PROT_READ,
                                          MAP_SHARED,
                                          shm_fd,
                                          0);

    if (shm == MAP_FAILED) {
        perror("mmap");
        close(shm_fd);
        return 1;
    }

    char mqtt_port_str[16];
    snprintf(mqtt_port_str, sizeof(mqtt_port_str), "%d", cfg.mqtt_port);

    const char* mqtt_args[] = {
        cfg.mqtt_host,
        mqtt_port_str,
        cfg.mqtt_topic
    };

    pthread_t mt;
    pthread_create(&mt, nullptr, mqtt_thread, (void*)mqtt_args);
    pthread_detach(mt);

    int srv = socket(AF_INET, SOCK_STREAM, 0);

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(cfg.http_port);

    bind(srv, (sockaddr*)&addr, sizeof(addr));
    listen(srv, 8);

    printf("Web viewer: http://0.0.0.0:%d\n", cfg.http_port);

    while (true) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;

        handle_client(fd, shm);
    }

    munmap(shm, sizeof(SharedFrame));
    close(shm_fd);

    return 0;
}
