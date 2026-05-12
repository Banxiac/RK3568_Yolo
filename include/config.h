#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>

struct AppConfig {
    // capture
    char video_device[64] = "/dev/video9";
    int  capture_width    = 640;
    int  capture_height   = 480;

    // inference
    float box_thresh = 0.25f;
    float nms_thresh = 0.45f;

    // mqtt
    char mqtt_host[64]  = "127.0.0.1";
    int  mqtt_port      = 1883;
    char mqtt_topic[64] = "edge/detect";

    // web
    int  http_port    = 8080;
    int  jpeg_quality = 70;
};

// 解析 INI 文件，key=value，# 开头为注释，返回 false 表示文件打开失败
inline bool load_config(const char* path, AppConfig& cfg) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '[' || *p == '\n') continue;
        char key[64], val[128];
        if (sscanf(p, "%63[^=]=%127[^\n]", key, val) != 2) continue;
        // trim trailing spaces
        for (int i = strlen(key)-1; i >= 0 && key[i] == ' '; i--) key[i] = 0;
        char* v = val; while (*v == ' ') v++;

        if      (!strcmp(key, "video_device"))   strncpy(cfg.video_device, v, 63);
        else if (!strcmp(key, "capture_width"))  cfg.capture_width  = atoi(v);
        else if (!strcmp(key, "capture_height")) cfg.capture_height = atoi(v);
        else if (!strcmp(key, "box_thresh"))     cfg.box_thresh     = atof(v);
        else if (!strcmp(key, "nms_thresh"))     cfg.nms_thresh     = atof(v);
        else if (!strcmp(key, "mqtt_host"))      strncpy(cfg.mqtt_host,  v, 63);
        else if (!strcmp(key, "mqtt_port"))      cfg.mqtt_port      = atoi(v);
        else if (!strcmp(key, "mqtt_topic"))     strncpy(cfg.mqtt_topic, v, 63);
        else if (!strcmp(key, "http_port"))      cfg.http_port      = atoi(v);
        else if (!strcmp(key, "jpeg_quality"))   cfg.jpeg_quality   = atoi(v);
    }
    fclose(f);
    return true;
}
