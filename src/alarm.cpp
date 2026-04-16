#include "alarm.h"
#include <mosquitto.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

void alarm_thread(RingQueue<InferResult, 4>& queue,
                  std::atomic<bool>& running,
                  const char* mqtt_host, int mqtt_port,
                  const char* topic)
{
    mosquitto_lib_init();
    struct mosquitto* mosq = mosquitto_new(nullptr, true, nullptr);
    if (!mosq) { fprintf(stderr, "mosquitto_new failed\n"); return; }

    if (mosquitto_connect(mosq, mqtt_host, mqtt_port, 60) != MOSQ_ERR_SUCCESS) {
        fprintf(stderr, "mqtt connect failed: %s:%d\n", mqtt_host, mqtt_port);
        mosquitto_destroy(mosq);
        mosquitto_lib_cleanup();
        return;
    }
    mosquitto_loop_start(mosq);

    char payload[2048];
    InferResult result;
    while (running.load()) {
        if (!queue.pop(result)) { usleep(5000); continue; }
        // 构造JSON
        int pos = snprintf(payload, sizeof(payload), "{\"seq\":%lu,\"objects\":[", result.seq);
        for (int i = 0; i < result.count && pos < (int)sizeof(payload) - 64; i++) {
            auto& r = result.results[i];
            pos += snprintf(payload + pos, sizeof(payload) - pos,
                "%s{\"class\":\"%s\",\"conf\":%.3f,\"box\":[%d,%d,%d,%d]}",
                i ? "," : "", r.name, r.prop, r.left, r.top, r.right, r.bottom);
        }
        snprintf(payload + pos, sizeof(payload) - pos, "]}");

        mosquitto_publish(mosq, nullptr, topic, strlen(payload), payload, 0, false);
    }

    mosquitto_loop_stop(mosq, true);
    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
}
