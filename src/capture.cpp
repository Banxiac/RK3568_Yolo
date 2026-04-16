#include "capture.h"
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <linux/videodev2.h>
#include <cstdio>
#include <cstring>
#include <vector>

static int xioctl(int fd, int req, void* arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

void capture_thread(RingQueue<RawFrame, 4>& queue, std::atomic<bool>& running) {
    const char* dev = "/dev/video9";
    const int W = 640, H = 480, NBUF = 4;

    int fd = open(dev, O_RDWR);
    if (fd < 0) { perror("open"); return; }

    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = W; fmt.fmt.pix.height = H;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    xioctl(fd, VIDIOC_S_FMT, &fmt);

    v4l2_requestbuffers req{};
    req.count = NBUF; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(fd, VIDIOC_REQBUFS, &req);

    struct { void* start; size_t len; } bufs[NBUF];
    for (int i = 0; i < NBUF; i++) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP; buf.index = i;
        xioctl(fd, VIDIOC_QUERYBUF, &buf);
        bufs[i].len = buf.length;
        bufs[i].start = mmap(nullptr, buf.length, PROT_READ|PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        xioctl(fd, VIDIOC_QBUF, &buf);
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd, VIDIOC_STREAMON, &type);

    uint64_t seq = 0;
    while (running.load()) {
        fd_set fds; FD_ZERO(&fds); FD_SET(fd, &fds);
        timeval tv{2, 0};
        if (select(fd+1, &fds, nullptr, nullptr, &tv) <= 0) continue;

        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0) continue;

        RawFrame frame;
        frame.width = W; frame.height = H; frame.seq = seq++;
        memcpy(frame.data, bufs[buf.index].start, buf.bytesused);
        queue.push(frame);  // 满则丢弃最新帧

        xioctl(fd, VIDIOC_QBUF, &buf);
    }

    xioctl(fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < NBUF; i++) munmap(bufs[i].start, bufs[i].len);
    close(fd);
}
