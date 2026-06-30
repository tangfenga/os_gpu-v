#pragma once

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

namespace vgpu_server {

inline uint64_t ReadUint64Env(const char *name, uint64_t fallback) {
    const char *env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(env, &end, 10);
    if (!end || *end != '\0') {
        return fallback;
    }
    return static_cast<uint64_t>(value);
}

inline bool DetailedPerfRequested() {
    const char *env = std::getenv("VGPU_PERF_DETAIL");
    return env && std::strcmp(env, "1") == 0;
}

inline void BindRingWorkerToCpu(uint64_t session_id) {
#if defined(__linux__)
    const unsigned int cpus = std::thread::hardware_concurrency();
    if (cpus <= 1) {
        return;
    }
    const uint64_t base = ReadUint64Env("VGPU_RING_CPU_BASE", 1);
    const int cpu = static_cast<int>((base + session_id - 1) % cpus);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (DetailedPerfRequested()) {
        std::cout << "[vgpu] op=BindRingWorker sid=" << session_id
                  << " cpu=" << cpu
                  << " result=" << rc << std::endl;
    }
#else
    (void)session_id;
#endif
}

class RingIdleBackoff {
public:
    void Reset() {
        idle_iters_ = 0;
    }

    void Wait() {
        ++idle_iters_;
        if (idle_iters_ < 1000) {
            return;
        }
        if (idle_iters_ < 5000) {
            std::this_thread::yield();
            return;
        }
        if (idle_iters_ < 10000) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }

private:
    uint32_t idle_iters_ = 0;
};

}  // namespace vgpu_server

