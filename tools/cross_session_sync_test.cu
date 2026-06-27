#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <unistd.h>

__global__ void spin_kernel(unsigned long long cycles, unsigned int *out) {
    const unsigned long long start = clock64();
    while (clock64() - start < cycles) {
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = 0x5a5a1234u;
    }
}

using Clock = std::chrono::steady_clock;

static void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s (%d)\n", what, cudaGetErrorString(err), static_cast<int>(err));
        std::exit(1);
    }
}

static double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

static void write_signal(const char *path) {
    FILE *file = std::fopen(path, "w");
    if (!file) {
        std::perror("fopen signal");
        std::exit(1);
    }
    std::fprintf(file, "ready\n");
    std::fclose(file);
}

static void wait_signal(const char *path, int timeout_ms) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (Clock::now() < deadline) {
        if (access(path, F_OK) == 0) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::fprintf(stderr, "timed out waiting for signal file %s\n", path);
    std::exit(1);
}

static int holder_main(const char *signal_path, unsigned long long cycles, int hold_ms) {
    cudaStream_t stream = nullptr;
    unsigned int *d_out = nullptr;
    unsigned int h_out = 0;
    const unsigned int zero = 0;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check(cudaMalloc(&d_out, sizeof(unsigned int)), "cudaMalloc");
    check(cudaMemcpyAsync(d_out, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream), "init H2D");
    check(cudaStreamSynchronize(stream), "warmup sync");

    const auto launch_begin = Clock::now();
    spin_kernel<<<1, 1, 0, stream>>>(cycles, d_out);
    check(cudaGetLastError(), "spin launch");
    write_signal(signal_path);
    std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    check(cudaStreamSynchronize(stream), "holder stream sync");
    const auto launch_end = Clock::now();
    check(cudaMemcpy(&h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost), "D2H out");
    if (h_out != 0x5a5a1234u) {
        std::fprintf(stderr, "holder output mismatch got=0x%x\n", h_out);
        return 1;
    }
    check(cudaFree(d_out), "cudaFree");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::printf("cross_session_holder pid=%d total_us=%.3f pass=1\n",
                static_cast<int>(getpid()), elapsed_us(launch_begin, launch_end));
    return 0;
}

static int sync_probe_main(const char *signal_path, double threshold_us) {
    unsigned int *d_tmp = nullptr;
    const unsigned int zero = 0;
    check(cudaMalloc(&d_tmp, sizeof(unsigned int)), "probe cudaMalloc");
    check(cudaMemcpy(d_tmp, &zero, sizeof(zero), cudaMemcpyHostToDevice), "probe init H2D");
    check(cudaDeviceSynchronize(), "probe warmup device sync");
    wait_signal(signal_path, 10000);

    const auto begin = Clock::now();
    check(cudaDeviceSynchronize(), "probe measured device sync");
    const auto end = Clock::now();
    const double sync_us = elapsed_us(begin, end);
    check(cudaFree(d_tmp), "probe cudaFree");
    const bool pass = sync_us < threshold_us;
    std::printf("cross_session_sync_probe pid=%d sync_us=%.3f threshold_us=%.3f pass=%d\n",
                static_cast<int>(getpid()), sync_us, threshold_us, pass ? 1 : 0);
    return pass ? 0 : 2;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s holder <signal_path> [cycles] [hold_ms]\n"
                     "       %s sync_probe <signal_path> [threshold_us]\n",
                     argv[0], argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    if (mode == "holder") {
        const unsigned long long cycles = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1500000000ull;
        const int hold_ms = argc > 4 ? std::atoi(argv[4]) : 1500;
        return holder_main(argv[2], cycles, hold_ms);
    }
    if (mode == "sync_probe") {
        const double threshold_us = argc > 3 ? std::atof(argv[3]) : 200000.0;
        return sync_probe_main(argv[2], threshold_us);
    }
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 1;
}
