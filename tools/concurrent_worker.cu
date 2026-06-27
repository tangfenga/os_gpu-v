#include <cuda_runtime.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void matmul_kernel(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

using Clock = std::chrono::steady_clock;

static void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "worker_error pid=%d op=%s cuda_error=%s code=%d\n",
                     static_cast<int>(getpid()), what, cudaGetErrorString(err), static_cast<int>(err));
        std::exit(1);
    }
}

static double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

static void run_vector(int iterations, int n) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 1024);
        h_b[i] = static_cast<float>((i * 3) % 2048);
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");
    const void *first_vptr = d_a;

    check(cudaMemcpyAsync(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D a");
    check(cudaMemcpyAsync(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D b");
    check(cudaStreamSynchronize(stream), "initial sync");

    const int block = 256;
    const int grid = (n + block - 1) / block;
    check(cudaEventRecord(start, stream), "event start");
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        vector_add_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
        check(cudaGetLastError(), "vector launch");
    }
    check(cudaMemcpyAsync(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream), "D2H c");
    check(cudaEventRecord(stop, stream), "event stop");
    check(cudaStreamSynchronize(stream), "final sync");
    const auto end = Clock::now();

    for (int i = 0; i < n; i += std::max(1, n / 32)) {
        const float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            std::fprintf(stderr, "worker_error pid=%d mode=vector mismatch index=%d expected=%f got=%f\n",
                         static_cast<int>(getpid()), i, expected, h_c[i]);
            std::exit(1);
        }
    }
    float event_ms = 0.0f;
    check(cudaEventElapsedTime(&event_ms, start, stop), "event elapsed");

    const double total_us = elapsed_us(begin, end);
    std::printf("worker_result pid=%d mode=vector iterations=%d n=%d first_vptr=%p total_us=%.3f throughput_ops_s=%.3f event_ms=%.3f pass=1\n",
                static_cast<int>(getpid()), iterations, n, first_vptr, total_us,
                static_cast<double>(iterations) / (total_us / 1000000.0), event_ms);

    check(cudaFree(d_c), "cudaFree d_c");
    check(cudaFree(d_b), "cudaFree d_b");
    check(cudaFree(d_a), "cudaFree d_a");
    check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check(cudaEventDestroy(start), "cudaEventDestroy start");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

static void run_matmul(int iterations, int n) {
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);
    std::vector<float> h_a(count, 1.0f);
    std::vector<float> h_b(count, 2.0f);
    std::vector<float> h_c(count, 0.0f);

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");
    const void *first_vptr = d_a;

    check(cudaMemcpyAsync(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D a");
    check(cudaMemcpyAsync(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D b");
    check(cudaStreamSynchronize(stream), "initial sync");

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    check(cudaEventRecord(start, stream), "event start");
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        matmul_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
        check(cudaGetLastError(), "matmul launch");
    }
    check(cudaMemcpyAsync(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream), "D2H c");
    check(cudaEventRecord(stop, stream), "event stop");
    check(cudaStreamSynchronize(stream), "final sync");
    const auto end = Clock::now();

    const float expected = static_cast<float>(n) * 2.0f;
    for (size_t i = 0; i < count; i += std::max<size_t>(1, count / 32)) {
        if (std::fabs(h_c[i] - expected) > 1e-4f) {
            std::fprintf(stderr, "worker_error pid=%d mode=matmul mismatch index=%zu expected=%f got=%f\n",
                         static_cast<int>(getpid()), i, expected, h_c[i]);
            std::exit(1);
        }
    }
    float event_ms = 0.0f;
    check(cudaEventElapsedTime(&event_ms, start, stop), "event elapsed");

    const double total_us = elapsed_us(begin, end);
    std::printf("worker_result pid=%d mode=matmul iterations=%d n=%d first_vptr=%p total_us=%.3f throughput_ops_s=%.3f event_ms=%.3f pass=1\n",
                static_cast<int>(getpid()), iterations, n, first_vptr, total_us,
                static_cast<double>(iterations) / (total_us / 1000000.0), event_ms);

    check(cudaFree(d_c), "cudaFree d_c");
    check(cudaFree(d_b), "cudaFree d_b");
    check(cudaFree(d_a), "cudaFree d_a");
    check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check(cudaEventDestroy(start), "cudaEventDestroy start");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

static void run_invalid_pointer_error() {
    cudaStream_t stream = nullptr;
    float *d_a = nullptr;
    float *d_b = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check(cudaMalloc(&d_a, sizeof(float)), "cudaMalloc d_a");
    check(cudaMalloc(&d_b, sizeof(float)), "cudaMalloc d_b");
    const cudaError_t submit_err = cudaMemcpyAsync(d_b, reinterpret_cast<void *>(0x12345678ull), sizeof(float),
                                                   cudaMemcpyDeviceToDevice, stream);
    cudaError_t sync_err = cudaSuccess;
    if (submit_err == cudaSuccess) {
        sync_err = cudaStreamSynchronize(stream);
    }
    if (submit_err == cudaSuccess && sync_err == cudaSuccess) {
        std::fprintf(stderr, "worker_error pid=%d mode=error invalid pointer did not surface\n",
                     static_cast<int>(getpid()));
        std::exit(1);
    }
    check(cudaStreamSynchronize(stream), "error clear sync");
    check(cudaFree(d_b), "cudaFree d_b");
    check(cudaFree(d_a), "cudaFree d_a");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::printf("worker_result pid=%d mode=error invalid_pointer_submit_error=%d invalid_pointer_sync_error=%d pass=1\n",
                static_cast<int>(getpid()), static_cast<int>(submit_err), static_cast<int>(sync_err));
}

static void run_crash_after_alloc(int n) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    cudaStream_t stream = nullptr;
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");
    vector_add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_a, d_b, d_c, n);
    check(cudaGetLastError(), "crash vector launch");
    std::printf("worker_result pid=%d mode=crash n=%d first_vptr=%p exiting_without_cleanup=1\n",
                static_cast<int>(getpid()), n, d_a);
    std::fflush(stdout);
    _exit(2);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s vector|matmul|error|crash [iterations] [n]\n", argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    const int n = argc > 3 ? std::atoi(argv[3]) : (mode == "matmul" ? 128 : 1 << 20);
    if (mode == "vector") {
        run_vector(iterations, n);
    } else if (mode == "matmul") {
        run_matmul(iterations, n);
    } else if (mode == "error") {
        run_invalid_pointer_error();
    } else if (mode == "crash") {
        run_crash_after_alloc(n);
    } else {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        return 1;
    }
    return 0;
}
