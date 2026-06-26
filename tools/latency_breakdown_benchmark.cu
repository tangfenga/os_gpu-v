#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void tiny_kernel(float *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] += 1.0f;
    }
}

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

template <typename Fn>
static std::vector<double> samples_us(int iters, Fn fn) {
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0);
    }
    return samples;
}

static void print_stats(const char *name, size_t bytes, std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    double sum = 0.0;
    for (double value : samples) {
        sum += value;
    }
    auto percentile = [&](double p) {
        const size_t idx = std::min(samples.size() - 1, static_cast<size_t>(p * samples.size()));
        return samples[idx];
    };
    std::printf(
        "%s bytes=%zu avg_us=%.3f p50_us=%.3f p95_us=%.3f p99_us=%.3f max_us=%.3f\n",
        name,
        bytes,
        sum / static_cast<double>(samples.size()),
        percentile(0.50),
        percentile(0.95),
        percentile(0.99),
        samples.back());
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    if (iters <= 0) {
        std::fprintf(stderr, "usage: %s [iters]\n", argv[0]);
        return 1;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    check_cuda(cudaMalloc(&d_a, 64ull * 1024ull * 1024ull), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, 64ull * 1024ull * 1024ull), "cudaMalloc d_b");

    float *h_a = nullptr;
    float *h_b = nullptr;
    check_cuda(cudaHostAlloc(reinterpret_cast<void **>(&h_a), 64ull * 1024ull * 1024ull, cudaHostAllocDefault), "cudaHostAlloc h_a");
    check_cuda(cudaHostAlloc(reinterpret_cast<void **>(&h_b), 64ull * 1024ull * 1024ull, cudaHostAllocDefault), "cudaHostAlloc h_b");
    for (size_t i = 0; i < (64ull * 1024ull * 1024ull) / sizeof(float); ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = 0.0f;
    }

    check_cuda(cudaMemcpy(d_a, h_a, 4 * 1024 * 1024, cudaMemcpyHostToDevice), "warmup H2D");
    tiny_kernel<<<1, 32>>>(d_a);
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    print_stats("device_sync_empty", 0, samples_us(iters, [&] {
        check_cuda(cudaDeviceSynchronize(), "device sync empty");
    }));
    print_stats("kernel_launch_only", 0, samples_us(iters, [&] {
        tiny_kernel<<<1, 32>>>(d_a);
        check_cuda(cudaGetLastError(), "kernel launch only");
    }));
    check_cuda(cudaDeviceSynchronize(), "post launch only sync");
    print_stats("kernel_launch_plus_sync", 0, samples_us(iters, [&] {
        tiny_kernel<<<1, 32>>>(d_a);
        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaDeviceSynchronize(), "kernel sync");
    }));
    print_stats("d2d_async_submit", 4 * 1024 * 1024, samples_us(iters, [&] {
        check_cuda(cudaMemcpyAsync(d_b, d_a, 4 * 1024 * 1024, cudaMemcpyDeviceToDevice, 0), "D2D async submit");
    }));
    check_cuda(cudaDeviceSynchronize(), "post d2d async sync");
    print_stats("d2d_sync", 4 * 1024 * 1024, samples_us(iters, [&] {
        check_cuda(cudaMemcpy(d_b, d_a, 4 * 1024 * 1024, cudaMemcpyDeviceToDevice), "D2D sync");
    }));

    const size_t sizes[] = {
        4,
        4ull * 1024ull,
        64ull * 1024ull,
        1ull * 1024ull * 1024ull,
        4ull * 1024ull * 1024ull,
        16ull * 1024ull * 1024ull,
        64ull * 1024ull * 1024ull,
    };
    for (size_t bytes : sizes) {
        print_stats("h2d_hostalloc", bytes, samples_us(iters, [&] {
            check_cuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "H2D hostalloc");
        }));
        print_stats("d2h_hostalloc", bytes, samples_us(iters, [&] {
            check_cuda(cudaMemcpy(h_b, d_a, bytes, cudaMemcpyDeviceToHost), "D2H hostalloc");
        }));
    }

    check_cuda(cudaFreeHost(h_b), "cudaFreeHost h_b");
    check_cuda(cudaFreeHost(h_a), "cudaFreeHost h_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_a), "cudaFree d_a");
    return 0;
}
