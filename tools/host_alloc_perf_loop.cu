#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ void host_alloc_perf_vec_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

template <typename Fn>
static double avg_us(int iters, Fn fn) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    const auto end = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    return static_cast<double>(us) / static_cast<double>(iters);
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : (1 << 20);
    const int iters = argc > 2 ? std::atoi(argv[2]) : 30;
    if (n <= 0 || iters <= 0) {
        std::fprintf(stderr, "usage: %s [elements] [iters]\n", argv[0]);
        return 1;
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *h_a = nullptr;
    float *h_b = nullptr;
    float *h_c = nullptr;
    check_cuda(cudaHostAlloc(reinterpret_cast<void **>(&h_a), bytes, cudaHostAllocDefault), "cudaHostAlloc h_a");
    check_cuda(cudaMallocHost(reinterpret_cast<void **>(&h_b), bytes), "cudaMallocHost h_b");
    check_cuda(cudaHostAlloc(reinterpret_cast<void **>(&h_c), bytes, cudaHostAllocDefault), "cudaHostAlloc h_c");

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
        h_c[i] = 0.0f;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    check_cuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "warmup H2D a");
    check_cuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "warmup H2D b");
    check_cuda(cudaMemcpy(d_c, d_a, bytes, cudaMemcpyDeviceToDevice), "warmup D2D");
    check_cuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "warmup D2H");

    const int block = 256;
    const int grid = (n + block - 1) / block;
    host_alloc_perf_vec_add<<<grid, block>>>(d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError(), "warmup kernel launch");
    check_cuda(cudaDeviceSynchronize(), "warmup kernel sync");

    const double h2d_us = avg_us(iters, [&] {
        check_cuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "H2D");
    });
    const double d2d_us = avg_us(iters, [&] {
        check_cuda(cudaMemcpy(d_c, d_a, bytes, cudaMemcpyDeviceToDevice), "D2D");
    });
    const double d2h_us = avg_us(iters, [&] {
        check_cuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "D2H");
    });
    const double kernel_us = avg_us(iters, [&] {
        host_alloc_perf_vec_add<<<grid, block>>>(d_a, d_b, d_c, n);
        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaDeviceSynchronize(), "kernel sync");
    });

    check_cuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "final D2H");
    for (int i = 0; i < n; ++i) {
        const float expected = static_cast<float>(3 * i);
        if (h_c[i] != expected) {
            std::fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, h_c[i], expected);
            return 1;
        }
    }

    std::printf("host_alloc_perf_loop elements=%d bytes=%zu iters=%d\n", n, bytes, iters);
    std::printf("H2D_avg_us=%.3f\n", h2d_us);
    std::printf("D2D_avg_us=%.3f\n", d2d_us);
    std::printf("D2H_avg_us=%.3f\n", d2h_us);
    std::printf("Kernel_sync_avg_us=%.3f\n", kernel_us);

    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_c), "cudaFree d_c");
    check_cuda(cudaFreeHost(h_a), "cudaFreeHost h_a");
    check_cuda(cudaFreeHost(h_b), "cudaFreeHost h_b");
    check_cuda(cudaFreeHost(h_c), "cudaFreeHost h_c");
    return 0;
}
