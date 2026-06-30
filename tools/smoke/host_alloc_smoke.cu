#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

__global__ void host_alloc_add(const float *a, const float *b, float *c, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr size_t bytes = n * sizeof(float);

    float *h_a = nullptr;
    float *h_b = nullptr;
    float *h_c = nullptr;
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    check_cuda(cudaHostAlloc(reinterpret_cast<void **>(&h_a), bytes, cudaHostAllocDefault), "cudaHostAlloc h_a");
    check_cuda(cudaMallocHost(reinterpret_cast<void **>(&h_b), bytes), "cudaMallocHost h_b");
    check_cuda(cudaHostAlloc(reinterpret_cast<void **>(&h_c), bytes, cudaHostAllocDefault), "cudaHostAlloc h_c");

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
        h_c[i] = -1.0f;
    }

    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_c), bytes), "cudaMalloc d_c");

    check_cuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_a->d_a");
    check_cuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy h_b->d_b");
    host_alloc_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_c->h_c");

    for (int i = 0; i < n; ++i) {
        const float expected = static_cast<float>(3 * i);
        if (h_c[i] != expected) {
            std::fprintf(stderr, "mismatch at %d: got %.1f expected %.1f\n", i, h_c[i], expected);
            std::exit(1);
        }
    }

    check_cuda(cudaFree(d_c), "cudaFree d_c");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFreeHost(h_c), "cudaFreeHost h_c");
    check_cuda(cudaFreeHost(h_b), "cudaFreeHost h_b");
    check_cuda(cudaFreeHost(h_a), "cudaFreeHost h_a");

    std::puts("host alloc smoke OK");
    return 0;
}
