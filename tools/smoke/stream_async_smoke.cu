#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void stream_scale_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * 2.0f + b[i];
    }
}

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *h_a = static_cast<float *>(std::malloc(bytes));
    float *h_b = static_cast<float *>(std::malloc(bytes));
    float *h_c = static_cast<float *>(std::malloc(bytes));
    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(3 * i);
        h_c[i] = 0.0f;
    }

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    check_cuda(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream), "copy a H2D async");
    check_cuda(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream), "copy b H2D async");

    const int block = 256;
    const int grid = (n + block - 1) / block;
    stream_scale_add<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaStreamSynchronize(stream), "stream synchronize");

    check_cuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "copy c D2H");

    for (int i = 0; i < n; ++i) {
        const float expected = static_cast<float>(5 * i);
        if (h_c[i] != expected) {
            std::fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, h_c[i], expected);
            return 1;
        }
    }

    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_c), "cudaFree d_c");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    std::printf("stream async smoke test passed: %d elements\n", n);
    return 0;
}
