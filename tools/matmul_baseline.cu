#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

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

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    const int n = 128;
    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);

    float *h_a = static_cast<float *>(std::malloc(bytes));
    float *h_b = static_cast<float *>(std::malloc(bytes));
    float *h_c = static_cast<float *>(std::malloc(bytes));
    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < count; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_c[i] = 0.0f;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    check_cuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "copy a H2D");
    check_cuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "copy b H2D");

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel sync");

    check_cuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "copy c D2H");

    const float expected = static_cast<float>(n) * 2.0f;
    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(h_c[i] - expected) > 1e-4f) {
            std::fprintf(stderr, "mismatch at %zu: got %f expected %f\n", i, h_c[i], expected);
            return 1;
        }
    }

    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_c), "cudaFree d_c");
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    std::printf("matmul baseline passed: %dx%d\n", n, n);
    return 0;
}

