#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
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

int main() {
    int driver_version = 0;
    int runtime_version = 0;
    cudaError_t driver_err = cudaDriverGetVersion(&driver_version);
    cudaError_t runtime_err = cudaRuntimeGetVersion(&runtime_version);
    std::printf("cudaDriverGetVersion: err=%d version=%d\n", driver_err, driver_version);
    std::printf("cudaRuntimeGetVersion: err=%d version=%d\n", runtime_err, runtime_version);

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
        h_b[i] = static_cast<float>(2 * i);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    check_cuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "copy a H2D");
    check_cuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "copy b H2D");

    const int block = 256;
    const int grid = (n + block - 1) / block;
    vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel sync");

    check_cuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "copy c D2H");

    for (int i = 0; i < n; ++i) {
        const float expected = static_cast<float>(3 * i);
        if (h_c[i] != expected) {
            std::fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, h_c[i], expected);
            return 1;
        }
    }

    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_c), "cudaFree d_c");
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    std::printf("vector_add smoke test passed: %d elements\n", n);
    return 0;
}
