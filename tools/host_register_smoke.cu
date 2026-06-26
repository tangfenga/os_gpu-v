#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr size_t bytes = n * sizeof(float);

    float *h = static_cast<float *>(std::malloc(bytes));
    float *out = static_cast<float *>(std::malloc(bytes));
    if (!h || !out) {
        std::fprintf(stderr, "malloc failed\n");
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        h[i] = static_cast<float>(i);
        out[i] = -1.0f;
    }

    float *d = nullptr;
    check_cuda(cudaHostRegister(h, bytes, cudaHostRegisterDefault), "cudaHostRegister");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d), bytes), "cudaMalloc");
    check_cuda(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    check_cuda(cudaMemcpy(out, d, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    for (int i = 0; i < n; ++i) {
        if (out[i] != h[i]) {
            std::fprintf(stderr, "mismatch at %d\n", i);
            return 1;
        }
    }
    check_cuda(cudaFree(d), "cudaFree");
    check_cuda(cudaHostUnregister(h), "cudaHostUnregister");
    std::free(out);
    std::free(h);
    std::puts("host register smoke OK");
    return 0;
}
