#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *h_in = static_cast<float *>(std::malloc(bytes));
    float *h_out = static_cast<float *>(std::malloc(bytes));
    if (!h_in || !h_out) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i) * 0.5f;
        h_out[i] = 0.0f;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");

    check_cuda(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice), "H2D");
    check_cuda(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice), "D2D");
    check_cuda(cudaMemcpy(h_out, d_b, bytes, cudaMemcpyDeviceToHost), "D2H");

    for (int i = 0; i < n; ++i) {
        if (h_out[i] != h_in[i]) {
            std::fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, h_out[i], h_in[i]);
            return 1;
        }
    }

    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    std::free(h_in);
    std::free(h_out);

    std::printf("memcpy baseline passed: %zu bytes H2D/D2D/D2H\n", bytes);
    return 0;
}

