#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void event_busy_add(const float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int j = 0; j < 32; ++j) {
            x = x * 1.000001f + 1.0f;
        }
        b[i] = x;
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

    float *h_in = static_cast<float *>(std::malloc(bytes));
    float *h_out = static_cast<float *>(std::malloc(bytes));
    if (!h_in || !h_out) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i);
        h_out[i] = 0.0f;
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    float *d_in = nullptr;
    float *d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");

    check_cuda(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream), "H2D async");
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");

    const int block = 256;
    const int grid = (n + block - 1) / block;
    event_busy_add<<<grid, block, 0, stream>>>(d_in, d_out, n);
    check_cuda(cudaGetLastError(), "kernel launch");

    check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    if (ms < 0.0f) {
        std::fprintf(stderr, "invalid elapsed time: %f ms\n", ms);
        return 1;
    }

    check_cuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "D2H");
    if (h_out[0] == 0.0f) {
        std::fprintf(stderr, "unexpected zero output\n");
        return 1;
    }

    check_cuda(cudaFree(d_in), "cudaFree d_in");
    check_cuda(cudaFree(d_out), "cudaFree d_out");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    std::free(h_in);
    std::free(h_out);

    std::printf("event timing smoke test passed: %.3f ms\n", ms);
    return 0;
}
