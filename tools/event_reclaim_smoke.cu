#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <vector>

__global__ void add_offset(const float *in, float *out, float offset, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + offset;
    }
}

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

static void check_values(const std::vector<float> &out, float offset, const char *what) {
    for (size_t i = 0; i < out.size(); ++i) {
        const float expected = static_cast<float>(i) + offset;
        if (out[i] != expected) {
            std::fprintf(
                stderr,
                "%s mismatch at %zu: got %.1f expected %.1f\n",
                what,
                i,
                out[i],
                expected);
            std::exit(1);
        }
    }
}

int main() {
    constexpr int n = 1 << 20;
    constexpr size_t bytes = n * sizeof(float);

    std::vector<float> in(n);
    std::vector<float> out(n, -1.0f);
    for (int i = 0; i < n; ++i) {
        in[i] = static_cast<float>(i);
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_in), bytes), "cudaMalloc d_in");
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&d_out), bytes), "cudaMalloc d_out");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check_cuda(cudaEventCreate(&event), "cudaEventCreate");

    check_cuda(cudaMemcpyAsync(d_in, in.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D async sync-path");
    add_offset<<<(n + 255) / 256, 256, 0, stream>>>(d_in, d_out, 3.0f, n);
    check_cuda(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream), "D2H async sync-path");
    check_cuda(cudaEventRecord(event, stream), "cudaEventRecord sync-path");
    check_cuda(cudaEventSynchronize(event), "cudaEventSynchronize");
    check_values(out, 3.0f, "cudaEventSynchronize reclaim");

    std::fill(out.begin(), out.end(), -1.0f);
    check_cuda(cudaMemcpyAsync(d_in, in.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D async query-path");
    add_offset<<<(n + 255) / 256, 256, 0, stream>>>(d_in, d_out, 7.0f, n);
    check_cuda(cudaMemcpyAsync(out.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream), "D2H async query-path");
    check_cuda(cudaEventRecord(event, stream), "cudaEventRecord query-path");

    for (;;) {
        cudaError_t query = cudaEventQuery(event);
        if (query == cudaSuccess) {
            break;
        }
        if (query != cudaErrorNotReady) {
            check_cuda(query, "cudaEventQuery");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check_values(out, 7.0f, "cudaEventQuery reclaim");

    check_cuda(cudaEventDestroy(event), "cudaEventDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    check_cuda(cudaFree(d_out), "cudaFree d_out");
    check_cuda(cudaFree(d_in), "cudaFree d_in");

    std::puts("event reclaim smoke OK");
    return 0;
}
