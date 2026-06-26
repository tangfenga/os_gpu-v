#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <vector>

__global__ void scale_kernel(const float *in, float *out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 3.0f + 1.0f;
    }
}

int main() {
    constexpr int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> input(n);
    std::vector<float> output(n, -1.0f);
    for (int i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i % 251) * 0.5f;
    }

    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamCreateWithFlags failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    err = cudaMalloc(reinterpret_cast<void **>(&d_in), bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_in failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaMalloc(reinterpret_cast<void **>(&d_out), bytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc d_out failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMemcpyAsync(d_in, input.data(), bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync H2D failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    scale_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_in, d_out, n);
    err = cudaMemcpyAsync(output.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync D2H failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        const float expected = input[i] * 3.0f + 1.0f;
        if (std::fabs(output[i] - expected) > 1e-5f) {
            std::cerr << "mismatch at " << i << ": got " << output[i]
                      << " expected " << expected << std::endl;
            return 1;
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    std::cout << "d2h async smoke test passed: " << n << " elements" << std::endl;
    return 0;
}
