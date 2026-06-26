#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <thread>

int main() {
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, 4 * 1024 * 1024);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::printf("session timeout smoke allocated pointer %p\n", ptr);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(10));

    cudaFree(ptr);
    return 0;
}
