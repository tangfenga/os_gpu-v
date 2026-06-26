#include <cuda_runtime.h>

#include <cstdio>

int main() {
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, 4 * 1024 * 1024);
    if (err != cudaErrorMemoryAllocation) {
        std::fprintf(stderr,
                     "expected cudaErrorMemoryAllocation, got %d (%s)\n",
                     err,
                     cudaGetErrorString(err));
        if (err == cudaSuccess) {
            cudaFree(ptr);
        }
        return 1;
    }

    std::printf("memory limit smoke test passed\n");
    return 0;
}
