#include <cuda_runtime.h>

#include <cstdio>

static int expect_error(cudaError_t err, cudaError_t expected, const char *what) {
    if (err != expected) {
        std::fprintf(stderr,
                     "%s: got %d (%s), expected %d (%s)\n",
                     what,
                     err,
                     cudaGetErrorString(err),
                     expected,
                     cudaGetErrorString(expected));
        return 1;
    }
    return 0;
}

int main() {
    int failures = 0;

    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, 4096);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed unexpectedly: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "first cudaFree failed unexpectedly: %s\n", cudaGetErrorString(err));
        return 1;
    }

    failures += expect_error(cudaFree(ptr), cudaErrorInvalidDevicePointer, "double free");
    failures += expect_error(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(0xCADA1000BAD0ull)),
        cudaErrorInvalidResourceHandle,
        "invalid stream synchronize");
    failures += expect_error(
        cudaEventQuery(reinterpret_cast<cudaEvent_t>(0xCADA2000BAD0ull)),
        cudaErrorInvalidResourceHandle,
        "invalid event query");

    if (failures != 0) {
        return 1;
    }

    std::printf("stability negative smoke test passed\n");
    return 0;
}
