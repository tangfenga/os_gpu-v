#include <cuda_runtime.h>

#include <cstdio>

static int expect_any_error(cudaError_t err, const char *what) {
    if (err == cudaSuccess) {
        std::fprintf(stderr,
                     "%s: got success, expected an error\n",
                     what);
        return 1;
    }
    std::printf("%s returned %d (%s)\n", what, err, cudaGetErrorString(err));
    return 0;
}

static int expect_one_of(cudaError_t err, cudaError_t first, cudaError_t second, const char *what) {
    if (err != first && err != second) {
        std::fprintf(stderr,
                     "%s: got %d (%s), expected %d (%s) or %d (%s)\n",
                     what,
                     err,
                     cudaGetErrorString(err),
                     first,
                     cudaGetErrorString(first),
                     second,
                     cudaGetErrorString(second));
        return 1;
    }
    std::printf("%s returned %d (%s)\n", what, err, cudaGetErrorString(err));
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

    failures += expect_one_of(cudaFree(ptr), cudaErrorInvalidDevicePointer, cudaErrorInvalidValue, "double free");
    failures += expect_any_error(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(0xCADA1000BAD0ull)),
        "invalid stream synchronize");
    failures += expect_any_error(
        cudaEventQuery(reinterpret_cast<cudaEvent_t>(0xCADA2000BAD0ull)),
        "invalid event query");

    if (failures != 0) {
        return 1;
    }

    std::printf("stability negative smoke test passed\n");
    return 0;
}
