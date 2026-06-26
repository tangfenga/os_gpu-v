#include <cuda_runtime.h>

#include <cstdio>

int main() {
    int driver_version = 0;
    int runtime_version = 0;
    int device_count = 0;
    int device = -1;

    cudaError_t err = cudaDriverGetVersion(&driver_version);
    std::printf("cudaDriverGetVersion: err=%d (%s) version=%d\n",
                err, cudaGetErrorString(err), driver_version);
    if (err != cudaSuccess) {
        return 1;
    }

    err = cudaRuntimeGetVersion(&runtime_version);
    std::printf("cudaRuntimeGetVersion: err=%d (%s) version=%d\n",
                err, cudaGetErrorString(err), runtime_version);
    if (err != cudaSuccess) {
        return 1;
    }

    err = cudaGetDeviceCount(&device_count);
    std::printf("cudaGetDeviceCount: err=%d (%s) count=%d\n",
                err, cudaGetErrorString(err), device_count);
    if (err != cudaSuccess || device_count <= 0) {
        return 1;
    }

    err = cudaSetDevice(0);
    std::printf("cudaSetDevice(0): err=%d (%s)\n", err, cudaGetErrorString(err));
    if (err != cudaSuccess) {
        return 1;
    }

    err = cudaGetDevice(&device);
    std::printf("cudaGetDevice: err=%d (%s) device=%d\n",
                err, cudaGetErrorString(err), device);
    if (err != cudaSuccess || device != 0) {
        return 1;
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, 0);
    std::printf("cudaGetDeviceProperties: err=%d (%s) name=%s sm=%d mem=%zu\n",
                err,
                cudaGetErrorString(err),
                prop.name,
                prop.multiProcessorCount,
                prop.totalGlobalMem);
    if (err != cudaSuccess) {
        return 1;
    }

    std::printf("last error before clear: %d\n", cudaPeekAtLastError());
    std::printf("last error clear: %d\n", cudaGetLastError());
    std::printf("last error after clear: %d\n", cudaPeekAtLastError());
    return 0;
}

