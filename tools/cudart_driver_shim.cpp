#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mutex>
#include <unordered_map>
#include <vector>

struct dim3 {
    unsigned int x, y, z;
};

using cudaError_t = int;
using cudaStream_t = void *;
using CUdevice = int;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CUdeviceptr = unsigned long long;
using CUresult = int;

static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorInvalidValue = 1;
static constexpr cudaError_t cudaErrorMemoryAllocation = 2;
static constexpr cudaError_t cudaErrorInitializationError = 3;
static constexpr cudaError_t cudaErrorInvalidDevicePointer = 17;
static constexpr cudaError_t cudaErrorInvalidMemcpyDirection = 21;
static constexpr cudaError_t cudaErrorLaunchFailure = 719;
static constexpr cudaError_t cudaErrorUnknown = 999;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

using cuInit_t = CUresult (*)(unsigned int);
using cuDriverGetVersion_t = CUresult (*)(int *);
using cuDeviceGetCount_t = CUresult (*)(int *);
using cuDeviceGet_t = CUresult (*)(CUdevice *, int);
using cuDeviceGetName_t = CUresult (*)(char *, int, CUdevice);
using cuDeviceTotalMem_t = CUresult (*)(size_t *, CUdevice);
using cuGetErrorName_t = CUresult (*)(CUresult, const char **);
using cuCtxCreate_t = CUresult (*)(CUcontext *, unsigned int, CUdevice);
using cuCtxSynchronize_t = CUresult (*)();
using cuMemAlloc_t = CUresult (*)(CUdeviceptr *, size_t);
using cuMemFree_t = CUresult (*)(CUdeviceptr);
using cuMemcpyHtoD_t = CUresult (*)(CUdeviceptr, const void *, size_t);
using cuMemcpyDtoH_t = CUresult (*)(void *, CUdeviceptr, size_t);
using cuMemcpyDtoD_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
using cuModuleLoadFatBinary_t = CUresult (*)(CUmodule *, const void *);
using cuModuleGetFunction_t = CUresult (*)(CUfunction *, CUmodule, const char *);
using cuLaunchKernel_t = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, CUstream, void **, void **);

static void *g_cuda = nullptr;
static CUcontext g_ctx = nullptr;
static CUmodule g_module = nullptr;
static void *g_fat_cubin = nullptr;
static cudaError_t g_last_error = cudaSuccess;

static cuInit_t p_cuInit = nullptr;
static cuDriverGetVersion_t p_cuDriverGetVersion = nullptr;
static cuDeviceGetCount_t p_cuDeviceGetCount = nullptr;
static cuDeviceGet_t p_cuDeviceGet = nullptr;
static cuDeviceGetName_t p_cuDeviceGetName = nullptr;
static cuDeviceTotalMem_t p_cuDeviceTotalMem = nullptr;
static cuGetErrorName_t p_cuGetErrorName = nullptr;
static cuCtxCreate_t p_cuCtxCreate = nullptr;
static cuCtxSynchronize_t p_cuCtxSynchronize = nullptr;
static cuMemAlloc_t p_cuMemAlloc = nullptr;
static cuMemFree_t p_cuMemFree = nullptr;
static cuMemcpyHtoD_t p_cuMemcpyHtoD = nullptr;
static cuMemcpyDtoH_t p_cuMemcpyDtoH = nullptr;
static cuMemcpyDtoD_t p_cuMemcpyDtoD = nullptr;
static cuModuleLoadFatBinary_t p_cuModuleLoadFatBinary = nullptr;
static cuModuleGetFunction_t p_cuModuleGetFunction = nullptr;
static cuLaunchKernel_t p_cuLaunchKernel = nullptr;

struct LaunchConfig {
    dim3 grid{1, 1, 1};
    dim3 block{1, 1, 1};
    size_t shared = 0;
    cudaStream_t stream = nullptr;
};

static thread_local std::vector<LaunchConfig> g_launch_configs;

static std::mutex &global_mutex() {
    static std::mutex *mu = new std::mutex;
    return *mu;
}

static std::unordered_map<const void *, const char *> &host_to_device_name() {
    static auto *map = new std::unordered_map<const void *, const char *>;
    return *map;
}

static void *must_symbol(const char *name) {
    void *sym = dlsym(g_cuda, name);
    if (!sym) {
        fprintf(stderr, "cudart_driver_shim: missing libcuda symbol %s\n", name);
        abort();
    }
    return sym;
}

static cudaError_t map_result(CUresult result) {
    if (result == 0) {
        return cudaSuccess;
    }
    if (result == 2) {
        return cudaErrorMemoryAllocation;
    }
    if (result == 1) {
        return cudaErrorInvalidValue;
    }
    return cudaErrorUnknown;
}

static cudaError_t ensure_driver_locked() {
    if (g_ctx) {
        return cudaSuccess;
    }

    if (!g_cuda) {
        g_cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!g_cuda) {
            g_cuda = dlopen("/usr/lib/wsl/lib/libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        }
    }
    if (!g_cuda) {
        fprintf(stderr, "cudart_driver_shim: dlopen libcuda failed: %s\n", dlerror());
        return cudaErrorInitializationError;
    }

    p_cuInit = reinterpret_cast<cuInit_t>(must_symbol("cuInit"));
    p_cuDriverGetVersion = reinterpret_cast<cuDriverGetVersion_t>(must_symbol("cuDriverGetVersion"));
    p_cuDeviceGetCount = reinterpret_cast<cuDeviceGetCount_t>(must_symbol("cuDeviceGetCount"));
    p_cuDeviceGet = reinterpret_cast<cuDeviceGet_t>(must_symbol("cuDeviceGet"));
    p_cuDeviceGetName = reinterpret_cast<cuDeviceGetName_t>(must_symbol("cuDeviceGetName"));
    p_cuDeviceTotalMem = reinterpret_cast<cuDeviceTotalMem_t>(must_symbol("cuDeviceTotalMem_v2"));
    p_cuGetErrorName = reinterpret_cast<cuGetErrorName_t>(dlsym(g_cuda, "cuGetErrorName"));
    p_cuCtxCreate = reinterpret_cast<cuCtxCreate_t>(must_symbol("cuCtxCreate_v2"));
    p_cuCtxSynchronize = reinterpret_cast<cuCtxSynchronize_t>(must_symbol("cuCtxSynchronize"));
    p_cuMemAlloc = reinterpret_cast<cuMemAlloc_t>(must_symbol("cuMemAlloc_v2"));
    p_cuMemFree = reinterpret_cast<cuMemFree_t>(must_symbol("cuMemFree_v2"));
    p_cuMemcpyHtoD = reinterpret_cast<cuMemcpyHtoD_t>(must_symbol("cuMemcpyHtoD_v2"));
    p_cuMemcpyDtoH = reinterpret_cast<cuMemcpyDtoH_t>(must_symbol("cuMemcpyDtoH_v2"));
    p_cuMemcpyDtoD = reinterpret_cast<cuMemcpyDtoD_t>(must_symbol("cuMemcpyDtoD_v2"));
    p_cuModuleLoadFatBinary =
        reinterpret_cast<cuModuleLoadFatBinary_t>(must_symbol("cuModuleLoadFatBinary"));
    p_cuModuleGetFunction =
        reinterpret_cast<cuModuleGetFunction_t>(must_symbol("cuModuleGetFunction"));
    p_cuLaunchKernel = reinterpret_cast<cuLaunchKernel_t>(must_symbol("cuLaunchKernel"));

    CUresult result = p_cuInit(0);
    if (result != 0) {
        const char *name = nullptr;
        if (p_cuGetErrorName) {
            p_cuGetErrorName(result, &name);
        }
        fprintf(stderr, "cudart_driver_shim: cuInit failed: %s (%d)\n",
                name ? name : "UNKNOWN", result);
        return map_result(result);
    }

    int count = 0;
    result = p_cuDeviceGetCount(&count);
    if (result != 0) {
        fprintf(stderr, "cudart_driver_shim: cuDeviceGetCount failed: %d\n", result);
        return map_result(result);
    }

    CUdevice dev = 0;
    result = p_cuDeviceGet(&dev, 0);
    if (result != 0) {
        fprintf(stderr, "cudart_driver_shim: cuDeviceGet failed: %d\n", result);
        return map_result(result);
    }
    result = p_cuCtxCreate(&g_ctx, 0, dev);
    if (result != 0) {
        fprintf(stderr, "cudart_driver_shim: cuCtxCreate failed: %d\n", result);
    }
    return map_result(result);
}

static cudaError_t ensure_driver() {
    std::lock_guard<std::mutex> lock(global_mutex());
    cudaError_t err = ensure_driver_locked();
    g_last_error = err;
    return err;
}

extern "C" const char *cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "no error";
        case cudaErrorInvalidValue:
            return "invalid argument";
        case cudaErrorMemoryAllocation:
            return "out of memory";
        case cudaErrorInitializationError:
            return "initialization error";
        case cudaErrorInvalidDevicePointer:
            return "invalid device pointer";
        case cudaErrorInvalidMemcpyDirection:
            return "invalid copy direction for memcpy";
        case cudaErrorLaunchFailure:
            return "unspecified launch failure";
        default:
            return "unknown error";
    }
}

extern "C" cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = ensure_driver();
    if (err != cudaSuccess) {
        *driverVersion = 0;
        return err;
    }
    int version = 0;
    CUresult result = p_cuDriverGetVersion(&version);
    *driverVersion = version;
    g_last_error = map_result(result);
    return g_last_error;
}

extern "C" cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) {
        return cudaErrorInvalidValue;
    }
    *runtimeVersion = 12000;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = ensure_driver();
    if (err != cudaSuccess) {
        return err;
    }
    CUdeviceptr ptr = 0;
    CUresult result = p_cuMemAlloc(&ptr, size);
    *devPtr = reinterpret_cast<void *>(static_cast<uintptr_t>(ptr));
    g_last_error = map_result(result);
    return g_last_error;
}

extern "C" cudaError_t cudaFree(void *devPtr) {
    cudaError_t err = ensure_driver();
    if (err != cudaSuccess) {
        return err;
    }
    CUdeviceptr ptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(devPtr));
    CUresult result = p_cuMemFree(ptr);
    g_last_error = map_result(result);
    return g_last_error;
}

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    cudaError_t err = ensure_driver();
    if (err != cudaSuccess) {
        return err;
    }

    CUresult result = 0;
    switch (kind) {
        case cudaMemcpyHostToHost:
            memcpy(dst, src, count);
            result = 0;
            break;
        case cudaMemcpyHostToDevice:
            result = p_cuMemcpyHtoD(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(dst)),
                                    src, count);
            break;
        case cudaMemcpyDeviceToHost:
            result = p_cuMemcpyDtoH(dst,
                                    static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(src)),
                                    count);
            break;
        case cudaMemcpyDeviceToDevice:
            result = p_cuMemcpyDtoD(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(dst)),
                                    static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(src)),
                                    count);
            break;
        default:
            g_last_error = cudaErrorInvalidMemcpyDirection;
            return g_last_error;
    }
    g_last_error = map_result(result);
    return g_last_error;
}

extern "C" cudaError_t cudaDeviceSynchronize() {
    cudaError_t err = ensure_driver();
    if (err != cudaSuccess) {
        return err;
    }
    CUresult result = p_cuCtxSynchronize();
    g_last_error = map_result(result);
    return g_last_error;
}

extern "C" cudaError_t cudaGetLastError() {
    cudaError_t err = g_last_error;
    g_last_error = cudaSuccess;
    return err;
}

extern "C" void **__cudaRegisterFatBinary(void *fatCubin) {
    std::lock_guard<std::mutex> lock(global_mutex());
    g_fat_cubin = fatCubin;
    return reinterpret_cast<void **>(&g_fat_cubin);
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

extern "C" void __cudaInitModule(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                        const char *hostFun,
                                        char *deviceFun,
                                        const char *deviceName,
                                        int thread_limit,
                                        void *tid,
                                        void *bid,
                                        void *bDim,
                                        void *gDim,
                                        int *wSize) {
    (void)fatCubinHandle;
    (void)deviceFun;
    (void)thread_limit;
    (void)tid;
    (void)bid;
    (void)bDim;
    (void)gDim;
    (void)wSize;
    std::lock_guard<std::mutex> lock(global_mutex());
    host_to_device_name()[hostFun] = deviceName;
}

extern "C" cudaError_t __cudaPushCallConfiguration(dim3 gridDim,
                                                    dim3 blockDim,
                                                    size_t sharedMem,
                                                    cudaStream_t stream) {
    g_launch_configs.push_back({gridDim, blockDim, sharedMem, stream});
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
                                                   dim3 *blockDim,
                                                   size_t *sharedMem,
                                                   cudaStream_t *stream) {
    if (g_launch_configs.empty()) {
        g_last_error = cudaErrorInvalidValue;
        return g_last_error;
    }
    LaunchConfig cfg = g_launch_configs.back();
    g_launch_configs.pop_back();
    if (gridDim) {
        *gridDim = cfg.grid;
    }
    if (blockDim) {
        *blockDim = cfg.block;
    }
    if (sharedMem) {
        *sharedMem = cfg.shared;
    }
    if (stream) {
        *stream = cfg.stream;
    }
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

extern "C" cudaError_t cudaLaunchKernel(const void *func,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void **args,
                                         size_t sharedMem,
                                         cudaStream_t stream) {
    cudaError_t err = ensure_driver();
    if (err != cudaSuccess) {
        return err;
    }

    const char *name = nullptr;
    {
        std::lock_guard<std::mutex> lock(global_mutex());
        auto &names = host_to_device_name();
        auto it = names.find(func);
        if (it != names.end()) {
            name = it->second;
        }
    }
    if (!name) {
        g_last_error = cudaErrorInvalidValue;
        return g_last_error;
    }
    if (!g_module) {
        if (!g_fat_cubin) {
            g_last_error = cudaErrorInvalidValue;
            return g_last_error;
        }
        CUresult load_result = p_cuModuleLoadFatBinary(&g_module, g_fat_cubin);
        if (load_result != 0) {
            g_last_error = map_result(load_result);
            return g_last_error;
        }
    }

    CUfunction kernel = nullptr;
    CUresult result = p_cuModuleGetFunction(&kernel, g_module, name);
    if (result != 0) {
        g_last_error = map_result(result);
        return g_last_error;
    }

    result = p_cuLaunchKernel(kernel,
                              gridDim.x, gridDim.y, gridDim.z,
                              blockDim.x, blockDim.y, blockDim.z,
                              static_cast<unsigned int>(sharedMem),
                              reinterpret_cast<CUstream>(stream),
                              args,
                              nullptr);
    g_last_error = map_result(result);
    return g_last_error;
}
