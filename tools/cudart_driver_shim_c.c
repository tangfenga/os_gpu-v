#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct dim3 {
    unsigned int x, y, z;
} dim3;

typedef int cudaError_t;
typedef void *cudaStream_t;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;
typedef int CUresult;

enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorUnknown = 999,
};

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
} cudaMemcpyKind;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDriverGetVersion_t)(int *);
typedef CUresult (*cuDeviceGetCount_t)(int *);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDeviceGetName_t)(char *, int, CUdevice);
typedef CUresult (*cuDeviceTotalMem_t)(size_t *, CUdevice);
typedef CUresult (*cuCtxCreate_t)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyDtoD_t)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*cuModuleLoadFatBinary_t)(CUmodule *, const void *);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuLaunchKernel_t)(CUfunction,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, CUstream, void **, void **);

struct FunctionMapEntry {
    const void *host;
    const char *device_name;
};

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared;
    cudaStream_t stream;
};

static void *g_cuda;
static CUcontext g_ctx;
static CUmodule g_module;
static void *g_fat_cubin;
static cudaError_t g_last_error = cudaSuccess;
static struct FunctionMapEntry g_functions[256];
static int g_function_count;
static struct LaunchConfig g_launch_configs[64];
static int g_launch_config_count;

static cuInit_t p_cuInit;
static cuDriverGetVersion_t p_cuDriverGetVersion;
static cuDeviceGetCount_t p_cuDeviceGetCount;
static cuDeviceGet_t p_cuDeviceGet;
static cuDeviceGetName_t p_cuDeviceGetName;
static cuDeviceTotalMem_t p_cuDeviceTotalMem;
static cuCtxCreate_t p_cuCtxCreate;
static cuCtxDestroy_t p_cuCtxDestroy;
static cuCtxSynchronize_t p_cuCtxSynchronize;
static cuMemAlloc_t p_cuMemAlloc;
static cuMemFree_t p_cuMemFree;
static cuMemcpyHtoD_t p_cuMemcpyHtoD;
static cuMemcpyDtoH_t p_cuMemcpyDtoH;
static cuMemcpyDtoD_t p_cuMemcpyDtoD;
static cuModuleLoadFatBinary_t p_cuModuleLoadFatBinary;
static cuModuleGetFunction_t p_cuModuleGetFunction;
static cuLaunchKernel_t p_cuLaunchKernel;

static void *must_symbol(const char *name) {
    void *sym = dlsym(g_cuda, name);
    if (!sym) {
        fprintf(stderr, "cudart_driver_shim_c: missing libcuda symbol %s\n", name);
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

static cudaError_t ensure_driver_locked(void) {
    CUresult result;
    CUdevice dev = 0;
    int count = 0;

    if (g_ctx) {
        return cudaSuccess;
    }

    if (!g_cuda) {
        g_cuda = dlopen("libcuda.so.1", RTLD_NOW);
    }
    if (!g_cuda) {
        fprintf(stderr, "cudart_driver_shim_c: dlopen libcuda failed: %s\n", dlerror());
        return cudaErrorInitializationError;
    }

    p_cuInit = (cuInit_t)must_symbol("cuInit");
    p_cuDriverGetVersion = (cuDriverGetVersion_t)must_symbol("cuDriverGetVersion");
    p_cuDeviceGetCount = (cuDeviceGetCount_t)must_symbol("cuDeviceGetCount");
    p_cuDeviceGet = (cuDeviceGet_t)must_symbol("cuDeviceGet");
    p_cuDeviceGetName = (cuDeviceGetName_t)must_symbol("cuDeviceGetName");
    p_cuDeviceTotalMem = (cuDeviceTotalMem_t)must_symbol("cuDeviceTotalMem_v2");
    p_cuCtxCreate = (cuCtxCreate_t)must_symbol("cuCtxCreate_v2");
    p_cuCtxDestroy = (cuCtxDestroy_t)must_symbol("cuCtxDestroy_v2");
    p_cuCtxSynchronize = (cuCtxSynchronize_t)must_symbol("cuCtxSynchronize");
    p_cuMemAlloc = (cuMemAlloc_t)must_symbol("cuMemAlloc_v2");
    p_cuMemFree = (cuMemFree_t)must_symbol("cuMemFree_v2");
    p_cuMemcpyHtoD = (cuMemcpyHtoD_t)must_symbol("cuMemcpyHtoD_v2");
    p_cuMemcpyDtoH = (cuMemcpyDtoH_t)must_symbol("cuMemcpyDtoH_v2");
    p_cuMemcpyDtoD = (cuMemcpyDtoD_t)must_symbol("cuMemcpyDtoD_v2");
    p_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_t)must_symbol("cuModuleLoadFatBinary");
    p_cuModuleGetFunction = (cuModuleGetFunction_t)must_symbol("cuModuleGetFunction");
    p_cuLaunchKernel = (cuLaunchKernel_t)must_symbol("cuLaunchKernel");
    if (!p_cuInit || !p_cuDriverGetVersion || !p_cuDeviceGetCount || !p_cuDeviceGet ||
        !p_cuDeviceGetName || !p_cuDeviceTotalMem || !p_cuCtxCreate || !p_cuCtxDestroy ||
        !p_cuCtxSynchronize || !p_cuMemAlloc || !p_cuMemFree ||
        !p_cuMemcpyHtoD || !p_cuMemcpyDtoH || !p_cuMemcpyDtoD ||
        !p_cuModuleLoadFatBinary || !p_cuModuleGetFunction || !p_cuLaunchKernel) {
        return cudaErrorInitializationError;
    }

    result = p_cuInit(0);
    if (result != 0) {
        fprintf(stderr, "cudart_driver_shim_c: cuInit failed: %d\n", result);
        return map_result(result);
    }
    result = p_cuDeviceGetCount(&count);
    if (result != 0 || count <= 0) {
        fprintf(stderr, "cudart_driver_shim_c: cuDeviceGetCount failed: %d count=%d\n",
                result, count);
        return result == 0 ? cudaErrorInitializationError : map_result(result);
    }
    result = p_cuDeviceGet(&dev, 0);
    if (result != 0) {
        fprintf(stderr, "cudart_driver_shim_c: cuDeviceGet failed: %d\n", result);
        return map_result(result);
    }
    result = p_cuCtxCreate(&g_ctx, 0, dev);
    if (result != 0) {
        fprintf(stderr, "cudart_driver_shim_c: cuCtxCreate failed: %d\n", result);
    }
    return map_result(result);
}

static cudaError_t ensure_driver(void) {
    cudaError_t err;
    err = ensure_driver_locked();
    g_last_error = err;
    return err;
}

const char *cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "no error";
        case cudaErrorInvalidValue:
            return "invalid argument";
        case cudaErrorMemoryAllocation:
            return "out of memory";
        case cudaErrorInitializationError:
            return "initialization error";
        case cudaErrorInvalidMemcpyDirection:
            return "invalid copy direction for memcpy";
        default:
            return "unknown error";
    }
}

cudaError_t cudaDriverGetVersion(int *driverVersion) {
    CUresult result;
    if (!driverVersion) {
        return cudaErrorInvalidValue;
    }
    if (ensure_driver() != cudaSuccess) {
        *driverVersion = 0;
        return g_last_error;
    }
    result = p_cuDriverGetVersion(driverVersion);
    g_last_error = map_result(result);
    return g_last_error;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) {
        return cudaErrorInvalidValue;
    }
    *runtimeVersion = 12000;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    CUdeviceptr ptr = 0;
    CUresult result;
    if (!devPtr) {
        return cudaErrorInvalidValue;
    }
    if (ensure_driver() != cudaSuccess) {
        return g_last_error;
    }
    result = p_cuMemAlloc(&ptr, size);
    *devPtr = (void *)(uintptr_t)ptr;
    g_last_error = map_result(result);
    return g_last_error;
}

cudaError_t cudaFree(void *devPtr) {
    CUdeviceptr ptr;
    CUresult result;
    if (ensure_driver() != cudaSuccess) {
        return g_last_error;
    }
    ptr = (CUdeviceptr)(uintptr_t)devPtr;
    result = p_cuMemFree(ptr);
    g_last_error = map_result(result);
    return g_last_error;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    CUresult result = 0;
    if (ensure_driver() != cudaSuccess) {
        return g_last_error;
    }
    switch (kind) {
        case cudaMemcpyHostToHost:
            memcpy(dst, src, count);
            break;
        case cudaMemcpyHostToDevice:
            result = p_cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst, src, count);
            break;
        case cudaMemcpyDeviceToHost:
            result = p_cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src, count);
            break;
        case cudaMemcpyDeviceToDevice:
            result = p_cuMemcpyDtoD((CUdeviceptr)(uintptr_t)dst,
                                    (CUdeviceptr)(uintptr_t)src,
                                    count);
            break;
        default:
            g_last_error = cudaErrorInvalidMemcpyDirection;
            return g_last_error;
    }
    g_last_error = map_result(result);
    return g_last_error;
}

cudaError_t cudaDeviceSynchronize(void) {
    CUresult result;
    if (ensure_driver() != cudaSuccess) {
        return g_last_error;
    }
    result = p_cuCtxSynchronize();
    g_last_error = map_result(result);
    return g_last_error;
}

cudaError_t cudaGetLastError(void) {
    cudaError_t err = g_last_error;
    g_last_error = cudaSuccess;
    return err;
}

void **__cudaRegisterFatBinary(void *fatCubin) {
    g_fat_cubin = fatCubin;
    return (void **)&g_fat_cubin;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

void __cudaInitModule(void **fatCubinHandle) {
    (void)fatCubinHandle;
}

void __cudaRegisterFunction(void **fatCubinHandle,
                            const char *hostFun,
                            char *deviceFun,
                            const char *deviceName,
                            int thread_limit,
                            void *tid,
                            void *bid,
                            void *bDim,
                            void *gDim,
                            int *wSize) {
    int i;
    (void)fatCubinHandle;
    (void)deviceFun;
    (void)thread_limit;
    (void)tid;
    (void)bid;
    (void)bDim;
    (void)gDim;
    (void)wSize;
    for (i = 0; i < g_function_count; ++i) {
        if (g_functions[i].host == hostFun) {
            g_functions[i].device_name = deviceName;
            return;
        }
    }
    if (g_function_count < (int)(sizeof(g_functions) / sizeof(g_functions[0]))) {
        g_functions[g_function_count].host = hostFun;
        g_functions[g_function_count].device_name = deviceName;
        ++g_function_count;
    }
}

cudaError_t __cudaPushCallConfiguration(dim3 gridDim,
                                        dim3 blockDim,
                                        size_t sharedMem,
                                        cudaStream_t stream) {
    if (g_launch_config_count >= (int)(sizeof(g_launch_configs) / sizeof(g_launch_configs[0]))) {
        g_last_error = cudaErrorInvalidValue;
        return g_last_error;
    }
    g_launch_configs[g_launch_config_count].grid = gridDim;
    g_launch_configs[g_launch_config_count].block = blockDim;
    g_launch_configs[g_launch_config_count].shared = sharedMem;
    g_launch_configs[g_launch_config_count].stream = stream;
    ++g_launch_config_count;
    g_last_error = cudaSuccess;
    return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
                                       dim3 *blockDim,
                                       size_t *sharedMem,
                                       cudaStream_t *stream) {
    struct LaunchConfig cfg;
    if (g_launch_config_count <= 0) {
        g_last_error = cudaErrorInvalidValue;
        return g_last_error;
    }
    --g_launch_config_count;
    cfg = g_launch_configs[g_launch_config_count];
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

cudaError_t cudaLaunchKernel(const void *func,
                             dim3 gridDim,
                             dim3 blockDim,
                             void **args,
                             size_t sharedMem,
                             cudaStream_t stream) {
    const char *name = NULL;
    CUfunction kernel = NULL;
    CUresult result;
    int i;

    if (ensure_driver() != cudaSuccess) {
        return g_last_error;
    }

    for (i = 0; i < g_function_count; ++i) {
        if (g_functions[i].host == func) {
            name = g_functions[i].device_name;
            break;
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
        result = p_cuModuleLoadFatBinary(&g_module, g_fat_cubin);
        if (result != 0) {
            g_last_error = map_result(result);
            return g_last_error;
        }
    }

    result = p_cuModuleGetFunction(&kernel, g_module, name);
    if (result != 0) {
        g_last_error = map_result(result);
        return g_last_error;
    }
    result = p_cuLaunchKernel(kernel,
                              gridDim.x, gridDim.y, gridDim.z,
                              blockDim.x, blockDim.y, blockDim.z,
                              (unsigned int)sharedMem,
                              (CUstream)stream,
                              args,
                              NULL);
    g_last_error = map_result(result);
    return g_last_error;
}
