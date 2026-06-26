#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int CUdevice;
typedef int CUresult;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDriverGetVersion_t)(int *);
typedef CUresult (*cuDeviceGetCount_t)(int *);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDeviceGetName_t)(char *, int, CUdevice);
typedef CUresult (*cuDeviceTotalMem_t)(size_t *, CUdevice);
typedef const char *(*cuGetErrorName_t)(CUresult);
typedef unsigned long long CUdeviceptr;
typedef CUresult (*cuCtxCreate_t)(void **, unsigned int, CUdevice);
typedef CUresult (*cuCtxDestroy_t)(void *);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void *, CUdeviceptr, size_t);

static void *load_symbol(void *lib, const char *name) {
    void *sym = dlsym(lib, name);
    if (!sym) {
        fprintf(stderr, "missing symbol: %s\n", name);
        exit(2);
    }
    return sym;
}

static const char *error_name(cuGetErrorName_t get_error_name, CUresult result) {
    const char *name = NULL;
    if (get_error_name) {
        name = get_error_name(result);
    }
    return name ? name : "UNKNOWN";
}

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)load_symbol(lib, "cuInit");
    cuDriverGetVersion_t cuDriverGetVersion =
        (cuDriverGetVersion_t)load_symbol(lib, "cuDriverGetVersion");
    cuDeviceGetCount_t cuDeviceGetCount =
        (cuDeviceGetCount_t)load_symbol(lib, "cuDeviceGetCount");
    cuDeviceGet_t cuDeviceGet = (cuDeviceGet_t)load_symbol(lib, "cuDeviceGet");
    cuDeviceGetName_t cuDeviceGetName =
        (cuDeviceGetName_t)load_symbol(lib, "cuDeviceGetName");
    cuDeviceTotalMem_t cuDeviceTotalMem =
        (cuDeviceTotalMem_t)load_symbol(lib, "cuDeviceTotalMem_v2");
    cuGetErrorName_t cuGetErrorName =
        (cuGetErrorName_t)dlsym(lib, "cuGetErrorName");
    cuCtxCreate_t cuCtxCreate = (cuCtxCreate_t)load_symbol(lib, "cuCtxCreate_v2");
    cuCtxDestroy_t cuCtxDestroy = (cuCtxDestroy_t)load_symbol(lib, "cuCtxDestroy_v2");
    cuMemAlloc_t cuMemAlloc = (cuMemAlloc_t)load_symbol(lib, "cuMemAlloc_v2");
    cuMemFree_t cuMemFree = (cuMemFree_t)load_symbol(lib, "cuMemFree_v2");
    cuMemcpyHtoD_t cuMemcpyHtoD =
        (cuMemcpyHtoD_t)load_symbol(lib, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_t cuMemcpyDtoH =
        (cuMemcpyDtoH_t)load_symbol(lib, "cuMemcpyDtoH_v2");

    CUresult result = cuInit(0);
    if (result != 0) {
        fprintf(stderr, "cuInit failed: %s (%d)\n", error_name(cuGetErrorName, result), result);
        return 3;
    }

    int driver_version = 0;
    result = cuDriverGetVersion(&driver_version);
    if (result != 0) {
        fprintf(stderr, "cuDriverGetVersion failed: %s (%d)\n",
                error_name(cuGetErrorName, result), result);
        return 4;
    }

    int count = 0;
    result = cuDeviceGetCount(&count);
    if (result != 0) {
        fprintf(stderr, "cuDeviceGetCount failed: %s (%d)\n",
                error_name(cuGetErrorName, result), result);
        return 5;
    }

    printf("CUDA driver version: %d\n", driver_version);
    printf("CUDA device count: %d\n", count);

    for (int i = 0; i < count; ++i) {
        CUdevice device;
        char name[256] = {0};
        size_t total_mem = 0;

        result = cuDeviceGet(&device, i);
        if (result != 0) {
            fprintf(stderr, "cuDeviceGet(%d) failed: %s (%d)\n",
                    i, error_name(cuGetErrorName, result), result);
            return 6;
        }

        result = cuDeviceGetName(name, sizeof(name), device);
        if (result != 0) {
            fprintf(stderr, "cuDeviceGetName(%d) failed: %s (%d)\n",
                    i, error_name(cuGetErrorName, result), result);
            return 7;
        }

        result = cuDeviceTotalMem(&total_mem, device);
        if (result != 0) {
            fprintf(stderr, "cuDeviceTotalMem(%d) failed: %s (%d)\n",
                    i, error_name(cuGetErrorName, result), result);
            return 8;
        }

        printf("GPU %d: %s, total memory: %.2f GiB\n",
               i, name, (double)total_mem / 1024.0 / 1024.0 / 1024.0);

        if (i == 0) {
            void *ctx = NULL;
            CUdeviceptr dptr = 0;
            int host_in[4] = {1, 2, 3, 4};
            int host_out[4] = {0, 0, 0, 0};

            result = cuCtxCreate(&ctx, 0, device);
            if (result != 0) {
                fprintf(stderr, "cuCtxCreate failed: %s (%d)\n",
                        error_name(cuGetErrorName, result), result);
                return 9;
            }
            result = cuMemAlloc(&dptr, sizeof(host_in));
            if (result != 0) {
                fprintf(stderr, "cuMemAlloc failed: %s (%d)\n",
                        error_name(cuGetErrorName, result), result);
                return 10;
            }
            result = cuMemcpyHtoD(dptr, host_in, sizeof(host_in));
            if (result != 0) {
                fprintf(stderr, "cuMemcpyHtoD failed: %s (%d)\n",
                        error_name(cuGetErrorName, result), result);
                return 11;
            }
            result = cuMemcpyDtoH(host_out, dptr, sizeof(host_out));
            if (result != 0) {
                fprintf(stderr, "cuMemcpyDtoH failed: %s (%d)\n",
                        error_name(cuGetErrorName, result), result);
                return 12;
            }
            for (int j = 0; j < 4; ++j) {
                if (host_out[j] != host_in[j]) {
                    fprintf(stderr, "driver memcpy mismatch at %d\n", j);
                    return 13;
                }
            }
            result = cuMemFree(dptr);
            if (result != 0) {
                fprintf(stderr, "cuMemFree failed: %s (%d)\n",
                        error_name(cuGetErrorName, result), result);
                return 14;
            }
            result = cuCtxDestroy(ctx);
            if (result != 0) {
                fprintf(stderr, "cuCtxDestroy failed: %s (%d)\n",
                        error_name(cuGetErrorName, result), result);
                return 15;
            }
            printf("Driver API malloc/H2D/D2H/free smoke test passed\n");
        }
    }

    dlclose(lib);
    return 0;
}
