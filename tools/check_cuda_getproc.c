#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int CUresult;
typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDriverGetVersion_t)(int *);
typedef CUresult (*cuGetProcAddress_t)(const char *, void **, int, unsigned long long);

static void *load_symbol(void *lib, const char *name) {
    void *sym = dlsym(lib, name);
    if (!sym) {
        fprintf(stderr, "missing symbol: %s\n", name);
        exit(2);
    }
    return sym;
}

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen libcuda.so.1 failed: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)load_symbol(lib, "cuInit");
    cuDriverGetVersion_t direct =
        (cuDriverGetVersion_t)load_symbol(lib, "cuDriverGetVersion");
    CUresult result = cuInit(0);
    printf("cuInit -> %d\n", result);

    cuGetProcAddress_t get_proc =
        (cuGetProcAddress_t)load_symbol(lib, "cuGetProcAddress");

    int version = -1;
    result = direct(&version);
    printf("direct cuDriverGetVersion -> result=%d version=%d\n", result, version);

    void *proc = NULL;
    result = get_proc("cuDriverGetVersion", &proc, 12000, 0);
    printf("cuGetProcAddress(cuDriverGetVersion, 12000, default) -> result=%d proc=%p\n",
           result, proc);
    if (proc) {
        version = -1;
        result = ((cuDriverGetVersion_t)proc)(&version);
        printf("proc cuDriverGetVersion -> result=%d version=%d\n", result, version);
    }

    proc = NULL;
    result = get_proc("cuDriverGetVersion", &proc, 12000, 1);
    printf("cuGetProcAddress(cuDriverGetVersion, 12000, per-thread default stream) -> result=%d proc=%p\n",
           result, proc);
    if (proc) {
        version = -1;
        result = ((cuDriverGetVersion_t)proc)(&version);
        printf("ptds proc cuDriverGetVersion -> result=%d version=%d\n", result, version);
    }

    return 0;
}
