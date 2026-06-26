#include <dlfcn.h>
#include <stdio.h>

typedef int (*cuInit_t)(unsigned int);

int main(void) {
    void *lib = dlopen("/usr/lib/wsl/lib/libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        printf("dlopen failed: %s\n", dlerror());
        return 1;
    }
    cuInit_t cuInit = (cuInit_t)dlsym(lib, "cuInit");
    if (!cuInit) {
        printf("dlsym failed: %s\n", dlerror());
        return 2;
    }
    int r = cuInit(0);
    printf("cuInit explicit path RTLD_LOCAL -> %d\n", r);
    return r == 0 ? 0 : 3;
}

