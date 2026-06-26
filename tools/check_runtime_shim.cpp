#include <stdio.h>

using cudaError_t = int;

extern "C" cudaError_t cudaDriverGetVersion(int *driverVersion);
extern "C" cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
extern "C" const char *cudaGetErrorString(cudaError_t error);

int main() {
    int driver = 0;
    int runtime = 0;
    cudaError_t err = cudaDriverGetVersion(&driver);
    printf("cudaDriverGetVersion: err=%d (%s) version=%d\n",
           err, cudaGetErrorString(err), driver);
    err = cudaRuntimeGetVersion(&runtime);
    printf("cudaRuntimeGetVersion: err=%d (%s) version=%d\n",
           err, cudaGetErrorString(err), runtime);
    return driver > 0 ? 0 : 1;
}
