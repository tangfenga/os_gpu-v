#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

const char *kTinyKernelPtx = R"ptx(
.version 7.0
.target sm_50
.address_size 64

.visible .entry tiny_kernel(
    .param .u64 out_param
) {
    .reg .pred %p;
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    .reg .f32 %f<3>;

    ld.param.u64 %rd1, [out_param];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    or.b32 %r1, %r1, %r2;
    setp.ne.u32 %p, %r1, 0;
    @%p bra DONE;
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.f32 %f1, [%rd2];
    add.f32 %f2, %f1, 0f3F800000;
    st.global.f32 [%rd2], %f2;
DONE:
    ret;
}
)ptx";

using Clock = std::chrono::steady_clock;

void check_driver(CUresult result, const char *what) {
    if (result == CUDA_SUCCESS) {
        return;
    }
    const char *name = nullptr;
    const char *text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::fprintf(stderr, "%s failed: %s: %s\n", what, name ? name : "CUDA_ERROR_UNKNOWN", text ? text : "");
    std::exit(1);
}

template <typename Fn>
std::vector<double> collect_samples_us(int iters, int warmup, CUstream stream, Fn fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    check_driver(cuStreamSynchronize(stream), "warmup stream sync");

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto begin = Clock::now();
        fn();
        const auto end = Clock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0);
    }
    return samples;
}

double percentile(const std::vector<double> &samples, double p) {
    if (samples.empty()) {
        return 0.0;
    }
    const size_t idx = std::min(samples.size() - 1, static_cast<size_t>(p * samples.size()));
    return samples[idx];
}

void print_stats(const char *name, size_t bytes, std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    double sum = 0.0;
    for (double value : samples) {
        sum += value;
    }
    const double mean = samples.empty() ? 0.0 : sum / static_cast<double>(samples.size());
    std::printf(
        "group=server_local_driver metric=%s bytes=%zu mean_us=%.3f p50_us=%.3f p95_us=%.3f p99_us=%.3f\n",
        name,
        bytes,
        mean,
        percentile(samples, 0.50),
        percentile(samples, 0.95),
        percentile(samples, 0.99));
}

}  // namespace

int main(int argc, char **argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const size_t bytes = argc > 2 ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) : 4ull * 1024ull * 1024ull;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 20;
    if (iters <= 0 || warmup < 0 || bytes == 0) {
        std::fprintf(stderr, "usage: %s [iters] [bytes] [warmup]\n", argv[0]);
        return 1;
    }

    check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    check_driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    CUstream stream = nullptr;
    check_driver(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

    CUdeviceptr d_a = 0;
    CUdeviceptr d_b = 0;
    check_driver(cuMemAlloc(&d_a, bytes), "cuMemAlloc d_a");
    check_driver(cuMemAlloc(&d_b, bytes), "cuMemAlloc d_b");

    CUmodule module = nullptr;
    CUfunction function = nullptr;
    check_driver(cuModuleLoadDataEx(&module, kTinyKernelPtx, 0, nullptr, nullptr), "cuModuleLoadDataEx");
    check_driver(cuModuleGetFunction(&function, module, "tiny_kernel"), "cuModuleGetFunction");

    void *kernel_args[] = {&d_a};
    check_driver(cuMemcpyDtoDAsync(d_b, d_a, bytes, stream), "initial D2D");
    check_driver(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, stream, kernel_args, nullptr), "initial launch");
    check_driver(cuStreamSynchronize(stream), "initial sync");

    print_stats("d2d_submit_only", bytes, collect_samples_us(iters, warmup, stream, [&] {
        check_driver(cuMemcpyDtoDAsync(d_b, d_a, bytes, stream), "cuMemcpyDtoDAsync");
    }));
    check_driver(cuStreamSynchronize(stream), "post D2D sync");

    print_stats("kernel_launch_only", 0, collect_samples_us(iters, warmup, stream, [&] {
        check_driver(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, stream, kernel_args, nullptr), "cuLaunchKernel");
    }));
    check_driver(cuStreamSynchronize(stream), "post launch sync");

    check_driver(cuModuleUnload(module), "cuModuleUnload");
    check_driver(cuMemFree(d_b), "cuMemFree d_b");
    check_driver(cuMemFree(d_a), "cuMemFree d_a");
    check_driver(cuStreamDestroy(stream), "cuStreamDestroy");
    check_driver(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return 0;
}
