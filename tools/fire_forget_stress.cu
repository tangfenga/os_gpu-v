#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>

__global__ void set_index_kernel(int *out, int index, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[index] = value;
    }
}

__global__ void add_one_kernel(int *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(out, 1);
    }
}

constexpr uint64_t kTraceDefault = 0;
constexpr uint64_t kTraceD2DSync = 2;
constexpr uint64_t kTraceKernelSync = 3;

using set_trace_label_fn = void (*)(uint64_t);

static set_trace_label_fn trace_label_fn() {
    static set_trace_label_fn fn =
        reinterpret_cast<set_trace_label_fn>(dlsym(RTLD_DEFAULT, "vgpuSetTraceLabel"));
    return fn;
}

static void set_trace_label(uint64_t label) {
    if (auto fn = trace_label_fn()) {
        fn(label);
    }
}

class ScopedTraceLabel {
public:
    explicit ScopedTraceLabel(uint64_t label) {
        set_trace_label(label);
    }

    ~ScopedTraceLabel() {
        set_trace_label(kTraceDefault);
    }
};

static void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s (%d)\n", what, cudaGetErrorString(err), static_cast<int>(err));
        std::exit(1);
    }
}

static void expect_error(cudaError_t err, const char *what) {
    if (err == cudaSuccess) {
        std::fprintf(stderr, "%s unexpectedly succeeded\n", what);
        std::exit(1);
    }
    std::printf("error_case %s returned %s (%d)\n", what, cudaGetErrorString(err), static_cast<int>(err));
}

static double elapsed_us(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

static void run_d2d_stress(int count) {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags d2d");

    std::vector<int> src(count);
    std::vector<int> dst(count, 0);
    for (int i = 0; i < count; ++i) {
        src[i] = 0x13570000 ^ (i * 17);
    }

    int *d_src = nullptr;
    int *d_dst = nullptr;
    check(cudaMalloc(&d_src, sizeof(int) * count), "cudaMalloc d_src");
    check(cudaMalloc(&d_dst, sizeof(int) * count), "cudaMalloc d_dst");
    check(cudaMemcpyAsync(d_src, src.data(), sizeof(int) * count, cudaMemcpyHostToDevice, stream), "H2D d_src");
    check(cudaMemcpyAsync(d_dst, dst.data(), sizeof(int) * count, cudaMemcpyHostToDevice, stream), "H2D d_dst init");
    check(cudaStreamSynchronize(stream), "initial d2d sync");

    const auto submit_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        check(cudaMemcpyAsync(d_dst + i, d_src + i, sizeof(int), cudaMemcpyDeviceToDevice, stream), "D2D submit");
    }
    const auto submit_end = std::chrono::steady_clock::now();
    const auto sync_begin = std::chrono::steady_clock::now();
    {
        ScopedTraceLabel label(kTraceD2DSync);
        check(cudaStreamSynchronize(stream), "D2D drain sync");
    }
    const auto sync_end = std::chrono::steady_clock::now();
    check(cudaMemcpy(dst.data(), d_dst, sizeof(int) * count, cudaMemcpyDeviceToHost), "D2H d2d dst");

    for (int i = 0; i < count; ++i) {
        if (dst[i] != src[i]) {
            std::fprintf(stderr, "D2D mismatch count=%d index=%d expected=%d got=%d\n", count, i, src[i], dst[i]);
            std::exit(1);
        }
    }

    check(cudaFree(d_dst), "cudaFree d_dst");
    check(cudaFree(d_src), "cudaFree d_src");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy d2d");
    const double submit_us = elapsed_us(submit_begin, submit_end);
    const double sync_us = elapsed_us(sync_begin, sync_end);
    std::printf("stress d2d count=%d submit_us=%.3f submit_per_cmd_us=%.3f sync_us=%.3f total_us=%.3f throughput_cmd_s=%.3f ok\n",
                count,
                submit_us,
                submit_us / static_cast<double>(count),
                sync_us,
                submit_us + sync_us,
                static_cast<double>(count) / ((submit_us + sync_us) / 1000000.0));
}

static void run_kernel_stress(int count) {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags kernel");

    int *d_out = nullptr;
    std::vector<int> init(count, 0);
    check(cudaMalloc(&d_out, sizeof(int) * count), "cudaMalloc d_out");
    check(cudaMemcpyAsync(d_out, init.data(), sizeof(int) * count, cudaMemcpyHostToDevice, stream), "H2D d_out init");
    check(cudaStreamSynchronize(stream), "initial kernel sync");

    const auto submit_begin = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        const int value = 0x24680000 ^ (i * 31);
        set_index_kernel<<<1, 1, 0, stream>>>(d_out, i, value);
        check(cudaGetLastError(), "kernel submit");
    }
    const auto submit_end = std::chrono::steady_clock::now();
    const auto sync_begin = std::chrono::steady_clock::now();
    {
        ScopedTraceLabel label(kTraceKernelSync);
        check(cudaStreamSynchronize(stream), "kernel drain sync");
    }
    const auto sync_end = std::chrono::steady_clock::now();

    std::vector<int> out(count);
    check(cudaMemcpy(out.data(), d_out, sizeof(int) * count, cudaMemcpyDeviceToHost), "D2H kernel out");
    for (int i = 0; i < count; ++i) {
        const int expected = 0x24680000 ^ (i * 31);
        if (out[i] != expected) {
            std::fprintf(stderr, "kernel payload mismatch count=%d index=%d expected=%d got=%d\n", count, i, expected, out[i]);
            std::exit(1);
        }
    }

    int *d_counter = nullptr;
    int host_counter = 0;
    check(cudaMalloc(&d_counter, sizeof(int)), "cudaMalloc d_counter");
    check(cudaMemcpyAsync(d_counter, &host_counter, sizeof(int), cudaMemcpyHostToDevice, stream), "H2D d_counter init");
    check(cudaStreamSynchronize(stream), "counter init sync");
    for (int i = 0; i < count; ++i) {
        add_one_kernel<<<1, 1, 0, stream>>>(d_counter);
        check(cudaGetLastError(), "counter kernel submit");
    }
    {
        ScopedTraceLabel label(kTraceKernelSync);
        check(cudaStreamSynchronize(stream), "counter drain sync");
    }
    check(cudaMemcpy(&host_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost), "D2H counter");
    if (host_counter != count) {
        std::fprintf(stderr, "kernel lost command count=%d got=%d\n", count, host_counter);
        std::exit(1);
    }

    check(cudaFree(d_counter), "cudaFree d_counter");
    check(cudaFree(d_out), "cudaFree d_out");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy kernel");
    const double submit_us = elapsed_us(submit_begin, submit_end);
    const double sync_us = elapsed_us(sync_begin, sync_end);
    std::printf("stress kernel count=%d submit_us=%.3f submit_per_cmd_us=%.3f sync_us=%.3f total_us=%.3f throughput_cmd_s=%.3f ok\n",
                count,
                submit_us,
                submit_us / static_cast<double>(count),
                sync_us,
                submit_us + sync_us,
                static_cast<double>(count) / ((submit_us + sync_us) / 1000000.0));
}

static void run_error_semantics() {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags error");

    int *d_a = nullptr;
    int *d_b = nullptr;
    check(cudaMalloc(&d_a, sizeof(int)), "cudaMalloc error d_a");
    check(cudaMalloc(&d_b, sizeof(int)), "cudaMalloc error d_b");

    check(cudaMemcpyAsync(d_b, reinterpret_cast<void *>(0x12345678ull), sizeof(int), cudaMemcpyDeviceToDevice, stream),
          "invalid D2D submit should be fire-and-forget success");
    expect_error(cudaStreamSynchronize(stream), "invalid_d2d_stream_sync");
    check(cudaStreamSynchronize(stream), "pending error cleared after invalid D2D");

    cudaStream_t bogus_stream = reinterpret_cast<cudaStream_t>(0xCAFECAFEull);
    set_index_kernel<<<1, 1, 0, bogus_stream>>>(d_a, 0, 7);
    check(cudaGetLastError(), "invalid stream launch submit should be fire-and-forget success");
    expect_error(cudaDeviceSynchronize(), "invalid_stream_device_sync");
    check(cudaDeviceSynchronize(), "pending error cleared after invalid stream");

    check(cudaMemcpyAsync(d_b, reinterpret_cast<void *>(0x11110000ull), sizeof(int), cudaMemcpyDeviceToDevice, stream),
          "first invalid D2D submit");
    set_index_kernel<<<1, 1, 0, bogus_stream>>>(d_a, 0, 9);
    check(cudaGetLastError(), "second invalid launch submit");
    expect_error(cudaDeviceSynchronize(), "first_pending_error_not_overwritten");
    check(cudaDeviceSynchronize(), "pending error cleared after overwrite test");

    check(cudaFree(d_b), "cudaFree error d_b");
    check(cudaFree(d_a), "cudaFree error d_a");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy error");
    std::printf("error semantics ok\n");
}

int main() {
    const int counts[] = {1, 16, 64, 256, 1024, 2048};
    for (int count : counts) {
        run_d2d_stress(count);
        run_kernel_stress(count);
    }
    run_error_semantics();
    std::printf("fire_forget_stress passed\n");
    return 0;
}
