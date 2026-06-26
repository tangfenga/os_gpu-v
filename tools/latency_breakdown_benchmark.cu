#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void tiny_kernel(float *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] += 1.0f;
    }
}

constexpr uint64_t kTraceDefault = 0;
constexpr uint64_t kTraceEmptyStreamSync = 1;
constexpr uint64_t kTraceD2DSync = 2;
constexpr uint64_t kTraceKernelSync = 3;
constexpr uint64_t kTraceBatchDrainSync = 4;

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

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

template <typename Fn>
static std::vector<double> collect_samples_us(int iters, int warmup, cudaStream_t stream, Fn fn) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    check_cuda(cudaStreamSynchronize(stream), "warmup stream sync");

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0);
    }
    return samples;
}

static double percentile(const std::vector<double> &samples, double p) {
    if (samples.empty()) {
        return 0.0;
    }
    const size_t idx = std::min(samples.size() - 1, static_cast<size_t>(p * samples.size()));
    return samples[idx];
}

static void print_stats(const char *name, size_t bytes, std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    double sum = 0.0;
    for (double value : samples) {
        sum += value;
    }
    const double mean = samples.empty() ? 0.0 : sum / static_cast<double>(samples.size());
    std::printf(
        "metric=%s bytes=%zu mean_us=%.3f p50_us=%.3f p95_us=%.3f p99_us=%.3f\n",
        name,
        bytes,
        mean,
        percentile(samples, 0.50),
        percentile(samples, 0.95),
        percentile(samples, 0.99));
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const size_t bytes = argc > 2 ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) : 4ull * 1024ull * 1024ull;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 20;
    if (iters <= 0 || warmup < 0 || bytes == 0) {
        std::fprintf(stderr, "usage: %s [iters] [bytes] [warmup]\n", argv[0]);
        return 1;
    }

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    void *d_a = nullptr;
    void *d_b = nullptr;
    void *h_a = nullptr;
    void *h_b = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check_cuda(cudaHostAlloc(&h_a, bytes, cudaHostAllocDefault), "cudaHostAlloc h_a");
    check_cuda(cudaHostAlloc(&h_b, bytes, cudaHostAllocDefault), "cudaHostAlloc h_b");
    std::memset(h_a, 0x5a, bytes);
    std::memset(h_b, 0, bytes);

    check_cuda(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream), "initial H2D");
    tiny_kernel<<<1, 32, 0, stream>>>(static_cast<float *>(d_a));
    check_cuda(cudaGetLastError(), "initial launch");
    check_cuda(cudaStreamSynchronize(stream), "initial stream sync");

    print_stats("empty_stream_sync", 0, collect_samples_us(iters, warmup, stream, [&] {
        ScopedTraceLabel label(kTraceEmptyStreamSync);
        check_cuda(cudaStreamSynchronize(stream), "empty stream sync");
    }));

    print_stats("h2d_pinned", bytes, collect_samples_us(iters, warmup, stream, [&] {
        check_cuda(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream), "H2D pinned");
        check_cuda(cudaStreamSynchronize(stream), "H2D stream sync");
    }));

    print_stats("d2h_pinned", bytes, collect_samples_us(iters, warmup, stream, [&] {
        check_cuda(cudaMemcpyAsync(h_b, d_a, bytes, cudaMemcpyDeviceToHost, stream), "D2H pinned");
        check_cuda(cudaStreamSynchronize(stream), "D2H stream sync");
    }));

    print_stats("d2d_submit_only", bytes, collect_samples_us(iters, warmup, stream, [&] {
        check_cuda(cudaMemcpyAsync(d_b, d_a, bytes, cudaMemcpyDeviceToDevice, stream), "D2D submit only");
    }));
    check_cuda(cudaStreamSynchronize(stream), "post D2D submit-only stream sync");

    print_stats("d2d_stream_sync", bytes, collect_samples_us(iters, warmup, stream, [&] {
        ScopedTraceLabel label(kTraceD2DSync);
        check_cuda(cudaMemcpyAsync(d_b, d_a, bytes, cudaMemcpyDeviceToDevice, stream), "D2D async");
        check_cuda(cudaStreamSynchronize(stream), "D2D stream sync");
    }));

    print_stats("kernel_launch_only", 0, collect_samples_us(iters, warmup, stream, [&] {
        tiny_kernel<<<1, 32, 0, stream>>>(static_cast<float *>(d_a));
        check_cuda(cudaGetLastError(), "kernel launch only");
    }));
    check_cuda(cudaStreamSynchronize(stream), "post launch-only stream sync");

    print_stats("kernel_launch_stream_sync", 0, collect_samples_us(iters, warmup, stream, [&] {
        ScopedTraceLabel label(kTraceKernelSync);
        tiny_kernel<<<1, 32, 0, stream>>>(static_cast<float *>(d_a));
        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaStreamSynchronize(stream), "kernel stream sync");
    }));

    print_stats("batch_drain_sync", 0, collect_samples_us(iters, warmup, stream, [&] {
        ScopedTraceLabel label(kTraceBatchDrainSync);
        for (int i = 0; i < 8; ++i) {
            tiny_kernel<<<1, 32, 0, stream>>>(static_cast<float *>(d_a));
            check_cuda(cudaGetLastError(), "batch launch");
        }
        check_cuda(cudaStreamSynchronize(stream), "batch stream sync");
    }));

    check_cuda(cudaFreeHost(h_b), "cudaFreeHost h_b");
    check_cuda(cudaFreeHost(h_a), "cudaFreeHost h_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}
