#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void transform_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

using Clock = std::chrono::steady_clock;

static void check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s (%d)\n", what, cudaGetErrorString(err), static_cast<int>(err));
        std::exit(1);
    }
}

static double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

static double percentile(const std::vector<double> &sorted, double p) {
    if (sorted.empty()) {
        return 0.0;
    }
    const size_t idx = std::min(sorted.size() - 1, static_cast<size_t>(p * sorted.size()));
    return sorted[idx];
}

static void print_stats(
    const char *mode,
    int repeats,
    size_t bytes,
    std::vector<double> totals_us) {
    std::sort(totals_us.begin(), totals_us.end());
    double sum = 0.0;
    for (double value : totals_us) {
        sum += value;
    }
    const double mean = sum / static_cast<double>(totals_us.size());
    const double median = percentile(totals_us, 0.50);
    const double p95 = percentile(totals_us, 0.95);
    const double min_v = totals_us.front();
    const double max_v = totals_us.back();
    const double mib_per_iter = static_cast<double>(bytes * 2) / 1024.0 / 1024.0;
    const double throughput_mib_s =
        mib_per_iter * static_cast<double>(repeats) / (median / 1000000.0);

    std::printf(
        "pipeline_stats mode=%s repeats=%d trials=%zu bytes_per_copy=%zu median_us=%.3f mean_us=%.3f p95_us=%.3f min_us=%.3f max_us=%.3f median_per_iter_us=%.3f throughput_mib_s=%.3f\n",
        mode,
        repeats,
        totals_us.size(),
        bytes,
        median,
        mean,
        p95,
        min_v,
        max_v,
        median / static_cast<double>(repeats),
        throughput_mib_s);
}

struct PipelineState {
    int elements = 0;
    size_t bytes = 0;
    int grid = 0;
    int block = 256;
    cudaStream_t stream = nullptr;
    float *h_in = nullptr;
    float *h_out = nullptr;
    float *d_in = nullptr;
    float *d_out = nullptr;
};

static void verify_output(const PipelineState &state, const char *mode, int repeats) {
    for (int i = 0; i < state.elements; i += std::max(1, state.elements / 16)) {
        const float expected = state.h_in[i] * 2.0f + 1.0f;
        if (state.h_out[i] != expected) {
            std::fprintf(
                stderr,
                "pipeline mismatch mode=%s repeats=%d index=%d expected=%f got=%f\n",
                mode,
                repeats,
                i,
                expected,
                state.h_out[i]);
            std::exit(1);
        }
    }
}

static void async_pipeline_once(PipelineState &state) {
    check(cudaMemcpyAsync(state.d_in, state.h_in, state.bytes, cudaMemcpyHostToDevice, state.stream), "async H2D");
    transform_kernel<<<state.grid, state.block, 0, state.stream>>>(state.d_in, state.d_out, state.elements);
    check(cudaGetLastError(), "async launch");
    check(cudaMemcpyAsync(state.h_out, state.d_out, state.bytes, cudaMemcpyDeviceToHost, state.stream), "async D2H");
    check(cudaStreamSynchronize(state.stream), "async sync");
}

static void sync_pipeline_once(PipelineState &state) {
    check(cudaMemcpy(state.d_in, state.h_in, state.bytes, cudaMemcpyHostToDevice), "sync H2D");
    transform_kernel<<<state.grid, state.block>>>(state.d_in, state.d_out, state.elements);
    check(cudaGetLastError(), "sync launch");
    check(cudaMemcpy(state.h_out, state.d_out, state.bytes, cudaMemcpyDeviceToHost), "sync D2H");
    check(cudaStreamSynchronize(nullptr), "sync stream sync");
}

static void warmup(PipelineState &state, int warmup_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        async_pipeline_once(state);
    }
}

static void print_first_call_breakdown(PipelineState &state) {
    const auto h2d_begin = Clock::now();
    check(cudaMemcpyAsync(state.d_in, state.h_in, state.bytes, cudaMemcpyHostToDevice, state.stream), "first H2D");
    const auto h2d_end = Clock::now();

    const auto kernel_begin = Clock::now();
    transform_kernel<<<state.grid, state.block, 0, state.stream>>>(state.d_in, state.d_out, state.elements);
    check(cudaGetLastError(), "first launch");
    const auto kernel_end = Clock::now();

    const auto d2h_begin = Clock::now();
    check(cudaMemcpyAsync(state.h_out, state.d_out, state.bytes, cudaMemcpyDeviceToHost, state.stream), "first D2H");
    const auto d2h_end = Clock::now();

    const auto sync_begin = Clock::now();
    check(cudaStreamSynchronize(state.stream), "first stream sync");
    const auto sync_end = Clock::now();

    std::printf(
        "first_call h2d_submit_us=%.3f kernel_submit_us=%.3f d2h_submit_us=%.3f stream_sync_us=%.3f\n",
        elapsed_us(h2d_begin, h2d_end),
        elapsed_us(kernel_begin, kernel_end),
        elapsed_us(d2h_begin, d2h_end),
        elapsed_us(sync_begin, sync_end));
    verify_output(state, "first_call", 1);
}

static void print_steady_single_breakdown(PipelineState &state) {
    const auto h2d_begin = Clock::now();
    check(cudaMemcpyAsync(state.d_in, state.h_in, state.bytes, cudaMemcpyHostToDevice, state.stream), "steady H2D");
    const auto h2d_end = Clock::now();

    const auto kernel_begin = Clock::now();
    transform_kernel<<<state.grid, state.block, 0, state.stream>>>(state.d_in, state.d_out, state.elements);
    check(cudaGetLastError(), "steady launch");
    const auto kernel_end = Clock::now();

    const auto d2h_begin = Clock::now();
    check(cudaMemcpyAsync(state.h_out, state.d_out, state.bytes, cudaMemcpyDeviceToHost, state.stream), "steady D2H");
    const auto d2h_end = Clock::now();

    const auto sync_begin = Clock::now();
    check(cudaStreamSynchronize(state.stream), "steady sync");
    const auto sync_end = Clock::now();

    std::printf(
        "steady_single_breakdown h2d_submit_us=%.3f kernel_submit_us=%.3f d2h_submit_us=%.3f stream_sync_us=%.3f total_us=%.3f\n",
        elapsed_us(h2d_begin, h2d_end),
        elapsed_us(kernel_begin, kernel_end),
        elapsed_us(d2h_begin, d2h_end),
        elapsed_us(sync_begin, sync_end),
        elapsed_us(h2d_begin, sync_end));
    verify_output(state, "steady_breakdown", 1);
}

template <typename Fn>
static void run_trials(PipelineState &state, const char *mode, int repeats, int trials, Fn fn) {
    std::vector<double> totals;
    totals.reserve(trials);
    for (int t = 0; t < trials; ++t) {
        const auto begin = Clock::now();
        for (int r = 0; r < repeats; ++r) {
            fn();
        }
        const auto end = Clock::now();
        verify_output(state, mode, repeats);
        totals.push_back(elapsed_us(begin, end));
    }
    print_stats(mode, repeats, state.bytes, std::move(totals));
}

int main(int argc, char **argv) {
    const int elements = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    const int warmup_iters = argc > 2 ? std::atoi(argv[2]) : 200;
    const int trials = argc > 3 ? std::atoi(argv[3]) : 10;
    if (elements <= 0 || warmup_iters < 0 || trials <= 0) {
        std::fprintf(stderr, "usage: %s [elements] [warmup_iters] [trials]\n", argv[0]);
        return 1;
    }

    PipelineState state;
    state.elements = elements;
    state.bytes = static_cast<size_t>(elements) * sizeof(float);
    state.grid = (elements + state.block - 1) / state.block;

    const auto stream_begin = Clock::now();
    check(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    const auto stream_end = Clock::now();

    const auto host_alloc_begin = Clock::now();
    check(cudaHostAlloc(&state.h_in, state.bytes, cudaHostAllocDefault), "cudaHostAlloc h_in");
    check(cudaHostAlloc(&state.h_out, state.bytes, cudaHostAllocDefault), "cudaHostAlloc h_out");
    const auto host_alloc_end = Clock::now();

    const auto malloc_begin = Clock::now();
    check(cudaMalloc(&state.d_in, state.bytes), "cudaMalloc d_in");
    check(cudaMalloc(&state.d_out, state.bytes), "cudaMalloc d_out");
    const auto malloc_end = Clock::now();

    for (int i = 0; i < elements; ++i) {
        state.h_in[i] = static_cast<float>(i % 1024) * 0.25f;
        state.h_out[i] = 0.0f;
    }

    std::printf(
        "setup stream_create_us=%.3f cudaHostAlloc_us=%.3f cudaMalloc_us=%.3f bytes_per_copy=%zu warmup_iters=%d trials=%d\n",
        elapsed_us(stream_begin, stream_end),
        elapsed_us(host_alloc_begin, host_alloc_end),
        elapsed_us(malloc_begin, malloc_end),
        state.bytes,
        warmup_iters,
        trials);

    print_first_call_breakdown(state);
    warmup(state, warmup_iters);
    print_steady_single_breakdown(state);

    const int repeat_values[] = {1, 10, 100, 1000};
    for (int repeats : repeat_values) {
        run_trials(state, "async", repeats, trials, [&] { async_pipeline_once(state); });
    }
    for (int repeats : repeat_values) {
        run_trials(state, "sync", repeats, trials, [&] { sync_pipeline_once(state); });
    }

    check(cudaFree(state.d_out), "cudaFree d_out");
    check(cudaFree(state.d_in), "cudaFree d_in");
    check(cudaFreeHost(state.h_out), "cudaFreeHost h_out");
    check(cudaFreeHost(state.h_in), "cudaFreeHost h_in");
    check(cudaStreamDestroy(state.stream), "cudaStreamDestroy");
    return 0;
}
