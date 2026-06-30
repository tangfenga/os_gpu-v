#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void matmul_kernel(const float *a, const float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
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
    const size_t idx = std::min(sorted.size() - 1, static_cast<size_t>(p * sorted.size()));
    return sorted[idx];
}

static void print_stats(int n, int trials, std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    double sum = 0.0;
    for (double v : samples) {
        sum += v;
    }
    const double mean = sum / static_cast<double>(samples.size());
    const double median = percentile(samples, 0.50);
    const double p95 = percentile(samples, 0.95);
    std::printf("matrix_benchmark n=%d trials=%d median_us=%.3f mean_us=%.3f p95_us=%.3f min_us=%.3f max_us=%.3f pass=1\n",
                n, trials, median, mean, p95, samples.front(), samples.back());
}

int main(int argc, char **argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 256;
    const int trials = argc > 2 ? std::atoi(argv[2]) : 10;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 20;
    if (n <= 0 || trials <= 0 || warmup < 0) {
        std::fprintf(stderr, "usage: %s [n] [trials] [warmup]\n", argv[0]);
        return 1;
    }

    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(float);
    std::vector<float> h_a(count, 1.0f);
    std::vector<float> h_b(count, 2.0f);
    std::vector<float> h_c(count, 0.0f);

    cudaStream_t stream = nullptr;
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    auto run_once = [&]() {
        check(cudaMemcpyAsync(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D a");
        check(cudaMemcpyAsync(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice, stream), "H2D b");
        matmul_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
        check(cudaGetLastError(), "matmul launch");
        check(cudaMemcpyAsync(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream), "D2H c");
        check(cudaStreamSynchronize(stream), "stream sync");
    };

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    std::vector<double> samples;
    samples.reserve(trials);
    for (int t = 0; t < trials; ++t) {
        const auto begin = Clock::now();
        run_once();
        const auto end = Clock::now();
        samples.push_back(elapsed_us(begin, end));
    }

    const float expected = static_cast<float>(n) * 2.0f;
    for (size_t i = 0; i < count; i += std::max<size_t>(1, count / 64)) {
        if (std::fabs(h_c[i] - expected) > 1e-4f) {
            std::fprintf(stderr, "matrix mismatch n=%d index=%zu expected=%f got=%f\n",
                         n, i, expected, h_c[i]);
            return 1;
        }
    }

    print_stats(n, trials, std::move(samples));
    check(cudaFree(d_c), "cudaFree d_c");
    check(cudaFree(d_b), "cudaFree d_b");
    check(cudaFree(d_a), "cudaFree d_a");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}
