#include <cuda_runtime_api.h>
#include <grpcpp/grpcpp.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../shared/vgpu_shm_ring.h"
#include "vgpu.grpc.pb.h"

namespace {

constexpr int kRuntimeVersion = 12000;
constexpr int kFallbackDriverVersion = 12060;
constexpr uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr uint32_t kFatbinDataMagic = 0xba55ed50u;
constexpr uint64_t kMaxFatbinBytes = 256ull * 1024ull * 1024ull;
constexpr uint64_t kDefaultShmSize = 64ull * 1024ull * 1024ull;
constexpr uint64_t kDefaultShmThreshold = 64ull * 1024ull;
constexpr uint64_t kDefaultShmBlockSize = 4ull * 1024ull * 1024ull;

using Clock = std::chrono::steady_clock;

thread_local cudaError_t tls_last_error = cudaSuccess;
thread_local dim3 tls_grid_dim(1, 1, 1);
thread_local dim3 tls_block_dim(1, 1, 1);
thread_local size_t tls_shared_mem = 0;
thread_local cudaStream_t tls_stream = nullptr;
thread_local uint64_t tls_trace_tag = vgpu_shm::kTraceDefault;

cudaError_t SetLastError(cudaError_t error) {
    tls_last_error = error;
    return error;
}

cudaError_t FromWireError(int cuda_error) {
    return static_cast<cudaError_t>(cuda_error);
}

std::string ServerAddress() {
    const char *env = std::getenv("VGPU_SERVER");
    if (env && env[0] != '\0') {
        return env;
    }
    return "127.0.0.1:50051";
}

uint64_t RequestedMemoryLimit() {
    const char *env = std::getenv("VGPU_MEMORY_LIMIT");
    if (!env || env[0] == '\0') {
        return 0;
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(env, &end, 10);
    if (!end || *end != '\0') {
        return 0;
    }
    return static_cast<uint64_t>(value);
}

uint64_t ReadUint64Env(const char *name, uint64_t fallback) {
    const char *env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(env, &end, 10);
    if (!end || *end != '\0') {
        return fallback;
    }
    return static_cast<uint64_t>(value);
}

bool ShmDataPlaneRequested() {
    const char *env = std::getenv("VGPU_DATA_PLANE");
    return env && std::strcmp(env, "shm") == 0;
}

bool DetailedPerfRequested() {
    const char *env = std::getenv("VGPU_PERF_DETAIL");
    return env && std::strcmp(env, "1") == 0;
}

std::string MakeShmName() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "/vgpu_" + std::to_string(static_cast<unsigned long long>(getuid())) +
           "_" + std::to_string(static_cast<unsigned long long>(getpid())) +
           "_" + std::to_string(static_cast<unsigned long long>(now));
}

size_t DivRoundUp(size_t value, size_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t ElapsedUs(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

uint64_t NowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

bool RingTraceRequested() {
    const char *env = std::getenv("VGPU_RING_TRACE");
    return env && std::strcmp(env, "1") == 0;
}

bool InitTraceRequested() {
    const char *env = std::getenv("VGPU_INIT_TRACE");
    return env && std::strcmp(env, "1") == 0;
}

void CopyStringToCudaName(char *dst, size_t dst_size, const std::string &src) {
    if (dst_size == 0) {
        return;
    }
    const size_t n = std::min(dst_size - 1, src.size());
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

const char *RingOpName(uint64_t op) {
    switch (op) {
        case vgpu_shm::kRingOpMemcpyShm:
            return "memcpy_shm";
        case vgpu_shm::kRingOpMemcpyD2D:
            return "d2d";
        case vgpu_shm::kRingOpDeviceSynchronize:
            return "device_sync";
        case vgpu_shm::kRingOpStreamSynchronize:
            return "stream_sync";
        case vgpu_shm::kRingOpEventSynchronize:
            return "event_sync";
        case vgpu_shm::kRingOpEventQuery:
            return "event_query";
        case vgpu_shm::kRingOpLaunchKernel:
            return "launch_kernel";
        default:
            return "unknown";
    }
}

const char *TraceTagName(uint64_t tag) {
    switch (tag) {
        case vgpu_shm::kTraceEmptyStreamSync:
            return "empty_stream_sync";
        case vgpu_shm::kTraceD2DSync:
            return "d2d_sync";
        case vgpu_shm::kTraceKernelSync:
            return "kernel_sync";
        case vgpu_shm::kTraceBatchDrainSync:
            return "batch_drain_sync";
        default:
            return "default";
    }
}

struct RingTraceBucket {
    uint64_t count = 0;
    uint64_t queue_depth_sum = 0;
    uint64_t queue_depth_max = 0;
    uint64_t submit_to_dequeue_ns = 0;
    uint64_t dequeue_to_context_ns = 0;
    uint64_t context_ns = 0;
    uint64_t context_to_lookup_ns = 0;
    uint64_t lookup_ns = 0;
    uint64_t lookup_to_rebuild_ns = 0;
    uint64_t rebuild_ns = 0;
    uint64_t rebuild_to_driver_ns = 0;
    uint64_t driver_api_ns = 0;
    uint64_t driver_to_complete_ns = 0;
    uint64_t complete_to_client_ns = 0;
    uint64_t total_ns = 0;
};

class RingTraceStats {
public:
    void Record(const vgpu_shm::RingEntry &entry) {
        if (!RingTraceRequested() ||
            entry.t_client_submit_ns == 0 ||
            entry.t_server_dequeue_ns == 0 ||
            entry.t_context_start_ns == 0 ||
            entry.t_context_end_ns == 0 ||
            entry.t_lookup_start_ns == 0 ||
            entry.t_lookup_end_ns == 0 ||
            entry.t_rebuild_start_ns == 0 ||
            entry.t_rebuild_end_ns == 0 ||
            entry.t_driver_start_ns == 0 ||
            entry.t_driver_end_ns == 0 ||
            entry.t_server_complete_ns == 0 ||
            entry.t_client_done_ns == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        const uint64_t key = (entry.trace_tag << 32) ^ entry.op;
        RingTraceBucket &bucket = buckets_[key];
        ++bucket.count;
        bucket.queue_depth_sum += entry.queue_depth_at_submit;
        bucket.queue_depth_max = std::max(bucket.queue_depth_max, entry.queue_depth_at_submit);
        bucket.submit_to_dequeue_ns += Delta(entry.t_client_submit_ns, entry.t_server_dequeue_ns);
        bucket.dequeue_to_context_ns += Delta(entry.t_server_dequeue_ns, entry.t_context_start_ns);
        bucket.context_ns += Delta(entry.t_context_start_ns, entry.t_context_end_ns);
        bucket.context_to_lookup_ns += Delta(entry.t_context_end_ns, entry.t_lookup_start_ns);
        bucket.lookup_ns += Delta(entry.t_lookup_start_ns, entry.t_lookup_end_ns);
        bucket.lookup_to_rebuild_ns += Delta(entry.t_lookup_end_ns, entry.t_rebuild_start_ns);
        bucket.rebuild_ns += Delta(entry.t_rebuild_start_ns, entry.t_rebuild_end_ns);
        bucket.rebuild_to_driver_ns += Delta(entry.t_rebuild_end_ns, entry.t_driver_start_ns);
        bucket.driver_api_ns += Delta(entry.t_driver_start_ns, entry.t_driver_end_ns);
        bucket.driver_to_complete_ns += Delta(entry.t_driver_end_ns, entry.t_server_complete_ns);
        bucket.complete_to_client_ns += Delta(entry.t_server_complete_ns, entry.t_client_done_ns);
        bucket.total_ns += Delta(entry.t_client_submit_ns, entry.t_client_done_ns);
    }

    void Print() {
        if (!RingTraceRequested()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto &entry : buckets_) {
            const RingTraceBucket &b = entry.second;
            if (b.count == 0) {
                continue;
            }
            const uint64_t op = entry.first & 0xffffffffull;
            const uint64_t tag = entry.first >> 32;
            const double n = static_cast<double>(b.count);
            std::fprintf(
                stderr,
                "ring_trace op=%s tag=%s count=%llu queue_depth_avg=%.3f queue_depth_max=%llu submit_to_dequeue_us=%.3f dequeue_to_context_us=%.3f context_us=%.3f context_to_lookup_us=%.3f lookup_us=%.3f lookup_to_rebuild_us=%.3f rebuild_us=%.3f rebuild_to_driver_us=%.3f driver_api_us=%.3f driver_to_complete_us=%.3f complete_to_client_us=%.3f total_us=%.3f\n",
                RingOpName(op),
                TraceTagName(tag),
                static_cast<unsigned long long>(b.count),
                static_cast<double>(b.queue_depth_sum) / n,
                static_cast<unsigned long long>(b.queue_depth_max),
                static_cast<double>(b.submit_to_dequeue_ns) / n / 1000.0,
                static_cast<double>(b.dequeue_to_context_ns) / n / 1000.0,
                static_cast<double>(b.context_ns) / n / 1000.0,
                static_cast<double>(b.context_to_lookup_ns) / n / 1000.0,
                static_cast<double>(b.lookup_ns) / n / 1000.0,
                static_cast<double>(b.lookup_to_rebuild_ns) / n / 1000.0,
                static_cast<double>(b.rebuild_ns) / n / 1000.0,
                static_cast<double>(b.rebuild_to_driver_ns) / n / 1000.0,
                static_cast<double>(b.driver_api_ns) / n / 1000.0,
                static_cast<double>(b.driver_to_complete_ns) / n / 1000.0,
                static_cast<double>(b.complete_to_client_ns) / n / 1000.0,
                static_cast<double>(b.total_ns) / n / 1000.0);
        }
    }

private:
    static uint64_t Delta(uint64_t start, uint64_t end) {
        return end >= start ? end - start : 0;
    }

    std::mutex mu_;
    std::unordered_map<uint64_t, RingTraceBucket> buckets_;
};

void ShutdownAtExit();
RingTraceStats &RingTrace();

struct FatbinWrapper {
    uint32_t magic;
    uint32_t version;
    const void *data;
    void *unused;
};

struct KernelInfo {
    uint64_t module_id = 0;
    std::string kernel_name;
    std::vector<size_t> param_sizes;
};

struct ModuleInfo {
    uint64_t server_module_id = 0;
    std::string fatbin;
    std::unordered_map<std::string, std::vector<size_t>> param_sizes_by_kernel;
};

struct ShmState {
    struct HostAllocation {
        size_t offset = 0;
        size_t blocks_used = 0;
        size_t bytes = 0;
        unsigned int flags = 0;
        bool owned = false;
        bool in_shm = false;
    };

    struct PendingBlock {
        size_t offset = 0;
        size_t blocks_used = 0;
        size_t bytes = 0;
        uintptr_t stream_id = 0;
        void *host_dst = nullptr;
        uint64_t seq = 0;
    };

    bool requested = false;
    bool enabled = false;
    bool ring_enabled = false;
    std::string name;
    int fd = -1;
    void *base = nullptr;
    size_t size = 0;
    size_t data_offset = 0;
    size_t data_size = 0;
    size_t threshold = kDefaultShmThreshold;
    size_t block_size = kDefaultShmBlockSize;
    std::vector<uint8_t> block_free;
    std::unordered_map<uintptr_t, HostAllocation> host_allocations;
    std::vector<PendingBlock> pending_blocks;
    uint64_t next_seq = 1;
    std::mutex mu;
    std::mutex ring_mu;
    std::condition_variable cv;
};

struct EventFence {
    uintptr_t stream_id = 0;
    uint64_t shm_seq_cutoff = 0;
};

bool ReadU32(const unsigned char *ptr, uint32_t *out) {
    std::memcpy(out, ptr, sizeof(*out));
    return true;
}

bool ReadU64(const unsigned char *ptr, uint64_t *out) {
    std::memcpy(out, ptr, sizeof(*out));
    return true;
}

std::string ExtractFatbinData(const void *fat_cubin) {
    if (!fat_cubin) {
        return {};
    }

    const auto *wrapper = static_cast<const FatbinWrapper *>(fat_cubin);
    if (wrapper->magic != kFatbinWrapperMagic || !wrapper->data) {
        return {};
    }

    const auto *data = static_cast<const unsigned char *>(wrapper->data);
    uint32_t magic = 0;
    uint64_t payload_size = 0;
    ReadU32(data, &magic);
    ReadU64(data + 8, &payload_size);
    if (magic != kFatbinDataMagic || payload_size > kMaxFatbinBytes) {
        return {};
    }

    const uint64_t total_size = payload_size + 16;
    if (total_size > kMaxFatbinBytes) {
        return {};
    }
    return std::string(reinterpret_cast<const char *>(data), static_cast<size_t>(total_size));
}

size_t ParamTypeSize(const std::string &type) {
    if (type == "u64" || type == "s64" || type == "b64" || type == "f64") {
        return 8;
    }
    if (type == "u32" || type == "s32" || type == "b32" || type == "f32") {
        return 4;
    }
    if (type == "u16" || type == "s16" || type == "b16" || type == "f16") {
        return 2;
    }
    if (type == "u8" || type == "s8" || type == "b8") {
        return 1;
    }
    return 0;
}

void SkipMangledType(const std::string &encoded, size_t *cursor) {
    if (*cursor >= encoded.size()) {
        return;
    }

    const char c = encoded[*cursor];
    if (c == 'P' || c == 'K' || c == 'V' || c == 'r') {
        ++(*cursor);
        SkipMangledType(encoded, cursor);
        return;
    }
    if (c == 'S') {
        ++(*cursor);
        while (*cursor < encoded.size() && encoded[*cursor] != '_') {
            ++(*cursor);
        }
        if (*cursor < encoded.size()) {
            ++(*cursor);
        }
        return;
    }
    ++(*cursor);
}

std::vector<size_t> ParseMangledParamSizes(const std::string &kernel_name) {
    std::vector<size_t> sizes;
    if (kernel_name.rfind("_Z", 0) != 0) {
        return sizes;
    }

    size_t cursor = 2;
    while (cursor < kernel_name.size() && kernel_name[cursor] >= '0' && kernel_name[cursor] <= '9') {
        size_t name_len = 0;
        while (cursor < kernel_name.size() && kernel_name[cursor] >= '0' && kernel_name[cursor] <= '9') {
            name_len = name_len * 10 + static_cast<size_t>(kernel_name[cursor] - '0');
            ++cursor;
        }
        cursor += name_len;
        break;
    }

    while (cursor < kernel_name.size()) {
        const char c = kernel_name[cursor];
        if (c == 'P') {
            sizes.push_back(8);
            ++cursor;
            SkipMangledType(kernel_name, &cursor);
            continue;
        }
        if (c == 'S') {
            sizes.push_back(8);
            SkipMangledType(kernel_name, &cursor);
            continue;
        }
        if (c == 'K' || c == 'V' || c == 'r') {
            ++cursor;
            continue;
        }
        if (c == 'i' || c == 'j' || c == 'f') {
            sizes.push_back(4);
            ++cursor;
            continue;
        }
        if (c == 'l' || c == 'm' || c == 'x' || c == 'y' || c == 'd') {
            sizes.push_back(8);
            ++cursor;
            continue;
        }
        if (c == 's' || c == 't') {
            sizes.push_back(2);
            ++cursor;
            continue;
        }
        if (c == 'a' || c == 'b' || c == 'c' || c == 'h') {
            sizes.push_back(1);
            ++cursor;
            continue;
        }
        return {};
    }
    return sizes;
}

std::vector<size_t> ParseKernelParamSizes(const std::string &fatbin, const std::string &kernel_name) {
    std::vector<size_t> sizes;

    size_t name_pos = std::string::npos;
    size_t search = 0;
    while (true) {
        const size_t entry_pos = fatbin.find(".entry", search);
        if (entry_pos == std::string::npos) {
            return ParseMangledParamSizes(kernel_name);
        }
        const size_t candidate = fatbin.find(kernel_name, entry_pos);
        if (candidate == std::string::npos) {
            return ParseMangledParamSizes(kernel_name);
        }
        const size_t next_entry = fatbin.find(".entry", entry_pos + 6);
        if (next_entry == std::string::npos || candidate < next_entry) {
            name_pos = candidate;
            break;
        }
        search = next_entry;
    }

    const size_t params_begin = fatbin.find('(', name_pos);
    if (params_begin == std::string::npos) {
        return sizes;
    }
    const size_t params_end = fatbin.find(')', params_begin);
    if (params_end == std::string::npos || params_end <= params_begin) {
        return sizes;
    }

    const std::string params = fatbin.substr(params_begin + 1, params_end - params_begin - 1);
    size_t cursor = 0;
    while (true) {
        const size_t marker = params.find(".param", cursor);
        if (marker == std::string::npos) {
            break;
        }
        const size_t dot = params.find('.', marker + 6);
        if (dot == std::string::npos) {
            break;
        }
        size_t end = dot + 1;
        while (end < params.size()) {
            const char c = params[end];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
                break;
            }
            ++end;
        }
        const size_t size = ParamTypeSize(params.substr(dot + 1, end - dot - 1));
        if (size == 0) {
            return ParseMangledParamSizes(kernel_name);
        }
        sizes.push_back(size);
        cursor = end;
    }
    const std::vector<size_t> mangled_sizes = ParseMangledParamSizes(kernel_name);
    if (sizes.size() < mangled_sizes.size()) {
        return mangled_sizes;
    }
    return sizes;
}

class RuntimeProxyClient {
public:
    cudaError_t EnsureSession() {
        std::lock_guard<std::mutex> lock(mu_);
        if (initialized_) {
            return cudaSuccess;
        }

        const auto init_start = Clock::now();
        const std::string address = ServerAddress();
        grpc::ChannelArguments args;
        args.SetMaxReceiveMessageSize(-1);
        args.SetMaxSendMessageSize(-1);
        auto channel = grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args);
        stub_ = vgpu::VgpuRuntime::NewStub(channel);

        const auto shm_start = Clock::now();
        PrepareShmForSession();
        const uint64_t shm_prepare_us = ElapsedUs(shm_start);

        vgpu::CreateSessionRequest request;
        request.set_client_pid(static_cast<uint32_t>(getpid()));
        request.set_client_name("libcudart_proxy");
        request.set_requested_memory_limit(RequestedMemoryLimit());
        if (shm_.requested && shm_.base) {
            request.set_shm_name(shm_.name);
            request.set_shm_size(static_cast<uint64_t>(shm_.size));
        }

        vgpu::CreateSessionReply reply;
        grpc::ClientContext context;
        const auto create_session_start = Clock::now();
        grpc::Status status = stub_->CreateSession(&context, request, &reply);
        const uint64_t create_session_us = ElapsedUs(create_session_start);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] CreateSession RPC failed: "
                      << status.error_message() << std::endl;
            CleanupShm();
            return cudaErrorUnknown;
        }
        if (reply.cuda_error() != cudaSuccess) {
            std::cerr << "[cudart_proxy] CreateSession returned CUDA error "
                      << reply.cuda_error() << ": " << reply.message() << std::endl;
            CleanupShm();
            return FromWireError(reply.cuda_error());
        }

        session_id_ = reply.session_id();
        shm_.enabled = reply.shm_enabled() && shm_.base;
        shm_.data_offset = static_cast<size_t>(reply.shm_data_offset());
        shm_.data_size = static_cast<size_t>(reply.shm_data_size());
        const auto configure_shm_start = Clock::now();
        ConfigureShmArena();
        const uint64_t configure_shm_us = ElapsedUs(configure_shm_start);
        if (!shm_.enabled) {
            CleanupShm();
        }
        initialized_ = true;
        if (!shutdown_registered_) {
            std::atexit(ShutdownAtExit);
            shutdown_registered_ = true;
        }
        if (DetailedPerfRequested()) {
            std::cerr << "[cudart_proxy] connected to " << address
                      << " session=" << session_id_
                      << " data_plane=" << (shm_.enabled ? "shm" : "grpc")
                      << std::endl;
        }
        if (InitTraceRequested()) {
            std::fprintf(
                stderr,
                "init_trace side=client op=EnsureSession total_us=%llu shm_prepare_us=%llu create_session_rpc_us=%llu configure_shm_arena_us=%llu shm_enabled=%d data_size=%llu\n",
                static_cast<unsigned long long>(ElapsedUs(init_start)),
                static_cast<unsigned long long>(shm_prepare_us),
                static_cast<unsigned long long>(create_session_us),
                static_cast<unsigned long long>(configure_shm_us),
                shm_.enabled ? 1 : 0,
                static_cast<unsigned long long>(shm_.data_size));
        }
        return cudaSuccess;
    }

    void Shutdown() {
        std::lock_guard<std::mutex> lock(mu_);
        if (!initialized_ || !stub_) {
            return;
        }

        vgpu::DestroySessionRequest request;
        request.set_session_id(session_id_);
        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->DestroySession(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] DestroySession RPC failed: "
                      << status.error_message() << std::endl;
        }
        CleanupShm();
        ClearEventFences();
        initialized_ = false;
        session_id_ = 0;
    }

    cudaError_t GetDeviceCount(int *count) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::GetDeviceCountRequest request;
        request.set_session_id(session_id_);
        vgpu::GetDeviceCountReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->GetDeviceCount(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] GetDeviceCount RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }

        err = FromWireError(reply.cuda_error());
        if (err == cudaSuccess) {
            *count = reply.count();
        }
        return err;
    }

    cudaError_t GetDeviceProperties(cudaDeviceProp *prop, int device) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::GetDevicePropertiesRequest request;
        request.set_session_id(session_id_);
        request.set_device(device);
        vgpu::GetDevicePropertiesReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->GetDeviceProperties(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] GetDeviceProperties RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }

        err = FromWireError(reply.cuda_error());
        if (err != cudaSuccess) {
            return err;
        }

        std::memset(prop, 0, sizeof(*prop));
        CopyStringToCudaName(prop->name, sizeof(prop->name), reply.name());
        prop->totalGlobalMem = static_cast<size_t>(reply.total_global_mem());
        prop->multiProcessorCount = reply.multi_processor_count();
        prop->major = reply.major();
        prop->minor = reply.minor();
        prop->warpSize = reply.warp_size();
        prop->maxThreadsPerBlock = reply.max_threads_per_block();
        for (int i = 0; i < std::min<int>(3, reply.max_threads_dim_size()); ++i) {
            prop->maxThreadsDim[i] = reply.max_threads_dim(i);
        }
        for (int i = 0; i < std::min<int>(3, reply.max_grid_size_size()); ++i) {
            prop->maxGridSize[i] = reply.max_grid_size(i);
        }
        prop->clockRate = reply.clock_rate();
        prop->memoryClockRate = reply.memory_clock_rate();
        prop->memoryBusWidth = reply.memory_bus_width();
        return cudaSuccess;
    }

    cudaError_t Malloc(void **dev_ptr, size_t size) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::MallocRequest request;
        request.set_session_id(session_id_);
        request.set_size(static_cast<uint64_t>(size));

        vgpu::MallocReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->Malloc(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] Malloc RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }

        err = FromWireError(reply.cuda_error());
        if (err == cudaSuccess) {
            *dev_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(reply.device_ptr()));
        }
        return err;
    }

    cudaError_t Free(void *dev_ptr) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::FreeRequest request;
        request.set_session_id(session_id_);
        request.set_device_ptr(reinterpret_cast<uintptr_t>(dev_ptr));

        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->Free(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] Free RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        return FromWireError(reply.cuda_error());
    }

    cudaError_t HostAlloc(void **ptr, size_t size, unsigned int flags) {
        if (!ptr) {
            return cudaErrorInvalidValue;
        }
        *ptr = nullptr;
        if (size == 0) {
            return cudaErrorInvalidValue;
        }
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }
        if (!shm_.enabled || !shm_.base) {
            return cudaErrorMemoryAllocation;
        }

        size_t shm_offset = 0;
        size_t blocks_used = 0;
        if (!AllocateShmBlocks(size, &shm_offset, &blocks_used, true)) {
            return cudaErrorMemoryAllocation;
        }

        void *host_ptr = static_cast<char *>(shm_.base) + shm_offset;
        {
            std::lock_guard<std::mutex> lock(shm_.mu);
            shm_.host_allocations[reinterpret_cast<uintptr_t>(host_ptr)] =
                ShmState::HostAllocation{shm_offset, blocks_used, size, flags, true, true};
        }
        *ptr = host_ptr;
        return cudaSuccess;
    }

    cudaError_t FreeHost(void *ptr) {
        if (!ptr) {
            return cudaSuccess;
        }
        std::lock_guard<std::mutex> lock(shm_.mu);
        auto it = shm_.host_allocations.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == shm_.host_allocations.end()) {
            return cudaErrorInvalidValue;
        }
        const ShmState::HostAllocation allocation = it->second;
        shm_.host_allocations.erase(it);
        if (allocation.owned) {
            FreeShmBlocksLocked(allocation.offset, allocation.blocks_used);
            shm_.cv.notify_all();
        }
        return cudaSuccess;
    }

    cudaError_t HostRegister(void *ptr, size_t size, unsigned int flags) {
        if (!ptr || size == 0) {
            return cudaErrorInvalidValue;
        }
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        size_t shm_offset = 0;
        const bool in_shm = HostPointerToShmOffset(ptr, size, &shm_offset);
        std::lock_guard<std::mutex> lock(shm_.mu);
        shm_.host_allocations[reinterpret_cast<uintptr_t>(ptr)] =
            ShmState::HostAllocation{shm_offset, 0, size, flags, false, in_shm};
        return cudaSuccess;
    }

    cudaError_t HostUnregister(void *ptr) {
        if (!ptr) {
            return cudaErrorInvalidValue;
        }
        std::lock_guard<std::mutex> lock(shm_.mu);
        auto it = shm_.host_allocations.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == shm_.host_allocations.end() || it->second.owned) {
            return cudaErrorInvalidValue;
        }
        shm_.host_allocations.erase(it);
        return cudaSuccess;
    }

    cudaError_t Memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
        return MemcpyWithStream(dst, src, count, kind, nullptr, false);
    }

    cudaError_t MemcpyAsync(
        void *dst,
        const void *src,
        size_t count,
        cudaMemcpyKind kind,
        cudaStream_t stream) {
        return MemcpyWithStream(dst, src, count, kind, stream, true);
    }

    cudaError_t MemcpyWithStream(
        void *dst,
        const void *src,
        size_t count,
        cudaMemcpyKind kind,
        cudaStream_t stream,
        bool async) {
        if (count == 0) {
            return cudaSuccess;
        }

        if (kind == cudaMemcpyHostToHost) {
            if (!dst || !src) {
                return cudaErrorInvalidValue;
            }
            std::memcpy(dst, src, count);
            return cudaSuccess;
        }

        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        size_t direct_offset = 0;
        const bool direct_host_shm =
            (kind == cudaMemcpyHostToDevice && HostPointerToShmOffset(src, count, &direct_offset)) ||
            (kind == cudaMemcpyDeviceToHost && HostPointerToShmOffset(dst, count, &direct_offset));
        if (direct_host_shm || CanUseShmMemcpy(count, kind, async)) {
            cudaError_t shm_err = MemcpyViaShm(dst, src, count, kind, stream, async);
            if (!async || shm_err != cudaErrorMemoryAllocation) {
                return shm_err;
            }
        }

        if (kind == cudaMemcpyDeviceToDevice) {
            if (!dst || !src) {
                return cudaErrorInvalidValue;
            }
            cudaError_t ring_err = cudaSuccess;
            if (SubmitD2DRing(
                    reinterpret_cast<uintptr_t>(dst),
                    reinterpret_cast<uintptr_t>(src),
                    count,
                    reinterpret_cast<uintptr_t>(stream),
                    async,
                    &ring_err)) {
                return ring_err;
            }
        }

        vgpu::MemcpyRequest request;
        request.set_session_id(session_id_);
        request.set_count(static_cast<uint64_t>(count));
        request.set_kind(static_cast<int>(kind));
        request.set_stream_id(reinterpret_cast<uintptr_t>(stream));
        request.set_async(async);

        switch (kind) {
            case cudaMemcpyHostToDevice:
                if (!dst || !src) {
                    return cudaErrorInvalidValue;
                }
                request.set_dst_device_ptr(reinterpret_cast<uintptr_t>(dst));
                request.set_host_data(src, count);
                break;
            case cudaMemcpyDeviceToHost:
                if (!dst || !src) {
                    return cudaErrorInvalidValue;
                }
                request.set_src_device_ptr(reinterpret_cast<uintptr_t>(src));
                break;
            case cudaMemcpyDeviceToDevice:
                if (!dst || !src) {
                    return cudaErrorInvalidValue;
                }
                request.set_dst_device_ptr(reinterpret_cast<uintptr_t>(dst));
                request.set_src_device_ptr(reinterpret_cast<uintptr_t>(src));
                break;
            default:
                return cudaErrorInvalidMemcpyDirection;
        }

        vgpu::MemcpyReply reply;
        grpc::ClientContext context;
        grpc::Status status = async
            ? stub_->MemcpyAsync(&context, request, &reply)
            : stub_->Memcpy(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] Memcpy RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }

        err = FromWireError(reply.cuda_error());
        if (err != cudaSuccess) {
            return err;
        }

        if (kind == cudaMemcpyDeviceToHost) {
            if (reply.host_data().size() != count) {
                return cudaErrorUnknown;
            }
            std::memcpy(dst, reply.host_data().data(), count);
        }
        return cudaSuccess;
    }

    cudaError_t StreamCreate(cudaStream_t *stream, unsigned int flags) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::StreamCreateRequest request;
        request.set_session_id(session_id_);
        request.set_flags(flags);

        vgpu::StreamCreateReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->StreamCreate(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] StreamCreate RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        err = FromWireError(reply.cuda_error());
        if (err == cudaSuccess) {
            *stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(reply.stream_id()));
        }
        return err;
    }

    cudaError_t StreamDestroy(cudaStream_t stream) {
        if (stream == nullptr) {
            return cudaSuccess;
        }
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::StreamDestroyRequest request;
        request.set_session_id(session_id_);
        request.set_stream_id(reinterpret_cast<uintptr_t>(stream));

        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->StreamDestroy(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] StreamDestroy RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        cudaError_t out = FromWireError(reply.cuda_error());
        if (out == cudaSuccess) {
            ReleasePendingShmBlocksForStream(reinterpret_cast<uintptr_t>(stream));
            ClearEventFencesForStream(reinterpret_cast<uintptr_t>(stream));
        }
        return out;
    }

    cudaError_t StreamSynchronize(cudaStream_t stream) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        cudaError_t ring_err = cudaSuccess;
        if (SubmitSyncRing(
                vgpu_shm::kRingOpStreamSynchronize,
                reinterpret_cast<uintptr_t>(stream),
                0,
                &ring_err)) {
            if (ring_err == cudaSuccess) {
                ReleasePendingShmBlocksForStream(reinterpret_cast<uintptr_t>(stream));
            }
            return ring_err;
        }

        vgpu::StreamSynchronizeRequest request;
        request.set_session_id(session_id_);
        request.set_stream_id(reinterpret_cast<uintptr_t>(stream));

        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->StreamSynchronize(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] StreamSynchronize RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        cudaError_t out = FromWireError(reply.cuda_error());
        if (out == cudaSuccess) {
            ReleasePendingShmBlocksForStream(reinterpret_cast<uintptr_t>(stream));
        }
        return out;
    }

    cudaError_t EventCreate(cudaEvent_t *event, unsigned int flags) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::EventCreateRequest request;
        request.set_session_id(session_id_);
        request.set_flags(flags);

        vgpu::EventCreateReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->EventCreate(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] EventCreate RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        err = FromWireError(reply.cuda_error());
        if (err == cudaSuccess) {
            *event = reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(reply.event_id()));
        }
        return err;
    }

    cudaError_t EventRecord(cudaEvent_t event, cudaStream_t stream) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        const uintptr_t event_id = reinterpret_cast<uintptr_t>(event);
        const uintptr_t stream_id = reinterpret_cast<uintptr_t>(stream);
        const uint64_t shm_seq_cutoff = MaxPendingShmSeqForStream(stream_id);

        vgpu::EventRecordRequest request;
        request.set_session_id(session_id_);
        request.set_event_id(event_id);
        request.set_stream_id(stream_id);

        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->EventRecord(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] EventRecord RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        cudaError_t out = FromWireError(reply.cuda_error());
        if (out == cudaSuccess) {
            std::lock_guard<std::mutex> lock(event_mu_);
            event_fences_[event_id] = EventFence{stream_id, shm_seq_cutoff};
        }
        return out;
    }

    cudaError_t EventSynchronize(cudaEvent_t event) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        cudaError_t ring_err = cudaSuccess;
        if (SubmitSyncRing(
                vgpu_shm::kRingOpEventSynchronize,
                0,
                reinterpret_cast<uintptr_t>(event),
                &ring_err)) {
            if (ring_err == cudaSuccess) {
                ReleasePendingShmBlocksForEvent(reinterpret_cast<uintptr_t>(event));
            }
            return ring_err;
        }

        vgpu::EventSynchronizeRequest request;
        request.set_session_id(session_id_);
        request.set_event_id(reinterpret_cast<uintptr_t>(event));

        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->EventSynchronize(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] EventSynchronize RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        cudaError_t out = FromWireError(reply.cuda_error());
        if (out == cudaSuccess) {
            ReleasePendingShmBlocksForEvent(reinterpret_cast<uintptr_t>(event));
        }
        return out;
    }

    cudaError_t EventQuery(cudaEvent_t event) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        cudaError_t ring_err = cudaSuccess;
        if (SubmitSyncRing(
                vgpu_shm::kRingOpEventQuery,
                0,
                reinterpret_cast<uintptr_t>(event),
                &ring_err)) {
            if (ring_err == cudaSuccess) {
                ReleasePendingShmBlocksForEvent(reinterpret_cast<uintptr_t>(event));
            }
            return ring_err;
        }

        vgpu::EventQueryRequest request;
        request.set_session_id(session_id_);
        request.set_event_id(reinterpret_cast<uintptr_t>(event));

        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->EventQuery(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] EventQuery RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        cudaError_t out = FromWireError(reply.cuda_error());
        if (out == cudaSuccess) {
            ReleasePendingShmBlocksForEvent(reinterpret_cast<uintptr_t>(event));
        }
        return out;
    }

    cudaError_t EventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop) {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::EventElapsedTimeRequest request;
        request.set_session_id(session_id_);
        request.set_start_event_id(reinterpret_cast<uintptr_t>(start));
        request.set_stop_event_id(reinterpret_cast<uintptr_t>(stop));

        vgpu::EventElapsedTimeReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->EventElapsedTime(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] EventElapsedTime RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        err = FromWireError(reply.cuda_error());
        if (err == cudaSuccess) {
            *ms = reply.milliseconds();
        }
        return err;
    }

    cudaError_t EventDestroy(cudaEvent_t event) {
        if (event == nullptr) {
            return cudaSuccess;
        }
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        vgpu::EventDestroyRequest request;
        request.set_session_id(session_id_);
        request.set_event_id(reinterpret_cast<uintptr_t>(event));

        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->EventDestroy(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] EventDestroy RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        cudaError_t out = FromWireError(reply.cuda_error());
        if (out == cudaSuccess) {
            std::lock_guard<std::mutex> lock(event_mu_);
            event_fences_.erase(reinterpret_cast<uintptr_t>(event));
        }
        return out;
    }

    cudaError_t DeviceSynchronize() {
        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        cudaError_t ring_err = cudaSuccess;
        if (SubmitSyncRing(vgpu_shm::kRingOpDeviceSynchronize, 0, 0, &ring_err)) {
            if (ring_err == cudaSuccess) {
                ReleaseAllPendingShmBlocks();
            }
            return ring_err;
        }

        vgpu::DeviceSynchronizeRequest request;
        request.set_session_id(session_id_);
        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->DeviceSynchronize(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] DeviceSynchronize RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        cudaError_t out = FromWireError(reply.cuda_error());
        if (out == cudaSuccess) {
            ReleaseAllPendingShmBlocks();
        }
        return out;
    }

    void **RegisterFatBinary(const void *fat_cubin) {
        const auto total_start = Clock::now();
        const uint64_t local_handle = next_local_module_handle_++;
        ModuleInfo module;
        const auto extract_start = Clock::now();
        module.fatbin = ExtractFatbinData(fat_cubin);
        const uint64_t extract_us = ElapsedUs(extract_start);

        cudaError_t err = EnsureSession();
        uint64_t register_rpc_us = 0;
        if (err == cudaSuccess && !module.fatbin.empty()) {
            vgpu::RegisterModuleRequest request;
            request.set_session_id(session_id_);
            request.set_fatbin(module.fatbin.data(), module.fatbin.size());

            vgpu::RegisterModuleReply reply;
            grpc::ClientContext context;
            const auto rpc_start = Clock::now();
            grpc::Status status = stub_->RegisterModule(&context, request, &reply);
            register_rpc_us = ElapsedUs(rpc_start);
            if (status.ok() && reply.cuda_error() == cudaSuccess) {
                module.server_module_id = reply.module_id();
                if (DetailedPerfRequested()) {
                    std::cerr << "[cudart_proxy] registered module id="
                              << module.server_module_id
                              << " fatbin_bytes=" << module.fatbin.size() << std::endl;
                }
            } else {
                std::cerr << "[cudart_proxy] RegisterModule failed: "
                          << (status.ok() ? reply.message() : status.error_message()) << std::endl;
            }
        }

        {
            std::lock_guard<std::mutex> lock(reg_mu_);
            modules_.emplace(local_handle, std::move(module));
        }
        if (InitTraceRequested()) {
            std::fprintf(
                stderr,
                "init_trace side=client op=RegisterFatBinary total_us=%llu extract_fatbin_us=%llu register_module_rpc_us=%llu\n",
                static_cast<unsigned long long>(ElapsedUs(total_start)),
                static_cast<unsigned long long>(extract_us),
                static_cast<unsigned long long>(register_rpc_us));
        }
        return reinterpret_cast<void **>(static_cast<uintptr_t>(local_handle));
    }

    void RegisterFunction(void **fat_cubin_handle, const char *host_fun, const char *device_name) {
        const uint64_t local_handle = reinterpret_cast<uintptr_t>(fat_cubin_handle);
        const uintptr_t host_fun_key = reinterpret_cast<uintptr_t>(host_fun);
        const std::string kernel_name = device_name ? device_name : "";
        if (local_handle == 0 || host_fun_key == 0 || kernel_name.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(reg_mu_);
        auto module_it = modules_.find(local_handle);
        if (module_it == modules_.end()) {
            return;
        }

        ModuleInfo &module = module_it->second;
        auto param_it = module.param_sizes_by_kernel.find(kernel_name);
        if (param_it == module.param_sizes_by_kernel.end()) {
            param_it = module.param_sizes_by_kernel
                .emplace(kernel_name, ParseKernelParamSizes(module.fatbin, kernel_name))
                .first;
        }

        kernels_[host_fun_key] = KernelInfo{
            module.server_module_id,
            kernel_name,
            param_it->second,
        };
        if (InitTraceRequested()) {
            std::fprintf(
                stderr,
                "init_trace side=client op=RegisterFunction kernel=%s params=%zu\n",
                kernel_name.c_str(),
                param_it->second.size());
        }
        if (DetailedPerfRequested()) {
            std::cerr << "[cudart_proxy] registered kernel name=" << kernel_name
                      << " module=" << module.server_module_id
                      << " params=" << param_it->second.size() << std::endl;
        }
    }

    void UnregisterFatBinary(void **fat_cubin_handle) {
        const uint64_t local_handle = reinterpret_cast<uintptr_t>(fat_cubin_handle);
        std::lock_guard<std::mutex> lock(reg_mu_);
        modules_.erase(local_handle);
    }

    cudaError_t LaunchKernel(
        const void *func,
        dim3 grid_dim,
        dim3 block_dim,
        void **args,
        size_t shared_mem,
        cudaStream_t stream) {
        if (!func || !args) {
            return cudaErrorInvalidValue;
        }

        KernelInfo kernel;
        {
            std::lock_guard<std::mutex> lock(reg_mu_);
            auto it = kernels_.find(reinterpret_cast<uintptr_t>(func));
            if (it == kernels_.end()) {
                return cudaErrorInvalidDeviceFunction;
            }
            kernel = it->second;
        }

        if (kernel.module_id == 0 || kernel.param_sizes.empty()) {
            return cudaErrorInvalidDeviceFunction;
        }

        cudaError_t err = EnsureSession();
        if (err != cudaSuccess) {
            return err;
        }

        cudaError_t ring_err = cudaSuccess;
        if (SubmitLaunchKernelRing(kernel, grid_dim, block_dim, args, shared_mem, stream, &ring_err)) {
            if (ring_err != cudaSuccess) {
                std::cerr << "[cudart_proxy] LaunchKernel ring failed kernel="
                          << kernel.kernel_name << " module=" << kernel.module_id
                          << " args=" << kernel.param_sizes.size()
                          << " error=" << ring_err << std::endl;
            }
            return ring_err;
        }

        vgpu::LaunchKernelRequest request;
        request.set_session_id(session_id_);
        request.set_module_id(kernel.module_id);
        request.set_kernel_name(kernel.kernel_name);
        request.mutable_grid_dim()->set_x(grid_dim.x);
        request.mutable_grid_dim()->set_y(grid_dim.y);
        request.mutable_grid_dim()->set_z(grid_dim.z);
        request.mutable_block_dim()->set_x(block_dim.x);
        request.mutable_block_dim()->set_y(block_dim.y);
        request.mutable_block_dim()->set_z(block_dim.z);
        request.set_shared_mem(static_cast<uint64_t>(shared_mem));
        request.set_stream_id(reinterpret_cast<uintptr_t>(stream));

        for (size_t i = 0; i < kernel.param_sizes.size(); ++i) {
            if (!args[i]) {
                return cudaErrorInvalidValue;
            }
            vgpu::KernelArg *arg = request.add_args();
            arg->set_data(args[i], kernel.param_sizes[i]);
        }

        vgpu::StatusReply reply;
        grpc::ClientContext context;
        grpc::Status status = stub_->LaunchKernel(&context, request, &reply);
        if (!status.ok()) {
            std::cerr << "[cudart_proxy] LaunchKernel RPC failed: "
                      << status.error_message() << std::endl;
            return cudaErrorUnknown;
        }
        if (reply.cuda_error() != cudaSuccess) {
            std::cerr << "[cudart_proxy] LaunchKernel failed kernel="
                      << kernel.kernel_name << " module=" << kernel.module_id
                      << " args=" << kernel.param_sizes.size()
                      << " error=" << reply.cuda_error()
                      << " message=" << reply.message() << std::endl;
        }
        return FromWireError(reply.cuda_error());
    }

private:
    void PrepareShmForSession() {
        shm_.requested = ShmDataPlaneRequested();
        shm_.enabled = false;
        shm_.threshold = static_cast<size_t>(ReadUint64Env("VGPU_SHM_THRESHOLD", kDefaultShmThreshold));
        if (!shm_.requested) {
            return;
        }

        CleanupShm();
        shm_.name = MakeShmName();
        shm_.size = static_cast<size_t>(ReadUint64Env("VGPU_SHM_SIZE", kDefaultShmSize));
        shm_.block_size = static_cast<size_t>(ReadUint64Env("VGPU_SHM_BLOCK_SIZE", kDefaultShmBlockSize));
        if (shm_.block_size == 0) {
            shm_.block_size = static_cast<size_t>(kDefaultShmBlockSize);
        }
        shm_.fd = shm_open(shm_.name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (shm_.fd < 0) {
            std::cerr << "[cudart_proxy] shm_open failed, falling back to grpc data plane" << std::endl;
            return;
        }
        if (ftruncate(shm_.fd, static_cast<off_t>(shm_.size)) != 0) {
            std::cerr << "[cudart_proxy] ftruncate shm failed, falling back to grpc data plane" << std::endl;
            CleanupShm();
            return;
        }
        shm_.base = mmap(nullptr, shm_.size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_.fd, 0);
        if (shm_.base == MAP_FAILED) {
            shm_.base = nullptr;
            std::cerr << "[cudart_proxy] mmap shm failed, falling back to grpc data plane" << std::endl;
            CleanupShm();
            return;
        }
        shm_.data_offset = 0;
        shm_.data_size = shm_.size;
    }

    void ConfigureShmArena() {
        std::lock_guard<std::mutex> lock(shm_.mu);
        shm_.block_free.clear();
        shm_.ring_enabled = false;
        if (!shm_.enabled || shm_.data_size == 0 || shm_.block_size == 0) {
            return;
        }
        const size_t blocks = shm_.data_size / shm_.block_size;
        if (blocks == 0) {
            shm_.enabled = false;
            return;
        }
        shm_.block_free.assign(blocks, 1);
        shm_.ring_enabled = vgpu_shm::HeaderLooksReady(vgpu_shm::Header(shm_.base));
    }

    void CleanupShm() {
        std::lock_guard<std::mutex> lock(shm_.mu);
        if (shm_.base) {
            munmap(shm_.base, shm_.size);
            shm_.base = nullptr;
        }
        if (shm_.fd >= 0) {
            close(shm_.fd);
            shm_.fd = -1;
        }
        if (!shm_.name.empty()) {
            shm_unlink(shm_.name.c_str());
            shm_.name.clear();
        }
        shm_.enabled = false;
        shm_.ring_enabled = false;
        shm_.size = 0;
        shm_.data_offset = 0;
        shm_.data_size = 0;
        shm_.block_free.clear();
        shm_.host_allocations.clear();
        shm_.pending_blocks.clear();
        shm_.next_seq = 1;
        shm_.cv.notify_all();
    }

    bool CanUseShmMemcpy(size_t count, cudaMemcpyKind kind, bool async) const {
        if (!shm_.enabled || !shm_.base || count < shm_.threshold || count > shm_.data_size ||
            shm_.block_size == 0 || shm_.block_free.empty()) {
            return false;
        }
        if (async) {
            return kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost;
        }
        return kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToHost;
    }

    bool HostPointerToShmOffset(const void *ptr, size_t count, size_t *offset) {
        if (!ptr || !offset || count == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(shm_.mu);
        if (!shm_.enabled || !shm_.base || shm_.data_size == 0) {
            return false;
        }
        const auto base = reinterpret_cast<uintptr_t>(shm_.base);
        const auto begin = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t data_begin = base + shm_.data_offset;
        const uintptr_t data_end = data_begin + shm_.data_size;
        if (begin < data_begin || begin > data_end || count > data_end - begin) {
            return false;
        }
        *offset = static_cast<size_t>(begin - base);
        return true;
    }

    bool TryAllocateShmBlocksLocked(size_t count, size_t *offset, size_t *blocks_used) {
        const size_t needed = DivRoundUp(count, shm_.block_size);
        if (needed == 0 || needed > shm_.block_free.size()) {
            return false;
        }

        size_t run_start = 0;
        size_t run_len = 0;
        for (size_t i = 0; i < shm_.block_free.size(); ++i) {
            if (shm_.block_free[i]) {
                if (run_len == 0) {
                    run_start = i;
                }
                ++run_len;
                if (run_len == needed) {
                    for (size_t j = run_start; j < run_start + needed; ++j) {
                        shm_.block_free[j] = 0;
                    }
                    *offset = shm_.data_offset + run_start * shm_.block_size;
                    *blocks_used = needed;
                    return true;
                }
            } else {
                run_len = 0;
            }
        }
        return false;
    }

    bool AllocateShmBlocks(size_t count, size_t *offset, size_t *blocks_used, bool wait_for_space) {
        std::unique_lock<std::mutex> lock(shm_.mu);
        if (!shm_.enabled || !shm_.base || count > shm_.data_size || shm_.block_size == 0) {
            return false;
        }
        if (!wait_for_space) {
            return TryAllocateShmBlocksLocked(count, offset, blocks_used);
        }
        shm_.cv.wait(lock, [&] {
            return !shm_.enabled || TryAllocateShmBlocksLocked(count, offset, blocks_used);
        });
        return shm_.enabled && *blocks_used != 0;
    }

    void FreeShmBlocksLocked(size_t offset, size_t blocks_used) {
        if (blocks_used == 0 || offset < shm_.data_offset || shm_.block_size == 0) {
            return;
        }
        const size_t first_block = (offset - shm_.data_offset) / shm_.block_size;
        for (size_t i = first_block; i < first_block + blocks_used && i < shm_.block_free.size(); ++i) {
            shm_.block_free[i] = 1;
        }
    }

    void FreeShmBlocks(size_t offset, size_t blocks_used) {
        std::lock_guard<std::mutex> lock(shm_.mu);
        FreeShmBlocksLocked(offset, blocks_used);
        shm_.cv.notify_all();
    }

    void CopyOutPendingBlockLocked(const ShmState::PendingBlock &pending) {
        if (pending.host_dst && shm_.base && pending.bytes > 0) {
            const char *slot = static_cast<const char *>(shm_.base) + pending.offset;
            std::memcpy(pending.host_dst, slot, pending.bytes);
        }
    }

    void QueuePendingShmBlock(
        size_t offset,
        size_t blocks_used,
        size_t bytes,
        cudaStream_t stream,
        void *host_dst) {
        std::lock_guard<std::mutex> lock(shm_.mu);
        shm_.pending_blocks.push_back(ShmState::PendingBlock{
            offset,
            blocks_used,
            bytes,
            reinterpret_cast<uintptr_t>(stream),
            host_dst,
            shm_.next_seq++,
        });
    }

    uint64_t MaxPendingShmSeqForStream(uintptr_t stream_id) {
        std::lock_guard<std::mutex> lock(shm_.mu);
        uint64_t max_seq = 0;
        for (const auto &pending : shm_.pending_blocks) {
            if (pending.stream_id == stream_id && pending.seq > max_seq) {
                max_seq = pending.seq;
            }
        }
        return max_seq;
    }

    void ReleasePendingShmBlocksForStream(uintptr_t stream_id) {
        std::lock_guard<std::mutex> lock(shm_.mu);
        auto it = shm_.pending_blocks.begin();
        while (it != shm_.pending_blocks.end()) {
            if (it->stream_id == stream_id) {
                CopyOutPendingBlockLocked(*it);
                FreeShmBlocksLocked(it->offset, it->blocks_used);
                it = shm_.pending_blocks.erase(it);
            } else {
                ++it;
            }
        }
        shm_.cv.notify_all();
    }

    void ReleasePendingShmBlocksForStreamUpTo(uintptr_t stream_id, uint64_t seq_cutoff) {
        if (seq_cutoff == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(shm_.mu);
        auto it = shm_.pending_blocks.begin();
        while (it != shm_.pending_blocks.end()) {
            if (it->stream_id == stream_id && it->seq <= seq_cutoff) {
                CopyOutPendingBlockLocked(*it);
                FreeShmBlocksLocked(it->offset, it->blocks_used);
                it = shm_.pending_blocks.erase(it);
            } else {
                ++it;
            }
        }
        shm_.cv.notify_all();
    }

    void ReleasePendingShmBlocksForEvent(uintptr_t event_id) {
        EventFence fence;
        {
            std::lock_guard<std::mutex> lock(event_mu_);
            auto it = event_fences_.find(event_id);
            if (it == event_fences_.end()) {
                return;
            }
            fence = it->second;
        }
        ReleasePendingShmBlocksForStreamUpTo(fence.stream_id, fence.shm_seq_cutoff);
    }

    void ClearEventFencesForStream(uintptr_t stream_id) {
        std::lock_guard<std::mutex> lock(event_mu_);
        auto it = event_fences_.begin();
        while (it != event_fences_.end()) {
            if (it->second.stream_id == stream_id) {
                it = event_fences_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void ClearEventFences() {
        std::lock_guard<std::mutex> lock(event_mu_);
        event_fences_.clear();
    }

    void ReleaseAllPendingShmBlocks() {
        std::lock_guard<std::mutex> lock(shm_.mu);
        for (const auto &pending : shm_.pending_blocks) {
            CopyOutPendingBlockLocked(pending);
            FreeShmBlocksLocked(pending.offset, pending.blocks_used);
        }
        shm_.pending_blocks.clear();
        shm_.cv.notify_all();
    }

    template <typename FillFn>
    bool SubmitRingEntry(FillFn fill, cudaError_t *out, uint64_t *ring_wait_us, bool wait_completion = true) {
        const auto ring_start = Clock::now();
        if (!shm_.ring_enabled || !shm_.base) {
            return false;
        }
        std::lock_guard<std::mutex> ring_lock(shm_.ring_mu);
        auto *header = vgpu_shm::Header(shm_.base);
        auto *entries = vgpu_shm::Entries(shm_.base);
        if (!vgpu_shm::HeaderLooksReady(header)) {
            return false;
        }

        uint64_t head = 0;
        uint64_t tail_snapshot = 0;
        for (;;) {
            if (vgpu_shm::LoadAcquire(&header->stop) != 0) {
                return false;
            }
            head = vgpu_shm::LoadAcquire(&header->head);
            const uint64_t tail = vgpu_shm::LoadAcquire(&header->tail);
            tail_snapshot = tail;
            if (head - tail < vgpu_shm::kRingCapacity) {
                break;
            }
            std::this_thread::yield();
        }

        vgpu_shm::RingEntry *entry = &entries[head % vgpu_shm::kRingCapacity];
        *entry = vgpu_shm::RingEntry{};
        vgpu_shm::StoreRelaxed(&entry->done, 0);
        entry->cuda_error = cudaErrorUnknown;
        entry->seq = head + 1;
        entry->trace_tag = tls_trace_tag;
        entry->queue_depth_at_submit = head - tail_snapshot;
        fill(entry);
        entry->t_client_submit_ns = NowNs();
        vgpu_shm::StoreRelease(&header->head, head + 1);

        if (!wait_completion) {
            *out = cudaSuccess;
            if (ring_wait_us) {
                *ring_wait_us = ElapsedUs(ring_start);
            }
            return true;
        }

        while (vgpu_shm::LoadAcquire(&entry->done) == 0) {
            if (vgpu_shm::LoadAcquire(&header->stop) != 0) {
                return false;
            }
            std::this_thread::yield();
        }
        entry->t_client_done_ns = NowNs();
        RingTrace().Record(*entry);
        *out = FromWireError(entry->cuda_error);
        if (ring_wait_us) {
            *ring_wait_us = ElapsedUs(ring_start);
        }
        return true;
    }

    bool SubmitMemcpyShmRing(
        uint64_t dst_device_ptr,
        uint64_t src_device_ptr,
        size_t count,
        cudaMemcpyKind kind,
        size_t shm_offset,
        uintptr_t stream_id,
        bool async,
        cudaError_t *out,
        uint64_t *ring_wait_us) {
        return SubmitRingEntry([&](vgpu_shm::RingEntry *entry) {
            entry->op = vgpu_shm::kRingOpMemcpyShm;
            entry->kind = static_cast<int32_t>(kind);
            entry->async = async ? 1 : 0;
            entry->dst_device_ptr = dst_device_ptr;
            entry->src_device_ptr = src_device_ptr;
            entry->count = static_cast<uint64_t>(count);
            entry->shm_offset = static_cast<uint64_t>(shm_offset);
            entry->stream_id = stream_id;
        }, out, ring_wait_us);
    }

    bool SubmitD2DRing(
        uint64_t dst_device_ptr,
        uint64_t src_device_ptr,
        size_t count,
        uintptr_t stream_id,
        bool async,
        cudaError_t *out) {
        return SubmitRingEntry([&](vgpu_shm::RingEntry *entry) {
            entry->op = vgpu_shm::kRingOpMemcpyD2D;
            entry->kind = static_cast<int32_t>(cudaMemcpyDeviceToDevice);
            entry->async = async ? 1 : 0;
            entry->dst_device_ptr = dst_device_ptr;
            entry->src_device_ptr = src_device_ptr;
            entry->count = static_cast<uint64_t>(count);
            entry->stream_id = stream_id;
        }, out, nullptr, !async);
    }

    bool SubmitSyncRing(uint64_t op, uintptr_t stream_id, uintptr_t event_id, cudaError_t *out) {
        return SubmitRingEntry([&](vgpu_shm::RingEntry *entry) {
            entry->op = op;
            entry->stream_id = stream_id;
            entry->event_id = event_id;
        }, out, nullptr);
    }

    bool SubmitLaunchKernelRing(const KernelInfo &kernel,
                                dim3 grid_dim,
                                dim3 block_dim,
                                void **args,
                                size_t shared_mem,
                                cudaStream_t stream,
                                cudaError_t *out) {
        if (kernel.kernel_name.size() >= vgpu_shm::kMaxKernelName ||
            kernel.param_sizes.size() > vgpu_shm::kMaxKernelArgs) {
            return false;
        }

        size_t arg_bytes = 0;
        for (size_t size : kernel.param_sizes) {
            arg_bytes += size;
            if (arg_bytes > vgpu_shm::kMaxKernelArgBytes) {
                return false;
            }
        }

        return SubmitRingEntry([&](vgpu_shm::RingEntry *entry) {
            entry->op = vgpu_shm::kRingOpLaunchKernel;
            entry->module_id = kernel.module_id;
            entry->stream_id = reinterpret_cast<uintptr_t>(stream);
            entry->shared_mem = static_cast<uint64_t>(shared_mem);
            entry->grid_dim[0] = grid_dim.x;
            entry->grid_dim[1] = grid_dim.y;
            entry->grid_dim[2] = grid_dim.z;
            entry->block_dim[0] = block_dim.x;
            entry->block_dim[1] = block_dim.y;
            entry->block_dim[2] = block_dim.z;
            entry->arg_count = static_cast<uint32_t>(kernel.param_sizes.size());
            std::memcpy(entry->kernel_name, kernel.kernel_name.c_str(), kernel.kernel_name.size() + 1);
            size_t cursor = 0;
            for (size_t i = 0; i < kernel.param_sizes.size(); ++i) {
                entry->arg_offsets[i] = static_cast<uint32_t>(cursor);
                entry->arg_sizes[i] = static_cast<uint32_t>(kernel.param_sizes[i]);
                std::memcpy(entry->arg_data + cursor, args[i], kernel.param_sizes[i]);
                cursor += kernel.param_sizes[i];
            }
            entry->arg_bytes = static_cast<uint32_t>(cursor);
        }, out, nullptr, false);
    }

    cudaError_t MemcpyViaShm(
        void *dst,
        const void *src,
        size_t count,
        cudaMemcpyKind kind,
        cudaStream_t stream,
        bool async) {
        const auto api_start = Clock::now();
        uint64_t alloc_us = 0;
        uint64_t h2d_host_copy_us = 0;
        uint64_t ring_wait_us = 0;
        uint64_t rpc_wait_us = 0;
        uint64_t d2h_copyout_us = 0;
        bool ring_submitted = false;
        bool direct_host_shm = false;
        if (!dst || !src) {
            return cudaErrorInvalidValue;
        }

        size_t shm_offset = 0;
        size_t blocks_used = 0;
        const auto alloc_start = Clock::now();
        if (kind == cudaMemcpyHostToDevice) {
            direct_host_shm = HostPointerToShmOffset(src, count, &shm_offset);
        } else if (kind == cudaMemcpyDeviceToHost) {
            direct_host_shm = HostPointerToShmOffset(dst, count, &shm_offset);
        }
        if (!direct_host_shm) {
            if (!AllocateShmBlocks(count, &shm_offset, &blocks_used, !async)) {
                return async ? cudaErrorMemoryAllocation : cudaErrorInvalidValue;
            }
        }
        alloc_us = ElapsedUs(alloc_start);

        char *slot = static_cast<char *>(shm_.base) + shm_offset;
        if (kind == cudaMemcpyHostToDevice && !direct_host_shm) {
            const auto copy_start = Clock::now();
            std::memcpy(slot, src, count);
            h2d_host_copy_us = ElapsedUs(copy_start);
        }

        vgpu::MemcpyShmRequest request;
        request.set_session_id(session_id_);
        request.set_count(static_cast<uint64_t>(count));
        request.set_kind(static_cast<int>(kind));
        request.set_shm_offset(static_cast<uint64_t>(shm_offset));
        request.set_stream_id(reinterpret_cast<uintptr_t>(stream));
        request.set_async(async);
        if (kind == cudaMemcpyHostToDevice) {
            request.set_dst_device_ptr(reinterpret_cast<uintptr_t>(dst));
        } else if (kind == cudaMemcpyDeviceToHost) {
            request.set_src_device_ptr(reinterpret_cast<uintptr_t>(src));
        } else {
            if (!direct_host_shm) {
                FreeShmBlocks(shm_offset, blocks_used);
            }
            return cudaErrorInvalidMemcpyDirection;
        }

        cudaError_t err = cudaSuccess;
        ring_submitted = SubmitMemcpyShmRing(
            request.dst_device_ptr(),
            request.src_device_ptr(),
            count,
            kind,
            shm_offset,
            request.stream_id(),
            async,
            &err,
            &ring_wait_us);
        if (!ring_submitted) {
            const auto rpc_start = Clock::now();
            vgpu::StatusReply reply;
            grpc::ClientContext context;
            grpc::Status status = stub_->MemcpyShm(&context, request, &reply);
            rpc_wait_us = ElapsedUs(rpc_start);
            if (!status.ok()) {
                std::cerr << "[cudart_proxy] MemcpyShm RPC failed: "
                          << status.error_message() << std::endl;
                if (!direct_host_shm) {
                    FreeShmBlocks(shm_offset, blocks_used);
                }
                return cudaErrorUnknown;
            }
            err = FromWireError(reply.cuda_error());
        }
        if (err != cudaSuccess) {
            if (!direct_host_shm) {
                FreeShmBlocks(shm_offset, blocks_used);
            }
            return err;
        }
        if (kind == cudaMemcpyDeviceToHost && !direct_host_shm) {
            const auto copyout_start = Clock::now();
            std::memcpy(dst, slot, count);
            d2h_copyout_us = ElapsedUs(copyout_start);
        }
        if (async && !direct_host_shm) {
            QueuePendingShmBlock(
                shm_offset,
                blocks_used,
                count,
                stream,
                kind == cudaMemcpyDeviceToHost ? dst : nullptr);
        } else if (!direct_host_shm) {
            FreeShmBlocks(shm_offset, blocks_used);
        }
        if (DetailedPerfRequested()) {
            std::cerr << "[cudart_proxy_perf] op="
                      << (async ? "MemcpyViaShmAsync" : "MemcpyViaShm")
                      << " kind=" << static_cast<int>(kind)
                      << " bytes=" << count
                      << " stream=" << reinterpret_cast<uintptr_t>(stream)
                      << " ring=" << static_cast<int>(ring_submitted)
                      << " direct_host_shm=" << static_cast<int>(direct_host_shm)
                      << " alloc_us=" << alloc_us
                      << " h2d_host_copy_us=" << h2d_host_copy_us
                      << " ring_wait_us=" << ring_wait_us
                      << " rpc_wait_us=" << rpc_wait_us
                      << " d2h_copyout_us=" << d2h_copyout_us
                      << " total_us=" << ElapsedUs(api_start)
                      << std::endl;
        }
        return cudaSuccess;
    }

    std::mutex mu_;
    std::mutex reg_mu_;
    std::mutex event_mu_;
    bool initialized_ = false;
    bool shutdown_registered_ = false;
    uint64_t session_id_ = 0;
    uint64_t next_local_module_handle_ = 1;
    ShmState shm_;
    std::unique_ptr<vgpu::VgpuRuntime::Stub> stub_;
    std::unordered_map<uint64_t, ModuleInfo> modules_;
    std::unordered_map<uintptr_t, KernelInfo> kernels_;
    std::unordered_map<uintptr_t, EventFence> event_fences_;
};

RuntimeProxyClient &Client() {
    static auto *client = new RuntimeProxyClient;
    return *client;
}

RingTraceStats &RingTrace() {
    static auto *trace = new RingTraceStats;
    return *trace;
}

void ShutdownAtExit() {
    Client().Shutdown();
    RingTrace().Print();
}

}  // namespace

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *count = 0;
    return SetLastError(Client().GetDeviceCount(count));
}

extern "C" cudaError_t cudaGetDevice(int *device) {
    if (!device) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *device = 0;
    return SetLastError(cudaSuccess);
}

extern "C" cudaError_t cudaSetDevice(int device) {
    if (device != 0) {
        return SetLastError(cudaErrorInvalidDevice);
    }
    return SetLastError(cudaSuccess);
}

extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    if (!prop) {
        return SetLastError(cudaErrorInvalidValue);
    }
    return SetLastError(Client().GetDeviceProperties(prop, device));
}

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *devPtr = nullptr;
    return SetLastError(Client().Malloc(devPtr, size));
}

extern "C" cudaError_t cudaFree(void *devPtr) {
    return SetLastError(Client().Free(devPtr));
}

extern "C" cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    return SetLastError(Client().HostAlloc(pHost, size, flags));
}

extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
    return SetLastError(Client().HostAlloc(ptr, size, cudaHostAllocDefault));
}

extern "C" cudaError_t cudaFreeHost(void *ptr) {
    return SetLastError(Client().FreeHost(ptr));
}

extern "C" cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    return SetLastError(Client().HostRegister(ptr, size, flags));
}

extern "C" cudaError_t cudaHostUnregister(void *ptr) {
    return SetLastError(Client().HostUnregister(ptr));
}

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    return SetLastError(Client().Memcpy(dst, src, count, kind));
}

extern "C" cudaError_t cudaMemcpyAsync(
    void *dst,
    const void *src,
    size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
    return SetLastError(Client().MemcpyAsync(dst, src, count, kind, stream));
}

extern "C" cudaError_t cudaDeviceSynchronize(void) {
    return SetLastError(Client().DeviceSynchronize());
}

extern "C" cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    if (!stream) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *stream = nullptr;
    return SetLastError(Client().StreamCreate(stream, cudaStreamDefault));
}

extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
    if (!stream) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *stream = nullptr;
    return SetLastError(Client().StreamCreate(stream, flags));
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return SetLastError(Client().StreamDestroy(stream));
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return SetLastError(Client().StreamSynchronize(stream));
}

extern "C" cudaError_t cudaEventCreate(cudaEvent_t *event) {
    if (!event) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *event = nullptr;
    return SetLastError(Client().EventCreate(event, cudaEventDefault));
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    if (!event) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *event = nullptr;
    return SetLastError(Client().EventCreate(event, flags));
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) {
        return SetLastError(cudaErrorInvalidResourceHandle);
    }
    return SetLastError(Client().EventRecord(event, stream));
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event) {
        return SetLastError(cudaErrorInvalidResourceHandle);
    }
    return SetLastError(Client().EventSynchronize(event));
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (!event) {
        return SetLastError(cudaErrorInvalidResourceHandle);
    }
    return SetLastError(Client().EventQuery(event));
}

extern "C" cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms || !start || !end) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *ms = 0.0f;
    return SetLastError(Client().EventElapsedTime(ms, start, end));
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return SetLastError(Client().EventDestroy(event));
}

extern "C" void vgpuSetTraceLabel(uint64_t label) {
    tls_trace_tag = label;
}

extern "C" void **__cudaRegisterFatBinary(void *fatCubin) {
    return Client().RegisterFatBinary(fatCubin);
}

extern "C" void __cudaRegisterFatBinaryEnd(void **) {
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    Client().UnregisterFatBinary(fatCubinHandle);
}

extern "C" char __cudaInitModule(void **) {
    return 0;
}

extern "C" void __cudaRegisterFunction(
    void **fatCubinHandle,
    const char *hostFun,
    char *,
    const char *deviceName,
    int,
    uint3 *,
    uint3 *,
    dim3 *,
    dim3 *,
    int *) {
    Client().RegisterFunction(fatCubinHandle, hostFun, deviceName);
}

extern "C" unsigned __cudaPushCallConfiguration(
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem,
    cudaStream_t stream) {
    tls_grid_dim = gridDim;
    tls_block_dim = blockDim;
    tls_shared_mem = sharedMem;
    tls_stream = stream;
    return cudaSuccess;
}

extern "C" cudaError_t __cudaPopCallConfiguration(
    dim3 *gridDim,
    dim3 *blockDim,
    size_t *sharedMem,
    void *stream) {
    if (!gridDim || !blockDim || !sharedMem || !stream) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *gridDim = tls_grid_dim;
    *blockDim = tls_block_dim;
    *sharedMem = tls_shared_mem;
    *static_cast<cudaStream_t *>(stream) = tls_stream;
    return SetLastError(cudaSuccess);
}

extern "C" cudaError_t cudaLaunchKernel(
    const void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream) {
    return SetLastError(Client().LaunchKernel(func, gridDim, blockDim, args, sharedMem, stream));
}

extern "C" cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    if (!runtimeVersion) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *runtimeVersion = kRuntimeVersion;
    return SetLastError(cudaSuccess);
}

extern "C" cudaError_t cudaDriverGetVersion(int *driverVersion) {
    if (!driverVersion) {
        return SetLastError(cudaErrorInvalidValue);
    }
    *driverVersion = kFallbackDriverVersion;
    return SetLastError(cudaSuccess);
}

extern "C" cudaError_t cudaGetLastError(void) {
    cudaError_t error = tls_last_error;
    tls_last_error = cudaSuccess;
    return error;
}

extern "C" cudaError_t cudaPeekAtLastError(void) {
    return tls_last_error;
}

extern "C" const char *cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "no error";
        case cudaErrorInvalidValue:
            return "invalid argument";
        case cudaErrorMemoryAllocation:
            return "out of memory";
        case cudaErrorInitializationError:
            return "initialization error";
        case cudaErrorInvalidDevice:
            return "invalid device";
        case cudaErrorInvalidDevicePointer:
            return "invalid device pointer";
        case cudaErrorInvalidDeviceFunction:
            return "invalid device function";
        case cudaErrorInvalidResourceHandle:
            return "invalid resource handle";
        case cudaErrorNotReady:
            return "device not ready";
        case cudaErrorInvalidMemcpyDirection:
            return "invalid copy direction for memcpy";
        case cudaErrorLaunchFailure:
            return "unspecified launch failure";
        case cudaErrorNotSupported:
            return "operation not supported";
        case cudaErrorUnknown:
            return "unknown error";
        default:
            return "unrecognized cuda error";
    }
}

extern "C" const char *cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";
        case cudaErrorNotReady:
            return "cudaErrorNotReady";
        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";
        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";
        case cudaErrorNotSupported:
            return "cudaErrorNotSupported";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
        default:
            return "cudaErrorUnrecognized";
    }
}
