#include <grpcpp/grpcpp.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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

constexpr int kCudaSuccess = 0;
constexpr int kCudaErrorInvalidValue = 1;
constexpr uint64_t kVirtualPtrBase = 0xCADA00000000ull;
constexpr uint64_t kVirtualPtrStride = 0x1000ull;
constexpr uint64_t kVirtualStreamBase = 0xCADA10000000ull;
constexpr uint64_t kVirtualStreamStride = 0x1000ull;
constexpr uint64_t kVirtualEventBase = 0xCADA20000000ull;
constexpr uint64_t kVirtualEventStride = 0x1000ull;
constexpr uint64_t kDefaultSessionTimeoutMs = 30000;

int RuntimeError(CUresult result) {
    switch (result) {
        case CUDA_SUCCESS:
            return cudaSuccess;
        case CUDA_ERROR_OUT_OF_MEMORY:
            return cudaErrorMemoryAllocation;
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_DEINITIALIZED:
            return cudaErrorInitializationError;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_INVALID_DEVICE:
            return cudaErrorInvalidDevice;
        case CUDA_ERROR_INVALID_HANDLE:
            return cudaErrorInvalidResourceHandle;
        case CUDA_ERROR_NOT_READY:
            return cudaErrorNotReady;
        default:
            return cudaErrorUnknown;
    }
}

std::string DriverErrorString(CUresult result) {
    const char *name = nullptr;
    const char *text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);

    std::string out = name ? name : "CUDA_ERROR_UNKNOWN";
    if (text) {
        out += ": ";
        out += text;
    }
    return out;
}

bool IsValidShmName(const std::string &name) {
    if (name.rfind("/vgpu_", 0) != 0 || name.size() > 128) {
        return false;
    }
    for (char ch : name) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (!(std::isalnum(c) || ch == '_' || ch == '-' || ch == '/')) {
            return false;
        }
    }
    return true;
}

bool RangeFits(size_t offset, size_t count, size_t size) {
    return offset <= size && count <= size - offset;
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

using Clock = std::chrono::steady_clock;

uint64_t ElapsedUs(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

uint64_t NowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

bool DetailedPerfRequested() {
    const char *env = std::getenv("VGPU_PERF_DETAIL");
    return env && std::strcmp(env, "1") == 0;
}

bool PerfLogRequested() {
    const char *env = std::getenv("VGPU_PERF_LOG");
    return env && std::strcmp(env, "1") == 0;
}

bool InitTraceRequested() {
    const char *env = std::getenv("VGPU_INIT_TRACE");
    return env && std::strcmp(env, "1") == 0;
}

bool PinnedShmRequested() {
    const char *env = std::getenv("VGPU_PINNED_SHM");
    return !env || std::strcmp(env, "0") != 0;
}

void LogPerf(
    const char *op,
    uint64_t session_id,
    uint64_t bytes,
    uint64_t stream_id,
    int cuda_error,
    Clock::time_point start) {
    if (!PerfLogRequested()) {
        return;
    }
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start).count();
    std::cout << "[vgpu_perf] op=" << op
              << " sid=" << session_id
              << " bytes=" << bytes
              << " stream=" << stream_id
              << " cuda_error=" << cuda_error
              << " elapsed_us=" << elapsed_us << std::endl;
}

struct Allocation {
    CUdeviceptr device_ptr = 0;
    size_t size = 0;
};

struct ShmMapping {
    std::string name;
    int fd = -1;
    void *base = nullptr;
    size_t size = 0;
    void *pinned_base = nullptr;
    size_t pinned_size = 0;
    bool pinned = false;
};

struct SessionState {
    std::mutex mu;
    std::condition_variable cv;
    bool closing = false;
    uint64_t session_id = 0;
    uint32_t client_pid = 0;
    uint64_t memory_limit = 0;
    uint64_t memory_used = 0;
    uint32_t active_ops = 0;
    std::chrono::steady_clock::time_point last_seen = std::chrono::steady_clock::now();
    CUstream default_stream = nullptr;
    ShmMapping shm;
    bool ring_enabled = false;
    std::atomic<bool> ring_stop{false};
    std::atomic<int> pending_error{cudaSuccess};
    std::thread ring_thread;
    uint64_t next_virtual_ptr = kVirtualPtrBase + kVirtualPtrStride;
    std::unordered_map<uint64_t, Allocation> allocations;
    std::unordered_map<uint64_t, CUmodule> modules;
    std::unordered_map<uint64_t, CUstream> streams;
    std::unordered_map<uint64_t, CUevent> events;
};

struct FatbinWrapper {
    unsigned int magic;
    unsigned int version;
    const void *data;
    void *unused;
};

class VgpuRuntimeService final : public vgpu::VgpuRuntime::Service {
public:
    VgpuRuntimeService() : session_timeout_ms_(ReadSessionTimeoutMs()) {
        reaper_thread_ = std::thread([this] { ReaperLoop(); });
    }

    ~VgpuRuntimeService() override {
        stop_reaper_.store(true);
        if (reaper_thread_.joinable()) {
            reaper_thread_.join();
        }
    }

    grpc::Status CreateSession(
        grpc::ServerContext *,
        const vgpu::CreateSessionRequest *request,
        vgpu::CreateSessionReply *reply) override {
        const auto total_start = Clock::now();
        const uint64_t session_id = next_session_id_.fetch_add(1);
        CUstream default_stream = nullptr;
        ShmMapping shm;
        bool shm_enabled = false;
        bool ring_enabled = false;
        uint64_t shm_map_us = 0;
        uint64_t ring_init_us = 0;
        uint64_t cuda_context_us = 0;
        uint64_t default_stream_us = 0;
        uint64_t pin_shm_us = 0;
        if (!request->shm_name().empty() && request->shm_size() > 0 &&
            IsValidShmName(request->shm_name())) {
            const auto shm_map_start = Clock::now();
            const int fd = shm_open(request->shm_name().c_str(), O_RDWR, 0600);
            if (fd >= 0) {
                void *base = mmap(nullptr, static_cast<size_t>(request->shm_size()),
                                  PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (base != MAP_FAILED) {
                    shm.name = request->shm_name();
                    shm.fd = fd;
                    shm.base = base;
                    shm.size = static_cast<size_t>(request->shm_size());
                    shm_enabled = true;
                    if (shm.size > vgpu_shm::kControlBytes) {
                        const auto ring_init_start = Clock::now();
                        vgpu_shm::InitRing(shm.base);
                        ring_init_us = ElapsedUs(ring_init_start);
                        ring_enabled = true;
                    }
                } else {
                    close(fd);
                }
            }
            shm_map_us = ElapsedUs(shm_map_start);
        }
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            const auto cuda_context_start = Clock::now();
            CUresult result = EnsureCudaLocked();
            cuda_context_us = ElapsedUs(cuda_context_start);
            if (result == CUDA_SUCCESS) {
                const auto stream_start = Clock::now();
                result = cuStreamCreate(&default_stream, CU_STREAM_NON_BLOCKING);
                default_stream_us = ElapsedUs(stream_start);
            }
            if (result != CUDA_SUCCESS) {
                if (shm.base) {
                    munmap(shm.base, shm.size);
                }
                if (shm.fd >= 0) {
                    close(shm.fd);
                }
                reply->set_cuda_error(RuntimeError(result));
                reply->set_message(DriverErrorString(result));
                return grpc::Status::OK;
            }

            session = std::make_shared<SessionState>();
            session->session_id = session_id;
            session->client_pid = request->client_pid();
            session->memory_limit = request->requested_memory_limit();
            session->last_seen = std::chrono::steady_clock::now();
            session->default_stream = default_stream;
            session->shm = std::move(shm);
            session->ring_enabled = ring_enabled;
            const auto pin_start = Clock::now();
            TryPinShmArenaLocked(session);
            pin_shm_us = ElapsedUs(pin_start);
            sessions_.emplace(session_id, session);
        }

        if (session && session->ring_enabled) {
            session->ring_thread = std::thread([this, session] { RingWorkerLoop(session); });
        }

        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("session created for pid " + std::to_string(request->client_pid()));
        reply->set_session_id(session_id);
        reply->set_shm_enabled(shm_enabled);
        reply->set_shm_data_offset(ring_enabled ? vgpu_shm::kControlBytes : 0);
        reply->set_shm_data_size(
            shm_enabled
                ? request->shm_size() - (ring_enabled ? vgpu_shm::kControlBytes : 0)
                : 0);

        if (DetailedPerfRequested()) {
            std::cout << "[vgpu] op=CreateSession sid=" << session_id
                      << " pid=" << request->client_pid()
                      << " name=" << request->client_name()
                      << " memory_limit=" << request->requested_memory_limit()
                      << " shm_enabled=" << shm_enabled
                      << " ring_enabled=" << ring_enabled
                      << " shm_pinned=" << (session && session->shm.pinned)
                      << " shm_name=" << request->shm_name()
                      << " shm_size=" << request->shm_size()
                      << " timeout_ms=" << session_timeout_ms_ << std::endl;
        }
        if (InitTraceRequested()) {
            std::cout << "init_trace side=server op=CreateSession"
                      << " sid=" << session_id
                      << " total_us=" << ElapsedUs(total_start)
                      << " shm_map_us=" << shm_map_us
                      << " ring_init_us=" << ring_init_us
                      << " cuda_context_us=" << cuda_context_us
                      << " default_stream_us=" << default_stream_us
                      << " pin_shm_us=" << pin_shm_us
                      << " shm_enabled=" << shm_enabled
                      << " ring_enabled=" << ring_enabled
                      << std::endl;
        }
        return grpc::Status::OK;
    }

    grpc::Status DestroySession(
        grpc::ServerContext *,
        const vgpu::DestroySessionRequest *request,
        vgpu::StatusReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
                sessions_.erase(it);
            }
        }
        if (session) {
            CleanupSessionResources(session, "destroy");
        }

        reply->set_cuda_error(session ? kCudaSuccess : kCudaErrorInvalidValue);
        reply->set_message(session ? "session destroyed" : "session not found");

        if (DetailedPerfRequested()) {
            std::cout << "[vgpu] op=DestroySession sid=" << request->session_id()
                      << " found=" << static_cast<bool>(session) << std::endl;
        }
        return grpc::Status::OK;
    }

    grpc::Status GetDeviceCount(
        grpc::ServerContext *,
        const vgpu::GetDeviceCountRequest *request,
        vgpu::GetDeviceCountReply *reply) override {
        if (!HasSession(request->session_id())) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            reply->set_count(0);
            return grpc::Status::OK;
        }

        std::lock_guard<std::mutex> lock(mu_);
        CUresult result = EnsureCudaLocked();
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            reply->set_count(0);
            return grpc::Status::OK;
        }

        int count = 0;
        result = cuDeviceGetCount(&count);
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "device count" : DriverErrorString(result));
        reply->set_count(result == CUDA_SUCCESS ? count : 0);
        return grpc::Status::OK;
    }

    grpc::Status GetDeviceProperties(
        grpc::ServerContext *,
        const vgpu::GetDevicePropertiesRequest *request,
        vgpu::GetDevicePropertiesReply *reply) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (sessions_.find(request->session_id()) == sessions_.end()) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }

        CUresult result = EnsureCudaLocked();
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        CUdevice device{};
        result = cuDeviceGet(&device, request->device());
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        char name[256] = {};
        size_t total_mem = 0;
        int sm_count = 0;
        int major = 0;
        int minor = 0;
        int warp_size = 0;
        int max_threads = 0;
        int max_threads_dim[3] = {};
        int max_grid_size[3] = {};
        int clock_rate = 0;
        int memory_clock_rate = 0;
        int memory_bus_width = 0;

        cuDeviceGetName(name, sizeof(name), device);
        cuDeviceTotalMem(&total_mem, device);
        cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
        cuDeviceGetAttribute(&max_threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
        cuDeviceGetAttribute(&max_threads_dim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
        cuDeviceGetAttribute(&max_threads_dim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
        cuDeviceGetAttribute(&max_threads_dim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device);
        cuDeviceGetAttribute(&max_grid_size[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
        cuDeviceGetAttribute(&max_grid_size[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
        cuDeviceGetAttribute(&max_grid_size[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
        cuDeviceGetAttribute(&clock_rate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
        cuDeviceGetAttribute(&memory_clock_rate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
        cuDeviceGetAttribute(&memory_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);

        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("device properties");
        reply->set_name(name);
        reply->set_total_global_mem(total_mem);
        reply->set_multi_processor_count(sm_count);
        reply->set_major(major);
        reply->set_minor(minor);
        reply->set_warp_size(warp_size);
        reply->set_max_threads_per_block(max_threads);
        for (int value : max_threads_dim) {
            reply->add_max_threads_dim(value);
        }
        for (int value : max_grid_size) {
            reply->add_max_grid_size(value);
        }
        reply->set_clock_rate(clock_rate);
        reply->set_memory_clock_rate(memory_clock_rate);
        reply->set_memory_bus_width(memory_bus_width);
        return grpc::Status::OK;
    }

    grpc::Status Malloc(
        grpc::ServerContext *,
        const vgpu::MallocRequest *request,
        vgpu::MallocReply *reply) override {
        const auto op_start = Clock::now();
        if (request->size() == 0) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("cudaMalloc size must be greater than zero");
            LogPerf("Malloc", request->session_id(), request->size(), 0, kCudaErrorInvalidValue, op_start);
            return grpc::Status::OK;
        }

        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto session_it = sessions_.find(request->session_id());
            if (session_it != sessions_.end()) {
                session = session_it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            LogPerf("Malloc", request->session_id(), request->size(), 0, kCudaErrorInvalidValue, op_start);
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUresult result = EnsureCudaLocked();
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (session->closing) {
                reply->set_cuda_error(kCudaErrorInvalidValue);
                reply->set_message("session is closing");
                return grpc::Status::OK;
            }
            if (session->memory_limit > 0) {
                const bool over_limit = session->memory_used >= session->memory_limit ||
                    request->size() > session->memory_limit - session->memory_used;
                if (over_limit) {
                    reply->set_cuda_error(cudaErrorMemoryAllocation);
                    reply->set_message("session memory limit exceeded");
                    LogPerf("Malloc", request->session_id(), request->size(), 0, cudaErrorMemoryAllocation, op_start);
                    return grpc::Status::OK;
                }
            }
        }

        CUdeviceptr real_ptr = 0;
        result = cuMemAlloc(&real_ptr, static_cast<size_t>(request->size()));
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        uint64_t virtual_ptr = 0;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (session->closing) {
                cuMemFree(real_ptr);
                reply->set_cuda_error(kCudaErrorInvalidValue);
                reply->set_message("session is closing");
                return grpc::Status::OK;
            }
            virtual_ptr = NextVirtualPtrLocked(*session, static_cast<size_t>(request->size()));
            session->allocations.emplace(
                virtual_ptr,
                Allocation{real_ptr, static_cast<size_t>(request->size())});
            session->memory_used += request->size();
        }

        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("device memory allocated");
        reply->set_device_ptr(virtual_ptr);
        LogPerf("Malloc", request->session_id(), request->size(), 0, kCudaSuccess, op_start);
        return grpc::Status::OK;
    }

    grpc::Status Free(
        grpc::ServerContext *,
        const vgpu::FreeRequest *request,
        vgpu::StatusReply *reply) override {
        const auto op_start = Clock::now();
        if (request->device_ptr() == 0) {
            reply->set_cuda_error(kCudaSuccess);
            reply->set_message("cudaFree null pointer");
            LogPerf("Free", request->session_id(), 0, 0, kCudaSuccess, op_start);
            return grpc::Status::OK;
        }

        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto session_it = sessions_.find(request->session_id());
            if (session_it != sessions_.end()) {
                session = session_it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            LogPerf("Free", request->session_id(), 0, 0, kCudaErrorInvalidValue, op_start);
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUdeviceptr real_ptr = 0;
        size_t freed_size = 0;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            auto alloc_it = session->allocations.find(request->device_ptr());
            if (alloc_it == session->allocations.end()) {
                reply->set_cuda_error(cudaErrorInvalidDevicePointer);
                reply->set_message("device pointer not found");
                LogPerf("Free", request->session_id(), 0, 0, cudaErrorInvalidDevicePointer, op_start);
                return grpc::Status::OK;
            }
            real_ptr = alloc_it->second.device_ptr;
            freed_size = alloc_it->second.size;
            session->allocations.erase(alloc_it);
            if (session->memory_used >= freed_size) {
                session->memory_used -= freed_size;
            } else {
                session->memory_used = 0;
            }
        }

        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuMemFree(real_ptr);
        }
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("device memory freed");
        LogPerf("Free", request->session_id(), freed_size, 0, kCudaSuccess, op_start);
        return grpc::Status::OK;
    }

    grpc::Status Memcpy(
        grpc::ServerContext *,
        const vgpu::MemcpyRequest *request,
        vgpu::MemcpyReply *reply) override {
        return DoMemcpyRpc(*request, reply, false);
    }

    grpc::Status MemcpyAsync(
        grpc::ServerContext *,
        const vgpu::MemcpyRequest *request,
        vgpu::MemcpyReply *reply) override {
        return DoMemcpyRpc(*request, reply, true);
    }

    grpc::Status MemcpyShm(
        grpc::ServerContext *,
        const vgpu::MemcpyShmRequest *request,
        vgpu::StatusReply *reply) override {
        const auto op_start = Clock::now();
        const char *op_name = request->async() ? "MemcpyShmAsync" : "MemcpyShm";
        if (request->count() == 0) {
            reply->set_cuda_error(kCudaSuccess);
            reply->set_message("zero-byte memcpy");
            LogPerf(op_name, request->session_id(), 0, request->stream_id(), kCudaSuccess, op_start);
            return grpc::Status::OK;
        }

        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto session_it = sessions_.find(request->session_id());
            if (session_it != sessions_.end()) {
                session = session_it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            LogPerf(op_name, request->session_id(), request->count(), request->stream_id(), kCudaErrorInvalidValue, op_start);
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = DoMemcpyShm(session, *request);
        }

        const int cuda_error = RuntimeError(result);
        reply->set_cuda_error(cuda_error);
        reply->set_message(result == CUDA_SUCCESS ? "shm memcpy complete" : DriverErrorString(result));
        LogPerf(op_name, request->session_id(), request->count(), request->stream_id(), cuda_error, op_start);
        return grpc::Status::OK;
    }

    grpc::Status DeviceSynchronize(
        grpc::ServerContext *,
        const vgpu::DeviceSynchronizeRequest *request,
        vgpu::StatusReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUresult result = DoDeviceSynchronize(session);
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "context synchronized" : DriverErrorString(result));
        return grpc::Status::OK;
    }

    grpc::Status StreamCreate(
        grpc::ServerContext *,
        const vgpu::StreamCreateRequest *request,
        vgpu::StreamCreateReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUresult result = EnsureCudaLocked();
        CUstream stream = nullptr;
        if (result == CUDA_SUCCESS) {
            result = cuStreamCreate(&stream, request->flags());
        }
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        const uint64_t stream_id = NextVirtualStreamLocked();
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (session->closing) {
                cuStreamDestroy(stream);
                reply->set_cuda_error(kCudaErrorInvalidValue);
                reply->set_message("session is closing");
                return grpc::Status::OK;
            }
            session->streams.emplace(stream_id, stream);
        }

        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("stream created");
        reply->set_stream_id(stream_id);
        if (DetailedPerfRequested()) {
            std::cout << "[vgpu] op=StreamCreate sid=" << request->session_id()
                      << " stream=" << stream_id << std::endl;
        }
        return grpc::Status::OK;
    }

    grpc::Status StreamDestroy(
        grpc::ServerContext *,
        const vgpu::StreamDestroyRequest *request,
        vgpu::StatusReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);
        if (request->stream_id() == 0) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("cannot destroy default stream");
            return grpc::Status::OK;
        }

        CUstream stream = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            auto it = session->streams.find(request->stream_id());
            if (it == session->streams.end()) {
                reply->set_cuda_error(cudaErrorInvalidResourceHandle);
                reply->set_message("stream not found");
                return grpc::Status::OK;
            }
            stream = it->second;
            session->streams.erase(it);
        }

        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuStreamSynchronize(stream);
        }
        if (result == CUDA_SUCCESS) {
            result = cuStreamDestroy(stream);
        }
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "stream destroyed" : DriverErrorString(result));
        return grpc::Status::OK;
    }

    grpc::Status StreamSynchronize(
        grpc::ServerContext *,
        const vgpu::StreamSynchronizeRequest *request,
        vgpu::StatusReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUstream stream = nullptr;
        bool ok = false;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            ok = ResolveStreamLocked(*session, request->stream_id(), &stream);
        }
        if (!ok) {
            reply->set_cuda_error(cudaErrorInvalidResourceHandle);
            reply->set_message("stream not found");
            return grpc::Status::OK;
        }

        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuStreamSynchronize(stream);
        }
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "stream synchronized" : DriverErrorString(result));
        return grpc::Status::OK;
    }

    grpc::Status EventCreate(
        grpc::ServerContext *,
        const vgpu::EventCreateRequest *request,
        vgpu::EventCreateReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUresult result = EnsureCudaLocked();
        CUevent event = nullptr;
        if (result == CUDA_SUCCESS) {
            result = cuEventCreate(&event, request->flags());
        }
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        const uint64_t event_id = NextVirtualEventLocked();
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (session->closing) {
                cuEventDestroy(event);
                reply->set_cuda_error(kCudaErrorInvalidValue);
                reply->set_message("session is closing");
                return grpc::Status::OK;
            }
            session->events.emplace(event_id, event);
        }

        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("event created");
        reply->set_event_id(event_id);
        return grpc::Status::OK;
    }

    grpc::Status EventRecord(
        grpc::ServerContext *,
        const vgpu::EventRecordRequest *request,
        vgpu::StatusReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUevent event = nullptr;
        CUstream stream = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (!ResolveEventLocked(*session, request->event_id(), &event) ||
                !ResolveStreamLocked(*session, request->stream_id(), &stream)) {
                reply->set_cuda_error(cudaErrorInvalidResourceHandle);
                reply->set_message("event or stream not found");
                return grpc::Status::OK;
            }
        }

        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuEventRecord(event, stream);
        }
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "event recorded" : DriverErrorString(result));
        return grpc::Status::OK;
    }

    grpc::Status EventSynchronize(
        grpc::ServerContext *,
        const vgpu::EventSynchronizeRequest *request,
        vgpu::StatusReply *reply) override {
        CUevent event = nullptr;
        grpc::Status status = ResolveEventForRequest(request->session_id(), request->event_id(), &event, reply);
        if (!status.ok() || reply->cuda_error() != kCudaSuccess) {
            return status;
        }

        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuEventSynchronize(event);
        }
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "event synchronized" : DriverErrorString(result));
        return grpc::Status::OK;
    }

    grpc::Status EventQuery(
        grpc::ServerContext *,
        const vgpu::EventQueryRequest *request,
        vgpu::StatusReply *reply) override {
        CUevent event = nullptr;
        grpc::Status status = ResolveEventForRequest(request->session_id(), request->event_id(), &event, reply);
        if (!status.ok() || reply->cuda_error() != kCudaSuccess) {
            return status;
        }

        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuEventQuery(event);
        }
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "event complete" : DriverErrorString(result));
        return grpc::Status::OK;
    }

    grpc::Status EventElapsedTime(
        grpc::ServerContext *,
        const vgpu::EventElapsedTimeRequest *request,
        vgpu::EventElapsedTimeReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUevent start = nullptr;
        CUevent stop = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (!ResolveEventLocked(*session, request->start_event_id(), &start) ||
                !ResolveEventLocked(*session, request->stop_event_id(), &stop)) {
                reply->set_cuda_error(cudaErrorInvalidResourceHandle);
                reply->set_message("event not found");
                return grpc::Status::OK;
            }
        }

        float ms = 0.0f;
        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuEventElapsedTime(&ms, start, stop);
        }
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "event elapsed time" : DriverErrorString(result));
        reply->set_milliseconds(result == CUDA_SUCCESS ? ms : 0.0f);
        return grpc::Status::OK;
    }

    grpc::Status EventDestroy(
        grpc::ServerContext *,
        const vgpu::EventDestroyRequest *request,
        vgpu::StatusReply *reply) override {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(request->session_id());
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUevent event = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            auto it = session->events.find(request->event_id());
            if (it == session->events.end()) {
                reply->set_cuda_error(cudaErrorInvalidResourceHandle);
                reply->set_message("event not found");
                return grpc::Status::OK;
            }
            event = it->second;
            session->events.erase(it);
        }

        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuEventDestroy(event);
        }
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "event destroyed" : DriverErrorString(result));
        return grpc::Status::OK;
    }

    grpc::Status DoMemcpyRpc(
        const vgpu::MemcpyRequest &request,
        vgpu::MemcpyReply *reply,
        bool async_rpc) {
        const auto op_start = Clock::now();
        const char *op_name = async_rpc || request.async() ? "MemcpyAsync" : "Memcpy";
        if (request.count() == 0) {
            reply->set_cuda_error(kCudaSuccess);
            reply->set_message("zero-byte memcpy");
            LogPerf(op_name, request.session_id(), 0, request.stream_id(), kCudaSuccess, op_start);
            return grpc::Status::OK;
        }

        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto session_it = sessions_.find(request.session_id());
            if (session_it != sessions_.end()) {
                session = session_it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            LogPerf(op_name, request.session_id(), request.count(), request.stream_id(), kCudaErrorInvalidValue, op_start);
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUresult result = EnsureCudaLocked();
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        result = DoMemcpy(session, request, reply, async_rpc || request.async());
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            LogPerf(op_name, request.session_id(), request.count(), request.stream_id(), RuntimeError(result), op_start);
            return grpc::Status::OK;
        }

        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("memcpy complete");
        LogPerf(op_name, request.session_id(), request.count(), request.stream_id(), kCudaSuccess, op_start);
        return grpc::Status::OK;
    }

    grpc::Status RegisterModule(
        grpc::ServerContext *,
        const vgpu::RegisterModuleRequest *request,
        vgpu::RegisterModuleReply *reply) override {
        const auto total_start = Clock::now();
        if (request->fatbin().empty()) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("empty fatbin");
            return grpc::Status::OK;
        }

        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto session_it = sessions_.find(request->session_id());
            if (session_it != sessions_.end()) {
                session = session_it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUresult result = EnsureCudaLocked();
        uint64_t module_load_us = 0;
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        FatbinWrapper wrapper{0x466243b1u, 1u, request->fatbin().data(), nullptr};
        CUmodule module = nullptr;
        const auto module_load_start = Clock::now();
        result = cuModuleLoadFatBinary(&module, &wrapper);
        if (result != CUDA_SUCCESS) {
            result = cuModuleLoadData(&module, request->fatbin().data());
        }
        module_load_us = ElapsedUs(module_load_start);
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        const uint64_t module_id = next_module_id_.fetch_add(1);
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (session->closing) {
                cuModuleUnload(module);
                reply->set_cuda_error(kCudaErrorInvalidValue);
                reply->set_message("session is closing");
                return grpc::Status::OK;
            }
            session->modules.emplace(module_id, module);
        }
        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("module registered");
        reply->set_module_id(module_id);
        if (DetailedPerfRequested()) {
            std::cout << "[vgpu] op=RegisterModule sid=" << request->session_id()
                      << " module=" << module_id
                      << " bytes=" << request->fatbin().size() << std::endl;
        }
        if (InitTraceRequested()) {
            std::cout << "init_trace side=server op=RegisterModule"
                      << " sid=" << request->session_id()
                      << " module=" << module_id
                      << " total_us=" << ElapsedUs(total_start)
                      << " module_load_us=" << module_load_us
                      << " bytes=" << request->fatbin().size()
                      << std::endl;
        }
        return grpc::Status::OK;
    }

    grpc::Status LaunchKernel(
        grpc::ServerContext *,
        const vgpu::LaunchKernelRequest *request,
        vgpu::StatusReply *reply) override {
        const auto op_start = Clock::now();
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto session_it = sessions_.find(request->session_id());
            if (session_it != sessions_.end()) {
                session = session_it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }
        TouchSession(session);

        CUmodule module = nullptr;
        CUstream stream = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            auto module_it = session->modules.find(request->module_id());
            if (module_it == session->modules.end()) {
                reply->set_cuda_error(kCudaErrorInvalidValue);
                reply->set_message("module not found");
                return grpc::Status::OK;
            }
            if (!ResolveStreamLocked(*session, request->stream_id(), &stream)) {
                reply->set_cuda_error(cudaErrorInvalidResourceHandle);
                reply->set_message("stream not found");
                return grpc::Status::OK;
            }
            module = module_it->second;
        }

        CUresult result = EnsureCudaLocked();
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        CUfunction function = nullptr;
        result = cuModuleGetFunction(&function, module, request->kernel_name().c_str());
        if (result != CUDA_SUCCESS) {
            reply->set_cuda_error(RuntimeError(result));
            reply->set_message(DriverErrorString(result));
            return grpc::Status::OK;
        }

        std::vector<std::string> arg_storage;
        std::vector<void *> kernel_params;
        arg_storage.reserve(request->args_size());
        kernel_params.reserve(request->args_size());
        for (const auto &arg : request->args()) {
            arg_storage.emplace_back(arg.data());
            std::string &data = arg_storage.back();
            if (data.size() == sizeof(uint64_t)) {
                uint64_t maybe_virtual_ptr = 0;
                std::memcpy(&maybe_virtual_ptr, data.data(), sizeof(maybe_virtual_ptr));
                std::lock_guard<std::mutex> session_lock(session->mu);
                uint64_t ptr_offset = 0;
                const Allocation *allocation = FindAllocationLocked(*session, maybe_virtual_ptr, 0, &ptr_offset);
                if (allocation) {
                    CUdeviceptr real_ptr = allocation->device_ptr + ptr_offset;
                    std::memcpy(data.data(), &real_ptr, sizeof(real_ptr));
                }
            }
            kernel_params.push_back(data.data());
        }

        result = cuLaunchKernel(
            function,
            request->grid_dim().x(),
            request->grid_dim().y(),
            request->grid_dim().z(),
            request->block_dim().x(),
            request->block_dim().y(),
            request->block_dim().z(),
            static_cast<unsigned int>(request->shared_mem()),
            stream,
            kernel_params.data(),
            nullptr);
        reply->set_cuda_error(RuntimeError(result));
        reply->set_message(result == CUDA_SUCCESS ? "kernel launched" : DriverErrorString(result));
        LogPerf("LaunchKernel", request->session_id(), 0, request->stream_id(), RuntimeError(result), op_start);
        if (DetailedPerfRequested()) {
            std::cout << "[vgpu] op=LaunchKernel sid=" << request->session_id()
                      << " module=" << request->module_id()
                      << " kernel=" << request->kernel_name()
                      << " grid=(" << request->grid_dim().x() << ","
                      << request->grid_dim().y() << "," << request->grid_dim().z() << ")"
                      << " block=(" << request->block_dim().x() << ","
                      << request->block_dim().y() << "," << request->block_dim().z() << ")"
                      << " args=" << request->args_size()
                      << " result=" << DriverErrorString(result) << std::endl;
        }
        return grpc::Status::OK;
    }

private:
    static uint64_t ReadSessionTimeoutMs() {
        const char *env = std::getenv("VGPU_SESSION_TIMEOUT_MS");
        if (!env || env[0] == '\0') {
            return kDefaultSessionTimeoutMs;
        }
        char *end = nullptr;
        const unsigned long long value = std::strtoull(env, &end, 10);
        if (!end || *end != '\0' || value == 0) {
            return kDefaultSessionTimeoutMs;
        }
        return static_cast<uint64_t>(value);
    }

    void TouchSession(const std::shared_ptr<SessionState> &session) {
        if (!session) {
            return;
        }
        std::lock_guard<std::mutex> lock(session->mu);
        session->last_seen = std::chrono::steady_clock::now();
    }

    void TryPinShmArenaLocked(const std::shared_ptr<SessionState> &session) {
        if (!session || !PinnedShmRequested() || !session->shm.base || session->shm.size == 0) {
            return;
        }

        const size_t data_offset = session->ring_enabled ? vgpu_shm::kControlBytes : 0;
        if (data_offset >= session->shm.size) {
            return;
        }

        void *arena_base = static_cast<char *>(session->shm.base) + data_offset;
        const size_t arena_size = session->shm.size - data_offset;
        const CUresult result = cuMemHostRegister(arena_base, arena_size, CU_MEMHOSTREGISTER_PORTABLE);
        if (result == CUDA_SUCCESS) {
            session->shm.pinned_base = arena_base;
            session->shm.pinned_size = arena_size;
            session->shm.pinned = true;
            if (DetailedPerfRequested()) {
                std::cout << "[vgpu] op=PinShm sid=" << session->session_id
                          << " bytes=" << arena_size
                          << " offset=" << data_offset
                          << " result=success" << std::endl;
            }
        } else {
            if (DetailedPerfRequested()) {
                std::cout << "[vgpu] op=PinShm sid=" << session->session_id
                          << " bytes=" << arena_size
                          << " offset=" << data_offset
                          << " result=failed"
                          << " error=" << DriverErrorString(result) << std::endl;
            }
        }
    }

    void UnpinShmArenaLocked(const std::shared_ptr<SessionState> &session) {
        if (!session || !session->shm.pinned || !session->shm.pinned_base) {
            return;
        }
        const CUresult result = cuMemHostUnregister(session->shm.pinned_base);
        if (DetailedPerfRequested()) {
            std::cout << "[vgpu] op=UnpinShm sid=" << session->session_id
                      << " bytes=" << session->shm.pinned_size
                      << " cuda_error=" << RuntimeError(result)
                      << std::endl;
        }
        session->shm.pinned_base = nullptr;
        session->shm.pinned_size = 0;
        session->shm.pinned = false;
    }

    void StopRingWorker(const std::shared_ptr<SessionState> &session) {
        if (!session || !session->ring_enabled) {
            return;
        }
        session->ring_stop.store(true);
        if (session->shm.base) {
            auto *header = vgpu_shm::Header(session->shm.base);
            vgpu_shm::StoreRelease(&header->stop, 1);
        }
        if (session->ring_thread.joinable()) {
            session->ring_thread.join();
        }
        session->ring_enabled = false;
    }

    void StorePendingError(const std::shared_ptr<SessionState> &session, int cuda_error) {
        if (!session || cuda_error == cudaSuccess) {
            return;
        }
        int expected = cudaSuccess;
        session->pending_error.compare_exchange_strong(expected, cuda_error);
    }

    int TakePendingError(const std::shared_ptr<SessionState> &session) {
        if (!session) {
            return cudaSuccess;
        }
        return session->pending_error.exchange(cudaSuccess);
    }

    CUresult DoDeviceSynchronize(const std::shared_ptr<SessionState> &session) {
        (void)session;
        CUresult result = EnsureCudaLocked();
        if (result != CUDA_SUCCESS) {
            return result;
        }
        result = cuCtxSynchronize();
        if (result != CUDA_SUCCESS) {
            return result;
        }
        const int pending = TakePendingError(session);
        return pending == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }

    CUresult DoStreamSynchronize(const std::shared_ptr<SessionState> &session, uint64_t stream_id) {
        CUstream stream = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (!ResolveStreamLocked(*session, stream_id, &stream)) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
        }
        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuStreamSynchronize(stream);
        }
        if (result != CUDA_SUCCESS) {
            return result;
        }
        const int pending = TakePendingError(session);
        return pending == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }

    CUresult DoEventSynchronize(const std::shared_ptr<SessionState> &session, uint64_t event_id) {
        CUevent event = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (!ResolveEventLocked(*session, event_id, &event)) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
        }
        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuEventSynchronize(event);
        }
        return result;
    }

    CUresult DoEventQuery(const std::shared_ptr<SessionState> &session, uint64_t event_id) {
        CUevent event = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (!ResolveEventLocked(*session, event_id, &event)) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
        }
        CUresult result = EnsureCudaLocked();
        if (result == CUDA_SUCCESS) {
            result = cuEventQuery(event);
        }
        return result;
    }

    CUresult DoLaunchKernelRing(
        const std::shared_ptr<SessionState> &session,
        const vgpu_shm::RingEntry &entry) {
        if (entry.arg_count > vgpu_shm::kMaxKernelArgs ||
            entry.arg_bytes > vgpu_shm::kMaxKernelArgBytes ||
            entry.kernel_name[0] == '\0') {
            return CUDA_ERROR_INVALID_VALUE;
        }

        CUmodule module = nullptr;
        CUstream stream = nullptr;
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            auto module_it = session->modules.find(entry.module_id);
            if (module_it == session->modules.end() ||
                !ResolveStreamLocked(*session, entry.stream_id, &stream)) {
                return CUDA_ERROR_INVALID_HANDLE;
            }
            module = module_it->second;
        }

        CUresult result = EnsureCudaLocked();
        if (result != CUDA_SUCCESS) {
            return result;
        }

        CUfunction function = nullptr;
        result = cuModuleGetFunction(&function, module, entry.kernel_name);
        if (result != CUDA_SUCCESS) {
            return result;
        }

        std::vector<std::vector<unsigned char>> arg_storage;
        std::vector<void *> kernel_params;
        arg_storage.reserve(entry.arg_count);
        kernel_params.reserve(entry.arg_count);
        for (uint32_t i = 0; i < entry.arg_count; ++i) {
            const uint32_t offset = entry.arg_offsets[i];
            const uint32_t size = entry.arg_sizes[i];
            if (size == 0 || offset > entry.arg_bytes || size > entry.arg_bytes - offset) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            arg_storage.emplace_back(entry.arg_data + offset, entry.arg_data + offset + size);
            std::vector<unsigned char> &data = arg_storage.back();
            if (data.size() == sizeof(uint64_t)) {
                uint64_t maybe_virtual_ptr = 0;
                std::memcpy(&maybe_virtual_ptr, data.data(), sizeof(maybe_virtual_ptr));
                std::lock_guard<std::mutex> session_lock(session->mu);
                uint64_t ptr_offset = 0;
                const Allocation *allocation = FindAllocationLocked(*session, maybe_virtual_ptr, 0, &ptr_offset);
                if (allocation) {
                    CUdeviceptr real_ptr = allocation->device_ptr + ptr_offset;
                    std::memcpy(data.data(), &real_ptr, sizeof(real_ptr));
                }
            }
            kernel_params.push_back(data.data());
        }

        return cuLaunchKernel(
            function,
            entry.grid_dim[0],
            entry.grid_dim[1],
            entry.grid_dim[2],
            entry.block_dim[0],
            entry.block_dim[1],
            entry.block_dim[2],
            static_cast<unsigned int>(entry.shared_mem),
            stream,
            kernel_params.data(),
            nullptr);
    }

    CUresult DoMemcpyD2DRingFast(
        const std::shared_ptr<SessionState> &session,
        vgpu_shm::RingEntry *entry) {
        entry->t_context_start_ns = NowNs();
        CUresult result = EnsureCudaLocked();
        entry->t_context_end_ns = NowNs();
        if (result != CUDA_SUCCESS) {
            return result;
        }

        CUdeviceptr dst_ptr = 0;
        CUdeviceptr src_ptr = 0;
        CUstream stream = nullptr;
        uint64_t dst_offset = 0;
        uint64_t src_offset = 0;
        entry->t_lookup_start_ns = NowNs();
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            const Allocation *dst = FindAllocationLocked(*session, entry->dst_device_ptr, entry->count, &dst_offset);
            const Allocation *src = FindAllocationLocked(*session, entry->src_device_ptr, entry->count, &src_offset);
            if (!dst || !src || !ResolveStreamLocked(*session, entry->stream_id, &stream)) {
                entry->t_lookup_end_ns = NowNs();
                return CUDA_ERROR_INVALID_VALUE;
            }
            dst_ptr = dst->device_ptr + dst_offset;
            src_ptr = src->device_ptr + src_offset;
        }
        entry->t_lookup_end_ns = NowNs();
        entry->t_rebuild_start_ns = NowNs();
        entry->t_rebuild_end_ns = entry->t_rebuild_start_ns;

        entry->t_driver_start_ns = NowNs();
        result = cuMemcpyDtoDAsync(dst_ptr, src_ptr, static_cast<size_t>(entry->count), stream);
        entry->t_driver_end_ns = NowNs();
        if (result == CUDA_SUCCESS && entry->async == 0) {
            result = cuStreamSynchronize(stream);
        }
        return result;
    }

    CUresult DoLaunchKernelRingFast(
        const std::shared_ptr<SessionState> &session,
        vgpu_shm::RingEntry *entry) {
        if (entry->arg_count > vgpu_shm::kMaxKernelArgs ||
            entry->arg_bytes > vgpu_shm::kMaxKernelArgBytes ||
            entry->kernel_name[0] == '\0') {
            return CUDA_ERROR_INVALID_VALUE;
        }

        entry->t_context_start_ns = NowNs();
        CUresult result = EnsureCudaLocked();
        entry->t_context_end_ns = NowNs();
        if (result != CUDA_SUCCESS) {
            return result;
        }

        CUmodule module = nullptr;
        CUstream stream = nullptr;
        CUfunction function = nullptr;
        entry->t_lookup_start_ns = NowNs();
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            auto module_it = session->modules.find(entry->module_id);
            if (module_it == session->modules.end() ||
                !ResolveStreamLocked(*session, entry->stream_id, &stream)) {
                entry->t_lookup_end_ns = NowNs();
                return CUDA_ERROR_INVALID_HANDLE;
            }
            module = module_it->second;
        }
        result = cuModuleGetFunction(&function, module, entry->kernel_name);
        if (result != CUDA_SUCCESS) {
            entry->t_lookup_end_ns = NowNs();
            return result;
        }
        entry->t_lookup_end_ns = NowNs();

        unsigned char arg_storage[vgpu_shm::kMaxKernelArgBytes] = {};
        void *kernel_params[vgpu_shm::kMaxKernelArgs] = {};
        entry->t_rebuild_start_ns = NowNs();
        std::memcpy(arg_storage, entry->arg_data, entry->arg_bytes);
        for (uint32_t i = 0; i < entry->arg_count; ++i) {
            const uint32_t offset = entry->arg_offsets[i];
            const uint32_t size = entry->arg_sizes[i];
            if (size == 0 || offset > entry->arg_bytes || size > entry->arg_bytes - offset) {
                entry->t_rebuild_end_ns = NowNs();
                return CUDA_ERROR_INVALID_VALUE;
            }
            unsigned char *data = arg_storage + offset;
            if (size == sizeof(uint64_t)) {
                uint64_t maybe_virtual_ptr = 0;
                std::memcpy(&maybe_virtual_ptr, data, sizeof(maybe_virtual_ptr));
                std::lock_guard<std::mutex> session_lock(session->mu);
                uint64_t ptr_offset = 0;
                const Allocation *allocation = FindAllocationLocked(*session, maybe_virtual_ptr, 0, &ptr_offset);
                if (allocation) {
                    CUdeviceptr real_ptr = allocation->device_ptr + ptr_offset;
                    std::memcpy(data, &real_ptr, sizeof(real_ptr));
                }
            }
            kernel_params[i] = data;
        }
        entry->t_rebuild_end_ns = NowNs();

        entry->t_driver_start_ns = NowNs();
        result = cuLaunchKernel(
            function,
            entry->grid_dim[0],
            entry->grid_dim[1],
            entry->grid_dim[2],
            entry->block_dim[0],
            entry->block_dim[1],
            entry->block_dim[2],
            static_cast<unsigned int>(entry->shared_mem),
            stream,
            kernel_params,
            nullptr);
        entry->t_driver_end_ns = NowNs();
        return result;
    }

    CUresult DoStreamSynchronizeRingFast(
        const std::shared_ptr<SessionState> &session,
        vgpu_shm::RingEntry *entry) {
        entry->t_context_start_ns = NowNs();
        CUresult result = EnsureCudaLocked();
        entry->t_context_end_ns = NowNs();
        if (result != CUDA_SUCCESS) {
            return result;
        }

        CUstream stream = nullptr;
        entry->t_lookup_start_ns = NowNs();
        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (!ResolveStreamLocked(*session, entry->stream_id, &stream)) {
                entry->t_lookup_end_ns = NowNs();
                return CUDA_ERROR_INVALID_HANDLE;
            }
        }
        entry->t_lookup_end_ns = NowNs();
        entry->t_rebuild_start_ns = NowNs();
        entry->t_rebuild_end_ns = entry->t_rebuild_start_ns;

        entry->t_driver_start_ns = NowNs();
        result = cuStreamSynchronize(stream);
        entry->t_driver_end_ns = NowNs();
        if (result != CUDA_SUCCESS) {
            return result;
        }
        const int pending = TakePendingError(session);
        return pending == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }

    void TryBindRingWorker(uint64_t session_id) {
#if defined(__linux__)
        const unsigned int cpus = std::thread::hardware_concurrency();
        if (cpus <= 1) {
            return;
        }
        const uint64_t base = ReadUint64Env("VGPU_RING_CPU_BASE", 1);
        const int cpu = static_cast<int>((base + session_id - 1) % cpus);
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        if (DetailedPerfRequested()) {
            std::cout << "[vgpu] op=BindRingWorker sid=" << session_id
                      << " cpu=" << cpu
                      << " result=" << rc << std::endl;
        }
#else
        (void)session_id;
#endif
    }

    void RingWorkerLoop(std::shared_ptr<SessionState> session) {
        if (!session || !session->shm.base) {
            return;
        }
        TryBindRingWorker(session->session_id);
        auto *header = vgpu_shm::Header(session->shm.base);
        auto *entries = vgpu_shm::Entries(session->shm.base);
        if (!vgpu_shm::HeaderLooksReady(header)) {
            return;
        }

        uint32_t idle_iters = 0;
        while (!session->ring_stop.load()) {
            uint64_t tail = vgpu_shm::LoadAcquire(&header->tail);
            const uint64_t head = vgpu_shm::LoadAcquire(&header->head);
            if (tail == head) {
                if (vgpu_shm::LoadAcquire(&header->stop) != 0) {
                    break;
                }
                ++idle_iters;
                if (idle_iters < 1000) {
                    continue;
                }
                if (idle_iters < 5000) {
                    std::this_thread::yield();
                    continue;
                }
                if (idle_iters < 10000) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                    continue;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(20));
                continue;
            }
            idle_iters = 0;

            while (tail < head && !session->ring_stop.load()) {
                const auto op_start = Clock::now();
                vgpu_shm::RingEntry *entry = &entries[tail % vgpu_shm::kRingCapacity];
                entry->t_server_dequeue_ns = NowNs();
                int cuda_error = kCudaErrorInvalidValue;
                if (entry->op == vgpu_shm::kRingOpMemcpyShm) {
                    TouchSession(session);
                    vgpu::MemcpyShmRequest request;
                    request.set_session_id(session->session_id);
                    request.set_dst_device_ptr(entry->dst_device_ptr);
                    request.set_src_device_ptr(entry->src_device_ptr);
                    request.set_count(entry->count);
                    request.set_kind(entry->kind);
                    request.set_shm_offset(entry->shm_offset);
                    request.set_stream_id(entry->stream_id);
                    request.set_async(entry->async != 0);

                    entry->t_driver_start_ns = NowNs();
                    CUresult result = EnsureCudaLocked();
                    if (result == CUDA_SUCCESS) {
                        result = DoMemcpyShm(session, request);
                    }
                    entry->t_driver_end_ns = NowNs();
                    cuda_error = RuntimeError(result);
                    LogPerf(
                        request.async() ? "MemcpyShmRingAsync" : "MemcpyShmRing",
                        session->session_id,
                        request.count(),
                        request.stream_id(),
                        cuda_error,
                        op_start);
                } else if (entry->op == vgpu_shm::kRingOpMemcpyD2D) {
                    TouchSession(session);
                    CUresult result = DoMemcpyD2DRingFast(session, entry);
                    cuda_error = RuntimeError(result);
                    LogPerf(
                        entry->async ? "MemcpyD2DRingAsync" : "MemcpyD2DRing",
                        session->session_id,
                        entry->count,
                        entry->stream_id,
                        cuda_error,
                        op_start);
                } else if (entry->op == vgpu_shm::kRingOpDeviceSynchronize) {
                    TouchSession(session);
                    entry->t_driver_start_ns = NowNs();
                    cuda_error = RuntimeError(DoDeviceSynchronize(session));
                    entry->t_driver_end_ns = NowNs();
                    LogPerf("DeviceSynchronizeRing", session->session_id, 0, 0, cuda_error, op_start);
                } else if (entry->op == vgpu_shm::kRingOpStreamSynchronize) {
                    TouchSession(session);
                    cuda_error = RuntimeError(DoStreamSynchronizeRingFast(session, entry));
                    LogPerf("StreamSynchronizeRing", session->session_id, 0, entry->stream_id, cuda_error, op_start);
                } else if (entry->op == vgpu_shm::kRingOpEventSynchronize) {
                    TouchSession(session);
                    entry->t_driver_start_ns = NowNs();
                    cuda_error = RuntimeError(DoEventSynchronize(session, entry->event_id));
                    entry->t_driver_end_ns = NowNs();
                    LogPerf("EventSynchronizeRing", session->session_id, 0, 0, cuda_error, op_start);
                } else if (entry->op == vgpu_shm::kRingOpEventQuery) {
                    TouchSession(session);
                    entry->t_driver_start_ns = NowNs();
                    cuda_error = RuntimeError(DoEventQuery(session, entry->event_id));
                    entry->t_driver_end_ns = NowNs();
                    LogPerf("EventQueryRing", session->session_id, 0, 0, cuda_error, op_start);
                } else if (entry->op == vgpu_shm::kRingOpLaunchKernel) {
                    TouchSession(session);
                    cuda_error = RuntimeError(DoLaunchKernelRingFast(session, entry));
                    LogPerf("LaunchKernelRing", session->session_id, 0, entry->stream_id, cuda_error, op_start);
                } else {
                    entry->t_driver_start_ns = NowNs();
                    entry->t_driver_end_ns = entry->t_driver_start_ns;
                }

                entry->cuda_error = cuda_error;
                if ((entry->op == vgpu_shm::kRingOpMemcpyD2D && entry->async != 0) ||
                    entry->op == vgpu_shm::kRingOpLaunchKernel) {
                    StorePendingError(session, cuda_error);
                }
                entry->t_server_complete_ns = NowNs();
                vgpu_shm::StoreRelease(&entry->done, 1);
                ++tail;
                vgpu_shm::StoreRelease(&header->tail, tail);
            }
        }
    }

    void CleanupSessionResources(const std::shared_ptr<SessionState> &session, const char *reason) {
        if (!session) {
            return;
        }

        StopRingWorker(session);

        const uint64_t sid = session->session_id;
        CUresult init_result = EnsureCudaLocked();
        if (init_result != CUDA_SUCCESS) {
            std::cerr << "[vgpu] op=CleanupSession sid=" << sid
                      << " reason=" << reason
                      << " cuda_init_error=" << DriverErrorString(init_result) << std::endl;
            return;
        }

        std::lock_guard<std::mutex> session_lock(session->mu);
        if (session->closing && session->default_stream == nullptr &&
            session->allocations.empty() && session->modules.empty() &&
            session->streams.empty() && session->events.empty()) {
            return;
        }
        session->closing = true;

        if (session->default_stream) {
            cuStreamSynchronize(session->default_stream);
        }
        for (const auto &entry : session->streams) {
            cuStreamSynchronize(entry.second);
        }
        for (const auto &entry : session->streams) {
            cuStreamDestroy(entry.second);
        }
        session->streams.clear();

        for (const auto &entry : session->events) {
            cuEventDestroy(entry.second);
        }
        session->events.clear();

        if (session->default_stream) {
            cuStreamDestroy(session->default_stream);
            session->default_stream = nullptr;
        }

        for (const auto &entry : session->allocations) {
            cuMemFree(entry.second.device_ptr);
        }
        session->allocations.clear();
        session->memory_used = 0;

        for (const auto &entry : session->modules) {
            cuModuleUnload(entry.second);
        }
        session->modules.clear();

        UnpinShmArenaLocked(session);

        if (session->shm.base) {
            munmap(session->shm.base, session->shm.size);
            session->shm.base = nullptr;
            session->shm.size = 0;
        }
        if (session->shm.fd >= 0) {
            close(session->shm.fd);
            session->shm.fd = -1;
        }
        session->shm.name.clear();

        if (DetailedPerfRequested()) {
            std::cout << "[vgpu] op=CleanupSession sid=" << sid
                      << " reason=" << reason << std::endl;
        }
    }

    void ReaperLoop() {
        while (!stop_reaper_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            std::vector<std::shared_ptr<SessionState>> expired;
            const auto now = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(mu_);
                for (auto it = sessions_.begin(); it != sessions_.end();) {
                    std::shared_ptr<SessionState> session = it->second;
                    bool should_expire = false;
                    {
                        std::lock_guard<std::mutex> session_lock(session->mu);
                        const auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - session->last_seen).count();
                        should_expire = !session->closing &&
                            idle_ms >= static_cast<int64_t>(session_timeout_ms_);
                    }
                    if (should_expire) {
                        expired.push_back(session);
                        it = sessions_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            for (const auto &session : expired) {
                CleanupSessionResources(session, "timeout");
            }
        }
    }

    bool HasSession(uint64_t session_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return false;
        }
        TouchSession(it->second);
        return true;
    }

    CUresult EnsureCudaLocked() {
        std::lock_guard<std::mutex> cuda_lock(cuda_mu_);
        if (cuda_initialized_) {
            return cuCtxSetCurrent(context_);
        }

        CUresult result = cuInit(0);
        if (result != CUDA_SUCCESS) {
            return result;
        }

        int count = 0;
        result = cuDeviceGetCount(&count);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        if (count <= 0) {
            return CUDA_ERROR_INVALID_DEVICE;
        }

        result = cuDeviceGet(&device_, 0);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        result = cuDevicePrimaryCtxRetain(&context_, device_);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        result = cuCtxSetCurrent(context_);
        if (result != CUDA_SUCCESS) {
            return result;
        }

        cuda_initialized_ = true;
        return CUDA_SUCCESS;
    }

    uint64_t NextVirtualPtrLocked(SessionState &session, size_t size) {
        constexpr uint64_t kVirtualPtrAlignment = 0x1000ull;
        const uint64_t aligned_size =
            (static_cast<uint64_t>(size) + kVirtualPtrAlignment - 1) & ~(kVirtualPtrAlignment - 1);
        const uint64_t span = aligned_size + kVirtualPtrAlignment;
        const uint64_t virtual_ptr = session.next_virtual_ptr;
        session.next_virtual_ptr += span;
        return virtual_ptr;
    }

    uint64_t NextVirtualStreamLocked() {
        const uint64_t id = next_virtual_stream_id_.fetch_add(1);
        return kVirtualStreamBase + id * kVirtualStreamStride;
    }

    uint64_t NextVirtualEventLocked() {
        const uint64_t id = next_virtual_event_id_.fetch_add(1);
        return kVirtualEventBase + id * kVirtualEventStride;
    }

    bool ResolveStreamLocked(SessionState &session, uint64_t stream_id, CUstream *stream) {
        if (stream_id == 0) {
            *stream = session.default_stream;
            return *stream != nullptr;
        }
        auto it = session.streams.find(stream_id);
        if (it == session.streams.end()) {
            return false;
        }
        *stream = it->second;
        return true;
    }

    bool ResolveEventLocked(SessionState &session, uint64_t event_id, CUevent *event) {
        auto it = session.events.find(event_id);
        if (it == session.events.end()) {
            return false;
        }
        *event = it->second;
        return true;
    }

    grpc::Status ResolveEventForRequest(
        uint64_t session_id,
        uint64_t event_id,
        CUevent *event,
        vgpu::StatusReply *reply) {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = sessions_.find(session_id);
            if (it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            reply->set_cuda_error(kCudaErrorInvalidValue);
            reply->set_message("session not found");
            return grpc::Status::OK;
        }

        {
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (!ResolveEventLocked(*session, event_id, event)) {
                reply->set_cuda_error(cudaErrorInvalidResourceHandle);
                reply->set_message("event not found");
                return grpc::Status::OK;
            }
        }

        reply->set_cuda_error(kCudaSuccess);
        reply->set_message("event resolved");
        return grpc::Status::OK;
    }

    const Allocation *FindAllocationLocked(
        SessionState &session,
        uint64_t virtual_ptr,
        size_t count,
        uint64_t *offset_out = nullptr) {
        for (const auto &entry : session.allocations) {
            const uint64_t base = entry.first;
            const Allocation &allocation = entry.second;
            if (virtual_ptr < base) {
                continue;
            }
            const uint64_t offset = virtual_ptr - base;
            if (offset <= allocation.size && count <= allocation.size - offset) {
                if (offset_out) {
                    *offset_out = offset;
                }
                return &allocation;
            }
        }
        return nullptr;
    }

    CUresult DoMemcpy(
        const std::shared_ptr<SessionState> &session,
        const vgpu::MemcpyRequest &request,
        vgpu::MemcpyReply *reply,
        bool async_request) {
        const size_t count = static_cast<size_t>(request.count());
        CUstream stream = nullptr;

        switch (request.kind()) {
            case cudaMemcpyHostToDevice: {
                if (request.host_data().size() != count) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                CUdeviceptr dst_ptr = 0;
                uint64_t dst_offset = 0;
                {
                    std::lock_guard<std::mutex> session_lock(session->mu);
                    const Allocation *dst = FindAllocationLocked(*session, request.dst_device_ptr(), count, &dst_offset);
                    if (!dst || !ResolveStreamLocked(*session, request.stream_id(), &stream)) {
                        return CUDA_ERROR_INVALID_VALUE;
                    }
                    dst_ptr = dst->device_ptr + dst_offset;
                }
                CUresult result = cuMemcpyHtoDAsync(dst_ptr, request.host_data().data(), count, stream);
                if (result == CUDA_SUCCESS && !async_request) {
                    result = cuStreamSynchronize(stream);
                }
                return result;
            }
            case cudaMemcpyDeviceToHost: {
                CUdeviceptr src_ptr = 0;
                uint64_t src_offset = 0;
                {
                    std::lock_guard<std::mutex> session_lock(session->mu);
                    const Allocation *src = FindAllocationLocked(*session, request.src_device_ptr(), count, &src_offset);
                    if (!src || !ResolveStreamLocked(*session, request.stream_id(), &stream)) {
                        return CUDA_ERROR_INVALID_VALUE;
                    }
                    src_ptr = src->device_ptr + src_offset;
                }
                std::string data;
                data.resize(count);
                CUresult result = cuMemcpyDtoHAsync(data.data(), src_ptr, count, stream);
                if (result == CUDA_SUCCESS) {
                    result = cuStreamSynchronize(stream);
                }
                if (result == CUDA_SUCCESS) {
                    reply->set_host_data(std::move(data));
                }
                return result;
            }
            case cudaMemcpyDeviceToDevice: {
                CUdeviceptr dst_ptr = 0;
                CUdeviceptr src_ptr = 0;
                uint64_t dst_offset = 0;
                uint64_t src_offset = 0;
                {
                    std::lock_guard<std::mutex> session_lock(session->mu);
                    const Allocation *dst = FindAllocationLocked(*session, request.dst_device_ptr(), count, &dst_offset);
                    const Allocation *src = FindAllocationLocked(*session, request.src_device_ptr(), count, &src_offset);
                    if (!dst || !src || !ResolveStreamLocked(*session, request.stream_id(), &stream)) {
                        return CUDA_ERROR_INVALID_VALUE;
                    }
                    dst_ptr = dst->device_ptr + dst_offset;
                    src_ptr = src->device_ptr + src_offset;
                }
                CUresult result = cuMemcpyDtoDAsync(dst_ptr, src_ptr, count, stream);
                if (result == CUDA_SUCCESS && !async_request) {
                    result = cuStreamSynchronize(stream);
                }
                return result;
            }
            default:
                return CUDA_ERROR_INVALID_VALUE;
        }
    }

    CUresult DoMemcpyShm(
        const std::shared_ptr<SessionState> &session,
        const vgpu::MemcpyShmRequest &request) {
        const auto detail_start = Clock::now();
        uint64_t resolve_us = 0;
        uint64_t allocation_lookup_us = 0;
        uint64_t cuda_submit_us = 0;
        uint64_t cuda_sync_us = 0;
        const size_t count = static_cast<size_t>(request.count());
        const size_t shm_offset = static_cast<size_t>(request.shm_offset());
        CUstream stream = nullptr;
        void *host_ptr = nullptr;

        {
            const auto resolve_start = Clock::now();
            std::lock_guard<std::mutex> session_lock(session->mu);
            if (request.kind() != cudaMemcpyDeviceToDevice) {
                const size_t min_offset = session->ring_enabled ? vgpu_shm::kControlBytes : 0;
                if (!session->shm.base || shm_offset < min_offset ||
                    !RangeFits(shm_offset, count, session->shm.size)) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                host_ptr = static_cast<char *>(session->shm.base) + shm_offset;
            }
            if (!ResolveStreamLocked(*session, request.stream_id(), &stream)) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            resolve_us = ElapsedUs(resolve_start);
        }

        auto log_detail = [&](CUresult result) {
            if (!DetailedPerfRequested()) {
                return;
            }
            std::cout << "[vgpu_perf_detail] op=DoMemcpyShm"
                      << " sid=" << session->session_id
                      << " kind=" << request.kind()
                      << " async=" << request.async()
                      << " bytes=" << count
                      << " stream=" << request.stream_id()
                      << " pinned=" << static_cast<int>(session->shm.pinned)
                      << " resolve_us=" << resolve_us
                      << " allocation_lookup_us=" << allocation_lookup_us
                      << " cuda_submit_us=" << cuda_submit_us
                      << " cuda_sync_us=" << cuda_sync_us
                      << " total_us=" << ElapsedUs(detail_start)
                      << " cuda_error=" << RuntimeError(result)
                      << std::endl;
        };

        switch (request.kind()) {
            case cudaMemcpyHostToDevice: {
                CUdeviceptr dst_ptr = 0;
                uint64_t dst_offset = 0;
                {
                    const auto lookup_start = Clock::now();
                    std::lock_guard<std::mutex> session_lock(session->mu);
                    const Allocation *dst = FindAllocationLocked(*session, request.dst_device_ptr(), count, &dst_offset);
                    if (!dst) {
                        return CUDA_ERROR_INVALID_VALUE;
                    }
                    dst_ptr = dst->device_ptr + dst_offset;
                    allocation_lookup_us = ElapsedUs(lookup_start);
                }
                const auto submit_start = Clock::now();
                CUresult result = cuMemcpyHtoDAsync(dst_ptr, host_ptr, count, stream);
                cuda_submit_us = ElapsedUs(submit_start);
                if (result == CUDA_SUCCESS && !request.async()) {
                    const auto sync_start = Clock::now();
                    result = cuStreamSynchronize(stream);
                    cuda_sync_us = ElapsedUs(sync_start);
                }
                log_detail(result);
                return result;
            }
            case cudaMemcpyDeviceToHost: {
                CUdeviceptr src_ptr = 0;
                uint64_t src_offset = 0;
                {
                    const auto lookup_start = Clock::now();
                    std::lock_guard<std::mutex> session_lock(session->mu);
                    const Allocation *src = FindAllocationLocked(*session, request.src_device_ptr(), count, &src_offset);
                    if (!src) {
                        return CUDA_ERROR_INVALID_VALUE;
                    }
                    src_ptr = src->device_ptr + src_offset;
                    allocation_lookup_us = ElapsedUs(lookup_start);
                }
                const auto submit_start = Clock::now();
                CUresult result = cuMemcpyDtoHAsync(host_ptr, src_ptr, count, stream);
                cuda_submit_us = ElapsedUs(submit_start);
                if (result == CUDA_SUCCESS && !request.async()) {
                    const auto sync_start = Clock::now();
                    result = cuStreamSynchronize(stream);
                    cuda_sync_us = ElapsedUs(sync_start);
                }
                log_detail(result);
                return result;
            }
            case cudaMemcpyDeviceToDevice: {
                CUdeviceptr dst_ptr = 0;
                CUdeviceptr src_ptr = 0;
                uint64_t dst_offset = 0;
                uint64_t src_offset = 0;
                {
                    const auto lookup_start = Clock::now();
                    std::lock_guard<std::mutex> session_lock(session->mu);
                    const Allocation *dst = FindAllocationLocked(*session, request.dst_device_ptr(), count, &dst_offset);
                    const Allocation *src = FindAllocationLocked(*session, request.src_device_ptr(), count, &src_offset);
                    if (!dst || !src) {
                        return CUDA_ERROR_INVALID_VALUE;
                    }
                    dst_ptr = dst->device_ptr + dst_offset;
                    src_ptr = src->device_ptr + src_offset;
                    allocation_lookup_us = ElapsedUs(lookup_start);
                }
                const auto submit_start = Clock::now();
                CUresult result = cuMemcpyDtoDAsync(dst_ptr, src_ptr, count, stream);
                cuda_submit_us = ElapsedUs(submit_start);
                if (result == CUDA_SUCCESS && !request.async()) {
                    const auto sync_start = Clock::now();
                    result = cuStreamSynchronize(stream);
                    cuda_sync_us = ElapsedUs(sync_start);
                }
                log_detail(result);
                return result;
            }
            default:
                return CUDA_ERROR_INVALID_VALUE;
        }
    }

    std::atomic<uint64_t> next_session_id_{1};
    std::atomic<uint64_t> next_virtual_stream_id_{1};
    std::atomic<uint64_t> next_virtual_event_id_{1};
    std::atomic<uint64_t> next_module_id_{1};
    std::mutex mu_;
    std::mutex cuda_mu_;
    std::unordered_map<uint64_t, std::shared_ptr<SessionState>> sessions_;
    std::atomic<bool> stop_reaper_{false};
    std::thread reaper_thread_;
    uint64_t session_timeout_ms_ = kDefaultSessionTimeoutMs;
    bool cuda_initialized_ = false;
    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
};

}  // namespace

int main(int argc, char **argv) {
    std::string listen_addr = "0.0.0.0:50051";
    if (argc > 1) {
        listen_addr = argv[1];
    }

    VgpuRuntimeService service;
    grpc::ServerBuilder builder;
    builder.SetMaxReceiveMessageSize(-1);
    builder.SetMaxSendMessageSize(-1);
    builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    if (!server) {
        std::cerr << "failed to start vgpu_server at " << listen_addr << std::endl;
        return 1;
    }

    std::cout << "vgpu_server listening on " << listen_addr << std::endl;
    server->Wait();
    return 0;
}
