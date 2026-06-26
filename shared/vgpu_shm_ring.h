#pragma once

#include <cstddef>
#include <cstdint>

namespace vgpu_shm {

constexpr uint64_t kRingMagic = 0x5647505553484d52ull;  // "VGPUSHMR"
constexpr uint64_t kRingVersion = 1;
constexpr uint64_t kRingOpMemcpyShm = 1;
constexpr uint64_t kRingOpMemcpyD2D = 2;
constexpr uint64_t kRingOpDeviceSynchronize = 3;
constexpr uint64_t kRingOpStreamSynchronize = 4;
constexpr uint64_t kRingOpEventSynchronize = 5;
constexpr uint64_t kRingOpEventQuery = 6;
constexpr uint64_t kRingOpLaunchKernel = 7;
constexpr uint64_t kTraceDefault = 0;
constexpr uint64_t kTraceEmptyStreamSync = 1;
constexpr uint64_t kTraceD2DSync = 2;
constexpr uint64_t kTraceKernelSync = 3;
constexpr uint64_t kTraceBatchDrainSync = 4;
constexpr size_t kRingCapacity = 1024;
constexpr size_t kControlBytes = 1024ull * 1024ull;
constexpr size_t kMaxKernelName = 128;
constexpr size_t kMaxKernelArgs = 16;
constexpr size_t kMaxKernelArgBytes = 256;

struct alignas(64) RingHeader {
    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t capacity = 0;
    uint64_t entry_size = 0;
    uint64_t head = 0;
    uint64_t tail = 0;
    uint64_t stop = 0;
};

struct alignas(64) RingEntry {
    uint64_t seq = 0;
    uint64_t op = 0;
    uint64_t done = 0;
    int32_t cuda_error = 0;
    int32_t kind = 0;
    uint64_t async = 0;
    uint64_t dst_device_ptr = 0;
    uint64_t src_device_ptr = 0;
    uint64_t count = 0;
    uint64_t shm_offset = 0;
    uint64_t stream_id = 0;
    uint64_t event_id = 0;
    uint64_t module_id = 0;
    uint64_t shared_mem = 0;
    uint64_t trace_tag = 0;
    uint64_t queue_depth_at_submit = 0;
    uint64_t t_client_submit_ns = 0;
    uint64_t t_server_dequeue_ns = 0;
    uint64_t t_context_start_ns = 0;
    uint64_t t_context_end_ns = 0;
    uint64_t t_lookup_start_ns = 0;
    uint64_t t_lookup_end_ns = 0;
    uint64_t t_rebuild_start_ns = 0;
    uint64_t t_rebuild_end_ns = 0;
    uint64_t t_driver_start_ns = 0;
    uint64_t t_driver_end_ns = 0;
    uint64_t t_server_complete_ns = 0;
    uint64_t t_client_done_ns = 0;
    uint32_t grid_dim[3] = {1, 1, 1};
    uint32_t block_dim[3] = {1, 1, 1};
    uint32_t arg_count = 0;
    uint32_t arg_bytes = 0;
    uint32_t arg_offsets[kMaxKernelArgs] = {};
    uint32_t arg_sizes[kMaxKernelArgs] = {};
    char kernel_name[kMaxKernelName] = {};
    unsigned char arg_data[kMaxKernelArgBytes] = {};
};

static_assert(sizeof(RingHeader) + sizeof(RingEntry) * kRingCapacity <= kControlBytes);

inline uint64_t LoadAcquire(const uint64_t *ptr) {
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

inline uint64_t LoadRelaxed(const uint64_t *ptr) {
    return __atomic_load_n(ptr, __ATOMIC_RELAXED);
}

inline void StoreRelease(uint64_t *ptr, uint64_t value) {
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

inline void StoreRelaxed(uint64_t *ptr, uint64_t value) {
    __atomic_store_n(ptr, value, __ATOMIC_RELAXED);
}

inline RingHeader *Header(void *base) {
    return static_cast<RingHeader *>(base);
}

inline const RingHeader *Header(const void *base) {
    return static_cast<const RingHeader *>(base);
}

inline RingEntry *Entries(void *base) {
    return reinterpret_cast<RingEntry *>(static_cast<char *>(base) + sizeof(RingHeader));
}

inline const RingEntry *Entries(const void *base) {
    return reinterpret_cast<const RingEntry *>(static_cast<const char *>(base) + sizeof(RingHeader));
}

inline bool HeaderLooksReady(const RingHeader *header) {
    return header &&
        LoadAcquire(&header->magic) == kRingMagic &&
        LoadAcquire(&header->version) == kRingVersion &&
        LoadAcquire(&header->capacity) == kRingCapacity &&
        LoadAcquire(&header->entry_size) == sizeof(RingEntry);
}

inline void InitRing(void *base) {
    auto *header = Header(base);
    auto *entries = Entries(base);
    for (size_t i = 0; i < kRingCapacity; ++i) {
        entries[i] = RingEntry{};
    }
    StoreRelaxed(&header->version, kRingVersion);
    StoreRelaxed(&header->capacity, kRingCapacity);
    StoreRelaxed(&header->entry_size, sizeof(RingEntry));
    StoreRelaxed(&header->head, 0);
    StoreRelaxed(&header->tail, 0);
    StoreRelaxed(&header->stop, 0);
    StoreRelease(&header->magic, kRingMagic);
}

}  // namespace vgpu_shm
