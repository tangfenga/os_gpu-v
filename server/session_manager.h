#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vgpu_server {

constexpr uint64_t kVirtualPtrBase = 0xCADA00000000ull;
constexpr uint64_t kVirtualPtrStride = 0x1000ull;

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

class SessionManager {
public:
    uint64_t NextSessionId() {
        return next_session_id_.fetch_add(1);
    }

    void Add(const std::shared_ptr<SessionState> &session) {
        std::lock_guard<std::mutex> lock(mu_);
        sessions_.emplace(session->session_id, session);
    }

    std::shared_ptr<SessionState> Find(uint64_t session_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        return it == sessions_.end() ? nullptr : it->second;
    }

    std::shared_ptr<SessionState> Remove(uint64_t session_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            return nullptr;
        }
        auto session = it->second;
        sessions_.erase(it);
        return session;
    }

    std::vector<std::shared_ptr<SessionState>> RemoveExpired(
        std::chrono::steady_clock::time_point now,
        uint64_t timeout_ms) {
        std::vector<std::shared_ptr<SessionState>> expired;
        std::lock_guard<std::mutex> lock(mu_);
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            std::shared_ptr<SessionState> session = it->second;
            bool should_expire = false;
            {
                std::lock_guard<std::mutex> session_lock(session->mu);
                const auto idle_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - session->last_seen).count();
                should_expire = !session->closing && idle_ms >= static_cast<int64_t>(timeout_ms);
            }
            if (should_expire) {
                expired.push_back(session);
                it = sessions_.erase(it);
            } else {
                ++it;
            }
        }
        return expired;
    }

private:
    std::mutex mu_;
    std::atomic<uint64_t> next_session_id_{1};
    std::unordered_map<uint64_t, std::shared_ptr<SessionState>> sessions_;
};

inline void TouchSession(const std::shared_ptr<SessionState> &session) {
    if (!session) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    session->last_seen = std::chrono::steady_clock::now();
}

inline uint64_t NextVirtualPtrLocked(SessionState &session, size_t size) {
    constexpr uint64_t kVirtualPtrAlignment = 0x1000ull;
    const uint64_t aligned_size =
        (static_cast<uint64_t>(size) + kVirtualPtrAlignment - 1) & ~(kVirtualPtrAlignment - 1);
    const uint64_t span = aligned_size + kVirtualPtrAlignment;
    const uint64_t virtual_ptr = session.next_virtual_ptr;
    session.next_virtual_ptr += span;
    return virtual_ptr;
}

inline bool ResolveStreamLocked(SessionState &session, uint64_t stream_id, CUstream *stream) {
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

inline bool ResolveEventLocked(SessionState &session, uint64_t event_id, CUevent *event) {
    auto it = session.events.find(event_id);
    if (it == session.events.end()) {
        return false;
    }
    *event = it->second;
    return true;
}

inline const Allocation *FindAllocationLocked(
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

}  // namespace vgpu_server

