# 阶段三：POSIX Shared Memory 数据面接入

阶段三目标是在当前 WSL/Linux 同一内核环境中，把 H2D/D2H 大块数据从 gRPC protobuf `bytes` 迁移到 POSIX Shared Memory。控制面继续使用 gRPC；数据面使用 `shm_open` + `mmap`；RPC 只传 shm offset、size、方向、vptr、stream 等元数据。

## 0. 当前实现状态

阶段三最小闭环已经完成；当前代码已经继续推进到阶段四的固定块 data arena。阶段三完成内容包括：

- `proto/vgpu.proto` 已增加 `CreateSessionRequest.shm_name/shm_size`。
- `CreateSessionReply` 已增加 `shm_enabled/shm_data_offset/shm_data_size`。
- 已新增 `MemcpyShm` RPC。
- client 侧 `libcudart_proxy.so` 支持 `VGPU_DATA_PLANE=shm`。
- client 侧默认创建 64 MiB POSIX shm，环境变量 `VGPU_SHM_SIZE` 可调整大小。
- client 侧默认 64 KiB 及以上同步 H2D/D2H 走 shm，环境变量 `VGPU_SHM_THRESHOLD` 可调整阈值。
- server 侧按 session 打开并 `mmap` client 创建的 shm。
- server 侧 H2D 使用 `cuMemcpyHtoDAsync(device, shm_base + offset, size, stream)`。
- server 侧 D2H 使用 `cuMemcpyDtoHAsync(shm_base + offset, device, size, stream)` 并在同步 API 中等待完成。
- D2D 仍走原来的元数据 RPC，不经过 shm。
- 未设置 `VGPU_DATA_PLANE=shm` 或 shm 建立失败时，自动回退到原 gRPC bytes 路径。

已验证：

```text
VGPU_DATA_PLANE=shm memcpy baseline passed: 4194304 bytes H2D/D2D/D2H
VGPU_DATA_PLANE=shm vector_add smoke test passed: 1048576 elements
VGPU_DATA_PLANE=shm matmul baseline passed: 128x128
```

server 日志中可看到：

```text
[vgpu_perf] op=MemcpyShm sid=... bytes=4194304 stream=0 cuda_error=0 elapsed_us=...
```

4 MiB 循环 benchmark 初测：

```text
原生 CUDA H2D_avg_us=428.933
gRPC bytes H2D_avg_us=52989.333
POSIX shm 单 slot H2D_avg_us=3613.967

原生 CUDA D2H_avg_us=569.133
gRPC bytes D2H_avg_us=47893.433
POSIX shm 单 slot D2H_avg_us=4353.633
```

这说明阶段三已经消除了 protobuf `bytes` 承载大数据的主要开销，但单 slot + 同步 RPC + pageable shm 仍不是最终性能形态。

阶段三第一版当时的限制：

- 第一版使用单 slot，`shm_offset=0`。
- 为保证正确性，当前 shm 快路径只用于同步 `cudaMemcpy` 的 H2D/D2H。
- `cudaMemcpyAsync` 暂时保留原路径；真正异步 shm 需要阶段四的 ring buffer、buffer 生命周期和 CUDA event 回收。
- 阶段三当时没有启用 pinned shm 和 pinned host API，后续阶段四继续评估。

当前代码已经在阶段四解决这些限制：单 slot 已升级为固定块 data arena，async H2D/D2H 已接入 shm 生命周期管理，`cudaEventSynchronize` / 成功的 `cudaEventQuery` 已支持按 event 精确回收，`MemcpyShm` descriptor 已接入 per-session SPSC ring，server 侧已默认尝试 pinned shm 注册，`cudaHostAlloc` / `cudaMallocHost` 也已返回 shm-backed host buffer。尚未完成的是低 CPU 唤醒机制、更完整 pinned host API 语义和完整性能报告。

## 1. 设计边界

本阶段不再引入 KVM/QEMU、VSOCK、IVSHMEM。

POSIX shm 的适用范围：

```text
同一个 Linux 内核
├── client CUDA process
│   └── libcudart_proxy.so
└── vgpu_server
    └── CUDA Driver API
```

它能解决当前 WSL 开发版的性能瓶颈：避免大块 host 数据进入 protobuf/gRPC。它不能直接证明跨 VM guest/host 共享内存，因为 QEMU guest 的 `/dev/shm` 和 WSL host 的 `/dev/shm` 不是同一个内核命名空间。

## 2. 当前问题

阶段二的 H2D 路径：

```text
client user buffer
  -> protobuf bytes
  -> gRPC serialization
  -> TCP transport
  -> gRPC deserialization
  -> server buffer
  -> cuMemcpyHtoDAsync
```

阶段二的 D2H 路径：

```text
cuMemcpyDtoHAsync
  -> server buffer
  -> protobuf bytes reply
  -> gRPC serialization
  -> TCP transport
  -> gRPC deserialization
  -> client user buffer
```

4 MiB 测试中，当前 gRPC bytes 版已经比原生 CUDA 慢几十到上百倍。阶段三要先消除 protobuf/gRPC 承载大数据的问题。

## 3. 目标路径

H2D：

```text
client user buffer
  -> memcpy to POSIX shm data arena
  -> RPC: dst_vptr + shm_offset + size + stream_id
  -> server cuMemcpyHtoDAsync(device, shm_base + offset, size, stream)
```

D2H：

```text
RPC: src_vptr + shm_offset + size + stream_id
  -> server cuMemcpyDtoHAsync(shm_base + offset, device, size, stream)
  -> server waits until D2H complete for sync API
  -> RPC returns DONE
  -> client memcpy(shm_base + offset -> user buffer)
```

D2D：

```text
server cuMemcpyDtoDAsync(dst_device, src_device, size, stream)
```

D2D 不经过 POSIX shm。

## 4. shm 生命周期

第一版采用 client 创建 shm：

```text
client first CUDA API
  -> shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600)
  -> ftruncate(size)
  -> mmap
  -> CreateSession RPC 携带 shm_name/shm_size
  -> server shm_open(name, O_RDWR)
  -> server mmap
  -> server 记录 shm_base/shm_size
```

session 销毁：

```text
client DestroySession
  -> server synchronize streams
  -> server munmap
  -> server close fd
  -> client munmap
  -> client close fd
  -> client shm_unlink
```

异常退出：

- client 被 `kill -9` 后无法 `shm_unlink`。
- server timeout reaper 清理 GPU 资源和 server-side mapping。
- 残留 shm 对象由启动清理脚本或 server 定期清理带有 `/vgpu_` 前缀且超时的对象。

## 5. shm 命名

推荐名称：

```text
/vgpu_<uid>_<pid>_<random>
```

原因：

- 避免不同用户冲突。
- 避免 pid 复用带来的同名问题。
- `0600` 权限限制其他用户访问。
- `random` 使用 `getrandom` 或高质量随机数。

不要使用可预测的固定名称，例如 `/vgpu_shm`，否则多进程并发和安全隔离都会出问题。

## 6. 内存布局

每个 session 一块独立 shm：

```text
SessionShm
├── ShmHeader
│   ├── magic = "VGPU_SHM"
│   ├── version
│   ├── total_size
│   ├── session_id
│   ├── ring_offset
│   ├── ring_size
│   ├── arena_offset
│   ├── arena_size
│   └── flags
├── CommandRing
│   ├── atomic head
│   ├── atomic tail
│   └── ShmCommandDesc[N]
└── DataArena
    ├── bump/free-list allocator metadata
    └── payload bytes
```

第一版可以先不用 ring，只用一个临时 data arena slot：

```text
client memcpy host data -> shm offset 0
client 发 MemcpyShm RPC
server 根据 offset/size 拷贝
RPC 返回后 offset 0 可复用
```

这样能最快替换掉 gRPC bytes 大数据路径。ring 和 arena allocator 放到阶段四优化。

## 7. proto 修改

当前已实现的 `CreateSessionRequest` 字段：

```proto
string shm_name = 4;
uint64 shm_size = 5;
```

当前已实现的 `CreateSessionReply` 字段：

```proto
bool shm_enabled = 4;
uint64 shm_data_offset = 5;
uint64 shm_data_size = 6;
```

当前已实现的 `MemcpyShmRequest`：

```proto
message MemcpyShmRequest {
  uint64 session_id = 1;
  uint64 dst_device_ptr = 2;
  uint64 src_device_ptr = 3;
  uint64 count = 4;
  int32 kind = 5;
  uint64 shm_offset = 6;
  uint64 stream_id = 7;
  bool async = 8;
}
```

第一版建议新增 `MemcpyShm`，保留旧 `Memcpy`，便于 A/B 测试：

```text
VGPU_DATA_PLANE=grpc   使用旧路径
VGPU_DATA_PLANE=shm    使用 POSIX shm 路径
```

## 8. client 实现

client 增加：

```text
PosixShmClient
├── Create(size)
├── Map()
├── AllocateTemp(size, alignment)
├── FreeTemp(offset)
├── CopyIn(host_ptr, size) -> offset
└── CopyOut(offset, host_ptr, size)
```

`cudaMemcpy` 选择逻辑：

```text
H2H:
  local memcpy

H2D:
  if shm enabled and count >= threshold:
    offset = shm.CopyIn(src, count)
    MemcpyShm(dst_vptr, offset, count)
  else:
    old gRPC bytes Memcpy

D2H:
  if shm enabled and count >= threshold:
    offset = shm.AllocateTemp(count)
    MemcpyShm(src_vptr, offset, count)
    shm.CopyOut(offset, dst, count)
  else:
    old gRPC bytes Memcpy

D2D:
  MemcpyShm or old RPC only carries vptr metadata; no host bytes
```

推荐阈值：

```text
VGPU_SHM_THRESHOLD=65536
```

小于 64 KiB 的 memcpy 可以继续走 gRPC bytes，避免 shm allocator 和额外本地 memcpy 的固定开销。

## 9. server 实现

server `SessionState` 增加：

```cpp
struct ShmMapping {
    std::string name;
    int fd;
    void* base;
    size_t size;
    bool registered_with_cuda;
};
```

`CreateSession`：

```text
validate shm_name/shm_size
shm_open
mmap
validate header magic/version
optionally cudaHostRegister(data arena)
store mapping in SessionState
```

`MemcpyShm`：

```text
validate session
validate shm mapping exists
validate shm_offset + count <= shm_size
validate vptr belongs to current session
validate allocation bounds
dispatch cuMemcpyHtoDAsync/cuMemcpyDtoHAsync/cuMemcpyDtoDAsync
synchronize when API is synchronous
return cuda_error
```

安全重点：

- server 不能相信 client 传来的 offset/size。
- offset 必须落在 data arena 内，不能覆盖 header/ring。
- count 不能越过 allocation size。
- shm name 必须限制前缀和字符集，避免打开任意对象。

## 10. Pinned Memory 策略

第一版可以先不注册 pinned memory，只验证功能和去掉 protobuf/gRPC 大数据传输。

第二步启用：

```text
server cudaHostRegister(shm_data_arena, arena_size, cudaHostRegisterDefault)
```

注意事项：

- 地址和大小尽量 page 对齐。
- 不要每次 memcpy 都 register/unregister。
- session 生命周期内复用注册结果。
- session 销毁时统一 `cudaHostUnregister`。
- 如果 WSL CUDA 对 `cudaHostRegister` + POSIX shm 支持有限，必须 fallback 到未注册 shm，不影响正确性。

## 11. 阶段三测试

功能测试：

```bash
VGPU_DATA_PLANE=shm LD_PRELOAD=./build/libcudart_proxy.so \
VGPU_SERVER=127.0.0.1:50051 \
/tmp/vgpu_memcpy_proxy_test

VGPU_DATA_PLANE=shm LD_PRELOAD=./build/libcudart_proxy.so \
VGPU_SERVER=127.0.0.1:50051 \
/tmp/vgpu_vector_add_proxy_test

VGPU_DATA_PLANE=shm LD_PRELOAD=./build/libcudart_proxy.so \
VGPU_SERVER=127.0.0.1:50051 \
/tmp/vgpu_matmul_proxy_test
```

A/B 测试：

```bash
VGPU_DATA_PLANE=grpc ...
VGPU_DATA_PLANE=shm ...
```

异常测试：

- shm offset 越界。
- shm size 太小。
- shm name 非法。
- client 创建 shm 后未初始化 header。
- client 被 kill 后 server timeout 清理。
- 双进程同时使用 shm 数据面。

## 12. 阶段三验收标准

- `VGPU_DATA_PLANE=grpc` 旧路径仍可用。
- `VGPU_DATA_PLANE=shm` 新路径可用。
- H2D/D2H 不再通过 protobuf `bytes` 传大块数据。
- memcpy、vector_add、matmul 在 shm 路径下通过。
- 双进程各自使用独立 shm，不互相覆盖。
- server 对 shm offset/size 做严格校验。
- 日志能区分 `dataplane=grpc` 和 `dataplane=shm`。
