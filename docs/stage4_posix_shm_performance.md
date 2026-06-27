# 阶段四：POSIX Shared Memory 性能优化与测试

阶段四目标是在阶段三 POSIX shm 数据面正确的基础上，继续优化 H2D/D2H、并发调度和同步语义，并形成性能测试文档与稳定性测试文档。控制面继续使用 gRPC；大数据面使用 POSIX shm；不引入 KVM/QEMU、VSOCK、IVSHMEM。

阶段三第一版实现的是“单 slot + 同步 `MemcpyShm` RPC”：

```text
client memcpy(user buffer -> shm offset 0)
client 调 MemcpyShm RPC
server 从 shm offset 0 发起 CUDA copy
同步 API 中等待 copy 完成
client/server 复用 offset 0
```

阶段四从这个实现继续演进，而不是推倒重写。

## 0. 当前实现状态

截至当前代码版本，阶段四已经完成八步：client 侧固定块 data arena、async H2D 的 shm block 生命周期管理、async D2H 的延迟 copy-out、event 精确回收、`MemcpyShm` 控制面的 per-session SPSC ring 快路径、`cudaHostAlloc` / `cudaMallocHost` 的 shm-backed host buffer 快路径、高频控制 API 的 ring fast path，以及 D2D descriptor 的 ring fast path。

已实现：

- `ShmState` 增加 `block_size` 和 `block_free` 空闲表。
- 默认 `VGPU_SHM_BLOCK_SIZE=4194304`，即 4 MiB 一个 block。
- 支持通过环境变量 `VGPU_SHM_BLOCK_SIZE` 调整 block 大小。
- 同步 H2D/D2H 在发起 `MemcpyShm` 前先申请一个或多个连续 block。
- `MemcpyShm` RPC 期间不再持有全局 `shm_.mu`，只在申请/释放 block 时加锁。
- RPC 成功、失败或返回 CUDA 错误时都会释放 block。
- block 不足时同步等待空闲 block，避免复用未完成数据。
- `cudaMemcpyAsync(H2D)` 可以走 `MemcpyShmAsync`。
- async H2D 成功提交后，client 不立即释放 shm block，而是记录为 pending。
- `cudaStreamSynchronize(stream)` 成功后释放该 stream 的 pending block。
- `cudaStreamDestroy(stream)` 成功后释放该 stream 的 pending block。
- `cudaDeviceSynchronize()` 成功后释放所有 pending block。
- async H2D 如果临时没有空闲 shm block，会回退到旧 gRPC async 路径，避免阻塞用户线程。
- `cudaMemcpyAsync(D2H)` 可以走 `MemcpyShmAsync`。
- async D2H 成功提交后，client 记录 `shm block -> 用户 host buffer`。
- `cudaStreamSynchronize(stream)` 或 `cudaDeviceSynchronize()` 成功后，client 先从 shm copy-out 到用户 host buffer，再释放 block。
- async D2H 如果临时没有空闲 shm block，会回退到旧 gRPC async 路径。
- 每个 async shm block 带递增序号。
- `cudaEventRecord(event, stream)` 记录该 stream 当前 pending block 的序号截止点。
- `cudaEventSynchronize(event)` 成功后，只释放该 event 记录点之前的 pending block。
- `cudaEventQuery(event)` 返回 `cudaSuccess` 后执行同样的精确释放；返回 `cudaErrorNotReady` 时不释放。
- POSIX shm 前 1 MiB 作为 ring control 区，后续区域作为 data arena。
- server 在 `CreateSession` 时初始化 ring header，并为该 session 启动一个 ring worker。
- client 的 `MemcpyViaShm` 优先把 `MemcpyShm` descriptor 写入 ring；ring 不可用时自动回退到原 `MemcpyShm` gRPC。
- server ring worker 消费 descriptor 后复用现有 `DoMemcpyShm`，再把 `cuda_error` 和 `done` 写回 ring entry。
- `cudaLaunchKernel` 已加入 ring fast path。kernel 名称、grid/block 维度、shared memory、stream id 和小参数包通过 ring descriptor 提交；超过固定 descriptor 容量时自动回退 gRPC。
- `cudaDeviceSynchronize`、`cudaStreamSynchronize`、`cudaEventSynchronize`、`cudaEventQuery` 已加入 ring fast path。
- `cudaMemcpy(D2D)` 和 `cudaMemcpyAsync(D2D)` 已加入 ring fast path，数据仍完全在 server/GPU 侧完成，不经过 POSIX shm payload。
- server ring worker 已从无条件 `sleep_for(50us)` 改为 adaptive idle：先短暂 busy spin，再 yield，最后微睡眠，降低固定控制延迟。
- 同一个 client 进程内多线程同时提交 CUDA API 时，client 侧用 `ring_mu` 串行化 ring producer；跨进程仍然是每 session 独立 ring。
- server 默认尝试对 POSIX shm data arena 做 `cuMemHostRegister` pinned 注册，注册失败自动回退到 pageable shm。
- `VGPU_PINNED_SHM=0` 可关闭 pinned shm 注册，便于 A/B 测试。
- `VGPU_PERF_DETAIL=1` 可开启 client/server 细粒度性能拆账日志。
- `cudaHostAlloc` / `cudaMallocHost` 从 POSIX shm data arena 分配 host buffer。
- `cudaFreeHost` 释放对应 shm arena block。
- H2D 如果发现 source host pointer 已经位于 shm data arena，直接发送该 shm offset，不再执行 `user_host -> shm` 临时拷贝。
- D2H 如果发现 destination host pointer 已经位于 shm data arena，server 直接写入用户 buffer 对应的 shm offset，client 不再执行 `shm -> user_host` copy-out。
- `cudaHostRegister` / `cudaHostUnregister` 已提供兼容实现；普通 malloc 指针可以注册/注销，但由于该内存不是跨进程 shm，不能走 direct shm 快路径。

已验证：

```text
VGPU_DATA_PLANE=shm memcpy baseline passed: 4194304 bytes H2D/D2D/D2H
VGPU_DATA_PLANE=shm vector_add smoke test passed: 1048576 elements
VGPU_DATA_PLANE=shm matmul baseline passed: 128x128
VGPU_DATA_PLANE=shm 双进程 vector_add passed
VGPU_DATA_PLANE=shm VGPU_SHM_BLOCK_SIZE=1048576 memcpy baseline passed
VGPU_DATA_PLANE=shm stream async smoke test passed: 1048576 elements
VGPU_DATA_PLANE=shm VGPU_SHM_SIZE=8388608 VGPU_SHM_BLOCK_SIZE=4194304 stream async smoke test passed
VGPU_DATA_PLANE=shm d2h async smoke test passed: 1048576 elements
VGPU_DATA_PLANE=shm event reclaim smoke OK
VGPU_DATA_PLANE=shm ring-enabled memcpy/vector_add/matmul/event_timing passed
VGPU_DATA_PLANE=shm ring-enabled 双进程 vector_add passed
VGPU_DATA_PLANE=shm host alloc smoke OK
VGPU_DATA_PLANE=shm host register smoke OK
VGPU_DATA_PLANE=shm launch/sync/event/D2D ring fast path passed
VGPU_DATA_PLANE=shm 双进程 vector+vector/vector+matmul/matmul+matmul passed
VGPU_DATA_PLANE=shm matrix_mul_benchmark n=256 proxy/native 约 1.22x passed
VGPU_DATA_PLANE=shm fire_forget_stress count=2048 D2D/kernel passed
VGPU_DATA_PLANE=shm 10 分钟持续稳定性 1452 轮 0 失败 passed
```

当前 4 MiB loop benchmark，pinned shm 默认开启：

```text
POSIX shm ring pinned H2D_avg_us=956.433
POSIX shm ring pinned D2D_avg_us=637.033
POSIX shm ring pinned D2H_avg_us=894.233
POSIX shm ring pinned Kernel_sync_avg_us=1261.700
```

关闭 pinned shm 的对照结果：

```text
VGPU_PINNED_SHM=0 H2D_avg_us=943.400
VGPU_PINNED_SHM=0 D2D_avg_us=878.633
VGPU_PINNED_SHM=0 D2H_avg_us=1339.733
VGPU_PINNED_SHM=0 Kernel_sync_avg_us=1376.733
```

对比阶段三单 slot + `MemcpyShm` RPC 记录，H2D 从约 3614 us 降到约 956 us，D2H 从约 4354 us 降到约 894 us。pinned shm 对 H2D 的收益不明显，因为 H2D 当前大头是 client 侧 user buffer 到 shm 的 CPU memcpy；对 D2H 收益明显，从约 1340 us 降到约 894 us。

一次 4 MiB memcpy 的细粒度拆账示例：

```text
[cudart_proxy_perf] op=MemcpyViaShm kind=1 bytes=4194304 ring=1 alloc_us=3 h2d_host_copy_us=1026 ring_wait_us=580 total_us=1652
[cudart_proxy_perf] op=MemcpyViaShm kind=2 bytes=4194304 ring=1 alloc_us=0 ring_wait_us=405 d2h_copyout_us=513 total_us=948
[vgpu_perf_detail] op=DoMemcpyShm kind=1 pinned=1 cuda_submit_us=34 cuda_sync_us=389 total_us=429
[vgpu_perf_detail] op=DoMemcpyShm kind=2 pinned=1 cuda_submit_us=16 cuda_sync_us=350 total_us=371
```

结论：server 侧 CUDA copy 已经接近原生量级。对普通 malloc host buffer，剩余大头是 client 侧额外 host copy：H2D 的 `user_host -> shm` 和 D2H 的 `shm -> user_host`。`cudaHostAlloc` / `cudaMallocHost` 已经让用户 buffer 本身来自 shm/pinned arena，从路径上减少一次 CPU copy；后续继续降损耗应重点处理控制面同步开销、低 CPU 唤醒和更完整的 pinned host API 语义。

`cudaHostAlloc` / `cudaMallocHost` direct shm benchmark：

```text
host_alloc_perf_loop elements=1048576 bytes=4194304 iters=30
H2D_avg_us=688.267
D2D_avg_us=449.533
D2H_avg_us=674.833
Kernel_sync_avg_us=926.867
```

细粒度日志显示 direct shm 路径已经消除了 client 侧额外 host copy：

```text
[cudart_proxy_perf] op=MemcpyViaShm kind=1 bytes=4194304 ring=1 direct_host_shm=1 alloc_us=0 h2d_host_copy_us=0 total_us=666
[cudart_proxy_perf] op=MemcpyViaShm kind=2 bytes=4194304 ring=1 direct_host_shm=1 alloc_us=0 d2h_copyout_us=0 total_us=506
```

注意：`cudaHostRegister` 无法把一个普通 malloc 指针原地变成 server 可 mmap 的跨进程共享内存。当前实现对普通指针提供兼容注册/注销，但它仍然走临时 shm block 路径；只有 `cudaHostAlloc` / `cudaMallocHost` 返回的 shm-backed 指针能走 direct shm 快路径。

高频控制 API ring 化后的拆账 benchmark：

```text
iters=50
device_sync_empty bytes=0 avg_us=1.936 p50_us=1.852 p95_us=2.386 p99_us=4.194 max_us=4.194
kernel_launch_only bytes=0 avg_us=13.237 p50_us=10.237 p95_us=23.590 p99_us=62.173 max_us=62.173
kernel_launch_plus_sync bytes=0 avg_us=89.535 p50_us=44.166 p95_us=488.179 p99_us=994.265 max_us=994.265
d2d_async_submit bytes=4194304 avg_us=24.193 p50_us=9.369 p95_us=110.515 p99_us=170.922 max_us=170.922
d2d_sync bytes=4194304 avg_us=185.414 p50_us=92.484 p95_us=207.168 p99_us=3443.592 max_us=3443.592
h2d_hostalloc bytes=4 avg_us=65.556 p50_us=57.104 p95_us=159.259 p99_us=254.182 max_us=254.182
d2h_hostalloc bytes=4 avg_us=42.332 p50_us=40.226 p95_us=66.359 p99_us=76.815 max_us=76.815
h2d_hostalloc bytes=4096 avg_us=86.684 p50_us=46.881 p95_us=345.528 p99_us=394.832 max_us=394.832
d2h_hostalloc bytes=4096 avg_us=53.730 p50_us=48.254 p95_us=78.939 p99_us=256.809 max_us=256.809
h2d_hostalloc bytes=65536 avg_us=50.046 p50_us=48.857 p95_us=62.287 p99_us=71.715 max_us=71.715
d2h_hostalloc bytes=65536 avg_us=55.567 p50_us=48.181 p95_us=112.058 p99_us=137.411 max_us=137.411
h2d_hostalloc bytes=1048576 avg_us=151.454 p50_us=145.677 p95_us=204.228 p99_us=233.274 max_us=233.274
d2h_hostalloc bytes=1048576 avg_us=204.842 p50_us=163.491 p95_us=362.382 p99_us=393.632 max_us=393.632
h2d_hostalloc bytes=4194304 avg_us=590.623 p50_us=569.409 p95_us=927.143 p99_us=977.719 max_us=977.719
d2h_hostalloc bytes=4194304 avg_us=416.814 p50_us=391.323 p95_us=542.015 p99_us=788.783 max_us=788.783
h2d_hostalloc bytes=16777216 avg_us=1667.495 p50_us=1515.035 p95_us=2614.133 p99_us=2725.462 max_us=2725.462
d2h_hostalloc bytes=16777216 avg_us=1505.341 p50_us=1406.693 p95_us=1993.814 p99_us=2690.860 max_us=2690.860
h2d_hostalloc bytes=67108864 avg_us=6003.160 p50_us=5752.618 p95_us=7712.055 p99_us=7890.334 max_us=7890.334
d2h_hostalloc bytes=67108864 avg_us=6581.479 p50_us=6635.374 p95_us=8821.282 p99_us=9273.603 max_us=9273.603
```

这组数据说明：

- 空 `cudaDeviceSynchronize` 已降到约 2 us，证明同步 API 已经绕开 gRPC 往返。
- `kernel_launch_only` 平均约 13 us，说明 `cudaLaunchKernel` 已不再被一次 gRPC RPC 限制。
- `kernel_launch_plus_sync` 平均约 90 us、P50 约 44 us，但 P95/P99 仍有尾延迟，后续需要继续处理 worker 唤醒和日志干扰。
- `D2D async submit` 平均约 24 us、P50 约 9 us，descriptor 提交开销已经较低；`D2D + sync` 的 P99 有毫秒级抖动，需要单独继续分析同步等待和调度尾延迟。
- direct hostalloc 的 4 B / 4 KiB 小数据已经走 shm ring direct path，不再被 `VGPU_SHM_THRESHOLD` 错误挡回 gRPC。

当前仍未完成：

- futex/eventfd 等低 CPU 占用唤醒机制。
- 完整性能报告，例如多轮统计、方差、不同数据大小、并发进程数下的 API 总耗时、ring 等待耗时、CUDA Driver 调用耗时、host memcpy 耗时。
- 更完整的 pinned host API 语义，例如 `cudaHostGetDevicePointer`、mapped host memory 标志和更严格的注册生命周期校验。
- benchmark 专用低噪声日志模式。当前 server 每个 ring op 仍会输出日志，这会放大 P95/P99 尾延迟。

最终验收摘要见 [最终验收结果](final_acceptance_results.md)。本阶段当前已经覆盖双进程并发、矩阵乘法、ring wrap-around、错误隔离和持续稳定性；未覆盖的是无 GPU client 容器实测，原因是当前环境未安装 Docker。

## 1. 设计目标

阶段四要把阶段三第一版改成更接近最终可展示的高性能数据面：

```text
每个 client process
  -> 独立 session
  -> 独立 POSIX shm
  -> 独立 SPSC command ring
  -> 独立 data arena
  -> 独立 default CUstream
```

目标：

- 减少 gRPC 大块数据传输。
- 减少 server 全局锁竞争。
- 避免多个进程共用 CUDA null stream 造成隐式同步。
- 支持多进程同时 H2D/kernel/D2H。
- 给出原生 CUDA、gRPC bytes 版、POSIX shm 版三组性能对比。

## 2. Command Ring

阶段四已经引入 SPSC ring。每个 session 拥有一个 command ring，server 中该 session 的 worker 是 consumer。当前实现的 ring 覆盖：

- H2D/D2H `MemcpyShm` descriptor。
- D2D `cudaMemcpy` / `cudaMemcpyAsync` descriptor。
- `cudaLaunchKernel` 小参数 descriptor。
- `cudaDeviceSynchronize`。
- `cudaStreamSynchronize`。
- `cudaEventSynchronize`。
- `cudaEventQuery`。

仍走 gRPC 的主要是低频或大对象 API：session 创建、malloc/free、stream/event create/destroy、event record、fatbin/module 注册、device properties 查询等。

严格的 SPSC 含义是“一个 producer，一个 consumer”。本项目的自然 producer 是一个 client session。为了让同一进程内多个用户线程同时调用 CUDA API 时不破坏 ring，client 侧当前用 `ring_mu` 把提交动作串行化。因此，跨进程场景仍然是每进程独立 ring、无共享竞争；同进程多线程不是完全无锁，而是正确性优先的串行 producer。

```cpp
struct RingHeader {
    uint64_t magic;
    uint64_t version;
    uint64_t capacity;
    uint64_t entry_size;
    uint64_t head;
    uint64_t tail;
    uint64_t stop;
};

struct RingEntry {
    uint64_t seq;
    uint64_t op;
    uint64_t done;
    int32_t cuda_error;
    int32_t kind;
    uint64_t async;
    uint64_t dst_device_ptr;
    uint64_t src_device_ptr;
    uint64_t count;
    uint64_t shm_offset;
    uint64_t stream_id;
    uint64_t event_id;
    uint64_t module_id;
    uint64_t shared_mem;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t arg_count;
    uint32_t arg_bytes;
    char kernel_name[128];
    uint8_t arg_data[256];
};
```

发布规则：

```cpp
entry fields = descriptor;
entry.done = 0;
ring.head.store(next, release);
```

消费规则：

```cpp
head = ring.head.load(acquire);
entry = entries[tail % capacity];
DoMemcpyShm(entry);
entry.done.store(1, release);
ring.tail.store(tail + 1, release);
```

不要用普通变量替代原子。跨进程共享内存仍会受到编译器重排和 CPU 内存序影响。

## 3. Data Arena

data arena 存放 H2D/D2H payload。

第一版 allocator：

```text
fixed-size block allocator
├── block_size: 1 MiB 或 4 MiB
├── free bitmap
└── large copy 使用连续多个 block
```

比完全通用 malloc 更简单，足够覆盖比赛测试：

- 4 MiB memcpy。
- vector add 输入输出。
- matmul 输入输出。
- 多进程并发。

需要保证：

- block offset 按 64B 或 page 对齐。
- `offset + size` 不越过 arena。
- D2H descriptor `Done` 之前，client 不能读取输出 block。
- async H2D 在 server 接受 descriptor 前，client 不能复用输入 block。

## 4. 通知机制

当前实现使用 server per-session worker 轮询 ring。worker 不再固定 sleep 50 us，而是使用 adaptive idle：

```text
client 写 descriptor 到 ring
client release-store 更新 head
server worker poll head/tail
server 处理 descriptor
server 写 cuda_error 和 done
client 等待 done 后返回 API 结果
```

优点：

- 不需要新增 `NotifyShm` RPC。
- 大块 shm memcpy 的控制 descriptor 不再走 gRPC。
- kernel launch、sync/event query 和 D2D descriptor 不再走 gRPC。
- 每个 session 独立 SPSC ring，不需要跨进程锁或 CAS。

后续可选优化：

- futex/eventfd 睡眠，降低 CPU 占用并减少 tail latency。
- eventfd 可用于降低空闲 CPU 占用，但需要 fd 传递或同源创建机制，当前不作为主线。

## 5. H2D 流程

同步 `cudaMemcpy(H2D)`：

```text
client alloc arena block
client memcpy(user_host -> shm block)
client publish MemcpyH2D descriptor
server ring worker polls descriptor
server cuMemcpyHtoDAsync(device, shm block, size, stream)
server cuStreamSynchronize(stream)
server marks Done
client releases arena block
client returns cudaSuccess
```

异步 `cudaMemcpyAsync(H2D)`：

```text
client alloc arena block
client memcpy(user_host -> shm block)
client publish MemcpyH2D descriptor
server ring worker polls descriptor
server queues cuMemcpyHtoDAsync
server marks Done after descriptor is accepted by CUDA
client may return after descriptor accepted
arena block released when stream/device/event synchronization proves completion
```

当前实现已经支持 async H2D block 生命周期管理：API 返回后 block 保持 pending，直到 stream/device/event 同步后释放。如果 async 分配不到 shm block，会回退到旧 gRPC async 路径。

## 6. D2H 流程

同步 `cudaMemcpy(D2H)`：

```text
client alloc arena block
client publish MemcpyD2H descriptor
server ring worker polls descriptor
server cuMemcpyDtoHAsync(shm block, device, size, stream)
server cuStreamSynchronize(stream)
server marks Done
client memcpy(shm block -> user_host)
client releases arena block
client returns cudaSuccess
```

异步 `cudaMemcpyAsync(D2H)` 是难点，因为 Runtime 语义允许 API 返回后稍后完成，但用户 host buffer 必须在 stream 同步后可见。

推荐阶段四实现：

```text
cudaMemcpyAsync(D2H):
  server 先拷贝到 shm block
  client 记录 pending D2H: shm block -> user host pointer
  cudaStreamSynchronize / cudaDeviceSynchronize / cudaEventSynchronize / 成功的 cudaEventQuery:
    等待 server Done
    client CopyOut 到 user host pointer
    释放 block
```

当前实现采用这一路径。event 回收不是粗暴释放整个 stream，而是按 `cudaEventRecord` 时记录的 pending 序号截止点释放，避免把 event 之后提交的 D2H 提前 copy-out。

## 7. Pinned shm

server 尝试对 data arena 做一次性注册：

```cpp
cudaHostRegister(arena_base, arena_size, cudaHostRegisterDefault);
```

或 Driver API：

```cpp
cuMemHostRegister(arena_base, arena_size, CU_MEMHOSTREGISTER_PORTABLE);
```

策略：

- register 成功：日志 `shm_pinned=1`，后续 H2D/D2H 使用 pinned shared memory。
- register 失败：日志记录错误，fallback 到 pageable shm，功能继续。
- 不允许每次 memcpy register/unregister。
- session 销毁时 unregister。

WSL 的 CUDA pinned memory 行为需要实测，不能在文档里假设一定成功。

## 8. 多进程并发

每个进程独立：

```text
session_id
shm object
command ring
data arena
default CUstream
allocation table
stream table
event table
module table
```

进程 A 只写 ring A，进程 B 只写 ring B。数据面热路径没有跨进程锁竞争。

server 全局只保留低频锁：

- session table。
- CUDA context 设置。
- 日志输出。
- module cache，如果后续引入。

server worker 模型：

```text
方案 1：每 session 一个 worker
  简单，适合初版，进程数少时足够。

方案 2：固定 worker pool + active session queue
  更节省线程，适合压力测试。
```

本项目建议先做方案 1，文档记录可扩展到方案 2。

## 9. 性能日志

需要把一次 API 的耗时拆开：

client 侧：

```text
api_total_us
host_to_shm_us
shm_to_host_us
grpc_control_us
wait_done_us
bytes
kind
dataplane
```

server 侧：

```text
queue_wait_us
cuda_submit_us
cuda_sync_us
cuda_total_us
bytes
stream_id
session_id
shm_pinned
cuda_error
```

这样才能解释：

- gRPC bytes 慢在哪里。
- POSIX shm 快在哪里。
- CUDA 真正执行耗时占多少。
- 多进程并发时是否排队。

## 10. 测试矩阵

正确性：

- H2D 单向。
- D2H 单向。
- H2D + kernel + D2H。
- D2D 不经过 shm payload，但 descriptor 已走 ring fast path。
- vector add。
- matmul。
- stream async。
- event timing。
- 双进程同时 H2D/kernel/D2H。

异常：

- ring 满。
- data arena 不足。
- shm offset 越界。
- shm name 非法。
- client 写 descriptor 后退出。
- client 写 H2D 数据一半退出。
- server copy 出错。
- session timeout。
- 超出显存配额。

性能：

| 测试项 | 原生 CUDA | gRPC bytes 版 | POSIX shm 版 | 说明 |
| --- | ---: | ---: | ---: | --- |
| H2D 4 MiB | 已测 | 已测 | 已测 | direct hostalloc avg 约 591 us |
| D2H 4 MiB | 已测 | 已测 | 已测 | direct hostalloc avg 约 417 us |
| D2D 4 MiB async submit | 已测 | 已测 | 已测 | ring submit avg 约 24 us |
| D2D 4 MiB + sync | 已测 | 已测 | 已测 | avg 约 185 us，P99 仍有抖动 |
| Kernel launch only | 已测 | 已测 | 已测 | ring fast path avg 约 13 us |
| Kernel launch + sync | 已测 | 已测 | 已测 | avg 约 90 us，P50 约 44 us |
| H2D 64 MiB | 待测 | 待测 | 已测 | direct hostalloc avg 约 6003 us |
| D2H 64 MiB | 待测 | 待测 | 已测 | direct hostalloc avg 约 6581 us |
| 双进程 vector add | 已通过 | 已通过 | 已通过 | 并发 |

## 11. 验收标准

- POSIX shm 版 H2D/D2H 正确。
- POSIX shm 版 vector add、matmul、memcpy 全部通过。
- 双进程同时运行不互相覆盖 shm buffer。
- server 对 ring descriptor 和 data arena offset 做边界检查。
- pinned shm 成功时有日志，失败时能 fallback。
- POSIX shm 版 H2D/D2H 明显优于 gRPC bytes 版。
- 性能报告包含原生 CUDA、gRPC bytes 版、POSIX shm 版三组数据。
- 稳定性报告包含异常退出、非法参数、资源不足和并发压力测试。

## 12. 诚实边界

本阶段结果可以作为 WSL/Linux 同内核 vGPU 原型的最终数据面结果。

但 POSIX shm 不等价于跨 VM 共享内存：

```text
POSIX shm:
  同一个 Linux 内核内多个进程共享内存。

跨 VM 共享内存:
  需要 hypervisor 把同一段 host 内存映射给 guest 和 host。
```

因此报告中应写：

```text
本项目最终实现采用 POSIX Shared Memory 优化同内核 client/server 数据面；
该实现验证了轻量化 CUDA Runtime API 转发、资源虚拟化、多进程并发和共享内存数据面优化；
不声称完成 KVM/QEMU host/guest IVSHMEM 形态。
```
