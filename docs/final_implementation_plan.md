# 最终实现总文档

本文是项目最终实现路线的总入口。项目按四个阶段推进，当前路线统一改为在 WSL/Linux 同一内核环境中实现 CUDA Runtime API 拦截、gRPC 控制面和 POSIX Shared Memory 数据面。项目不再把 KVM/QEMU、VSOCK、IVSHMEM 作为后续主线。

四个阶段文档：

- [阶段一：WSL 单机 gRPC 功能闭环](stage1_wsl_grpc_functional.md)
- [阶段二：WSL 资源虚拟化与并发闭环](stage2_wsl_resource_concurrency.md)
- [阶段三：POSIX Shared Memory 数据面接入](stage3_posix_shm_dataplane.md)
- [阶段四：POSIX Shared Memory 性能优化与测试](stage4_posix_shm_performance.md)
- [最终验收结果](final_acceptance_results.md)

## 0. 当前进度快照

当前代码处于：

```text
阶段一：已完成
阶段二：kernel launch、双进程并发、stream/event、异常清理、显存配额、基础性能日志已完成
阶段三：POSIX shm 同步 H2D/D2H 数据面已完成，并保留 gRPC bytes 回退
阶段四：已完成固定块 data arena、async H2D/D2H 生命周期、event 精确回收、MemcpyShm/D2D/kernel launch/sync/event query 的 SPSC ring 控制快路径
最终验收：已完成矩阵乘法、双进程并发、多 session 隔离、ring capacity、错误隔离、异常退出和 10 分钟持续稳定性测试
```

已实现并验证：

| 能力 | 当前状态 |
| --- | --- |
| gRPC/CMake 工程 | 已完成 |
| `LD_PRELOAD` Runtime 查询拦截 | 已完成 |
| session 创建/销毁 | 已完成 |
| server CUDA Driver API 初始化 | 已完成 |
| 虚拟 device pointer 到真实 `CUdeviceptr` 映射 | 已完成 |
| `cudaMalloc` / `cudaFree` | 已完成 |
| `cudaMemcpy` H2D / D2H / D2D | 已完成并验证 |
| `cudaDeviceSynchronize` | 已完成，ring fast path 当前同步当前 session 的 stream |
| fatbin 注册和 module load | 已完成 |
| kernel symbol 映射 | 已完成 |
| `cudaLaunchKernel` | 已完成基础版，常见小参数 kernel 已走 ring fast path，超出 descriptor 容量自动回退 gRPC |
| `vector_add` | 已通过 |
| `matmul` | 已通过 |
| 双进程并发 kernel | 已通过 |
| 双进程 vector+vector/vector+matmul/matmul+matmul | 已通过 |
| 多 session 相同虚拟指针隔离 | 已通过 |
| 错误 session 隔离 | 已通过 |
| 10 分钟持续稳定性 | 已通过 |
| per-session stream | 已实现 |
| `cudaMemcpyAsync` / stream API / event API | 已实现基础版 |
| session timeout 清理 | 已实现并验证 |
| 显存配额 | 已实现并验证 |
| 基础性能日志 | 已实现，输出 `elapsed_us/bytes/session_id/op/cuda_error` |
| POSIX shm 数据面 | 已实现，支持同步 H2D/D2H 大块数据，保留 gRPC bytes 回退 |
| POSIX shm data arena | 已实现固定块 allocator，默认 4 MiB block，支持连续多 block |
| POSIX shm async H2D | 已实现，`cudaMemcpyAsync(H2D)` 可走 shm 数据面，在 stream/device/event sync 后回收 block |
| POSIX shm async D2H | 已实现，server 写入 shm，client 在 stream/device/event sync 后 copy-out 到用户 host buffer |
| POSIX shm event 精确回收 | 已实现，`cudaEventRecord` 记录 stream 上的 shm pending 序号截止点，`cudaEventSynchronize` / `cudaEventQuery` 成功后只释放 event 之前的 block |
| POSIX shm SPSC ring | 已实现 `MemcpyShm`、D2D、kernel launch、device/stream/event sync、event query 控制快路径；每个 session 独立 ring，server per-session worker 消费 descriptor |
| POSIX shm pinned arena | 已实现，server 默认尝试对 shm data arena 做 `cuMemHostRegister`，失败自动回退；`VGPU_PINNED_SHM=0` 可关闭 |
| POSIX shm 性能拆账 | 已实现基础版，`VGPU_PERF_DETAIL=1` 输出 client host copy、ring wait、server CUDA submit/sync 等分段耗时 |
| shm-backed pinned host API | 已实现 `cudaHostAlloc` / `cudaMallocHost` / `cudaFreeHost` direct shm 快路径；`cudaHostRegister` / `cudaHostUnregister` 已有兼容实现 |
| POSIX shm 后续优化 | 未完成 futex/eventfd 唤醒、低噪声 benchmark 日志模式、更完整 pinned host API 语义和完整性能报告 |

当前已验证命令输出包括：

```text
memcpy baseline passed: 4194304 bytes H2D/D2D/D2H
vector_add smoke test passed: 1048576 elements
matmul baseline passed: 128x128
VGPU_DATA_PLANE=shm memcpy baseline passed: 4194304 bytes H2D/D2D/D2H
VGPU_DATA_PLANE=shm vector_add smoke test passed: 1048576 elements
VGPU_DATA_PLANE=shm matmul baseline passed: 128x128
VGPU_DATA_PLANE=shm 双进程 vector_add passed
VGPU_DATA_PLANE=shm VGPU_SHM_BLOCK_SIZE=1048576 memcpy baseline passed
VGPU_DATA_PLANE=shm stream async smoke test passed: 1048576 elements
VGPU_DATA_PLANE=shm VGPU_SHM_SIZE=8388608 stream async smoke test passed
VGPU_DATA_PLANE=shm d2h async smoke test passed: 1048576 elements
VGPU_DATA_PLANE=shm event reclaim smoke OK
VGPU_DATA_PLANE=shm ring-enabled memcpy/vector_add/matmul/event_timing passed
VGPU_DATA_PLANE=shm launch/sync/event/D2D ring fast path passed
VGPU_DATA_PLANE=shm host alloc smoke OK
VGPU_DATA_PLANE=shm host register smoke OK
stream async smoke test passed: 1048576 elements
event timing smoke test passed: 6.962 ms
stability negative smoke test passed
memory limit smoke test passed
```

当前性能基线说明：

```text
原生 CUDA 4 MiB H2D: 约 429 us
当前 gRPC bytes 版 4 MiB H2D: 约 52989 us
当前 POSIX shm 单 slot 版 4 MiB H2D: 约 3614 us
当前 POSIX shm ring+pinned 版 4 MiB H2D: 约 956 us
当前 POSIX shm ring+pinned+HostAlloc 版 4 MiB H2D: 约 688 us
当前 POSIX shm ring+control fast path+HostAlloc 版 4 MiB H2D: 约 591 us
原生 CUDA 4 MiB D2H: 约 569 us
当前 gRPC bytes 版 4 MiB D2H: 约 47893 us
当前 POSIX shm 单 slot 版 4 MiB D2H: 约 4354 us
当前 POSIX shm ring+pinned 版 4 MiB D2H: 约 894 us
当前 POSIX shm ring+pinned+HostAlloc 版 4 MiB D2H: 约 675 us
当前 POSIX shm ring+control fast path+HostAlloc 版 4 MiB D2H: 约 417 us
当前 POSIX shm ring fast path kernel launch only: 平均约 13 us
当前 POSIX shm ring fast path kernel launch + sync: 平均约 90 us，P50 约 44 us
当前 POSIX shm ring fast path D2D async submit 4 MiB: 平均约 24 us，P50 约 9 us
当前 POSIX shm ring fast path D2D + sync 4 MiB: 平均约 185 us，P99 有毫秒级尾延迟
矩阵乘法 n=256 稳态 proxy/native median-of-medians: 约 1.22x
10 分钟持续稳定性: 1452 轮，0 失败
```

结论：gRPC bytes 版本功能正确，但大块 H2D/D2H 因 protobuf/gRPC 序列化和多次拷贝产生明显性能损耗。POSIX Shared Memory 已经把大块 host 数据移出 protobuf，并进一步加入固定块 data arena、async 生命周期、event 精确回收、per-session SPSC ring 控制快路径、server 侧 pinned shm、shm-backed `cudaHostAlloc`，以及 kernel launch/sync/event/D2D 的高频控制 fast path。当前 direct shm host buffer 已把 4 MiB H2D/D2H 压到约 591/417 us；kernel launch only 平均约 13 us，空 device sync 平均约 2 us。后续主要是减少尾延迟、低 CPU 唤醒机制、低噪声 benchmark 日志模式、更完整 pinned host API 语义和完整性能报告。

## 1. 最终目标

本项目实现 CUDA Runtime API 级轻量化 GPU 虚拟化原型，不做完整 GPU 硬件虚拟化，不做 PCIe passthrough，不依赖 MIG、MPS、SR-IOV。虚拟化粒度是 GPU 运行进程，每个进程对应一个 server session，只虚拟化该进程真实使用的 CUDA 资源。

最终系统形态：

```text
WSL/Linux client process
  CUDA 应用
    |
    | LD_PRELOAD
    v
  libcudart_proxy.so
    |
    | 控制面：gRPC/TCP
    | 数据面：POSIX Shared Memory
    v
WSL/Linux server process
  vgpu_server
    |
    | CUDA Driver API
    v
  NVIDIA GPU
```

client 侧不直接使用 CUDA Driver API，不持有真实 `CUdeviceptr`；server 侧拥有 NVIDIA GPU、driver、CUDA Toolkit 和 `vgpu_server`。client 应用看到的是 CUDA Runtime API 语义，实际显存分配、数据传输、kernel module 加载、kernel launch 和 stream/event 同步都由 server 执行。

边界说明：

- POSIX Shared Memory 适用于同一个 Linux 内核内的多个进程。
- 在 WSL 中，它可以作为当前项目的高性能数据面实现。
- 它不能直接跨 QEMU guest/host 内核共享内存，因此本路线不再声称完成真实跨 VM IVSHMEM 验证。
- 项目文档和测试报告按 WSL/Linux 同内核 vGPU 原型来组织。

## 2. 三个通信平面

| 平面 | 当前实现 | 最终实现 | 说明 |
| --- | --- | --- | --- |
| 控制平面 | gRPC over TCP + shm SPSC ring | gRPC over TCP + shm ring | session、malloc/free、stream/event 创建销毁、module 注册等低频 API 走 gRPC；MemcpyShm、D2D、kernel launch、sync/event query 走 ring |
| 数据平面 | protobuf `bytes`，可选 POSIX shm data arena | POSIX shm data arena | H2D/D2H 大块 host 数据 |
| 通知同步平面 | RPC 返回值 + ring done 字段 | ring done + 可选 futex/eventfd | shm descriptor 提交、D2H 完成、buffer 回收 |

控制平面和数据平面必须分开看。控制消息只有几十到几百字节，重点是低延迟；大块张量数据可能是 MiB 到 GiB 级，重点是减少拷贝和序列化。前两阶段使用 gRPC 同时承担控制和数据，便于先跑通功能；第三阶段开始把 H2D/D2H 数据面切到 POSIX shm。

推荐分层：

```text
Runtime Interceptor
├── ControlPlaneClient: gRPC stub
├── DataPlane
│   ├── GrpcBytesDataPlane      当前已有
│   └── PosixShmDataPlane       后续主线
└── Resource Mapper
    ├── vptr
    ├── vstream
    └── vevent
```

## 3. 核心资源模型

server 为每个 client 进程创建一个 session：

```cpp
struct Session {
    uint64_t session_id;
    uint32_t client_pid;
    uint64_t memory_limit;
    uint64_t memory_used;

    AllocationTable allocations;  // vptr -> CUdeviceptr
    StreamTable streams;          // vstream -> CUstream
    EventTable events;            // vevent -> CUevent
    ModuleTable modules;          // module_handle -> CUmodule
    FunctionTable functions;      // host_fun_id -> CUfunction

    // 第三阶段启用
    PosixShmChannel shm_channel;
};
```

客户机永远不拿真实 `CUdeviceptr`。`cudaMalloc` 返回的是 server 生成的虚拟指针 `vptr`。所有 `cudaMemcpy`、kernel 参数、`cudaFree` 都携带 `vptr`，server 查表后翻译成真实 device pointer。

推荐 `vptr` 编码：

```text
高位：session cookie 或 session index
低位：allocation sequence
```

这样能防止一个 session 伪造或误释放另一个 session 的 allocation，并且便于日志追踪。

## 4. CUDA Runtime API 支持顺序

第一批：查询和基础会话

- `cudaGetDeviceCount`
- `cudaGetDevice`
- `cudaSetDevice`
- `cudaGetDeviceProperties`
- `cudaRuntimeGetVersion`
- `cudaDriverGetVersion`
- `cudaGetLastError`
- `cudaPeekAtLastError`
- `cudaGetErrorString`
- `cudaGetErrorName`

第二批：显存和数据传输

- `cudaMalloc`
- `cudaFree`
- `cudaMemcpy`
- `cudaMemset`（未实现）

第三批：kernel 注册和启动

- `__cudaRegisterFatBinary`
- `__cudaRegisterFatBinaryEnd`
- `__cudaRegisterFunction`
- `__cudaUnregisterFatBinary`
- `__cudaInitModule`
- `__cudaPushCallConfiguration`
- `__cudaPopCallConfiguration`
- `cudaLaunchKernel`
- 可选兼容 `cudaConfigureCall`、`cudaSetupArgument`、`cudaLaunch`（未实现旧 API 兼容层）

第四批：并发和同步

- `cudaStreamCreate`
- `cudaStreamCreateWithFlags`
- `cudaStreamDestroy`
- `cudaStreamSynchronize`
- `cudaMemcpyAsync`
- `cudaDeviceSynchronize`
- `cudaEventCreate`
- `cudaEventCreateWithFlags`
- `cudaEventRecord`
- `cudaEventQuery`
- `cudaEventSynchronize`
- `cudaEventElapsedTime`
- `cudaEventDestroy`

第五批：POSIX shm 性能增强

- shm-backed H2D
- shm-backed D2H
- per-session shm arena
- per-session SPSC descriptor ring
- `cudaHostRegister` server 侧 pinned shm arena
- 可选 `cudaHostAlloc` / `cudaFreeHost`
- 可选 `cudaHostRegister` / `cudaHostUnregister`

## 5. 阶段边界

### 阶段一

在 WSL 内跑通 gRPC 功能闭环。client 和 server 在同一环境中运行，通过 `127.0.0.1:50051` 通信。目标是 `cudaMalloc`、`cudaFree`、`cudaMemcpy` 的 gRPC bytes 版本正确，server 能用 CUDA Driver API 操作真实 GPU。

当前状态：已完成，`memcpy_baseline` 已通过。

### 阶段二

仍在 WSL 内完成 kernel launch、多 session、多 stream 和清理。目标是 vector add、matmul、memcpy、双进程并发都能通过。

当前状态：已完成 kernel launch、双进程并发、per-session default stream、显式 stream API、`cudaMemcpyAsync` 基础版、event API 基础版、异常退出 timeout 清理、显存配额和基础性能日志；更细粒度 `rpc_us/cuda_us` 拆分和长时间压力测试尚未完成。

### 阶段三

在 WSL/Linux 同一内核内接入 POSIX Shared Memory 数据面。控制面继续使用 gRPC，CreateSession 时协商 shm 名称、大小和 channel 信息。H2D/D2H 的大块 host 数据不再放入 protobuf `bytes`，而是通过 `shm_offset + size` descriptor 描述。

当前状态：已完成。`CreateSession` 协商 POSIX shm，server mmap 同一 shm 对象；H2D/D2H 大块数据通过 `shm_offset + size` descriptor 描述。D2D 不经过 shm payload，当前已升级为 metadata-only ring fast path。

### 阶段四

围绕 POSIX shm 数据面做性能优化、并发压力测试和稳定性测试。重点包括 per-session SPSC ring、server worker 调度、pinned shm arena、D2H 完成同步、buffer 回收、性能损耗统计。

当前状态：已完成主要原型：固定块 data arena、async H2D/D2H、event 精确回收、per-session SPSC ring、server worker、pinned shm arena、shm-backed `cudaHostAlloc`、kernel launch/sync/event/D2D ring fast path 和基础性能拆账已实现并通过 memcpy/vector_add/matmul/stream/event/hostalloc/双进程测试。futex/eventfd 唤醒、低噪声 benchmark 日志模式、更完整 pinned host API 语义和完整性能报告仍未完成。

## 6. POSIX Shared Memory 设计原则

POSIX shm 由 client 创建或由 server 创建后传回名称，双方通过同一个 shm name 打开并 `mmap`：

```text
shm_open("/vgpu_<uid>_<pid>_<session>")
ftruncate(shm_fd, shm_size)
mmap(PROT_READ | PROT_WRITE, MAP_SHARED)
```

推荐阶段三第一版采用 client 创建：

```text
client first CUDA API
  -> create shm
  -> mmap shm
  -> INIT RPC 携带 shm_name/shm_size
  -> server shm_open
  -> server mmap
  -> server cudaHostRegister data arena
  -> reply session_id/channel_id/default_stream_id
```

每个 session 拥有独立 shm region：

```text
SessionShm
├── header
│   ├── magic
│   ├── version
│   ├── session_id
│   ├── ring_offset
│   ├── ring_size
│   ├── arena_offset
│   └── arena_size
├── command ring
└── data arena
```

每个 session 的 command ring 是 SPSC：

- client session 是 producer，只写 `head`。
- server worker 是 consumer，只写 `tail`。
- 需要 `std::atomic` 和 acquire/release 内存序。
- 跨进程热路径不用共享锁，也不需要 CAS。
- 同一个 client 进程内多线程同时调用 CUDA API 时，当前 client 用 `ring_mu` 串行化提交，保证一个 session ring 仍按单 producer 方式写入。后续如果要做到每线程完全无锁，需要改为每 client thread 一个 ring 或 MPSC 提交队列。

ring 只放 descriptor，大块数据放 data arena：

```cpp
struct ShmCommandDesc {
    std::atomic<uint32_t> state;
    uint32_t op;
    uint64_t seq;
    uint64_t src_vptr;
    uint64_t dst_vptr;
    uint64_t shm_offset;
    uint64_t size;
    uint64_t stream_id;
    uint64_t event_id;
    uint64_t module_id;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint8_t kernel_arg_data[256];
    int32_t cuda_error;
};
```

普通 malloc host buffer 的 H2D/D2H 不是完全零拷贝，因为用户 host buffer 仍需复制到 POSIX shm data arena。`cudaHostAlloc` / `cudaMallocHost` 当前直接返回 shm arena 内存，因此这类用户 buffer 可以跳过额外 CPU copy，更接近零拷贝。普通 `cudaHostRegister` 指针无法原地变成跨进程 shm，当前只提供兼容注册/注销。

## 7. 最终验收

功能验收：

- WSL/Linux 同内核环境中运行 CUDA Runtime 测试程序。
- `cudaMalloc` / `cudaFree` / `cudaMemcpy` 正确。
- vector add 和 matmul 结果与原生 CUDA 一致。
- H2D / D2H / D2D memcpy 正确。
- 双进程并发 kernel 不互相污染。
- POSIX shm H2D/D2H 与 gRPC bytes H2D/D2H 结果一致。

并发验收：

- 每个进程独立 session。
- 每个 session 独立 vptr table、stream table、module table、shm channel。
- 一个 session 异常退出后资源能清理。
- 一个 session 超配额不影响其他 session。
- 多进程同时使用 shm 数据面时 ring 和 data arena 不互相覆盖。

性能验收：

- 对比原生 CUDA、gRPC bytes 版 vGPU、POSIX shm 版 vGPU。
- D2D 全程 server 内部完成。
- H2D/D2H 使用 POSIX shm 后比 gRPC bytes 版明显降低开销。
- 报告中区分 API 总耗时、RPC 耗时、shm copy 耗时、CUDA Driver API 耗时、kernel event 时间和数据传输耗时。

文档验收：

- 总体设计文档。
- 四阶段实施文档。
- API 映射文档。
- RPC 协议文档。
- 虚拟显存文档。
- POSIX shm 数据平面文档。
- 性能测试报告。
- 稳定性测试报告。
