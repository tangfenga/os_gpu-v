# os_gpu-v 设计说明

## 1. 项目目标

`os_gpu-v` 实现了一个面向 CUDA Runtime API 的轻量化 GPU 进程虚拟化原型。系统以用户 GPU 进程为边界，只抽取并转发进程实际使用的 CUDA 资源和数据，包括显存分配、kernel launch、stream、event、Host/Device 数据传输和异步错误状态。

该方案避免把虚拟化边界下沉到 PCIe 设备、GPU MMU、中断或完整驱动栈，而是在 Runtime API 层保持 CUDA 程序的调用语义。用户程序通过代理库调用 CUDA Runtime API，server 端使用 CUDA Driver API 在宿主 GPU 上执行真实操作。

## 2. 总体架构

```text
CUDA application
      |
      | LD_PRELOAD
      v
libcudart_proxy.so
      |
      | gRPC control path
      | POSIX shared-memory data arena
      | per-session SPSC ring
      v
vgpu_server
      |
      | CUDA Driver API
      v
NVIDIA GPU
```

系统由三部分组成：

| 模块 | 作用 |
| --- | --- |
| `client/libcudart_proxy.cpp` | 导出 CUDA Runtime 同名函数，拦截应用的 Runtime API 调用 |
| `server/main.cpp` | 实现 gRPC service，维护 session，调用 CUDA Driver API |
| `shared/vgpu_shm_ring.h` | 定义 client/server 共享的 ring buffer 数据结构 |

## 3. Runtime API 拦截

client 侧通过 `LD_PRELOAD` 加载 `libcudart_proxy.so`。代理库导出与 CUDA Runtime API 相同的 C 符号，例如 `cudaMalloc`、`cudaMemcpy`、`cudaLaunchKernel`、`cudaStreamSynchronize` 等。应用原本进入 `libcudart.so` 的调用会优先进入代理库。

典型 `cudaMalloc` 路径如下：

```text
application cudaMalloc(&ptr, size)
  -> libcudart_proxy.so::cudaMalloc
  -> EnsureSession()
  -> gRPC Malloc(session_id, size)
  -> server cuMemAlloc
  -> return virtual pointer
```

kernel launch 路径额外依赖 CUDA fatbin 和 function registration。代理库记录 host function pointer 到 module/function 的映射，launch 时把 grid、block、shared memory、stream 和参数数据转发给 server。server 解析 module/function 后调用 `cuLaunchKernel`。

## 4. Session 和资源虚拟化

每个 client 进程在 server 中对应一个 session。session 是资源隔离和生命周期管理的基本单位。

每个 session 保存：

- allocation table
- module table
- stream table
- event table
- private default stream
- pending error
- shared memory mapping
- ring worker

### 4.1 Virtual Pointer

client 侧看到的 device pointer 是 virtual pointer，不是真实 `CUdeviceptr`。server 在当前 session 内维护映射：

```text
virtual pointer -> { real CUdeviceptr, allocation size }
```

两个 session 可以拿到相同 virtual pointer 数值，但查表时总是先定位 session，因此不会访问到其他进程的显存。

allocation lookup 支持指针偏移。server 会找到包含目标 virtual pointer 的 allocation，并检查 `offset + size <= allocation.size`，从而支持 `dptr + offset` 这类 CUDA 程序常见写法。

### 4.2 Stream 和 Event

stream 和 event 也作为 session-local handle 管理：

```text
stream_id -> CUstream
event_id  -> CUevent
```

`stream == 0` 被映射到当前 session 的 private non-blocking default stream，而不是 CUDA 全局 null stream。这样可以避免多个 session 之间因为 null stream 隐式同步而互相阻塞。

## 5. 通信路径

项目把通信拆成三条路径。

### 5.1 gRPC 控制面

低频、结构化、生命周期强的请求走 gRPC：

- CreateSession / DestroySession
- Malloc / Free
- StreamCreate / EventCreate
- RegisterModule

这类请求的参数结构清晰，调用频率相对较低，适合用 protobuf 表达。

### 5.2 Shared Memory 数据面

H2D/D2H payload 可能很大。若直接放入 protobuf bytes，会产生序列化和额外 CPU copy。项目通过 POSIX shared memory 建立 data arena：

```text
H2D: user buffer -> shm arena -> cuMemcpyHtoDAsync
D2H: cuMemcpyDtoHAsync -> shm arena -> user buffer
```

server 还会尝试将 data arena 注册为 pinned host memory，以提升 GPU DMA 访问效率。若注册失败，系统自动回退到普通 host memory 路径。

### 5.3 SPSC Ring 快路径

kernel launch、D2D copy、sync 和 event query 等请求 payload 小但频率高。项目为每个 session 建立 SPSC ring：

```text
client producer -> ring entry -> server ring worker
```

ring 使用 acquire/release 内存序保证跨进程可见性。producer 写完整 entry 后发布 head；consumer 看到 head 后读取 entry，执行完成后写 done 和错误码。

## 6. 并发隔离

多进程并发时，server 按 session 隔离资源。一个进程的 allocation、module、stream、event、pending error 和 ring worker 不会与另一个进程共享。

同步 API 的边界也是 session：

- `cudaStreamSynchronize(stream)` 只等待当前 session 的目标 stream。
- `cudaDeviceSynchronize()` 只等待当前 session 的 default stream 和显式 streams。
- `cudaEventSynchronize(event)` 只等待当前 session 的 event。

server 不把 `cudaDeviceSynchronize()` 实现为全局 `cuCtxSynchronize()`。这样一个 session 中的长 kernel 不会阻塞另一个 session 的 device sync。

## 7. 代码结构

```text
client/
  libcudart_proxy.cpp      Runtime API proxy
  probe.cpp                small gRPC probe

server/
  main.cpp                 gRPC service and CUDA Driver API executor
  session_manager.h        session and resource tables
  ring_worker.h            ring worker helpers

proto/
  vgpu.proto               client/server protocol

shared/
  vgpu_shm_ring.h          shared-memory ring layout

tools/
  acceptance tests and benchmarks
```
