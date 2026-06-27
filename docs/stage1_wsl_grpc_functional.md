# 阶段一：WSL 单机 gRPC 功能闭环

阶段一目标是在 WSL 内先跑通最小功能闭环。此阶段不追求真实 host/guest 隔离，也不实现 POSIX Shared Memory 数据面。重点是让 CUDA Runtime API 拦截库、gRPC 协议、server session、虚拟显存表和 CUDA Driver API 执行路径可用。

## 1. 拓扑

```text
WSL Ubuntu
├── CUDA 测试程序
├── libcudart_proxy.so
│   └── gRPC client
└── vgpu_server
    ├── gRPC service
    ├── session table
    ├── vptr allocation table
    └── CUDA Driver API
```

通信地址：

```text
VGPU_SERVER=127.0.0.1:50051
```

启动方式：

```bash
./build/vgpu_server 127.0.0.1:50051

LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so \
VGPU_SERVER=127.0.0.1:50051 \
/tmp/runtime_query_smoke
```

## 2. 当前实现状态

阶段一已经完成，并且当前代码已经超过阶段一边界，进入阶段二的 kernel launch 基础闭环。

已完成工程文件：

- `CMakeLists.txt`
- `proto/vgpu.proto`
- `server/main.cpp`
- `client/probe.cpp`
- `client/libcudart_proxy.cpp`
- `tools/runtime_query_smoke.cu`
- `tools/vector_add_smoke.cu`
- `tools/memcpy_baseline.cu`
- `tools/matmul_baseline.cu`

已完成构建目标：

- `build/vgpu_server`
- `build/vgpu_client_probe`
- `build/libcudart_proxy.so`

已完成 Runtime 查询拦截：

- `cudaGetDeviceCount`
- `cudaGetDevice`
- `cudaSetDevice`
- `cudaGetDeviceProperties_v2`
- `cudaRuntimeGetVersion`
- `cudaDriverGetVersion`
- `cudaGetLastError`
- `cudaPeekAtLastError`
- `cudaGetErrorString`
- `cudaGetErrorName`

已完成显存和 memcpy 拦截：

- `cudaMalloc`
- `cudaFree`
- `cudaMemcpy`
- `cudaDeviceSynchronize`

已完成 server CUDA Driver API 路径：

- `cuInit`
- `cuDeviceGet`
- `cuDevicePrimaryCtxRetain`
- `cuCtxSetCurrent`
- `cuMemAlloc`
- `cuMemFree`
- `cuMemcpyHtoD`
- `cuMemcpyDtoH`
- `cuMemcpyDtoD`
- `cuCtxSynchronize`

已完成 gRPC RPC：

- `CreateSession`
- `DestroySession`
- `GetDeviceCount`
- `GetDeviceProperties`
- `Malloc`
- `Free`
- `Memcpy`
- `DeviceSynchronize`

当前验证结果：

```text
memcpy baseline passed: 4194304 bytes H2D/D2D/D2H
```

## 3. 已实现的阶段一代码

### 3.1 proto

`proto/vgpu.proto` 已经包含：

```proto
rpc Malloc(MallocRequest) returns (MallocReply);
rpc Free(FreeRequest) returns (StatusReply);
rpc Memcpy(MemcpyRequest) returns (MemcpyReply);
rpc DeviceSynchronize(DeviceSynchronizeRequest) returns (StatusReply);
```

当前未实现 `cudaMemset`，它不属于阶段一验收必需项。

核心消息包含：

```text
MallocRequest: session_id, size
MallocReply: cuda_error, message, device_ptr
FreeRequest: session_id, device_ptr
MemcpyRequest: session_id, kind, dst_vptr, src_vptr, host_data, count
MemcpyReply: cuda_error, message, host_data
```

### 3.2 client 拦截实现

`client/libcudart_proxy.cpp` 已经实现：

- `cudaMalloc`
- `cudaFree`
- `cudaMemcpy`
- `cudaDeviceSynchronize`

处理规则：

- `cudaMalloc` 发送 size，返回 vptr，写入 `*devPtr`。
- `cudaFree` 把 pointer 当 vptr 发给 server。
- H2D：client 从用户 host pointer 读取 `count` 字节，放入 `host_data`。
- D2H：client 请求 server 返回 `host_data`，再 memcpy 到用户 host pointer。
- D2D：client 只传 src/dst vptr 和 count，不传 host data。
- H2H：client 本地 memcpy，不发 RPC。

### 3.3 server CUDA Driver API 实现

当前实现暂时集中在 `server/main.cpp`，没有拆成独立 `.h/.cpp` 模块。后续进入并发和 POSIX shm 数据面前，建议再拆分：

```text
server/session.h
server/session.cpp
server/cuda_executor.h
server/cuda_executor.cpp
server/memory_manager.h
server/memory_manager.cpp
```

当前 Driver API 初始化：

```cpp
cuInit(0);
cuDeviceGet(&device, 0);
cuDevicePrimaryCtxRetain(&ctx, device);
cuCtxSetCurrent(ctx);
```

当前每次 CUDA 操作前都会确保：

```cpp
cuCtxSetCurrent(primary_ctx);
```

`cudaMalloc` 映射：

```text
cuMemAlloc -> real CUdeviceptr
generate vptr
allocations[vptr] = {real_ptr, size}
return vptr
```

`cudaMemcpy` 映射：

```text
H2D -> cuMemcpyHtoD
D2H -> cuMemcpyDtoH
D2D -> cuMemcpyDtoD
```

## 4. 阶段一测试命令

基础构建：

```bash
cmake -S . -B build
cmake --build build -j
```

原生 CUDA 基线：

```bash
bash tools/run_cuda_baselines.sh
```

Runtime 查询：

```bash
nvcc -cudart shared tools/runtime_query_smoke.cu -o /tmp/runtime_query_smoke
./build/vgpu_server 127.0.0.1:50051
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so \
VGPU_SERVER=127.0.0.1:50051 \
/tmp/runtime_query_smoke
```

memcpy 测试目标：

```bash
nvcc -cudart shared tools/memcpy_baseline.cu -o /tmp/vgpu_memcpy_proxy_test
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so \
VGPU_SERVER=127.0.0.1:50051 \
/tmp/vgpu_memcpy_proxy_test
```

## 5. 阶段一验收标准

- [x] 查询类 API 通过 gRPC 返回。
- [x] `cudaMalloc` / `cudaFree` 通过 vptr 映射到 server allocation。
- [x] H2D / D2H / D2D 正确。
- [x] `memcpy_baseline` 在 LD_PRELOAD 下通过。
- [ ] 更完整的性能日志：rpc_us、cuda_us、bytes、session_id、op、cuda_error。
- 不要求 kernel launch。
- 不要求 KVM/QEMU。
- 不要求 POSIX Shared Memory 数据面。

说明：当前代码已经实现 kernel launch 基础路径，因此功能上已经超过阶段一验收。
