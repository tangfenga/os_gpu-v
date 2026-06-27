# 阶段二：WSL 资源虚拟化与并发闭环

阶段二目标是在 WSL 内完成完整 CUDA Runtime 常用路径：fatbin 注册、kernel launch、stream、session 清理和双进程并发。此阶段仍使用 gRPC/TCP 和 protobuf `bytes`，暂不引入 POSIX Shared Memory 数据面。

## 0. 当前进度

当前代码已经完成阶段二的主要功能闭环：kernel launch、双进程并发、per-session stream、`cudaMemcpyAsync` 基础版、event API 基础版和多 session 隔离验证。

已完成并验证：

- vGPU `vector_add` 通过。
- vGPU `matmul` 通过。
- vGPU `memcpy H2D / D2H / D2D` 通过。
- fatbin 注册、kernel symbol 映射、参数打包、`cudaLaunchKernel` 转发已形成闭环。
- 双进程 `vector_add` 并发通过。
- 双进程 `vector_add + vector_add`、`vector_add + matrix_mul`、`matrix_mul + matrix_mul` 并发通过。
- 两个进程首个虚拟指针值相同仍互不串扰，证明 allocation 查表按 session 隔离。
- 一个进程触发无效指针错误或异常退出，不影响另一个 session 和后续新 session。
- 显式 stream async 测试通过。
- event timing 测试通过。
- double free、无效 stream、无效 event 稳定性测试通过。
- session 显存配额测试通过。
- client 被 `kill -9` 后，server timeout reaper 能自动清理 session。
- server 已输出基础性能日志。

尚未完成：

- 更完整的性能统计拆分：当前已有 `elapsed_us`，尚未拆成 `rpc_us` 和 `cuda_us`。
- 更完整的稳定性测试矩阵：当前已覆盖无效 vptr、无效 stream/event、double free、异常退出和 10 分钟持续运行；后续可补更多 API 组合。

## 1. 目标能力

阶段二完成后，以下测试必须通过：

- vGPU `vector_add`
- vGPU `matmul`
- vGPU `memcpy H2D / D2H / D2D`
- 双进程并发 kernel
- session 异常退出后资源释放

## 2. Runtime 注册路径

NVCC 编译的 Runtime 程序会注册 fatbin 和 kernel function。client 必须拦截：

- `__cudaRegisterFatBinary`
- `__cudaRegisterFatBinaryEnd`
- `__cudaRegisterFunction`
- `__cudaUnregisterFatBinary`
- `__cudaInitModule`
- `__cudaPushCallConfiguration`
- `__cudaPopCallConfiguration`
- `cudaLaunchKernel`

client 保存：

```text
fatbin_handle -> server module_id
host_fun_id -> device_name
host_fun_id -> module_id
host_fun_id -> param_sizes
```

server 执行：

```text
RegisterModule -> cuModuleLoadFatBinary / cuModuleLoadData
LaunchKernel -> cuModuleGetFunction
LaunchKernel -> cuLaunchKernel
```

当前已验证日志形态：

```text
[cudart_proxy] registered module id=3 fatbin_bytes=2728
[cudart_proxy] registered kernel name=_Z10vector_addPKfS0_Pfi module=3 params=4
[vgpu] op=LaunchKernel sid=3 module=3 kernel=_Z10vector_addPKfS0_Pfi grid=(4096,1,1) block=(256,1,1) args=4 result=CUDA_SUCCESS: no error
```

## 3. Kernel 参数处理

`cudaLaunchKernel` 的问题是 `void **args` 不携带每个参数大小。实现顺序：

1. 已支持 nvcc triple-chevron 路径中的 `__cudaPushCallConfiguration` / `__cudaPopCallConfiguration` / `cudaLaunchKernel`。
2. client 从 fatbin 中尝试解析 `.entry` 参数声明。
3. fatbin 内 PTX 文本被压缩或截断时，fallback 到 Itanium C++ mangled name，推断常见指针和标量参数大小。
4. server launch 前把命中 allocation table 的 8 字节 vptr 参数替换成真实 `CUdeviceptr`。

server launch 前要把 device pointer 参数从 vptr 翻译成真实 `CUdeviceptr`：

```text
arg raw value == vptr
  -> lookup allocations[vptr]
  -> replace with real CUdeviceptr
```

如果参数大小等于指针大小且 raw value 命中当前 session 的 allocation table，可按 device pointer 处理。否则按普通标量传入。

当前参数处理限制：

- 支持常见指针参数、`int`、`float`、`double`、常见 8 字节整数。
- 不支持结构体按值传参的通用 ABI 解析。
- 不支持复杂模板 kernel 的完整 signature 解析。
- 不支持用户自定义 stream，当前只支持默认 stream / null stream。

## 4. Session 资源隔离

每个 client 进程一个 session：

```text
session_id
allocations: vptr -> real_ptr
streams: vstream -> CUstream
events: vevent -> CUevent
modules: module_handle -> CUmodule
functions: host_fun_id -> CUfunction
memory_limit
memory_used
unhealthy flag
```

当前已实现：

- 每个 client 进程一个 session。
- 每个 session 独立 allocation table。
- 每个 session 独立 module table。
- 每个 session 独立 default stream。
- 每个 session 独立 stream table。
- 每个 session 独立 event table。
- `memory_limit` / `memory_used` 配额。
- `last_seen` 和后台 timeout reaper。
- session 正常退出时释放 allocation 并 unload module。

当前未实现：

- unhealthy flag。目前错误通过 per-session pending error 返回，不额外维护 session unhealthy 状态。

隔离规则：

- 一个 session 不能访问另一个 session 的 vptr。
- 一个 session 不能销毁另一个 session 的 stream/event/module。
- allocation 越界 memcpy 返回 `cudaErrorInvalidValue` 或 `cudaErrorInvalidDevicePointer`。
- double free 返回 CUDA Runtime 错误；当前验证中返回 `cudaErrorInvalidDevicePointer`。
- 超出配额返回 `cudaErrorMemoryAllocation`。

## 5. Stream 设计

每个 session 创建一个 default stream：

```text
client stream 0 -> session.default_stream
```

不要把所有 session 的默认 stream 都映射到 null stream，否则会引入跨 session 隐式同步。

当前实现已经为每个 session 创建 `CU_STREAM_NON_BLOCKING` default stream。client 传 `stream == nullptr` 时映射到当前 session 的 default stream，不再把所有 session 合并到 CUDA null stream。

支持：

- `cudaStreamCreate`
- `cudaStreamCreateWithFlags`
- `cudaStreamDestroy`
- `cudaStreamSynchronize`
- `cudaMemcpyAsync`

server 映射：

```text
cudaStreamCreate -> cuStreamCreate
cudaStreamDestroy -> cuStreamSynchronize + cuStreamDestroy
cudaStreamSynchronize -> cuStreamSynchronize
cudaMemcpyAsync H2D -> cuMemcpyHtoDAsync
cudaMemcpyAsync D2D -> cuMemcpyDtoDAsync
```

D2H Async 初版可保守处理：server 完成拷贝后返回，后续再做真正异步回调。

## 6. 同步语义

`cudaDeviceSynchronize` 推荐同步当前 session 拥有的所有 stream，而不是直接 `cuCtxSynchronize`。如果初版使用 `cuCtxSynchronize`，必须在日志和文档中说明它会影响其他 session。

`cudaStreamSynchronize` 只同步指定 vstream 对应的真实 `CUstream`。

`cudaGetLastError` 和 `cudaPeekAtLastError` 仍由 client thread-local 状态维护。

当前状态：

- `cudaGetLastError` / `cudaPeekAtLastError` 已由 client thread-local 维护。
- `cudaDeviceSynchronize` 已改为同步当前 session 拥有的 default stream 和显式 stream。
- `cudaStreamSynchronize` 已实现，只同步指定 vstream。

## 7. 清理顺序

session 正常退出或超时后：

```text
mark session closing
stop accepting new requests
synchronize streams
destroy events
destroy streams
free allocations
unload modules
erase session
```

server 退出时才释放 primary context：

```cpp
cuDevicePrimaryCtxRelease(device);
```

当前状态：

- 正常退出时，client `atexit` 调用 `DestroySession`。
- server 在 `DestroySession` 中同步并销毁 stream/event，释放 allocation 并 unload module。
- server 后台 reaper 根据 `VGPU_SESSION_TIMEOUT_MS` 清理长时间无请求的 session。
- 已验证 client 被 `SIGKILL` 后，server 输出 `CleanupSession reason=timeout`。

## 8. 阶段二测试

功能测试：

```bash
nvcc -cudart shared tools/vector_add_smoke.cu -o /tmp/vgpu_vector_add_proxy_test
nvcc -cudart shared tools/matmul_baseline.cu -o /tmp/vgpu_matmul_proxy_test
nvcc -cudart shared tools/memcpy_baseline.cu -o /tmp/vgpu_memcpy_proxy_test
nvcc -cudart shared tools/stream_async_smoke.cu -o /tmp/vgpu_stream_async_test
nvcc -cudart shared tools/event_timing_smoke.cu -o /tmp/vgpu_event_timing_test
nvcc -cudart shared tools/stability_negative_smoke.cu -o /tmp/vgpu_stability_negative_test
nvcc -cudart shared tools/memory_limit_smoke.cu -o /tmp/vgpu_memory_limit_test
nvcc -cudart shared tools/session_timeout_smoke.cu -o /tmp/vgpu_session_timeout_test

./build/vgpu_server 127.0.0.1:50051

LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_vector_add_proxy_test
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_matmul_proxy_test
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_memcpy_proxy_test
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_stream_async_test
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_event_timing_test
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_stability_negative_test
VGPU_MEMORY_LIMIT=1024 LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_memory_limit_test
```

已验证输出：

```text
vector_add smoke test passed: 1048576 elements
matmul baseline passed: 128x128
memcpy baseline passed: 4194304 bytes H2D/D2D/D2H
stream async smoke test passed: 1048576 elements
event timing smoke test passed: 6.962 ms
stability negative smoke test passed
memory limit smoke test passed
```

并发测试：

```bash
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_vector_add_proxy_test &
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so VGPU_SERVER=127.0.0.1:50051 /tmp/vgpu_vector_add_proxy_test &
wait
```

并发测试已通过，两个进程均输出：

```text
vector_add smoke test passed: 1048576 elements
```

异常测试：

- kill client 后 server timeout 释放资源。
- double free 不崩溃，返回 `cudaErrorInvalidDevicePointer`。
- 无效 stream/event 不崩溃，返回 `cudaErrorInvalidResourceHandle`。
- 超出显存配额返回 `cudaErrorMemoryAllocation`。

timeout 清理测试：

```bash
VGPU_SESSION_TIMEOUT_MS=1000 ./build/vgpu_server 127.0.0.1:50051
timeout -s KILL 1 env LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so \
  VGPU_SERVER=127.0.0.1:50051 \
  /tmp/vgpu_session_timeout_test
```

server 预期日志：

```text
[vgpu] op=CleanupSession sid=1 reason=timeout
```

## 9. 阶段二验收标准

- [x] vector add 正确。
- [x] matmul 正确。
- [x] memcpy 正确。
- [x] 双进程并发正确。
- [x] D2D 不回传 client。
- [x] 每个 session 独立 stream/module/allocation/event 基础资源表。
- [x] session 正常退出后 GPU 显存释放。
- [x] session 异常退出后 timeout 清理。
- [x] 有基础性能日志：elapsed_us、bytes、session_id、op、cuda_error。
- [ ] 更细粒度性能日志：rpc_us、cuda_us 分离统计。

阶段二下一步优先级：

1. 补更严格的无效 vptr memcpy、跨 session vptr 访问、长时间压力测试。
2. 将性能日志拆成 `rpc_us`、`cuda_us`、排队时间和数据大小统计。
3. 准备阶段三 POSIX Shared Memory 数据面接入。
