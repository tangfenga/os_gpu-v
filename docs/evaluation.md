# 测试与性能评估

## 1. 测试目标

测试围绕赛题验收要求设计，覆盖以下内容：

- Runtime API 是否能够被拦截。
- Host/Device 数据传输是否正确。
- kernel launch 是否能在 server 端 GPU 上执行。
- 多进程并发时资源是否隔离。
- 同步 API 是否只等待当前 session。
- ring fast path 是否能承受容量回绕和连续提交。
- 端到端性能损耗是否控制在 50% 以内。
- 长时间运行是否稳定。

## 2. 工具清单

`tools/` 只保留验收和复现实验需要的程序：

| 文件 | 用途 |
| --- | --- |
| `runtime_query_smoke.cu` | 查询 Runtime/Driver 基础信息 |
| `vector_add_smoke.cu` | 向量加法，验证 malloc、H2D、kernel、D2H |
| `memcpy_baseline.cu` | H2D、D2H、D2D 拷贝正确性 |
| `matrix_mul_benchmark.cu` | 矩阵乘法功能和性能测试 |
| `stream_async_smoke.cu` | stream 和 async memcpy 测试 |
| `event_timing_smoke.cu` | event record/sync/elapsed 测试 |
| `host_alloc_smoke.cu` | shm-backed host allocation 测试 |
| `concurrent_worker.cu` | 并发、错误、异常退出 worker |
| `cross_session_sync_test.cu` | 跨 session 同步隔离测试 |
| `fire_forget_stress.cu` | ring capacity、wrap-around、异步错误测试 |
| `stability_negative_smoke.cu` | double free、invalid stream/event 负例 |
| `run_acceptance_validation.py` | 总体验收编排脚本 |
| `run_cross_session_sync_test.py` | holder/probe 同步隔离编排脚本 |

## 3. 构建方式

核心程序：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

同时构建 CUDA 测试：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVGPU_BUILD_CUDA_TESTS=ON
cmake --build build -j
```

CUDA 测试程序需要使用 shared cudart。若手动编译，使用：

```bash
nvcc -std=c++17 -cudart shared tools/vector_add_smoke.cu -o /tmp/vgpu_vector_add_smoke
```

## 4. 基础功能测试

基础测试覆盖 CUDA 程序的典型执行路径：

```text
cudaMalloc
cudaMemcpy HostToDevice
kernel launch
cudaDeviceSynchronize
cudaMemcpy DeviceToHost
cudaFree
```

向量加法验证一维 kernel，矩阵乘法验证二维 grid/block、多 device pointer 参数和标量参数。Memcpy 测试覆盖 HostToDevice、DeviceToHost 和 DeviceToDevice 三个方向。

## 5. 并发测试

并发测试通过两个独立 client 进程同时运行 worker：

| 组合 | 结果 |
| --- | --- |
| vector + vector | pass |
| vector + matmul | pass |
| matmul + matmul | pass |

测试中两个进程的首个 virtual pointer 可以相同。server 按 session 查 allocation table，因此相同 virtual pointer 数值不会造成显存串扰。

## 6. 同步隔离测试

跨 session 同步测试由 holder/probe 两个进程组成：

```text
holder: 在 session A 中 launch 一个约 1.5 秒长 kernel
probe:  在 session B 中调用 cudaDeviceSynchronize()
```

测试结果：

```text
holder total_us ~= 1500294.555
probe  sync_us  ~= 68.608
```

probe 的同步耗时远小于 holder kernel 时间，说明 `cudaDeviceSynchronize()` 只等待当前 session 的 streams。

## 7. Ring Stress 和错误语义

`fire_forget_stress.cu` 连续提交 D2D 和 kernel 命令，覆盖 1、16、64、256、1024、2048 等规模。`2048` 超过当前 ring capacity，用于验证 ring wrap-around、entry 复用和 server drain。

错误语义覆盖：

- invalid D2D
- invalid stream sync
- pending error 不被后续错误覆盖

异步 fast path 中，server 后台发现的错误写入当前 session 的 pending error，并在后续 sync API 中返回。

## 8. 性能结果

矩阵乘法使用 native Runtime 和 proxy Runtime 运行同一测试程序。多轮 median-of-medians 结果：

| 模式 | median-of-medians |
| --- | ---: |
| Native Runtime | 约 230.343 us |
| Proxy Runtime | 约 281.283 us |
| proxy/native | 约 1.221x |

性能损耗约 22.1%，低于 50% 目标。

主要优化来源：

- H2D/D2H payload 使用 shared memory data arena，减少 protobuf 序列化和额外 copy。
- D2D 和 kernel launch 使用 ring fast path，减少 gRPC 固定往返开销。
- 默认 stream 私有化，避免跨 session 隐式同步。
- fire-and-forget 路径减少异步 submit latency。

## 9. 稳定性测试

稳定性脚本持续运行 10 分钟，每轮执行：

- 两个 vector worker 并发运行。
- 一个错误 worker 触发负例。
- 后续 worker 验证 server 仍能继续服务新 session。

结果：

```text
rounds=1452
failures=0
mean_round_ms=413.330
```

负例测试结果：

```text
double free -> invalid device pointer
invalid stream synchronize -> invalid resource handle
invalid event query -> invalid resource handle
```
