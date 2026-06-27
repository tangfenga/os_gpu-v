# 最终验收结果

测试日期：2026-06-27

测试环境：WSL/Linux 同一内核原型。server 进程可访问 NVIDIA GPU；client 进程通过 `LD_PRELOAD=build/libcudart_proxy.so` 访问 vGPU server。当前环境未安装 Docker，无法实测“不挂载 GPU 设备和真实 libcuda 的 client 容器”，因此客户机隔离采用同机双进程 session 隔离验证，并保留容器部署方法作为后续环境要求。

## 1. 已完成赛题要求

| 要求 | 当前结果 |
| --- | --- |
| 学习并梳理虚拟化和 CUDA 基本流程 | 已完成，见四阶段文档和基础教学文档 |
| 截获客户机 CUDA Runtime API | 已完成，`LD_PRELOAD` 拦截 Runtime 常用路径 |
| 转发 kernel 到宿主机 GPU 成功运行 | 已完成，`vector_add`、`matrix_mul` 均通过 |
| 只虚拟化进程需要的 GPU 资源 | 已完成，每个 client process 对应独立 session，按 session 保存 allocation/module/function/stream/event |
| Host/Device 数据传输正确 | 已完成，H2D/D2H/D2D 和 async 路径均通过 |
| 双进程并发 | 已完成，vector+vector、vector+matmul、matmul+matmul 均通过 |
| 性能优化 | 已完成 POSIX shm data arena、pinned shm、shm-backed host alloc、SPSC ring fast path、fire-and-forget kernel/D2D submit |
| 稳定性测试 | 已完成基础负例、ring capacity、异常退出和 10 分钟持续运行 |

未完成或受限项：

- 当前不是 KVM/QEMU 跨内核 VM 方案，POSIX Shared Memory 只适用于同一 Linux 内核内的进程。
- 当前环境没有 Docker，未实测无 GPU client 容器。后续可用容器去掉 `/dev/dxg`、`/usr/lib/wsl/lib/libcuda.so*` 等 GPU/driver 挂载，只挂载代理库和测试程序。
- 尚未实现 futex/eventfd 低 CPU 唤醒、完整 pinned host API 语义和旧版 Runtime launch API。

## 2. 新增测试程序

| 文件 | 作用 |
| --- | --- |
| `tools/concurrent_worker.cu` | 单个 client worker，支持 `vector`、`matmul`、`error`、`crash` 四种模式 |
| `tools/matrix_mul_benchmark.cu` | native/proxy 共用矩阵乘法 benchmark，验证 2D grid/block、多 kernel 参数、较大显存、H2D/kernel/D2H |
| `tools/run_acceptance_validation.py` | 自动运行矩阵对比、双进程并发、错误隔离、异常退出隔离和持续稳定性 |
| `tools/fire_forget_stress.cu` | 连续提交 1/16/64/256/1024/2048 条 D2D 和 kernel，覆盖超过 ring capacity 的压力 |
| `tools/stability_negative_smoke.cu` | double free、无效 stream、无效 event 负例 |

所有 CUDA 测试程序必须使用 shared cudart 编译，否则 `LD_PRELOAD` 可能无法稳定接管 Runtime API：

```bash
nvcc -std=c++17 -cudart shared tools/concurrent_worker.cu -o /tmp/vgpu_concurrent_worker
nvcc -std=c++17 -cudart shared tools/matrix_mul_benchmark.cu -o /tmp/vgpu_matrix_mul_benchmark
nvcc -std=c++17 -cudart shared tools/fire_forget_stress.cu -ldl -o /tmp/vgpu_fire_forget_stress
nvcc -std=c++17 -cudart shared tools/stability_negative_smoke.cu -o /tmp/vgpu_stability_negative_test
```

## 3. 双进程并发结果

运行命令：

```bash
VGPU_INIT_TRACE=1 ./build/vgpu_server 127.0.0.1:50052
./tools/run_acceptance_validation.py \
  --server 127.0.0.1:50052 \
  --proxy-lib /home/hyj/code/os_hyj/build/libcudart_proxy.so \
  --skip-stability
```

结果：

| 组合 | 单独运行 A/us | 单独运行 B/us | 并发 A/us | 并发 B/us | same first vptr | 结果 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| vector + vector | 2633.948 | 2733.821 | 9154.807 | 9650.961 | 是 | pass |
| vector + matmul | 3855.145 | 1156.842 | 2679.256 | 1024.271 | 是 | pass |
| matmul + matmul | 606.807 | 805.685 | 1534.224 | 895.395 | 是 | pass |

解释：

- `same_first_vptr=1` 是刻意验证项。两个独立进程首个虚拟指针值相同，但 server 查表时按 session 隔离，因此不会访问对方显存。
- 三组并发均完成结果校验，没有丢命令、覆盖或死锁。
- 部分并发耗时高于单跑，主要来自同一块物理 GPU 上两个 workload 争用 SM/拷贝资源和系统调度，不是跨 session 全局同步导致的正确性失败。
- 错误隔离通过：一个进程触发无效指针错误，另一个长时间 vector worker 仍正确完成。
- 异常退出隔离通过：一个进程 `_exit(2)` 不释放资源，后续新进程仍能正常创建 session 并运行 vector。

## 4. 矩阵乘法结果

测试覆盖：

- `dim3 block(16, 16)` 和二维 grid。
- 3 个 device pointer 参数和 1 个 `int n` 标量参数。
- `cudaMalloc` 分配 A/B/C 三块矩阵显存。
- H2D A/B、kernel launch、D2H C、stream sync。
- native 和 proxy 使用同一份 `/tmp/vgpu_matrix_mul_benchmark`，只通过是否设置 `LD_PRELOAD` 区分。

单轮快速对比：

```text
matrix_compare n=256 native_median_us=277.768 proxy_median_us=308.715 ratio=1.111 pass=1
```

10 轮 median-of-medians 对比：

| 模式 | median-of-medians/us | 说明 |
| --- | ---: | --- |
| Native Runtime | 约 230.343 | 10 次运行，每次 10 个 timed trials |
| Proxy Runtime | 约 281.283 | 10 次运行，每次 10 个 timed trials |
| 倍率 | 约 1.221x | 满足不超过 1.5x |

有一次长时间稳定性前置快速比较出现 `1.780x` 抖动，未作为最终结论。多轮统计显示稳态矩阵乘法满足 1.5x。

## 5. Ring Capacity 和错误语义

运行命令：

```bash
VGPU_DATA_PLANE=shm \
LD_PRELOAD=/home/hyj/code/os_hyj/build/libcudart_proxy.so \
VGPU_SERVER=127.0.0.1:50052 \
/tmp/vgpu_fire_forget_stress
```

结果摘要：

| 压力项 | count | submit/us | sync/us | total/us | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| D2D | 1 | 0.775 | 242.541 | 243.316 | ok |
| kernel | 1 | 13.116 | 117.918 | 131.034 | ok |
| D2D | 1024 | 142.421 | 8105.600 | 8248.021 | ok |
| kernel | 1024 | 437.582 | 10738.362 | 11175.944 | ok |
| D2D | 2048 | 11970.516 | 11520.646 | 23491.162 | ok |
| kernel | 2048 | 6667.787 | 13370.099 | 20037.886 | ok |

`count=2048` 超过当前 ring capacity，验证了 wrap-around、payload 生命周期、server drain 和最终结果正确性。

错误语义结果：

```text
error_case invalid_d2d returned unknown error (999)
error_case invalid_stream_device_sync returned unknown error (999)
error_case first_pending_error_not_overwritten returned unknown error (999)
error semantics ok
fire_forget_stress passed
```

说明：当前 fast path 对部分异步错误以 pending error 形式在后续 sync 暴露；对明显无效虚拟指针，也允许 client 在 submit 阶段直接返回错误。两种方式都限定在当前 session 内，不污染其他 session。

## 6. 稳定性结果

负例：

```text
double free returned 17 (invalid device pointer)
invalid stream synchronize returned 400 (invalid resource handle)
invalid event query returned 400 (invalid resource handle)
stability negative smoke test passed
```

10 分钟持续运行：

```bash
./tools/run_acceptance_validation.py \
  --server 127.0.0.1:50052 \
  --proxy-lib /home/hyj/code/os_hyj/build/libcudart_proxy.so \
  --stability-seconds 600
```

结果：

```text
stability_sustained seconds=600 rounds=1452 failures=0 mean_round_ms=413.330 pass=1
```

持续测试每轮执行两个独立 vector client 并发运行，并插入一个错误 client。10 分钟内无失败、无死锁、无资源清理导致的后续运行失败。

## 7. 客户机隔离方式

当前已验证：

- 同机两个独立 client 进程分别创建独立 session。
- 每个 session 独立拥有 allocation、module、function、stream、event、pending error。
- 两个进程首个虚拟指针值可以相同，但查表按 session 隔离，不能访问对方显存。
- 一个进程无效指针错误不影响另一个进程。
- 一个进程异常退出后，server 后续仍可服务新 session。

当前未实测：

- 无 GPU client 容器。当前环境 `which docker` 无结果，不能创建“无 `/dev/dxg`、无真实 `libcuda`”的 client 容器。

后续容器验证建议：

```bash
# server 在 WSL host 中运行
VGPU_INIT_TRACE=1 ./build/vgpu_server 0.0.0.0:50052

# client 容器只挂载代理库、测试程序和必要 libcudart shared object，
# 不挂载 /dev/dxg，不挂载 /usr/lib/wsl/lib/libcuda.so*
docker run --rm --network host \
  -v "$PWD/build/libcudart_proxy.so:/opt/vgpu/libcudart_proxy.so:ro" \
  -v /tmp/vgpu_vector_add_proxy_test:/opt/vgpu/vector_add:ro \
  -e LD_PRELOAD=/opt/vgpu/libcudart_proxy.so \
  -e VGPU_SERVER=127.0.0.1:50052 \
  ubuntu:22.04 /opt/vgpu/vector_add
```

预期：

- 不设置 `LD_PRELOAD` 时，client 侧 CUDA 程序因无 GPU driver/libcuda 无法运行。
- 设置 `LD_PRELOAD` 后，CUDA Runtime 调用进入代理库，通过 server 完成 vector_add、matrix_mul、memcpy。

## 8. 修改文件列表

本轮修改：

- `server/main.cpp`：修正 virtual pointer 分配为 per-session，允许不同 session 出现相同虚拟指针值但互不串扰。
- `tools/concurrent_worker.cu`：新增双进程 worker，覆盖 vector、matmul、error、crash。
- `tools/matrix_mul_benchmark.cu`：新增 native/proxy 共用矩阵乘法 benchmark。
- `tools/run_acceptance_validation.py`：新增验收编排脚本。
- `tools/fire_forget_stress.cu`：错误语义测试兼容 submit-time error 和 sync-time pending error。
- `tools/stability_negative_smoke.cu`：负例测试按“必须报错”语义判定，兼容不同合法错误码。
- `docs/final_acceptance_results.md`：本文档。

## 9. 最终结论

当前版本已覆盖赛题核心验收路径：Runtime API 拦截、显存虚拟化、kernel 转发、H2D/D2H/D2D、矩阵乘法、双进程并发、多 session 隔离、错误隔离、异常退出和 10 分钟稳定性。矩阵乘法稳态 proxy/native 倍率约 1.22x，满足不超过 1.5x 的目标。

当前主要限制是部署形态仍是 WSL/Linux 同内核 POSIX shm 原型，不是跨 KVM/QEMU guest/host 的真实 VM 共享内存方案；无 GPU client 容器也受当前环境缺少 Docker 限制，尚未实测。
