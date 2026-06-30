# 轻量化 GPU 进程虚拟化最终报告

## 1. 项目概述

本项目实现了 CUDA Runtime API 层的轻量化 GPU 进程虚拟化原型。用户 CUDA 程序通过 `LD_PRELOAD` 加载代理库后，常用 Runtime API 调用会被代理库截获，并转发给 server。server 运行在可访问 NVIDIA GPU 和 CUDA Driver API 的环境中，负责创建真实 CUDA 资源并执行 kernel。

项目以 GPU 用户进程为基本单位组织虚拟化状态。每个 client 进程对应 server 中的一个 session。session 内维护该进程的 allocation、module、stream、event、shared memory 和 pending error。这样可以在不迁移完整 GPU 硬件状态的情况下，保留 CUDA 进程运行所需的关键资源和数据。

## 2. 对赛题任务的完成情况

| 赛题任务 | 完成情况 |
| --- | --- |
| 学习虚拟化与外设数据处理 | 采用 Runtime API 级虚拟化边界，client/server 分离，server 统一管理真实 GPU 资源 |
| 梳理 GPU 计算流程和数据结构 | 实现 allocation、module、function、stream、event、kernel args、pending error 的虚拟化 |
| 截获客户进程算子并转发到宿主 GPU | 通过 `LD_PRELOAD` 截获 Runtime API，通过 gRPC、shared memory 和 ring 转发请求 |
| 针对数据转发和并发同步优化性能 | 使用 shared memory data arena 和 per-session SPSC ring，device sync 限定在当前 session |

## 3. 核心设计

### 3.1 API 层虚拟化

本项目选择 CUDA Runtime API 作为虚拟化边界。用户程序仍然调用 `cudaMalloc`、`cudaMemcpy`、kernel launch、stream/event API。代理库负责把这些调用转换成项目内部协议，server 再用 CUDA Driver API 执行真实操作。

### 3.2 显存虚拟化

`cudaMalloc` 返回给 client 的不是真实 `CUdeviceptr`，而是 session-local virtual pointer。server 在当前 session 的 allocation table 中保存：

```text
virtual pointer -> real CUdeviceptr
```

后续 memcpy 和 kernel 参数处理时，server 会把 virtual pointer 翻译成真实 `CUdeviceptr`。该设计避免把真实 GPU 地址暴露给 client，也使每个进程的显存资源可以独立管理。

### 3.3 Kernel 转发

代理库截获 CUDA fatbin 和 function registration，记录 host function 与 kernel name/module 的关系。kernel launch 时，client 将 grid、block、shared memory、stream 和参数数据发送给 server。server 加载 module，查找 function，替换参数中的 virtual pointer，然后调用 `cuLaunchKernel`。

### 3.4 多进程隔离

每个进程有独立 session：

- 独立 allocation table
- 独立 module table
- 独立 stream/event table
- 独立 private default stream
- 独立 pending error
- 独立 ring worker

`cudaDeviceSynchronize()` 只同步当前 session 的 streams。跨 session 测试表明，一个 session 的长 kernel 不会阻塞另一个 session 的 device sync。

## 4. 性能优化

项目将通信拆为三类路径：

| 路径 | 适用场景 |
| --- | --- |
| gRPC | session、malloc/free、module、stream/event 创建等低频控制请求 |
| POSIX shared memory | H2D/D2H 大块数据 |
| SPSC ring | kernel launch、D2D、sync、event query 等高频小命令 |

H2D/D2H 走 shared memory，避免把大块数据放入 protobuf bytes。kernel launch 和 D2D 走 ring，减少 gRPC 往返延迟。server 还会尝试将 shared memory data arena 注册为 pinned host memory。

## 5. 验收结果

项目完成了 CUDA Runtime API 拦截、向量加法、矩阵乘法、H2D/D2H/D2D 拷贝、stream async、event timing、双进程并发、跨 session 同步隔离、ring wrap-around stress、负例错误处理和 10 分钟稳定性测试。矩阵乘法稳态测试中，native Runtime 的 median-of-medians 约为 230.343 us，proxy Runtime 约为 281.283 us，proxy/native 约为 1.221x，虚拟化损耗约 22.1%，满足赛题中性能损耗控制在 50% 以内的目标。

## 6. 工程结构

```text
client/     Runtime API proxy
server/     vGPU server and session resources
proto/      gRPC protocol
shared/     shared-memory ring definitions
tools/      acceptance tests and benchmarks
docs/       design, evaluation, final report
```
