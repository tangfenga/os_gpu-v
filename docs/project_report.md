# os_gpu-v 项目报告

## 1. 目标描述

本项目选择的赛题编号为 `proj43`，题目方向是轻量化 GPU 虚拟化。赛题希望以 GPU 运行进程为基本单位，根据进程实际需要迁移必要的 CUDA 运行信息，降低完整 GPU 虚拟化中大量设备状态和数据迁移带来的开销。

我们小组的目标是做出一个能跑通 CUDA 典型工作流的原型系统：用户程序仍然按普通 CUDA Runtime API 写法调用 `cudaMalloc`、`cudaMemcpy` 和 kernel launch；代理库截获这些调用后，把必要的显存信息、kernel 信息和同步信息发给 server；server 在可访问 NVIDIA GPU 的环境中执行真实 CUDA Driver API。这样，用户进程侧看到的是 CUDA Runtime 语义，真实 GPU 资源由 server 统一管理。

项目最终实现的是一个 CUDA Runtime API 层的 GPU 进程虚拟化原型。用户程序通过 `LD_PRELOAD` 加载 `libcudart_proxy.so` 后，常用 CUDA Runtime API 会被代理库截获。代理库为每个用户进程创建一个 session，将显存分配、数据拷贝、kernel launch、stream/event 同步等请求转发给 `vgpu_server`。server 侧使用 CUDA Driver API 操作真实 NVIDIA GPU，并按 session 维护每个进程的虚拟资源。

我们把重点放在赛题强调的“以 GPU 运行进程为基本迁移单位”和“仅迁移必要数据信息”上。项目把 CUDA 进程实际使用到的资源抽象成 virtual pointer、module、stream、event、pending error 和 shared-memory 数据块，在 Runtime API 层形成一套轻量化的资源表示。

## 2. 比赛题目分析和相关资料调研

赛题背景中提到，GPU 在云环境中价格较高，多用户共享 GPU 是实际需求；但完整 GPU 虚拟化会涉及大量设备状态和数据迁移，容易带来明显延迟。因此，我们最开始没有直接写代码，而是先把题目拆成两层：一层是“CUDA 程序真正依赖哪些运行时状态”，另一层是“这些状态如何在 client 和 server 之间表达”。

CUDA 程序的一般流程包括查询设备、分配显存、Host 到 Device 拷贝、kernel launch、Device 到 Host 拷贝、stream/event 同步以及资源释放。围绕这个流程，我们重点阅读了 CUDA Runtime API 和 CUDA Driver API 文档，对比两类接口的职责：Runtime API 更接近用户程序，适合作为拦截入口；Driver API 控制更细，适合在 server 端执行真实 GPU 操作。这个判断基本决定了项目的整体路线：client 侧模拟 Runtime，server 侧执行 Driver。

CRIUgpu 和 Cricket 给我们的主要参考是：GPU 进程迁移不能只看 CPU 内存，必须识别 CUDA 运行时中的显存、kernel、stream、event 等状态；同时，数据面传输不能全部走通用 RPC，否则大块 Host/Device 数据会成为瓶颈。结合这些资料，我们确定了 Runtime API 拦截、session 资源表、shared memory 数据面和 ring fast path 这几项主要设计。cuda-gdb 和 Nsight Compute 的资料主要帮助我们理解 CUDA 程序调试、kernel 执行和性能观察方式，虽然项目中没有把这些工具作为运行时依赖，但它们对分析 CUDA 执行流程有帮助。

本项目参考资料包括：

- CUDA Runtime API Manual
- CUDA Driver API Manual
- cuda-gdb Manual
- Nsight Compute
- CRIUgpu
- Cricket

## 3. 系统框架设计

系统由 client 代理库、server、协议层和共享内存 fast path 组成。

```text
CUDA application
      |
      | LD_PRELOAD
      v
libcudart_proxy.so
      |
      | gRPC control path
      | POSIX shared memory data path
      | per-session SPSC ring
      v
vgpu_server
      |
      | CUDA Driver API
      v
NVIDIA GPU
```

client 侧的 `libcudart_proxy.so` 导出 CUDA Runtime API 同名符号，例如 `cudaMalloc`、`cudaMemcpy`、`cudaLaunchKernel`、`cudaStreamSynchronize` 等。用户程序运行时通过动态链接机制优先进入代理库。代理库负责创建 session、维护 kernel registration 信息、管理 shared memory block，并把请求转换成 gRPC 请求或 ring entry。

server 侧以 session 为隔离单位。每个 client 进程对应一个 `SessionState`，其中包含 allocation table、module table、stream table、event table、private default stream、pending error、shared memory mapping 和 ring worker。server 收到请求后先定位 session，再在该 session 内查找虚拟资源并调用 CUDA Driver API。

显存虚拟化使用 virtual pointer。`cudaMalloc` 返回给 client 的不是真实 `CUdeviceptr`，而是 session-local handle。server 在 allocation table 中保存 virtual pointer 到真实 `CUdeviceptr` 的映射。后续 memcpy 或 kernel 参数中出现 virtual pointer 时，server 再把它翻译成真实 GPU 地址。

数据传输分为控制面和数据面。CreateSession、Malloc、Free、RegisterModule 这类低频控制请求走 gRPC。H2D/D2H 大块数据走 POSIX shared memory data arena。kernel launch、D2D copy、sync 等高频小命令走 per-session SPSC ring，以减少 RPC 固定开销。

多进程隔离通过 session 完成。每个 session 有独立资源表和 private default stream。`cudaDeviceSynchronize()` 只同步当前 session 拥有的 streams，不把其他 session 的 GPU 任务纳入等待范围。

## 4. 开发计划

项目开发按“先跑通功能，再补隔离，最后优化性能和稳定性”的顺序推进。

第一阶段先完成最小闭环：构建 gRPC server，代理 `cudaMalloc`、`cudaFree`、`cudaMemcpy`，使用 CUDA Driver API 在 server 侧完成显存分配和 H2D/D2H/D2D 拷贝。该阶段的目标是确认 Runtime API 拦截和 client/server 转发路径可行。

第二阶段加入 kernel launch。我们处理 CUDA fatbin 和 function registration，记录 host function 到 kernel name/module 的关系，在 server 端加载 module 并调用 `cuLaunchKernel`。这一阶段用 vector add 和 matrix multiply 验证 kernel 参数传递、grid/block 配置和多 device pointer 参数。

第三阶段完善 session 资源模型。server 为每个 client 进程建立独立 session，显存、module、stream、event 和 pending error 都放在 session 内管理。这个阶段重点解决多进程并发和资源隔离问题。

第四阶段做性能优化。大块 H2D/D2H payload 从 gRPC bytes 改为 POSIX shared memory；kernel launch、D2D 和 sync 走 per-session SPSC ring；server 尝试将 shared memory data arena 注册为 pinned host memory。最后补充压力测试、负例测试和 10 分钟稳定性测试。

## 5. 比赛过程中的重要进展

项目中比较关键的进展有四个。

第一个进展是完成 Runtime API 拦截。`LD_PRELOAD` 加载代理库后，用户程序中的 `cudaMalloc`、`cudaMemcpy` 等调用能够进入我们自己的函数。这一步让项目从普通 CUDA 程序变成可控的 API 转发系统。

第二个进展是建立 virtual pointer 和 session resource table。最初如果直接把真实 `CUdeviceptr` 暴露给 client，资源隔离和生命周期都不好处理。改为 virtual pointer 后，client 只持有虚拟 handle，server 按 session 翻译真实 GPU 资源，也方便处理 double free、invalid pointer 和进程退出后的资源回收。

第三个进展是解决跨 session 同步边界。CUDA 的 context 级同步容易把多个 session 的任务一起等待。项目把 `cudaDeviceSynchronize()` 改成只同步当前 session 的 default stream 和显式 stream，并用 holder/probe 测试验证：一个 session 中约 1.5 秒的长 kernel 不会阻塞另一个 session 的 device sync。

第四个进展是性能路径拆分。初版所有 payload 都走 RPC，功能可以跑通，但大数据和高频小命令开销明显。后续引入 shared memory data arena 和 SPSC ring 后，矩阵乘法稳态 proxy/native 约为 1.221x，性能损耗控制在赛题要求范围内。

## 6. 系统测试情况

测试程序集中放在 `tools/` 目录，保留的都是验收相关工具。

`runtime_query_smoke.cu` 用于验证 Runtime/Driver 基础查询。`vector_add_smoke.cu` 验证 `cudaMalloc`、H2D、kernel launch、D2H 和 `cudaFree` 的完整闭环。`memcpy_baseline.cu` 覆盖 HostToDevice、DeviceToHost、DeviceToDevice 三种拷贝。`matrix_mul_benchmark.cu` 验证二维 grid/block、多参数 kernel 和端到端性能。`stream_async_smoke.cu` 和 `event_timing_smoke.cu` 验证 stream/event 语义。`host_alloc_smoke.cu` 验证 host allocation 路径。`concurrent_worker.cu`、`cross_session_sync_test.cu`、`fire_forget_stress.cu` 和 `stability_negative_smoke.cu` 分别覆盖双进程并发、跨 session 同步隔离、ring wrap-around 和负例错误处理。

功能测试结果显示，vector add、matrix multiply、H2D/D2H/D2D memcpy、stream async、event timing 和 host alloc 路径均能通过。双进程并发测试覆盖 vector+vector、vector+matmul、matmul+matmul 三种组合，结果正确且没有资源串扰。跨 session 同步测试中，holder 进程运行约 1.5 秒长 kernel，probe 进程的 `cudaDeviceSynchronize()` 约几十微秒返回，说明同步边界按 session 生效。

性能测试使用矩阵乘法作为端到端指标。多轮统计中，native Runtime median-of-medians 约为 230.343 us，proxy Runtime 约为 281.283 us，proxy/native 约为 1.221x，虚拟化损耗约 22.1%。稳定性测试持续运行 10 分钟，共 1452 轮，失败数为 0。

## 7. 遇到的主要问题和解决方法

第一个问题是 kernel launch 的信息并不只来自 `cudaLaunchKernel` 本身。CUDA 程序启动时会注册 fatbin 和 function，kernel launch 时需要知道 module、kernel name、参数大小和参数数据。我们通过拦截 registration 过程保存 host function 到 module/function 的映射，再在 launch 时把必要信息发给 server。

第二个问题是 device pointer 不能直接跨进程使用。client 进程看到的地址和 server 进程中的 CUDA context 并不是同一个概念。解决方法是引入 virtual pointer，由 server 维护 session-local allocation table，并在 memcpy 和 kernel 参数处理时翻译成真实 `CUdeviceptr`。

第三个问题是多进程同步容易扩大边界。如果用 context 级同步，一个进程的 `cudaDeviceSynchronize()` 可能等待其他进程的 GPU 工作。我们改为收集当前 session 自己的 streams 并逐个同步，同时把默认流映射为 private non-blocking stream，避免 null stream 带来的隐式同步。

第四个问题是性能路径上的固定开销。大块数据走 protobuf bytes 会有序列化和多次 copy；高频 kernel launch 走 gRPC 会有固定往返延迟。对应的解决方法是：H2D/D2H 使用 shared memory data arena，kernel/D2D/sync 使用 per-session SPSC ring。

第五个问题是异步错误的返回时机。kernel launch 和 D2D fast path 可能先返回，server 后台执行时才发现错误。我们为每个 session 设置 pending error，后台错误写入当前 session，并在后续 stream/device sync 时返回。

## 8. 分工和协作

项目按三人小组的方式分工推进。

黄昱嘉主要负责 client 侧代理库，包括 `LD_PRELOAD` 拦截、Runtime API 包装、session 初始化、shared memory block 管理、kernel registration 记录和 last error 语义。

程昶斌主要负责 server 侧资源虚拟化，包括 gRPC service、SessionState、allocation/module/stream/event table、virtual pointer 翻译、CUDA Driver API 调用和资源清理。

石锐主要负责测试、性能优化和文档整理，包括 vector/matrix/memcpy 测试、双进程并发脚本、跨 session 同步测试、ring stress、稳定性测试、性能数据整理和项目报告。

实际开发中三个部分不是完全割裂的。比如 kernel launch 同时涉及 client registration、协议字段和 server module/function 查找；ring fast path 同时涉及 client submit、shared memory 数据结构和 server worker。因此每个阶段都会一起对接口、调试日志和测试输出，先保证功能正确，再根据 benchmark 调整路径。

## 9. 提交仓库目录和文件描述

```text
client/
  libcudart_proxy.cpp       CUDA Runtime API 代理库
  probe.cpp                 简单 gRPC 探测程序

server/
  main.cpp                  vGPU server 主体和 CUDA Driver API 执行逻辑
  session_manager.h         session 状态、资源表和生命周期管理
  ring_worker.h             ring worker 辅助逻辑

proto/
  vgpu.proto                client/server gRPC 协议

shared/
  vgpu_shm_ring.h           shared memory ring 数据结构

tools/
  runtime_query_smoke.cu
  vector_add_smoke.cu
  memcpy_baseline.cu
  matrix_mul_benchmark.cu
  stream_async_smoke.cu
  event_timing_smoke.cu
  host_alloc_smoke.cu
  concurrent_worker.cu
  cross_session_sync_test.cu
  fire_forget_stress.cu
  stability_negative_smoke.cu
  run_acceptance_validation.py
  run_cross_session_sync_test.py

docs/
  project_report.md         设计赛道项目报告

CMakeLists.txt              构建配置
README.md                   项目入口说明
```

`CMakeLists.txt` 默认构建 `vgpu_server`、`libcudart_proxy.so` 和 `vgpu_client_probe`。打开 `VGPU_BUILD_CUDA_TESTS=ON` 后，会构建保留的 CUDA 测试程序。

## 10. 比赛收获

通过这个项目，我们对 CUDA 程序的运行流程有了更具体的理解。以前只知道 `cudaMalloc`、`cudaMemcpy` 和 kernel launch 的使用方法，做完项目后才真正意识到 Runtime API 背后还有 module、function、stream、event、context、异步错误和资源生命周期等状态。

我们也加深了对虚拟化边界的理解。轻量化虚拟化不是简单地把所有东西都转发，而是要判断哪些状态是进程运行必须的，哪些可以通过 server 侧重建。virtual pointer 和 session resource table 是这个项目中最核心的抽象。

性能优化方面，我们学到不能只看“功能跑通”。同样的 API 转发，数据面走 protobuf 还是 shared memory，kernel launch 走 RPC 还是 ring，性能差别很明显。先画清楚数据路径，再减少 copy、序列化和固定往返开销，是这次优化中最有效的方法。

最后，测试对系统类项目非常关键。双进程并发、跨 session 同步、ring wrap-around、负例错误和长时间稳定性测试帮助我们发现并修正了很多边界问题。项目最终不仅能跑通示例程序，也能说明为什么多进程资源不会串扰、同步不会扩大到其他 session、异步错误能够按 session 返回。
