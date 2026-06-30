# os_gpu-v 项目报告

## 1. 目标描述

本项目选择的赛题编号为 `proj43`，题目方向是轻量化 GPU 虚拟化。赛题给出的核心问题是：GPU 在云环境中价格高、共享需求强，但完整 GPU 虚拟化会涉及大量设备状态、显存内容和运行时上下文迁移，迁移延迟很容易变大。因此，题目希望我们不要做一个“大而全”的 GPU 虚拟机，而是以 GPU 运行进程为单位，识别这个进程真正需要的 CUDA 运行信息，只迁移和转发必要的数据。

我们小组对题目的理解是：用户进程不应该感知后端 GPU 在哪里，也不应该修改原来的 CUDA 程序写法；系统需要在用户调用 CUDA Runtime API 时接管这些调用，把显存分配、数据传输、kernel launch、stream/event 同步等行为转成 server 端可以执行的请求。server 运行在能访问 NVIDIA GPU、CUDA driver 和 CUDA toolkit 的环境中，负责真正执行 CUDA Driver API，并把结果返回给用户进程。

最终实现的 `os_gpu-v` 是一个 CUDA Runtime API 层的轻量化 GPU 进程虚拟化原型。用户程序通过 `LD_PRELOAD` 加载 `libcudart_proxy.so` 后，常用 Runtime API 会先进入代理库。代理库为每个进程创建 session，并把请求转发给 `vgpu_server`。server 端按 session 管理资源表，在真实 GPU 上完成显存分配、数据拷贝和 kernel 执行。

项目的目标可以概括为四点：

- 功能上，支持 CUDA 程序常见路径，包括 device query、`cudaMalloc`、`cudaFree`、H2D/D2H/D2D memcpy、kernel launch、stream/event 和同步。
- 隔离上，按 GPU 运行进程建立 session，使多个用户进程的显存、stream、event 和错误状态互不串扰。
- 性能上，把大块数据和高频小命令从普通 RPC 路径中拆出来，使用 shared memory 和 ring fast path 降低传输开销。
- 文档和测试上，给出可复现的构建、运行、功能测试、并发测试、性能测试和稳定性测试说明，便于评审直接检查。

这个项目不是简单包装几个 CUDA API，而是围绕“一个 CUDA 进程实际依赖哪些状态”来设计资源模型。我们把运行中需要保留和转发的信息整理为 virtual pointer、module/function、stream、event、pending error、shared-memory data block 和 ring command。这样 client 侧只保存虚拟资源句柄，server 侧持有真实 GPU 资源，二者通过协议保持一致。

## 2. 比赛题目分析和相关资料调研

赛题背景强调虚拟化、外设驱动、GPU 计算流程和性能优化四个方面。我们开始时先没有急着写拦截库，而是把一个普通 CUDA 程序从源代码到 GPU 执行的过程拆开分析。

一个典型 CUDA 程序大致会经历以下步骤：先查询设备数量和属性，再用 `cudaMalloc` 分配 device memory；把输入数据通过 `cudaMemcpy` 从 host 拷到 device；调用 kernel，传入 grid/block 配置、shared memory 大小、stream 和参数数组；kernel 执行后再把结果从 device 拷回 host；最后同步、检查错误并释放资源。看起来用户代码只调用 Runtime API，但背后实际涉及 CUDA context、module、function、device pointer、stream、event、异步错误和资源生命周期。

围绕这些状态，我们阅读了 CUDA Runtime API 和 CUDA Driver API 文档。Runtime API 更接近用户程序，例如 `cudaMalloc`、`cudaMemcpy`、`cudaLaunchKernel` 都是用户最常写的接口，适合做 client 侧拦截入口。Driver API 更接近底层资源控制，例如 `cuMemAlloc`、`cuMemcpyHtoD`、`cuModuleLoadData`、`cuLaunchKernel`，适合放在 server 端执行真实 GPU 操作。这个对比决定了项目的整体架构：client 模拟 Runtime，server 调用 Driver。

我们也调研了 CRIUgpu 和 Cricket。CRIUgpu 的思路提醒我们，GPU 进程状态不能只看 CPU 内存，还要考虑 CUDA 上下文和显存对象；Cricket 给我们的启发是，可以把 CUDA API 调用转发到远端执行，并且需要特别处理数据传输路径。结合赛题“轻量化”的要求，我们没有把所有 GPU 状态都做成完整镜像，而是只为当前测试和典型 CUDA 运行流保留必要信息。

在性能资料方面，我们关注的重点不是单个 kernel 的计算性能，而是虚拟化路径本身的额外开销。大块 H2D/D2H 数据如果全部走 protobuf bytes，会出现序列化、内存拷贝和 RPC 往返成本；高频 kernel launch 如果每次都走普通 gRPC，也会让固定延迟变大。因此我们在设计阶段就把路径分成控制面和数据面：低频控制请求走 gRPC，大块数据走 POSIX shared memory，高频小命令走 per-session SPSC ring。

本项目主要参考资料如下：

- CUDA Runtime API Manual：用于确认 Runtime API 行为、返回值和错误语义。
- CUDA Driver API Manual：用于实现 server 端真实 GPU 操作和 module/kernel launch。
- cuda-gdb Manual：用于理解 CUDA 程序的调试方式和 kernel 运行信息。
- Nsight Compute：用于了解 CUDA kernel 性能观察方式。
- CRIUgpu：用于参考 GPU 进程 checkpoint/restore 中涉及的 CUDA 状态。
- Cricket：用于参考 CUDA API 远端执行和数据通路设计。

## 3. 系统框架设计

系统整体由四部分组成：client 代理库、gRPC 协议层、server 资源管理与执行层、shared memory/ring fast path。

```text
CUDA application
      |
      | LD_PRELOAD
      v
libcudart_proxy.so
      |
      | gRPC control path
      | POSIX shared memory data path
      | per-session SPSC ring fast path
      v
vgpu_server
      |
      | CUDA Driver API
      v
NVIDIA GPU
```

client 侧的核心是 `libcudart_proxy.so`。它导出 CUDA Runtime API 同名符号，用户程序运行时会因为 `LD_PRELOAD` 优先进入我们的代理函数。例如用户调用 `cudaMalloc(&ptr, size)` 时，代理库不会在 client 进程中分配真实 GPU 显存，而是向 server 发送 malloc 请求。server 分配真实 GPU memory 后返回一个 virtual pointer，client 把这个 virtual pointer 填回用户传入的 `ptr`。

server 侧的核心是 session。每个 client 进程连接 server 时创建一个独立 session。session 中保存 allocation table、module table、stream table、event table、private default stream、pending error、shared memory mapping 和 ring worker。这样，同一台 server 上可以同时服务多个用户进程，每个进程都只看到自己的虚拟 GPU 资源。

显存虚拟化使用 virtual pointer。真实 `CUdeviceptr` 只存在 server 进程中，client 拿到的是 session-local handle。后续 memcpy 或 kernel 参数中如果出现这个 handle，server 会先在当前 session 的 allocation table 中查找真实 device pointer，再调用 Driver API。这个设计解决了两个问题：第一，client 不需要知道 server 端真实地址；第二，不同 session 即使 virtual pointer 数值相似，也不会访问到彼此的资源表。

kernel launch 是项目中最关键的一段逻辑。CUDA Runtime 在程序启动阶段会注册 fatbin 和 host function，真正 launch 时只传入 host function 指针和参数数组。代理库需要记录 registration 信息，把 host function 映射到对应的 module/function。server 收到 launch 请求后，根据 module 和 function 信息加载 CUDA module，翻译参数中的 virtual pointer，然后调用 `cuLaunchKernel`。这部分工作让 vector add 和 matrix multiply 这类真实 kernel 能够在 server 侧执行。

数据路径分为三类：

- 控制面：CreateSession、Malloc、Free、RegisterModule 等低频请求走 gRPC，便于表达结构化信息。
- 数据面：H2D/D2H 大块 payload 使用 POSIX shared memory data arena，减少 protobuf bytes 带来的额外复制。
- 快路径：kernel launch、D2D copy 和部分同步命令走 per-session SPSC ring，减少高频小请求的 RPC 固定开销。

多进程并发依赖 session 隔离和 private default stream。CUDA 默认流有隐式同步语义，如果直接复用 context 级默认流，不同 client 进程可能互相影响。项目中每个 session 维护自己的 private non-blocking stream；`cudaDeviceSynchronize()` 只同步当前 session 的 stream 集合，不把其他 session 的任务纳入等待范围。

错误处理也按 session 保存。对于立即能判断的错误，例如 invalid device pointer、double free，可以直接返回 CUDA Runtime 对应错误码。对于异步路径上的错误，例如 ring worker 后台执行 kernel 或 D2D copy 时才发现的问题，会写入当前 session 的 pending error，并在后续 `cudaStreamSynchronize()`、`cudaDeviceSynchronize()` 或错误查询时返回。

## 4. 开发计划

项目开发按“最小闭环、kernel 支持、并发隔离、性能优化、验收整理”的顺序推进。这样做的原因是系统类项目很容易一开始就陷入很多接口细节，如果没有先跑通端到端闭环，后续很难判断问题出在 client、协议还是 server。

第一阶段是最小 CUDA 内存闭环。我们先搭建 gRPC server 和 client proxy，完成 `cudaMalloc`、`cudaFree` 和 `cudaMemcpy`。这一阶段只要求能让用户程序通过代理库申请显存、拷贝数据、释放资源。它验证了三个基本假设：`LD_PRELOAD` 能接管 Runtime API；server 能用 Driver API 操作真实 GPU；client/server 之间可以通过协议传递 CUDA 调用参数和返回值。

第二阶段是 kernel launch。我们加入 fatbin/function registration 记录、module 传输、function 查找和 kernel 参数翻译。这个阶段的测试从简单 vector add 开始，再扩展到 matrix multiply。vector add 主要验证单维 grid 和三个 device pointer 参数；matrix multiply 验证二维 grid/block、更多参数和更大的数据量。

第三阶段是 session 化和多进程隔离。初版功能跑通后，我们发现如果资源表是全局的，就很难保证多进程情况下的资源归属。因此 server 重构为每个 client 进程一个 session，所有 allocation、module、stream、event 和错误状态都挂在 session 下。这个阶段加入双进程并发和跨 session 同步测试。

第四阶段是性能优化。所有数据都走 gRPC 时功能正确，但大块 memcpy 和高频 kernel launch 的开销比较明显。我们把 H2D/D2H payload 改到 shared memory data arena，把 kernel/D2D/sync 请求改到 per-session SPSC ring，并在 server 侧加入 ring worker。优化目标不是追求超过 native，而是把虚拟化额外损耗控制在赛题要求的范围内。

第五阶段是验收整理。我们清理了仓库中一次性调试工具，只保留和功能、并发、性能、稳定性直接相关的测试程序；完善 CMake，使核心 target 可以通过标准 CMake 构建；补充 README 和项目报告，让评审能够从仓库入口直接看到题目编号、项目目标、构建方式、测试方式和最终结果。

## 5. 比赛过程中的重要进展

项目第一个关键进展是 Runtime API 拦截跑通。最初我们只是写了几个和 CUDA API 同名的函数，通过 `LD_PRELOAD` 验证用户程序确实会进入代理库。这个阶段看似简单，但意义很大，因为它说明用户程序可以保持原来的 CUDA 写法，虚拟化逻辑可以放在动态链接层完成。后面所有功能都建立在这个入口上。

第二个关键进展是显存从真实地址改为 virtual pointer。早期实现中我们尝试直接围绕真实 device pointer 思考，但很快发现它不适合作为 client 侧可见资源。真实 device pointer 属于 server 的 CUDA context，client 进程不能直接使用，也不应该暴露给其他 session。引入 virtual pointer 后，client 只持有一个虚拟句柄，server 在当前 session 的 allocation table 中翻译到真实 `CUdeviceptr`。这让资源隔离、double free 检查、invalid pointer 检查和退出清理都更清楚。

第三个关键进展是 kernel launch 端到端跑通。kernel launch 不是只转发一个函数名那么简单，CUDA Runtime 会在程序初始化时注册 fatbin 和 host function，launch 时再通过 host function 找到实际 kernel。我们在 client 侧记录 registration，在协议中传递 module/function 信息，在 server 端加载 module 并调用 `cuLaunchKernel`。vector add 跑通后，说明基本 kernel 参数翻译是正确的；matrix multiply 跑通后，说明更复杂的 grid/block 和参数组合也能工作。

第四个关键进展是跨 session 同步隔离。多进程并发不只是两个程序同时跑出正确答案，还要确认一个进程的同步不会把另一个进程的 GPU 工作也等待进去。我们为每个 session 使用 private default stream，并把 `cudaDeviceSynchronize()` 的语义限制在当前 session。cross-session 测试中，一个 holder 进程运行约 1.5 秒长 kernel，另一个 probe 进程调用 device sync 能在几十微秒级返回，说明同步边界没有扩大到其他 session。

第五个关键进展是 shared memory 和 ring fast path。初版所有请求都通过 gRPC 完成，调试方便但性能一般。后续我们把大块 H2D/D2H 数据放到 POSIX shared memory，client 和 server 只通过控制消息传递 block 信息；kernel launch、D2D copy 和部分同步命令进入 per-session SPSC ring，由 server 侧 worker 消费。这个改动后，矩阵乘法端到端 proxy/native 约为 1.221x，虚拟化损耗约 22.1%，达到赛题要求的 50% 以内目标。

第六个关键进展是补齐负例和稳定性测试。GPU 虚拟化原型不能只在正常输入下工作，还要能处理 invalid pointer、double free、invalid stream/event 等错误。我们加入 `stability_negative_smoke.cu` 测试负例场景，验证代理库对部分可识别错误（如 double free）能够返回 CUDA 错误码。稳定性测试连续运行 10 分钟，覆盖多轮 API 调用、资源创建释放和数据传输，最终失败数为 0。

## 6. 系统测试情况

测试程序集中放在 `tools/` 目录，按类型分为五个子目录：
- `tools/smoke/` — 基础功能测试，包括 `runtime_query_smoke.cu`（Runtime/Driver 查询）、`vector_add_smoke.cu`（malloc→H2D→kernel→D2H→free 闭环）、`memcpy_baseline.cu`（H2D/D2H/D2D 拷贝）、`host_alloc_smoke.cu`、`stream_async_smoke.cu` 和 `event_timing_smoke.cu`（stream/event 语义验证）。
- `tools/benchmark/` — `matrix_mul_benchmark.cu` 用于端到端性能对比。
- `tools/concurrency/` — `concurrent_worker.cu`（双进程并发）和 `cross_session_sync_test.cu`（跨 session 同步隔离）。
- `tools/stress/` — `fire_forget_stress.cu`（ring wrap-around 压力）和 `stability_negative_smoke.cu`（负例错误处理）。
- `tools/scripts/` — `run_acceptance_validation.py` 和 `run_cross_session_sync_test.py` 验证脚本。

并发和隔离测试中，双进程并发测试会同时启动两个通过 proxy 运行的 CUDA 进程，组合包括 vector+vector、vector+matmul、matmul+matmul，用来观察两个 session 是否可以并发运行、结果是否正确、资源是否串扰。跨 session 同步测试中，holder 进程提交长 kernel，probe 进程随后执行 `cudaDeviceSynchronize()`。结果显示 probe 不会等待 holder 的长 kernel，说明 session 同步隔离生效。

压力和负例测试中，`fire_forget_stress.cu` 主要压 ring wrap-around 和后台 worker 消费；`stability_negative_smoke.cu` 检查 double free、invalid device pointer、invalid stream、invalid event 等错误路径；acceptance 脚本把功能测试、并发测试、性能测试和稳定性测试串起来，减少人工重复操作。

验收结果可以用一段话概括：Runtime API 拦截、vector add、matrix multiply、H2D/D2H/D2D memcpy、stream async、event timing、双进程并发、跨 session 同步隔离、ring wrap-around stress、负例错误处理和 10 分钟稳定性测试均通过。矩阵乘法稳态性能中，native Runtime median-of-medians 约为 230.343 us，proxy Runtime median-of-medians 约为 281.283 us，proxy/native 约为 1.221x，虚拟化损耗约 22.1%，满足赛题中性能损耗控制在 50% 以内的目标。

这些测试覆盖了赛题给出的主要测试用例：向量加法和矩阵乘法用于验证 kernel launch；memcpy 测试用于验证 Host 到 Device、Device 到 Host、Device 到 Device 拷贝；双进程并发用于观察多用户场景下是否互相阻塞。除此之外，我们还增加了跨 session 同步、ring 压力和负例错误处理，主要是为了证明系统在边界情况下也能保持资源隔离和错误返回。

## 7. 遇到的主要问题和解决方法

第一个问题是 CUDA Runtime API 的拦截范围比最开始想象的更大。只拦截 `cudaMalloc`、`cudaMemcpy` 和 `cudaLaunchKernel` 不够，因为 kernel launch 依赖程序初始化时的 fatbin 和 function registration。如果没有记录这些注册信息，launch 时 server 不知道应该加载哪个 module、调用哪个 kernel。解决方法是在 client 侧补充 registration 拦截和映射表，把 host function、kernel name、module image 等信息保存下来，再在 launch 时随请求发送给 server。

第二个问题是 device pointer 的跨进程表达。client 进程中返回给用户的指针不可能直接作为 server 端真实 GPU 地址使用，反过来 server 端真实 `CUdeviceptr` 也不应该暴露给 client。我们最终使用 virtual pointer，并且让它只在当前 session 内有效。server 每次处理 memcpy 或 kernel 参数时都先查 allocation table，找不到就返回 invalid device pointer。这个设计也让负例测试中的伪造指针不会被解引用，而是被当成无效 handle 处理。

第三个问题是 kernel 参数翻译。CUDA kernel 参数数组里可能包含普通标量，也可能包含 device pointer。client 侧只能看到一段参数内存，server 侧必须知道哪些参数需要从 virtual pointer 翻译成真实 device pointer。我们通过记录 allocation table 和参数内容，在 server launch 前对参数进行重写，把属于当前 session 的 virtual pointer 替换为真实 `CUdeviceptr`，普通标量保持原样。vector add 和 matrix multiply 都依赖这一步。

第四个问题是默认流同步语义。CUDA 默认流在不同模式下可能带来隐式同步，如果处理不好，多进程并发会出现一个 session 的同步等待另一个 session 的工作。我们的解决方法是为每个 session 创建 private default stream，并且把 device synchronize 实现为同步当前 session 持有的 stream 集合。这样用户仍然看到 Runtime API 的同步结果，但同步边界被限制在自己的虚拟 GPU 进程内。

第五个问题是性能开销集中在数据传输和高频请求上。初版所有 payload 都走 gRPC，功能正确但复制次数和固定延迟比较明显。我们把 H2D/D2H 大块数据放到 shared memory data arena，避免把大块数据编码进 protobuf；把 kernel launch、D2D 和 sync 放到 SPSC ring，由 server worker 消费，减少高频小请求的 RPC 往返。

第六个问题是异步错误的返回时机。CUDA API 中很多错误并不一定在 launch 当下返回，可能在后续同步或错误查询时暴露。如果 ring worker 在后台发现错误，不能把它丢掉，也不能写到全局状态。我们为 session 增加 pending error，后台错误只记录到对应 session，后续同步或 query 时再返回给该 client。

第七个问题是开发环境变量对 gRPC 的影响。测试 `stability_negative_smoke` 时出现过连接异常，排查后发现不是伪造指针被解引用，而是进程继承了 HTTP proxy 环境变量，gRPC 尝试走本机代理端口导致连接失败。解决方法是在 client gRPC channel 参数中关闭 HTTP proxy，并在测试脚本启动子进程时清理 proxy 环境变量。修复后负例测试能稳定返回预期 CUDA 错误码。

第八个问题是 session 空闲超时和 ring 卡死。没有心跳机制时，低频调用的 client 可能被 server reaper 误清理；ring worker 崩溃时，client 在等待 slot 或 completion 时永不自旋。解决方法是为 client 添加保活后台线程（间隔为 `session_timeout_ms / 4`），并在 ring 操作中加入可配置超时（`VGPU_RING_TIMEOUT_US`，默认 5 秒），超时后标记 ring 不可用并自动回退到 gRPC 路径。

第九个问题是 allocation 表在大规模场景下的性能退化。原实现使用 `std::unordered_map` 线性遍历查找 virtual pointer，每次 memcpy 或 kernel launch 都需要扫描所有 allocation。在规模较大时 O(n) 复杂度可能成为瓶颈。解决方法是将 allocation 表改为 `std::map` 并使用 `upper_bound` 实现 O(log n) 查找。

第十个问题是共享内存残留。client 通过 `LD_PRELOAD` 创建 POSIX 共享内存段，正常退出时 `atexit` handler 调用 `shm_unlink` 清理。但如果 client 被 SIGKILL 杀死，共享内存段会永久残留在 `/dev/shm` 中。解决方法是在 server 端 `CleanupSessionResources` 中也调用 `shm_unlink`，确保无论是 client 主动销毁还是 server 超时回收，都能清理对应的共享内存段。

## 8. 分工和协作

本项目按照三人本科生小组的方式推进，成员为黄昱嘉、程昶斌、石锐。分工大体按 client、server、测试文档三条线展开，但很多接口需要一起联调。

黄昱嘉主要负责 client 侧代理库，包括 `LD_PRELOAD` 拦截、Runtime API 包装、session 初始化、shared memory block 管理、kernel registration 记录、last error 语义和部分测试联调。client 侧的难点在于既要让用户程序看到接近 CUDA Runtime 的行为，又要把真实执行转移到 server。

程昶斌主要负责 server 侧资源虚拟化，包括 gRPC service、`SessionState`、allocation/module/stream/event table、virtual pointer 翻译、CUDA Driver API 调用、资源清理和 ring worker 执行路径。server 侧的重点是保证每个 session 的资源边界清楚，并且在错误情况下不访问不属于当前 session 的资源。

石锐主要负责测试、性能优化和文档整理，包括 vector add、matrix multiply、memcpy、stream/event、双进程并发、cross-session sync、ring stress、稳定性测试、性能数据整理、README 和项目报告。测试部分不只是写 demo，而是根据赛题验收点把功能、并发、性能和稳定性串成可执行脚本。

协作过程中，我们通常先把接口字段和预期行为讨论清楚，再分别改 client、proto 和 server。比如 kernel launch 同时涉及 registration 拦截、协议字段、server module 加载和参数翻译；shared memory 数据面同时涉及 client 内存块、server mapping 和脚本运行方式；cross-session sync 同时涉及 stream 设计、server 同步逻辑和测试程序。每次做完一个阶段，都会用最小 smoke test 先验证，再跑完整 acceptance 脚本。

## 9. 提交仓库目录和文件描述

本次提交保留了项目运行和验收需要的文件，清理了本地一次性调试程序、生成文件和临时学习笔记。仓库目录如下：

```text
client/
  libcudart_proxy.cpp       CUDA Runtime API 代理库，负责拦截、session、协议请求和 shared memory 数据面
  probe.cpp                 简单 gRPC 探测程序，用于检查 server 连通性

server/
  main.cpp                  vGPU server 主体，包含 gRPC service 和 CUDA Driver API 执行逻辑
  session_manager.h         session 状态、资源表、virtual pointer 和生命周期管理
  ring_worker.h             per-session ring worker 辅助逻辑

proto/
  vgpu.proto                client/server 之间的 gRPC 与 protobuf 协议定义

shared/
  vgpu_shm_ring.h           shared memory ring 数据结构和命令格式

tools/
  smoke/
    runtime_query_smoke.cu
    vector_add_smoke.cu
    memcpy_baseline.cu
    stream_async_smoke.cu
    event_timing_smoke.cu
    host_alloc_smoke.cu
  stress/
    fire_forget_stress.cu
    stability_negative_smoke.cu
  concurrency/
    concurrent_worker.cu
    cross_session_sync_test.cu
  benchmark/
    matrix_mul_benchmark.cu
  scripts/
    run_acceptance_validation.py
    run_cross_session_sync_test.py

docs/
  project_report.md         设计赛道项目报告

CMakeLists.txt              标准 CMake 构建配置
README.md                   项目入口说明、题目编号、构建和验收运行方式
```

`CMakeLists.txt` 中的 CUDA 测试源文件路径也同步更新，构建时自动按子目录查找源文件。

除了基本功能外，系统还实现了以下可靠性改进：
- **Keepalive 心跳**：client 端后台线程定期（`session_timeout_ms / 4`）向 server 发送保活 RPC，防止低频调用的 session 被误清理。
- **Ring 超时回退**：ring 操作（slot 等待和完成等待）有可配置超时（默认 5 秒），超时后自动标记 ring 不可用并回退到 gRPC 路径。
- **O(log n) 指针查找**：将 allocation 表从无序线性扫描改为有序映射的 `upper_bound` 二分查找，避免大规模 allocation 下的性能退化。
- **Server 端共享内存清理**：session 清理时同步调用 `shm_unlink`，防止 client 异常退出导致 `/dev/shm` 中的共享内存段残留。

`CMakeLists.txt` 默认构建核心组件：`vgpu_server`、`libcudart_proxy.so` 和 `vgpu_client_probe`。当构建参数打开 `-DVGPU_BUILD_CUDA_TESTS=ON` 且环境中存在 `nvcc` 时，会构建 `tools/` 中保留的 CUDA 测试程序。这样仓库既可以作为普通开源项目从 README 开始构建，也可以在验收时直接打开 CUDA 测试 target。

`tools/` 目录经过整理后只保留与验收相关的工具。它们分别对应功能 smoke、性能 benchmark、并发验证、跨 session 隔离、ring 压力和负例稳定性。一次性调试代码没有放入提交版本，避免评审阅读时被无关文件干扰。

## 10. 比赛收获

通过这个项目，我们对 CUDA 程序的理解从“会调用 API”推进到“知道 API 背后需要哪些运行状态”。以前写 CUDA 程序时，`cudaMalloc`、`cudaMemcpy`、kernel launch 和 `cudaDeviceSynchronize` 只是固定流程；做完这个项目后，我们能更清楚地区分 Runtime API、Driver API、device pointer、module、function、stream、event、context 和异步错误之间的关系。

我们也更具体地理解了轻量化虚拟化的含义。轻量化不是少写几个功能，而是要找到真正需要虚拟化的边界：用户进程需要看到 Runtime API 和自己的 CUDA 资源，server 需要持有真实 GPU 资源，二者之间通过 virtual pointer、session 和协议建立映射。只迁移必要信息，才有可能降低数据量和延迟。

性能优化方面，我们最大的体会是不能只看功能正确。初版所有请求都走 gRPC 时，程序能跑通，但数据传输和高频请求的开销比较明显。把数据面拆到 shared memory，把高频命令拆到 ring 后，性能才接近可接受范围。系统设计里“数据怎么走”往往和“功能怎么做”同样重要。

并发隔离方面，我们认识到多进程支持不是简单地让两个进程同时运行。GPU 资源、默认流、同步语义、错误状态和退出清理都需要按 session 设计，否则很容易出现一个进程影响另一个进程的情况。cross-session sync 测试让我们更直观地看到隔离边界的重要性。

最后，测试和文档是系统项目的一部分。vector add、matrix multiply、memcpy 只能说明基本路径正确；双进程并发、ring wrap-around、负例错误处理和 10 分钟稳定性测试才能说明这个原型在边界情况下也能工作。整理 README、CMake 和项目报告的过程，也帮助我们把项目从“自己机器上能跑”整理成一个别人可以阅读、构建和验收的完整仓库。
