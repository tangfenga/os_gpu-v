# GPU 与 CUDA 基础概念

本文面向没有任何 GPU 和 CUDA 编程经验的读者，从最基础的概念开始，逐步介绍 CUDA 编程中涉及的术语和原理。

---

## 1. 什么是 GPU

### 1.1 CPU 与 GPU 的区别

**CPU（中央处理器）** 是你电脑的"大脑"，擅长处理复杂、有分支、串行的任务。一个 CPU 通常有 4~16 个"核心"，每个核心非常强大，像一个大学教授——能解决各种复杂问题，但一次只能处理一两件事情。

**GPU（图形处理器）** 最初是为了渲染游戏画面而设计的。游戏画面需要在极短时间内对屏幕上几百万个像素做类似的简单计算。因此 GPU 被设计成有几千个小核心，每个核心比较简单，像一个由几千个小学生组成的班级——每个人做不了复杂的数学题，但如果让每个人做一道简单的加法，几千个人同时做，速度非常快。

类比：
- CPU = 4 个博士一起干活，每人能力超强
- GPU = 1000 个小学生一起干活，每人只会做简单运算，但数量多，并行完成时总量惊人

### 1.2 为什么 GPU 适合做计算

很多科学计算和机器学习任务有"数据并行"的特征：

- 向量加法：C[i] = A[i] + B[i]，对 100 万个元素做同样的加法
- 矩阵乘法：大量乘加运算可以同时进行
- 图像处理：每个像素的处理方式相同

GPU 的几千个核心非常适合这种"对大量数据做相同简单操作"的场景。这就是 **GPGPU（通用 GPU 计算）** 的由来——用显卡做图形以外的事情。

### 1.3 显存（VRAM）

GPU 有自己的内存，叫**显存**（VRAM，Video RAM）。就像 CPU 有系统内存（插在主板上的 DDR 内存条），GPU 有焊在显卡上的显存。

关键点：
- CPU 不能直接访问显存中的数据
- GPU 不能直接读写系统内存中的数据
- 需要显式地把数据从系统内存**拷贝**到显存，GPU 才能处理
- 处理完后再把结果从显存**拷贝**回系统内存

这就是 `cudaMemcpy`（或 `cuMemcpyHtoD`、`cuMemcpyDtoH`）的作用——搬运数据。

---

## 2. CUDA 是什么

CUDA（Compute Unified Device Architecture）是 NVIDIA 公司推出的并行计算平台。它包含：

1. **硬件层面**：NVIDIA GPU 硬件本身
2. **驱动层面**：NVIDIA 显卡驱动，操作系统通过它与 GPU 通信
3. **编程层面**：一套 C/C++ 扩展，让你用 C 语言写运行在 GPU 上的程序

简单说：CUDA 让你可以用写 C 语言的方式，让 GPU 帮你做大规模并行计算。

### 2.1 为什么需要 CUDA

没有 CUDA 的时候，用 GPU 做计算非常痛苦——你需要把数据伪装成图形（比如把数组当纹理贴图），用图形 API（OpenGL/DirectX）去操作，再把结果从图形数据还原成普通数据。CUDA 直接提供了一套编程接口，让你绕过图形那一层。

---

## 3. CUDA 编程的核心概念

### 3.1 Host（主机）和 Device（设备）

这两个词在 CUDA 编程中非常关键：

| 术语 | 含义 | 通俗理解 |
|------|------|---------|
| **Host** | 你的 CPU + 系统内存 | 你的电脑本体 |
| **Device** | NVIDIA GPU + 显存 | 插在电脑上的显卡 |

你在普通 C 程序中写的代码运行在 Host 上。想让代码在 Device（GPU）上运行，需要写特殊的函数（Kernel），并显式地让 GPU 执行它。

### 3.2 Kernel（核函数）

Kernel 是在 GPU 上运行的函数。在代码中用 `__global__` 关键字标记：

```cpp
// 这是一个 Kernel，会在 GPU 上运行
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    // i 是当前线程的索引
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];   // 每个线程计算一个元素
    }
}
```

**Kernel 的核心思想**：你写的是"一个线程"要做什么（计算一个 C[i]），然后启动成千上万个线程，每个线程计算不同的 i。

启动 Kernel 的语法（所谓的"三重尖括号"）：
```cpp
vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
//         ^^^^^^^^^^^^^^
//         Kernel 启动配置，下面马上解释
```

### 3.3 Grid、Block、Thread（线程层次）

这是 CUDA 对"同时做很多事"的组织方式，从大到小三层：

```text
Grid（网格）
  ├── Block 0（线程块）
  │     ├── Thread 0
  │     ├── Thread 1
  │     ├── ...
  │     └── Thread 255
  ├── Block 1
  │     └── ...（256 个 Thread）
  ├── ...
  └── Block N-1
```

| 概念 | 数量 | 说明 |
|------|------|------|
| **Grid** | 1 个（一个 kernel launch） | 整个 kernel 任务 |
| **Block** | 最多几千个 | 一组线程，可共享内存 |
| **Thread** | 每个 Block 最多 1024 个 | 最小执行单元 |

**为什么需要三层结构？**

- 硬件上，GPU 的几千个核心被分成若干个"流多处理器"（SM）。每个 SM 一次处理一个 Block。
- 同一个 Block 内的 Thread 可以协作（同步、共享数据）。
- 不同 Block 之间不能直接协作（它们可能在不同时间执行）。

**计算全局索引的经典模式**：

```cpp
int i = threadIdx.x      // 我在本 Block 内的编号 (0~255)
      + blockIdx.x       // 本 Block 的编号 (0~N-1)
      * blockDim.x;      // 每个 Block 有多少线程 (256)
// 结果：i 的范围 = 0 ~ (N*256-1)
```

**实际例子**：你有 100 万个元素要做向量加法。
- `blockDim.x = 256`（每个 Block 256 个线程）
- `gridDim.x = ceil(1000000 / 256) = 3907`（需要 3907 个 Block）

每个线程计算 `C[i] = A[i] + B[i]`（一个元素的加法），100 万个线程同时（或分批）完成。

### 3.4 dim3 类型

在 CUDA 中，Grid 和 Block 可以是 1D、2D 或 3D 的，用 `dim3` 类型表示：

```cpp
// 1D：4096 个线程块，每个块 256 线程
dim3 grid(4096, 1, 1);
dim3 block(256, 1, 1);

// 2D：用于图像处理等场景
dim3 grid(64, 64, 1);
dim3 block(16, 16, 1);
// 总共 64x64=4096 个 Block，每个 Block 16x16=256 个 Thread
```

在 gRPC 设计文档的 proto 中，你可以看到：
```proto
message Dim3 {
  uint32 x = 1;
  uint32 y = 2;
  uint32 z = 3;
}
```

这对应了 CUDA 中 `dim3` 的三个维度。

### 3.5 Device Pointer（设备指针）

```cpp
float *d_A;  // d_ 前缀是约定俗成，表示这是个 device pointer
cudaMalloc(&d_A, size);  // 在 GPU 显存上分配空间，d_A 指向显存中的地址
```

**关键理解**：
- `d_A` 是一个指针变量，存在 Host（CPU 侧）的内存中
- `d_A` 的值（即它指向的地址）是 GPU 显存中的一个地址
- **Host 代码不能直接 `*d_A` 解引用这个指针**——CPU 不能直接访问显存
- 你需要 `cudaMemcpy` 把数据从显存拷到系统内存，才能读取

在 vGPU 项目中，client 侧没有真实 GPU，所以 `cudaMalloc` 返回的是**虚拟 device pointer**——server 给的一个编号，不是真实的 GPU 显存地址。

### 3.6 Stream（流）

Stream 是 CUDA 中管理**操作执行顺序**的概念。

- **默认**：所有操作在同一个 Stream 中，严格按顺序执行。必须先完成操作 A，才能开始操作 B。
- **多 Stream**：不同 Stream 中的操作可以并发执行。比如 Stream 1 在做内存拷贝（H2D），同时 Stream 2 在执行 Kernel。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 两个 Stream 可以同时进行不同的工作
cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1);
vectorAdd<<<grid, block, 0, stream2>>>(d_B, d_C, d_D, N);
// stream1 在拷数据，stream2 在执行 Kernel —— 同时进行！
```

**类比**：Stream 像流水线。单 Stream = 只有一个工人，做完一件事再做下一件。多 Stream = 多个工人，每人做不同的事。

### 3.7 Event（事件）

Event 是 CUDA 中的"时间戳标记"，用于：
1. **测量时间**：在 Stream 中插入 Event，记录 GPU 执行到此处的时间
2. **同步**：Host 可以等待某个 Event 完成

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);   // 在 stream 中标记"开始"
vectorAdd<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);    // 标记"结束"

cudaEventSynchronize(stop);       // 等待 GPU 执行到 stop

float ms;
cudaEventElapsedTime(&ms, start, stop);  // 计算两次标记之间的时间（毫秒）
printf("Kernel 耗时: %f ms\n", ms);
```

### 3.8 同步（Synchronization）

CUDA 中很多操作是**异步**的——Kernel 启动后，Host 代码立即继续执行，不等待 GPU 完成。这很有用（CPU 可以同时做别的事），但也意味着你需要显式同步：

- `cudaDeviceSynchronize()`：等待 GPU 上所有操作完成
- `cudaStreamSynchronize(stream)`：等待某个 Stream 上的操作完成

```cpp
kernel<<<grid, block>>>(d_A, d_B, d_C, N);
// Host 代码继续执行，Kernel 可能还没完成！

cudaDeviceSynchronize();  // 现在 Host 停下来等待，直到 Kernel 完成
// 现在可以安全地读取结果了
```

---

## 4. CUDA 的内存类型

### 4.1 两种最基本的内存

| 内存类型 | 位置 | 分配函数 | 谁可以访问 |
|----------|------|---------|-----------|
| Host Memory | 系统内存（内存条） | `malloc` / `new` | CPU |
| Device Memory（Global Memory） | 显存 | `cudaMalloc` | GPU |

### 4.2 数据传输：H2D、D2H、D2D、H2H

这些缩写非常常用：

| 缩写 | 全称 | 含义 | 操作 |
|------|------|------|------|
| **H2D** | Host to Device | 从系统内存到显存 | `cudaMemcpyHostToDevice` |
| **D2H** | Device to Host | 从显存到系统内存 | `cudaMemcpyDeviceToHost` |
| **D2D** | Device to Device | 从显存到显存（同一 GPU 内） | `cudaMemcpyDeviceToDevice` |
| **H2H** | Host to Host | 系统内存内拷贝 | 普通 `memcpy` |

**典型 CUDA 程序的数据流**：

```text
1. H2D: 把输入数据从 CPU 内存传到 GPU 显存
         A_h(在内存条) --cudaMemcpy--> A_d(在显存)

2. Kernel: GPU 进行处理
           C_d[i] = A_d[i] + B_d[i]   (全部在显存内)

3. D2H: 把结果从 GPU 显存传回 CPU 内存
        C_d(在显存) --cudaMemcpy--> C_h(在内存条)

4. 现在 CPU 可以读取/打印/保存结果了
```

**在 vGPU 项目中的意义**：

- H2D：client 把数据发给 server，server 写到 GPU 显存
- D2H：server 从 GPU 显存读数据，发给 client
- D2D：两个操作数都在 server GPU 上 → **数据不需要经过 client**，直接在 server 侧完成
- H2H：不需要 RPC，client 本地 `memcpy`

---

## 5. NVIDIA 软件栈

从顶层到底层：

```text
┌──────────────────────────────────┐
│  CUDA 应用程序（你的代码）         │
│  cudaMalloc / kernel<<<>>>       │
├──────────────────────────────────┤
│  CUDA Runtime API (libcudart.so) │  ← 用户友好的接口
├──────────────────────────────────┤
│  CUDA Driver API (libcuda.so)    │  ← 底层接口，控制更细
├──────────────────────────────────┤
│  NVIDIA 显卡驱动 (kernel module)  │  ← 操作系统层面的驱动
├──────────────────────────────────┤
│  NVIDIA GPU 硬件                  │  ← 真实的显卡
└──────────────────────────────────┘
```

### 5.1 Runtime API 和 Driver API 的简化理解

| | Runtime API | Driver API |
|---|---|---|
| 函数前缀 | `cuda` | `cu` |
| 库文件 | `libcudart.so` | `libcuda.so`（来自驱动） |
| 使用难度 | 简单，像普通 C 函数 | 复杂，需要手动管理 context |
| 功能 | 覆盖大部分需要 | 功能更全，控制更细 |
| 例子 | `cudaMalloc` | `cuMemAlloc` |

**为什么 vGPU 项目用 Runtime 拦截但 server 用 Driver API？**

- Client 拦截的是 Runtime API（因为普通 CUDA 程序都用这个）
- Server 用 Driver API 执行（因为 Driver API 更底层、控制更细，适合作为 GPU 的"代理执行者"）

---

## 6. Fat Binary（胖二进制）

### 6.1 普通 C 程序 vs CUDA 程序的编译

**普通 C 程序**：
```text
hello.c → gcc → hello（只包含 x86 CPU 指令）
```

**CUDA 程序**：
```text
vector_add.cu → nvcc → vector_add（包含两部分）
                         ├── x86 CPU 指令（Host 代码）
                         └── GPU 指令（Device 代码，即 Kernel）
```

一个 CUDA 编译出的可执行文件里既有 CPU 代码又有 GPU 代码。GPU 代码可以包含多个版本（对应不同架构的 GPU），所以叫 **"胖"** 二进制（Fat Binary）。

### 6.2 Fat Binary 的内容

Fat Binary 通常包含：
- **PTX**：中间表示，类似汇编但还没彻底翻译成机器码。可以在加载时由驱动翻译成目标 GPU 的机器码。
- **CUBIN**：已经翻译好的 GPU 机器码，针对特定 GPU 架构。

### 6.3 `__cudaRegisterFatBinary` 是做什么的

当 CUDA 程序启动时，CUDA Runtime 会自动调用 `__cudaRegisterFatBinary` 把编译时嵌入程序的 GPU 代码注册到 CUDA Runtime 中。这样后续 `kernel<<<>>>` 启动时，Runtime 知道去哪里找对应的 GPU 代码。

在 vGPU 项目中，这是最复杂的一步——client 没有 GPU，但程序仍然会尝试注册 fatbin。Client 的拦截库必须：
1. 捕获 fatbin 内容
2. 通过 gRPC 发送给 server
3. Server 在真实 GPU 上加载这个 fatbin

---

## 7. vGPU 项目中涉及的关键 CUDA 概念总结

| 概念 | 在 vGPU 项目中的作用 |
|------|---------------------|
| **Device Pointer** | client 用假的虚拟指针，server 用真实 GPU 指针，两者通过映射表关联 |
| **Stream** | 每个 client session 有独立的 stream，保证多进程不互相阻塞 |
| **Event** | 用于性能测量，记录 kernel 时间 |
| **H2D/D2H/D2D** | 决定数据是否需要经过网络传输；D2D 在 server 内部完成，是重要的性能优化点 |
| **Fat Binary** | client 拦截 `__cudaRegisterFatBinary`，把 GPU 代码发给 server 加载 |
| **Kernel Launch** | client 把 kernel 名和参数打包成 RPC 发给 server，server 调用真实 GPU 执行 |
| **Synchronization** | 共享 primary context 时，`cudaDeviceSynchronize` 会影响所有 session |

---

## 8. 延伸阅读顺序

1. NVIDIA CUDA C Programming Guide（官方编程指南，前 3 章足够）
2. 动手写一个 vector_add.cu，在真实 GPU 上跑通
3. 尝试用 CUDA Driver API 实现同样的功能（理解 Runtime 和 Driver 的差异）
4. 观察 `nvcc --cudart shared` 编译出的程序的符号表：`nm -D` 查看

理解了这些基础概念后，再回头看 vGPU 设计文档，你会发现每个设计决策都有明确的理由——为什么拦截的是 Runtime API、为什么 server 用 Driver API、为什么 D2D 不需要回传数据、为什么 fatbin 注册是最难的部分。

---

> **下一份文档**：[操作系统基础：动态链接与 LD_PRELOAD](basics_os_dynamic_linking.md)
