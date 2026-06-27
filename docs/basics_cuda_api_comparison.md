# CUDA Runtime API 与 Driver API 对比

本文详细对比 CUDA 的两套编程接口，解释它们的关系、差异，以及 vGPU 项目为什么 client 拦截 Runtime 而 server 使用 Driver API。

---

## 1. 两套 API 概述

CUDA 提供了两套 C 语言编程接口，它们层次不同、用途不同，但底层都操作同一个 GPU：

```text
┌─────────────────────────────────────┐
│  你的 CUDA 程序                      │
├─────────────────────────────────────┤
│  CUDA Runtime API (libcudart.so)    │  ← 高级接口，自动管理 context
│    函数前缀: cuda                    │     用户友好，隐藏了细节
├─────────────────────────────────────┤
│  CUDA Driver API (libcuda.so)       │  ← 低级接口，手动管理一切
│    函数前缀: cu                      │     功能更全，控制更精细
├─────────────────────────────────────┤
│  NVIDIA 显卡驱动 (kernel module)     │
├─────────────────────────────────────┤
│  NVIDIA GPU 硬件                    │
└─────────────────────────────────────┘
```

**快速区分**：看你用的函数名开头是 `cuda` 还是 `cu`：
- `cudaMalloc` = Runtime API
- `cuMemAlloc` = Driver API

---

## 2. Runtime API（高级接口）

### 2.1 特点

Runtime API 是为"大多数用户"设计的，目标是简单好用：

- 自动管理 CUDA Context（不需要你手动创建）
- `kernel<<<grid, block>>>()` 语法可以直接在 C++ 中写 kernel 调用
- 自动处理一些错误（如 deferred error 机制）
- 和 C++ 代码无缝集成

### 2.2 典型代码

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);

    // 1. 在 GPU 上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);   // 简单！不用管 context
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 2. 拷贝数据到 GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 3. 启动 kernel（三重尖括号是 Runtime API 的特色）
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    vectorAdd<<<grid, block>>>(d_A, d_B, d_C, N);
    //        ^^^^^^^^^^^^^^^^  Runtime API 专属语法

    // 4. 等待完成，拷回结果
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 5. 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

**Runtime API 隐藏了的事情**：
- CUDA Context 的创建和设置（自动完成）
- Module 加载（`nvcc` 编译时把 kernel 嵌入可执行文件，启动时自动注册）
- Kernel 参数的打包和传递（`<<<>>>` 语法内部帮你做了）

### 2.3 Runtime API 的 init 机制：隐式初始化

当你第一次调用任何 Runtime API 函数（比如 `cudaMalloc`），Runtime 内部自动做以下事情：

```text
1. 初始化 CUDA Driver
2. 创建一个 CUDA Context（绑定到当前线程）
3. 加载已注册的 fatbin module
4. 建立 kernel function 映射表
```

用户不用手动做任何初始化，这是 Runtime API 方便的原因。但这也意味着**拦截库不能随便放行第一个 cuda 调用**——如果放行了，Runtime 会尝试初始化 GPU，在 client 侧（没有 GPU 的机器上）会失败。

---

## 3. Driver API（低级接口）

### 3.1 特点

Driver API 是更低层的接口，直接和 NVIDIA 驱动对话：

- 需要手动创建和管理 CUDA Context
- 没有 `<<<>>>` 语法，kernel launch 需要手动设置参数
- Module（GPU 代码）需要手动加载（`cuModuleLoadData`）
- 功能比 Runtime API 更多（比如 checkpoint、JIT 编译等高级功能）
- 更细粒度的错误控制

### 3.2 典型代码（同样做 vector add）

```cpp
#include <cuda.h>      // 注意：是 cuda.h，不是 cuda_runtime.h
#include <stdio.h>
#include <vector>

int main() {
    // ==================== 第一步：手动初始化 ====================
    CUresult err;
    err = cuInit(0);                          // 手动初始化 Driver

    CUdevice device;
    err = cuDeviceGet(&device, 0);            // 获取 GPU 设备

    CUcontext ctx;
    err = cuCtxCreate(&ctx, 0, device);       // 手动创建 Context！
    // 现在这个线程绑定了 ctx，后续 GPU 操作都在此 Context 下

    // ==================== 第二步：加载 GPU 代码 ====================
    // 你需要提前把 vector_add.cu 编译成 PTX 文件：
    //   nvcc -ptx vector_add.cu -o vector_add.ptx
    // 然后把 PTX 内容读进来传给 Driver

    // （这里省略了从文件读取 PTX 的代码）
    const char *ptx_source = "...";           // PTX 文本内容
    CUmodule module;
    err = cuModuleLoadData(&module, ptx_source);  // 手动加载 Module

    CUfunction kernel;
    err = cuModuleGetFunction(&kernel, module, "vectorAdd");  // 手动获取 Kernel

    // ==================== 第三步：分配显存 ====================
    CUdeviceptr d_A, d_B, d_C;
    size_t size = N * sizeof(float);
    err = cuMemAlloc(&d_A, size);
    err = cuMemAlloc(&d_B, size);
    err = cuMemAlloc(&d_C, size);

    // ==================== 第四步：拷贝数据 ====================
    err = cuMemcpyHtoD(d_A, h_A, size);
    err = cuMemcpyHtoD(d_B, h_B, size);

    // ==================== 第五步：启动 Kernel ====================
    // 没有 <<<>>> 语法了！需要手动设置所有参数
    int N_val = N;

    void *kernelArgs[] = {
        &d_A,   // 参数 1: float *A  （注意：传的是 &d_A，不是 d_A）
        &d_B,   // 参数 2: float *B
        &d_C,   // 参数 3: float *C
        &N_val  // 参数 4: int N
    };

    err = cuLaunchKernel(
        kernel,              // CUfunction
        gridX, gridY, gridZ, // grid 维度
        blockX, blockY, blockZ, // block 维度
        0,                   // shared memory（动态共享内存大小）
        NULL,                // stream（NULL = 默认 stream）
        kernelArgs,          // kernel 参数（void** 数组）
        NULL                 // extra（通常 NULL）
    );

    // ==================== 第六步：同步和拷回 ====================
    err = cuCtxSynchronize();
    err = cuMemcpyDtoH(h_C, d_C, size);

    // ==================== 第七步：清理 ====================
    err = cuMemFree(d_A);
    err = cuMemFree(d_B);
    err = cuMemFree(d_C);
    err = cuModuleUnload(module);
    err = cuCtxDestroy(ctx);
}
```

**Driver API 让你手动做的事（Runtime 帮你做的）**：
- `cuInit` 初始化
- `cuCtxCreate` 创建 Context
- `cuModuleLoadData` 加载 PTX/cubin
- `cuModuleGetFunction` 获取 kernel 函数
- 手动构造 `void* kernelArgs[]` 数组
- `cuCtxDestroy` 销毁 Context

---

## 4. 逐项对比

| 维度 | Runtime API | Driver API |
|------|------------|------------|
| **头文件** | `<cuda_runtime.h>` | `<cuda.h>` |
| **库文件** | `libcudart.so` | `libcuda.so`（驱动自带） |
| **函数前缀** | `cudaXxx` | `cuXxx` |
| **Context** | 自动创建和管理 | 手动 `cuCtxCreate` / `cuCtxDestroy` |
| **Module** | 编译时嵌入 fatbin，启动时自动注册 | 手动 `cuModuleLoadData` / `cuModuleLoadFatBinary` |
| **Kernel Launch** | `kernel<<<grid, block>>>(args)` | `cuLaunchKernel(func, grid, block, shm, stream, args, extra)` |
| **错误处理** | `cudaGetLastError()` 延迟返回 | `CUresult` 每次调用立即返回 |
| **易用程度** | 简单，像普通 C 函数 | 复杂，需要理解 GPU 底层概念 |
| **功能覆盖** | 90% 的常用功能 | 100% 功能，包括高级特性 |
| **版本稳定性** | 不同 CUDA 版本 ABI 可能变 | 驱动 API 非常稳定 |
| **适用场景** | 应用开发者写 CUDA 程序 | 框架开发者（TensorFlow/PyTorch）或底层工具 |

### 4.1 函数名对照表

| 操作 | Runtime API | Driver API |
|------|-----------|-----------|
| 初始化 | 自动 | `cuInit` |
| 创建 Context | 自动 | `cuCtxCreate` |
| 分配显存 | `cudaMalloc` | `cuMemAlloc` |
| 释放显存 | `cudaFree` | `cuMemFree` |
| H2D 拷贝 | `cudaMemcpy(..., H2D)` | `cuMemcpyHtoD` |
| D2H 拷贝 | `cudaMemcpy(..., D2H)` | `cuMemcpyDtoH` |
| D2D 拷贝 | `cudaMemcpy(..., D2D)` | `cuMemcpyDtoD` |
| 异步 H2D | `cudaMemcpyAsync(..., H2D, stream)` | `cuMemcpyHtoDAsync` |
| 设置内存值 | `cudaMemset` | `cuMemsetD8` |
| 创建 Stream | `cudaStreamCreate` | `cuStreamCreate` |
| 销毁 Stream | `cudaStreamDestroy` | `cuStreamDestroy` |
| 同步 Stream | `cudaStreamSynchronize` | `cuStreamSynchronize` |
| 同步设备 | `cudaDeviceSynchronize` | `cuCtxSynchronize` |
| 创建 Event | `cudaEventCreate` | `cuEventCreate` |
| 记录 Event | `cudaEventRecord` | `cuEventRecord` |
| Event 计时 | `cudaEventElapsedTime` | `cuEventElapsedTime` |
| 加载 Module | 自动（fatbin 注册） | `cuModuleLoadData` / `cuModuleLoadFatBinary` |
| 获取 Function | 自动 | `cuModuleGetFunction` |
| 启动 Kernel | `cudaLaunchKernel` 或 `<<<>>>` | `cuLaunchKernel` |

---

## 5. CUDA Context 详解

### 5.1 Context 是什么

CUDA Context 是 GPU 程序执行的"环境"。类比：

- 一个进程 = 一个公司
- CUDA Context = 公司里的一块独立办公区域
- 在这个 Context 里分配的资源（显存、stream、module）都属于这个 Context

所有 GPU 操作都必须在某个 Context 内进行。没有 Context，`cuMemAlloc` 之类的函数会失败。

### 5.2 Context 和线程的绑定

**CUDA Driver API 的核心规则**：一个 CPU 线程在同一时刻只能有一个活跃的 CUDA Context。

```cpp
// 线程 A 切换到 Context 1
cuCtxSetCurrent(ctx1);
cuMemAlloc(...);    // 在 ctx1 的 GPU 上分配

// 线程 A 切换到 Context 2
cuCtxSetCurrent(ctx2);
cuMemAlloc(...);    // 在 ctx2 的 GPU 上分配
```

### 5.3 Primary Context vs 自定义 Context

**Primary Context**（主 Context）：
- GPU 0 自带一个 primary context
- 通过 `cuDevicePrimaryCtxRetain` 获取
- 所有使用这个 GPU 的程序可以共享同一个 primary context
- **优点**：不需要 context switch，性能好
- **缺点**：不隔离，一个程序的错误会影响其他程序

**自定义 Context**：
- 通过 `cuCtxCreate` 创建
- 相互隔离（一个 Context 中的资源不会影响其他 Context）
- **优点**：隔离好，安全
- **缺点**：创建和切换有开销

vGPU 项目选择 **primary context + 多 stream** 的折中方案：
- 所有 session 共享同一个 primary context（性能）
- 通过 session stream table、内存配额、资源映射表自己管理隔离（手动隔离）

---

## 6. 为什么 vGPU 采用"拦截 Runtime，执行 Driver"的架构

### 6.1 Client 侧：拦截 Runtime API

理由：
1. **普通 CUDA 程序用 Runtime API**：如果拦截 Driver API，大多数现成的 CUDA 程序不能用
2. **`<<<>>>` 语法只存在于 Runtime API**：比赛测试程序会用到
3. **LD_PRELOAD 拦截 `libcudart.so` 比拦截 `libcuda.so` 更稳定**：`libcudart.so` 是 CUDA Toolkit 的一部分，ABI 相对可控；`libcuda.so` 随驱动变化

### 6.2 Server 侧：使用 Driver API

理由：
1. **Driver API 更底层**：不需要隐式 Runtime 初始化、自动注册等复杂机制。server 只需要做"我被告知要做什么，我就做什么"。
2. **手动控制 Context**：server 可以在全局使用 primary context，精确控制 context 生命周期
3. **Module 加载更直接**：client 通过 gRPC 发来 fatbin 数据 → server 直接 `cuModuleLoadFatBinary`，没有 Runtime 的隐式注册在中间捣乱
4. **错误控制更精细**：`CUresult` 立即返回，不像 Runtime API 的 deferred error
5. **功能更全**：需要某些高级功能时（如查看 GPU 内存使用量），Driver API 可能提供 Runtime API 没有的接口

### 6.3 架构图

```text
Client 侧                                    Server 侧
───────                                     ────────
CUDA 程序
  kernel<<<grid, block>>>(args)
  cudaMalloc(&ptr, size)
       |
       | (LD_PRELOAD 拦截)
       v
libcudart_proxy.so                      vgpu_server
  ├ cudaMalloc()                         ├ cuInit(0)
  │   → MallocRequest                    ├ cuDevicePrimaryCtxRetain()
  │   ← MallocReply (vptr)               ├ cuCtxSetCurrent(primary_ctx)
  │                                      │
  ├ __cudaRegisterFatBinary()            ├ 收到 RegisterFatBinary
  │   → RegisterFatBinaryRequest          │   → cuModuleLoadFatBinary()
  │   ← RegisterFatBinaryReply (handle)  │
  │                                      │
  ├ cudaLaunchKernel()                   ├ 收到 LaunchKernel
  │   → LaunchKernelRequest               │   → cuLaunchKernel()
  │   ← StatusReply                       │
  │                                      │
  └ cudaMemcpy()                         └ 收到 Memcpy
      → MemcpyRequest                        → cuMemcpyHtoD/cuMemcpyDtoH
      ← MemcpyReply (data)                   ← 完成
```

---

## 7. Kernel Launch 的两种路径

这一点在 vGPU 实现中很关键，因为 client 拦截时必须理解这两种路径的差异。

### 7.1 新式路径：cudaLaunchKernel（CUDA 5.0+）

```cpp
// 一条函数调用完成所有参数设置
cudaLaunchKernel(
    (const void*)func,  // kernel 函数指针
    gridDim,            // dim3
    blockDim,           // dim3
    args,               // void** 指向参数数组
    sharedMem,          // 动态共享内存字节数
    stream              // CUDA stream
);
```

**问题**：`args` 是 `void**`，不包含每个参数的大小信息。拦截层怎么知道每个参数多大？

### 7.2 老式路径：cudaConfigureCall + cudaSetupArgument + cudaLaunch

```cpp
// 三步走（已被新式替代，但编译器可能仍生成此路径的代码）
cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
cudaSetupArgument(arg1, size1, offset1);  // 每个参数逐一设置
cudaSetupArgument(arg2, size2, offset2);  // 明确给出 size！
cudaLaunch(func);
```

**优点**：`cudaSetupArgument` 显式给出了每个参数的 size 和 offset，拦截层可以直接使用。

**这就是为什么 vGPU 设计文档中建议**：
1. 先支持 `cudaSetupArgument` 路径（容易拿到参数大小）
2. 再补 `cudaLaunchKernel`（需要从 fatbin metadata 或参数推断取得参数大小）

---

## 8. 错误码对应

Runtime API 和 Driver API 各有自己的错误码体系：

| Runtime 错误码（cudaError_t） | Driver 错误码（CUresult） | 含义 |
|---|---|---|---|
| `cudaSuccess` (0) | `CUDA_SUCCESS` (0) | 成功 |
| `cudaErrorInvalidValue` (1) | `CUDA_ERROR_INVALID_VALUE` (1) | 参数无效 |
| `cudaErrorMemoryAllocation` (2) | `CUDA_ERROR_OUT_OF_MEMORY` (2) | 显存不足 |
| `cudaErrorInvalidDevicePointer` (17) | `CUDA_ERROR_INVALID_DEVICE` (101) | 指针无效 |
| `cudaErrorLaunchFailure` (4) | `CUDA_ERROR_LAUNCH_FAILED` (4) | kernel 启动失败 |

注意：这两个错误码的数值**不一定对应**。在 vGPU server 中需要实现一个转换函数：

```cpp
cudaError_t MapCuResultToCudaError(CUresult r) {
    switch (r) {
        case CUDA_SUCCESS:              return cudaSuccess;
        case CUDA_ERROR_INVALID_VALUE:  return cudaErrorInvalidValue;
        case CUDA_ERROR_OUT_OF_MEMORY:  return cudaErrorMemoryAllocation;
        // ... 更多映射
        default:                        return cudaErrorUnknown;
    }
}
```

Server 返回 Runtime API 错误码给 client，client 直接传给应用，这样用户程序感知到的错误和原生 CUDA 一致。

---

## 9. 总结

| | Runtime API | Driver API |
|---|---|---|
| **谁用** | 普通 CUDA 程序员 | 框架/工具开发者 |
| **vGPU 中谁使用** | Client 侧（被拦截的对象） | Server 侧（执行工具） |
| **设计哲学** | 简单、自动、隐藏细节 | 强大、手动、完全控制 |
| **Kernel 语法** | `<<<>>>` | `cuLaunchKernel` |
| **Module 管理** | 自动注册 fatbin | 手动 load data |
| **Context 管理** | 自动 | 手动 |
| **隔离性** | 自动（但不可控） | 手动（完全可控） |

理解了这两套 API 的差异，就会明白 vGPU 项目的架构选择非常合理：
- **Client 拦截 Runtime**：因为目标程序用它，透明性好
- **Server 用 Driver**：因为它是"GPU 资源管理者"，需要完全控制

---

> **上一份文档**：[计算机网络基础：TCP、RPC 与 gRPC](basics_network_rpc.md)
> **下一份文档**：[CUDA 编译工具链：nvcc、PTX、Fatbin](basics_compilation_toolchain.md)
