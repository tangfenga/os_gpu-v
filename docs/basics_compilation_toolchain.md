# CUDA 编译工具链：nvcc、PTX、Fatbin 与构建工具

本文介绍 CUDA 程序的编译过程、涉及的中间文件格式，以及 vGPU 项目使用的构建工具（CMake、gRPC/protobuf 代码生成）。

---

## 1. 从 .cu 到可执行文件

### 1.1 CUDA 编译的特殊之处

普通 C 程序：
```text
hello.c  →  gcc  →  hello（只包含 x86 机器指令）
```

CUDA 程序（`.cu` 文件）：
```text
vector_add.cu  →  nvcc  →  可执行文件
                              ├── x86 机器指令（Host 代码，CPU 执行）
                              └── GPU 机器指令（Device 代码，GPU 执行）
```

一个 `.cu` 文件里混着两种代码：CPU 上运行的（Host Code）和 GPU 上运行的（Device Code）。NVCC 编译器负责把它们拆开、分别编译、再打包到一起。

### 1.2 NVCC 的编译流程

```text
vector_add.cu（一份源文件，混合了 Host 和 Device 代码）
       │
       │ nvcc 预处理
       v
┌──────────────────────────────┐
│ 分离 Host 代码和 Device 代码  │
└──────────────────────────────┘
       │                    │
       │ Host 代码          │ Device 代码（Kernel）
       v                    v
   gcc/g++ 编译          nvcc 编译为 PTX
       │                    │
       │                    │ PTX → ptxas → CUBIN
       │                    │ （可选，或保留 PTX）
       v                    v
    Host .o               Device .o（或嵌入 fatbin）
       │                    │
       └────────┬───────────┘
                │ 链接
                v
          vector_add（可执行文件）
          包含：Host 代码 + 嵌入的 Fat Binary
```

### 1.3 NVCC 常用命令

```bash
# 编译并链接（一步到位，静态链接 cudart）
nvcc vector_add.cu -o vector_add

# 编译并链接（动态链接 cudart，适合 LD_PRELOAD 拦截）
nvcc --cudart shared vector_add.cu -o vector_add

# 只编译不链接（生成 .o 文件）
nvcc -c vector_add.cu -o vector_add.o

# 编译为 PTX（不生成可执行文件，只生成 GPU 中间代码）
nvcc -ptx vector_add.cu -o vector_add.ptx

# 编译为 cubin（针对特定 GPU 架构的机器码）
nvcc -cubin vector_add.cu -o vector_add.cubin

# 指定目标 GPU 架构
nvcc -arch=sm_86 vector_add.cu -o vector_add   # RTX 30 系列用的 sm_86

# 查看 NVCC 版本
nvcc --version
```

---

## 2. PTX、CUBIN、Fatbin 的区别

这是理解 vGPU 中 fatbin 注册机制的核心知识。

### 2.1 PTX（Parallel Thread Execution）

PTX 是 NVIDIA GPU 的**中间表示**（IR，Intermediate Representation），类似汇编语言但不是最终的机器码。

```ptx
// vector_add.ptx（部分内容，简化示意）
.version 7.0
.target sm_50

.visible .entry _Z10vectorAddPfS_S_i(
    .param .u64 _Z10vectorAddPfS_S_i_param_0,
    .param .u64 _Z10vectorAddPfS_S_i_param_1,
    .param .u64 _Z10vectorAddPfS_S_i_param_2,
    .param .u32 _Z10vectorAddPfS_S_i_param_3
)
{
    // ... PTX 指令 ...
    ld.global.f32   %f1, [%rd1];    // 从显存加载
    ld.global.f32   %f2, [%rd2];
    add.f32         %f3, %f1, %f2;  // 浮点加法
    st.global.f32   [%rd3], %f3;    // 存回显存
    ret;
}
```

PTX 的特点：
- **人类可读**：是文本格式（虽然不是很好看懂）
- **跨 GPU 架构**：同一份 PTX 可以在不同 GPU 架构上运行
- **JIT 编译**：GPU 驱动在运行时把 PTX 翻译成当前 GPU 能执行的机器码
- **稍慢**：运行时 JIT 编译有额外开销

### 2.2 CUBIN（CUDA Binary）

CUBIN 是**已经编译好的 GPU 机器码**，针对特定 GPU 架构（如 sm_86 = RTX 3060/3070/3080）。

CUBIN 的特点：
- **二进制格式**：不可读
- **架构特定**：为 sm_86 编译的 cubin 不能在 sm_70 上运行
- **加载快**：不需要 JIT 编译，直接可以执行
- **无兼容性**：必须在编译时指定的 GPU 架构上运行

### 2.3 Fatbin（Fat Binary）

Fatbin 就是把**多个版本的 GPU 代码打包在一起**。

```text
vector_add.fatbin
  ├── PTX（通用，驱动 JIT 编译到任何支持的 GPU）
  ├── CUBIN for sm_70（Volta 架构）
  ├── CUBIN for sm_75（Turing 架构）
  ├── CUBIN for sm_80（Ampere A100）
  └── CUBIN for sm_86（Ampere RTX 30 系列）
```

**为什么叫"胖"（Fat）**：因为包含多个版本，文件比单一 cubin 大。

**运行时选择逻辑**：
1. GPU 驱动检查当前 GPU 是什么架构
2. 如果 fatbin 中有对应架构的 cubin → 直接加载 cubin（快）
3. 如果没有 cubin，有 PTX → 用驱动 JIT 编译 PTX（慢，但能跑）
4. 都没有 → 程序无法运行

**在 vGPU 项目中**：
- `__cudaRegisterFatBinary` 的参数就是 fatbin 数据
- Client 拦截后复制一份 fatbin 内容
- 通过 gRPC 发送给 server
- Server 调用 `cuModuleLoadFatBinary(fatbin_data)` 加载
- Server 的 GPU 可能是另一种架构，但 fatbin 自带了兼容性支持

### 2.4 三者的关系图

```text
源文件(.cu)
    │
    │ nvcc
    v
┌──── PTX ────┐         ← 中间表示，文本格式，跨架构
│  (可选保留)  │
└──────┬──────┘
       │ ptxas（PTX 汇编器）
       v
┌─── CUBIN ───┐         ← GPU 机器码，二进制，架构特定
│   sm_70     │
│   sm_86     │
│   ...       │
└──────┬──────┘
       │ 打包
       v
┌── Fatbin ───┐         ← 包含 PTX + 多架构 CUBIN
│  完整包      │          嵌入在可执行文件的数据段中
└─────────────┘
```

---

## 3. CUDA 程序的启动与 Kernel 注册

### 3.1 CUDA 程序为什么和普通程序不一样

普通 C 程序的 `main` 是第一个被执行的函数。但 CUDA Runtime 程序在 `main` 之前，有一系列自动执行的初始化：

```text
程序加载
  │
  │ 动态链接器加载 libcudart.so
  │ libcudart.so 内部构造函数初始化 Runtime
  v
┌────────────────────────────────────────────┐
│ CUDA Runtime 初始化（用户看不到这些代码）   │
│  1. cuInit(0)                              │
│  2. 遍历已注册的 fatbin                     │
│  3. 对每个 fatbin 调用 cuModuleLoadFatBinary│
│  4. 对每个 registered function：           │
│     cuModuleGetFunction(module, name)      │
│  5. 建立 host_func_ptr → CUfunction 的映射 │
└────────────────────────────────────────────┘
  │
  │ 注册函数（由编译器生成、嵌入程序的代码调用）
  │ 这些调用发生在 main 之前或 main 刚开始时
  │
  ├── __cudaRegisterFatBinary(fatbin_data)
  │     └ 告诉 Runtime："这里有一份 GPU 代码"
  │
  ├── __cudaRegisterFunction(hostFun, deviceFun, deviceName, ...)
  │     └ 告诉 Runtime："host 函数指针 hostFun 对应 GPU kernel deviceName"
  │
  ├── ...
  │
  v
main() 开始执行（用户代码）
  │
  ├── cudaMalloc(...)
  ├── kernel<<<grid, block>>>(args)
  │     │
  │     │ Runtime 内部：
  │     │   查 hostFun → CUfunction 映射
  │     │   打包参数
  │     │   调用 cuLaunchKernel
  │     │
  ├── cudaDeviceSynchronize()
  └── cudaMemcpy(..., D2H)
```

### 3.2 注册函数的签名（为什么要拦截它们）

```cpp
// 这些函数由 nvcc 编译器自动生成调用代码，程序员不直接写它们

// 注册 fatbin：返回一个不透明的 handle
void** __cudaRegisterFatBinary(void *fatCubin);

// 注册一个 kernel function：建立 host 函数指针和 device kernel 名之间的映射
void   __cudaRegisterFunction(
    void **fatCubinHandle,    // fatbin handle
    const char *hostFun,      // host 侧的函数桩（stub）的地址
    char *deviceFun,          // device kernel 的名称
    const char *deviceName,   // 可选的名称（可能和 deviceFun 不同）
    int thread_limit,         // 线程限制
    uint3 *tid, uint3 *bid,   // 其他元数据
    dim3 *bDim, dim3 *gDim,
    int *wSize
);

// 注销
void   __cudaUnregisterFatBinary(void **fatCubinHandle);
```

### 3.3 在 vGPU 项目中拦截这些函数的策略

```cpp
// libcudart_proxy.cpp 中的拦截逻辑（简化版）

// 全局的 fatbin registry（本地记录）
static std::map<void*, FatbinInfo> g_fatbin_registry;
static std::map<void*, std::string> g_func_registry;  // host_fun → device_name
static std::map<void*, uint64_t> g_module_map;         // host_fatbin → server module_handle

extern "C" void** __cudaRegisterFatBinary(void *fatCubin) {
    // 1. 获取 fatbin 数据
    //    fatCubin 指向 __fatDeviceText 或类似的结构
    //    需要根据 CUDA 版本确定如何提取数据和大小

    // 2. 保存到本地（为后续 register function 用）
    void** handle = new void*(fatCubin);  // 生成虚拟 handle
    g_fatbin_registry[handle] = { data, size };

    // 3. 通过 gRPC 发给 server 加载
    RegisterFatBinaryRequest req;
    req.set_session_id(g_session_id);
    req.set_fatbin(fatbin_data, fatbin_size);

    RegisterFatBinaryReply reply;
    g_stub->RegisterFatBinary(&context, req, &reply);

    // 4. 记录 server 返回的 module_handle
    g_module_map[handle] = reply.module_handle();

    return handle;
}

extern "C" void __cudaRegisterFunction(
    void **fatCubinHandle,
    const char *hostFun,
    char *deviceFun,
    const char *deviceName,
    int thread_limit,
    uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim,
    int *wSize
) {
    // 提取 kernel 名称（优先用 deviceName，其次 deviceFun）
    std::string kernel_name = deviceName ? deviceName : deviceFun;

    // 记录 host 函数指针到 kernel 名称的映射
    uint64_t host_fun_id = reinterpret_cast<uint64_t>(hostFun);
    g_func_registry[(void*)host_fun_id] = kernel_name;

    // 通过 gRPC 告诉 server
    RegisterFunctionRequest req;
    req.set_session_id(g_session_id);
    req.set_module_handle(g_module_map[fatCubinHandle]);
    req.set_host_fun_id(host_fun_id);
    req.set_device_name(kernel_name);

    StatusReply reply;
    g_stub->RegisterFunction(&context, req, &reply);

    // 打印调试日志（非常有用！）
    fprintf(stderr, "[vgpu] register function host=%p device=%s\n",
            hostFun, kernel_name.c_str());
}

// cudaLaunchKernel 被拦截时：
extern "C" cudaError_t cudaLaunchKernel(
    const void *func, dim3 grid, dim3 block,
    void **args, size_t sharedMem, cudaStream_t stream
) {
    uint64_t host_fun_id = reinterpret_cast<uint64_t>(func);

    // 查本地 registry 获取 kernel 名称
    std::string kernel_name = g_func_registry[(void*)host_fun_id];

    // 发 RPC
    LaunchKernelRequest req;
    req.set_session_id(g_session_id);
    req.set_device_name(kernel_name);
    // ... 设置 grid, block, args, shared_mem, stream
}
```

---

## 4. C++ Name Mangling（名称修饰）

### 4.1 为什么 kernel 名是像 `_Z10vectorAddPfS_S_i` 这样的

C++ 支持函数重载（同名函数不同参数）。为了让链接器区分 `void foo(int)` 和 `void foo(float)`，C++ 编译器把函数名"修饰"成包含参数类型信息的字符串。

```cpp
// 原始函数名
void vectorAdd(float *A, float *B, float *C, int N);

// C++ 编译器生成的符号名（用 nm 可以看到）
_Z10vectorAddPfS_S_i
// 解码：
// _Z         = C++ 符号前缀
// 10vectorAdd = 函数名 10 个字符 "vectorAdd"
// Pf          = 第一个参数 float*
// S_          = 第二个参数 float*（S_ 表示"和上一个参数一样"）
// S_          = 第三个参数 float*
// i           = 第四个参数 int
```

可以用 `c++filt` 工具还原：

```bash
echo "_Z10vectorAddPfS_S_i" | c++filt
# 输出：vectorAdd(float*, float*, float*, int)
```

### 4.2 在 vGPU 中的影响

Client 拦截到的 `__cudaRegisterFunction` 参数中，`deviceFun` 和 `deviceName` 可能就是 mangled name。这是正常的，server 加载 module 后用 mangled name 查找 kernel 即可。

---

## 5. 构建工具：CMake、gRPC、Protobuf

### 5.1 CMake 基础

CMake 是一个跨平台的构建系统生成器。你写一个 `CMakeLists.txt` 描述项目结构，CMake 自动生成 Makefile（Linux）或 Visual Studio 工程（Windows）。

**最小 CMakeLists.txt 示例**：

```cmake
cmake_minimum_required(VERSION 3.16)
project(vgpu LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 查找依赖包
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 编译 proto 文件
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS proto/vgpu.proto)
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS proto/vgpu.proto)

# 构建 server
add_executable(vgpu_server
    server/main.cpp
    server/cuda_executor.cpp
    server/session.cpp
    server/memory_manager.cpp
    ${PROTO_SRCS} ${GRPC_SRCS}
)
target_link_libraries(vgpu_server
    gRPC::grpc++
    CUDA::cuda_driver     # 链接 libcuda.so（Driver API）
)

# 构建 client 拦截库
add_library(cudart_proxy SHARED
    client/libcudart_proxy.cpp
    client/runtime_intercept.cpp
    client/fatbin_registry.cpp
    client/grpc_client.cpp
    ${PROTO_SRCS} ${GRPC_SRCS}
)
target_link_libraries(cudart_proxy
    gRPC::grpc++
)
```

**常用 CMake 命令**：

```bash
# 在 build 目录中配置项目
mkdir build && cd build
cmake ..

# 编译
cmake --build . -j$(nproc)

# 只编译特定 target
cmake --build . --target vgpu_server
```

### 5.2 Protobuf 代码生成

写一个 `.proto` 文件后，用 `protoc` 工具生成 C++ 代码：

```bash
# 安装 protobuf 编译器
sudo apt install protobuf-compiler

# 生成 C++ 代码
protoc --cpp_out=. vgpu.proto
# 生成 vgpu.pb.h 和 vgpu.pb.cc（消息类型的序列化/反序列化代码）
```

生成的代码中：
- `vgpu.pb.h/cc`：包含 `MallocRequest`、`MallocReply` 等消息类的定义
- 每个消息类有 `set_xxx()`、`xxx()` getter/setter
- 自动处理序列化（`SerializeToString`）和反序列化（`ParseFromString`）

### 5.3 gRPC 代码生成

```bash
# 安装 gRPC 插件
sudo apt install protobuf-compiler-grpc

# 生成 gRPC service 代码
protoc --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` vgpu.proto
# 生成 vgpu.grpc.pb.h 和 vgpu.grpc.pb.cc（RPC service 的 stub 和 service 基类）
```

生成的代码中：
- `VgpuRuntime::Stub`：client 用的代理类，有 `Malloc()`、`Free()` 等方法
- `VgpuRuntime::Service`：server 用的抽象基类，你需要继承并实现 `Malloc()`、`Free()` 等方法

### 5.4 构建环境准备（Ubuntu）

```bash
# 基础构建工具
sudo apt install build-essential cmake

# CUDA Toolkit（需要和 nvidia-smi 显示的版本匹配）
# 推荐从 NVIDIA 官方源安装
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6

# gRPC 和 Protobuf
sudo apt install protobuf-compiler libgrpc++-dev libprotobuf-dev
# 或者从源码编译以获得最新版本
```

验证安装：

```bash
nvcc --version
nvidia-smi
cmake --version
protoc --version
```

---

## 6. 调试工具速查

### 6.1 查看可执行文件/库文件信息

```bash
# 查看文件类型
file ./vector_add
# 输出: ELF 64-bit LSB executable, x86-64, ...

# 查看动态库依赖
ldd ./vector_add
# 输出: libcudart.so.12 => /usr/local/cuda-12/lib64/libcudart.so.12

# 查看导出符号
nm -D ./libcudart_proxy.so
# 列出所有动态符号（函数名等）

# 查看可执行文件中的 fatbin
cuobjdump ./vector_add
# 列出嵌入的 GPU 代码

# 查看 fatbin 详细信息
cuobjdump -ptx ./vector_add
# 导出嵌入的 PTX 代码

# 查看 ELF 段信息
readelf -S ./vector_add
# 可以看到 .nv_fatbin 等 CUDA 特有的段
```

### 6.2 运行时调试

```bash
# 查看 CUDA 程序加载了哪些库
LD_DEBUG=libs ./vector_add 2>&1 | grep -i cuda

# 查看程序中的所有系统调用
strace ./vector_add 2>&1 | head -50

# 查看动态链接器的符号解析过程
LD_DEBUG=symbols ./vector_add 2>&1 | grep cudaMalloc
```

### 6.3 NVIDIA 工具

```bash
# GPU 状态
nvidia-smi

# 持续监控 GPU（类似 htop）
nvidia-smi -l 1

# 查看 GPU 拓扑
nvidia-smi topo -m

# 查看 CUDA 相关信息
nvidia-smi -q | head -30

# 列出系统中所有 CUDA 相关库
ldconfig -p | grep -E 'libcuda|libcudart'
```

---

## 7. vGPU 项目文件编译关系图

```text
proto/vgpu.proto
    │
    │ protoc                    protoc + grpc plugin
    v                           v
vgpu.pb.h + vgpu.pb.cc      vgpu.grpc.pb.h + vgpu.grpc.pb.cc
(消息类型)                   (RPC service stub + 基类)
    │                           │
    └────────────┬──────────────┘
                 │ #include
    ┌────────────┼────────────┐
    v            v            v
client/      server/       common/
(拦截库)      (server)      (共享)

client 编译：
  libcudart_proxy.so
    ← client/*.cpp + proto generated code
    ← gRPC::grpc++ (gRPC client)
    （不链接 CUDA！client 侧不需要 GPU）

server 编译：
  vgpu_server
    ← server/*.cpp + proto generated code
    ← gRPC::grpc++ (gRPC server)
    ← CUDA::cuda_driver (libcuda.so - Driver API)
    ← CUDA::cudart (libcudart.so - 可能需要，取决于实现)
```

---

## 8. 关键概念速查

| 概念 | 一句话解释 |
|------|-----------|
| nvcc | NVIDIA CUDA 编译器，编译 .cu 文件 |
| PTX | GPU 的"汇编语言"，中间表示，文本格式 |
| CUBIN | GPU 的"机器码"，二进制，架构特定 |
| Fatbin | 包含 PTX + 多架构 cubin 的"胖"二进制包 |
| `--cudart shared` | nvcc 选项，动态链接 cudart（LD_PRELOAD 的必要前置条件） |
| Name Mangling | C++ 编译器修改函数名以支持重载 |
| `cuobjdump` | 工具，查看 CUDA 可执行文件中嵌入的 GPU 代码 |
| `nm -D` | 查看 .so 导出的符号 |
| `ldd` | 查看程序依赖哪些动态库 |
| CMake | 跨平台构建系统，通过 CMakeLists.txt 生成 Makefile |
| protoc | Protocol Buffers 编译器，.proto → .pb.h/.pb.cc |
| gRPC plugin | protoc 插件，.proto → .grpc.pb.h/.grpc.pb.cc |

---

> **上一篇文档**：[CUDA Runtime API 与 Driver API 对比](basics_cuda_api_comparison.md)
>
> 至此，五份基础文档涵盖了 vGPU 设计文档中涉及的全部前置知识。建议阅读顺序：
> 1. [GPU 与 CUDA 基础概念](basics_gpu_cuda.md)
> 2. [操作系统基础：动态链接与 LD_PRELOAD](basics_os_dynamic_linking.md)
> 3. [计算机网络基础：TCP、RPC 与 gRPC](basics_network_rpc.md)
> 4. [CUDA Runtime API 与 Driver API 对比](basics_cuda_api_comparison.md)
> 5. [CUDA 编译工具链：nvcc、PTX、Fatbin](basics_compilation_toolchain.md)（本文）
