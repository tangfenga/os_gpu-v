# 操作系统基础：动态链接与 LD_PRELOAD

本文面向没有操作系统课程基础的读者，从程序是如何运行的最基础概念讲起，逐步解释动态链接库和 LD_PRELOAD 的原理。这是理解 vGPU 项目 client 拦截层的必要知识。

---

## 1. 程序是怎么变成进程的

### 1.1 从源代码到可执行文件

你写的 C 代码先经过**编译**变成**可执行文件**。但那只是简化说法，实际过程分四步：

```text
源代码(.c) ──预处理──> 展开后的代码 ──编译──> 汇编代码(.s)
                                                    │
                                                    │ 汇编
                                                    v
可执行文件(./a.out) <──链接── 目标文件(.o) <────────┘
```

以最简单的程序为例：

```c
// hello.c
#include <stdio.h>
int main() {
    printf("Hello, World!\n");
    return 0;
}
```

```bash
# 一步到位（实际做了上面四步）
gcc hello.c -o hello

# 运行
./hello
# 输出：Hello, World!
```

仔细想一个问题：`printf` 的代码在哪里？

你只是 `#include <stdio.h>` 告诉了编译器 `printf` "长什么样"（声明），但 `printf` 真正的实现代码并不在你的 `hello.c` 里。那它在哪里？

### 1.2 库 = 别人写好的代码

`printf` 的代码在 C 标准库（`libc.so`）里。C 标准库是操作系统自带的，几乎每个 C 程序都会用到。

**库（Library）** 就是"别人写好的、你可以直接用的代码的集合"。

库有两种形式：

| | 静态库 | 动态库（共享库） |
|---|---|---|
| 文件后缀 | `.a`（Linux）或 `.lib`（Windows） | `.so`（Linux）或 `.dll`（Windows） |
| 怎么用 | 编译时复制到你的可执行文件中 | 程序运行时才加载 |
| 可执行文件大小 | 更大（包含了库代码） | 更小（不包含库代码） |
| 更新库 | 需要重新编译程序 | 替换 .so 即可，不用重编译 |
| 共享 | 每个程序各自一份 | 多个程序共用同一份内存 |

**类比**：
- 静态库 = 把整本词典复印一份装订在你的书里。书变厚了，但不需要找原词典。
- 动态库 = 你的书里只说"查《现代汉语词典》第 233 页"，你去图书馆找那本词典。书很薄，但必须词典在图书馆才能用。

### 1.3 动态链接的过程

当你运行 `./hello` 时：

```text
1. 操作系统读取 hello 可执行文件
2. 操作系统检查："这个程序需要哪些 .so 文件？"
   （hello 说：我需要 libc.so.6）
3. 动态链接器（ld.so）找到 /usr/lib/libc.so.6
4. 把 libc.so.6 加载到内存
5. hello 中的 printf 调用 → 指向内存中 libc.so 的 printf 实现
6. 程序开始运行
```

可以查看一个程序依赖哪些动态库：

```bash
ldd ./hello
# 输出类似：
#   linux-vdso.so.1  =>  (0x00007fff...)
#   libc.so.6        =>  /lib/x86_64-linux-gnu/libc.so.6 (0x00007f...)
#   /lib64/ld-linux-x86-64.so.2 (0x00007f...)
```

也可以查看一个 .so 导出了哪些函数名：

```bash
nm -D /lib/x86_64-linux-gnu/libc.so.6 | grep printf
# 输出类似：
#   000000000005c780 T printf
```

---

## 2. .so 文件到底是啥

### 2.1 ELF 格式

Linux 的可执行文件和 .so 文件都是 **ELF**（Executable and Linkable Format）格式。

ELF 文件里包含：
- **代码段（.text）**：编译好的机器指令
- **数据段（.data、.bss）**：全局变量
- **符号表**：记录了这个文件"提供了哪些函数"和"需要哪些外部函数"
- **重定位信息**：告诉动态链接器哪些地址需要修正

### 2.2 符号（Symbol）

**符号** 就是一个函数名或全局变量名。

每个 .so 文件导出一些符号（它提供的函数），也可能需要一些外部符号（它依赖的其他 .so 的函数）。

```text
libcudart_proxy.so:
  提供的符号（导出）: cudaMalloc, cudaFree, cudaMemcpy, ...
  需要的符号（依赖）: grpc::CreateChannel, ...

libcudart.so（真实 CUDA Runtime）:
  提供的符号（导出）: cudaMalloc, cudaFree, cudaMemcpy, ...
  需要的符号（依赖）: libcuda.so 中的 Driver API 函数
```

### 2.3 动态链接器如何找到函数

当程序调用 `cudaMalloc` 时：

```text
1. 动态链接器在所有已加载的 .so 中搜索符号 "cudaMalloc"
2. 找到后在内部记录："cudaMalloc 的代码在 libcudart.so 偏移 0x12345 处"
3. 把程序中对 cudaMalloc 的调用指向那个地址
4. 下次调用直接跳过去，很快
```

这个过程叫**符号解析**（Symbol Resolution）。

---

## 3. LD_PRELOAD：偷梁换柱

### 3.1 符号解析的优先级

动态链接器搜索符号时有一个优先级：

```text
1. LD_PRELOAD 指定的 .so（最先搜索！）
2. 可执行文件本身
3. /etc/ld.so.cache 中记录的库
4. /lib 和 /usr/lib 中的库
```

**`LD_PRELOAD` 是优先级最高的**——在搜索任何其他库之前，动态链接器先看 `LD_PRELOAD` 指定的 .so 中有没有这个符号。

### 3.2 LD_PRELOAD 能干什么

```bash
# 正常运行（使用真实的 libcudart.so 中的 cudaMalloc）
./vector_add

# 使用 LD_PRELOAD（优先用我们的 libcudart_proxy.so 中的 cudaMalloc）
LD_PRELOAD=./libcudart_proxy.so ./vector_add
```

加了 `LD_PRELOAD` 后：
- 程序调用 `cudaMalloc` → 调用的是 `libcudart_proxy.so` 里的（我们的拦截版本）
- 我们的 `cudaMalloc` 内部可以做任何事（比如发送 gRPC 请求），然后返回一个假的结果
- 程序完全不知道 `cudaMalloc` 被"偷换"了

**类比**：
- 正常情况：你打电话给餐馆（程序调用函数）→ 餐馆接电话（真实函数执行）
- LD_PRELOAD：你打电话给餐馆 → 中间人先接了电话（拦截函数），假装是餐馆，记录你的订单后转给真正的厨房（转发给 server），再把结果告诉你
- 你完全不知道接电话的不是餐馆

### 3.3 一个简单的 LD_PRELOAD 例子

假设我们想拦截 `malloc`，记录每次分配的大小：

```c
// malloc_hook.c
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>

// 原始的 malloc 函数指针
static void *(*real_malloc)(size_t) = NULL;

// 我们自己的 malloc（拦截版本）
void *malloc(size_t size) {
    // 第一次调用时，获取真实的 malloc
    if (!real_malloc) {
        real_malloc = dlsym(RTLD_NEXT, "malloc");
    }

    // 做我们想做的事：记录日志
    printf("[HOOK] malloc(%zu) called\n", size);

    // 调用真实的 malloc 完成实际工作
    void *ptr = real_malloc(size);
    printf("[HOOK] malloc returned %p\n", ptr);
    return ptr;
}
```

编译和使用：

```bash
# 编译成共享库
gcc -shared -fPIC malloc_hook.c -o malloc_hook.so -ldl

# 使用 LD_PRELOAD 运行任何程序
LD_PRELOAD=./malloc_hook.so ls
# 你会看到每次 malloc 的日志！
```

### 3.4 dlsym(RTLD_NEXT, ...) 的作用

在上面的代码中，`dlsym(RTLD_NEXT, "malloc")` 的意思是：

> "跳过我自己（当前的 .so），找到下一个名为 `malloc` 的符号"

这样我们的拦截函数可以"转发"到真实的 `malloc`——先做我们的事（记录日志），再调用真实函数（真正分配内存）。

**如果不用 `RTLD_NEXT`**：我们的 `malloc` 调用 `dlsym(RTLD_DEFAULT, "malloc")` 可能会找到自己，造成无限递归。

在 vGPU 项目中，我们通常**不需要**转发到真实 cudart——因为 client 没有 GPU，真实 cudart 也只会失败。我们的拦截函数直接把请求发给 gRPC server。

---

## 4. 动态链接在 vGPU 项目中的应用

### 4.1 LD_PRELOAD 拦截 CUDA Runtime

```bash
# client 侧不支持直接 GPU，但普通 CUDA 程序仍然链接 libcudart.so
# 我们用自己的库"假装"是 libcudart
LD_PRELOAD=./libcudart_proxy.so ./vector_add
```

`libcudart_proxy.so` 导出了与 `libcudart.so` 完全相同的函数名：
- `cudaMalloc`
- `cudaFree`
- `cudaMemcpy`
- `cudaLaunchKernel`
- `__cudaRegisterFatBinary`
- `__cudaRegisterFunction`
- ...

动态链接器优先搜索 `LD_PRELOAD` 指定的库，找到了我们的版本，就不会再去搜索真实 `libcudart.so`。

### 4.2 为什么要用 `nvcc --cudart shared`

CUDA 程序有两种链接方式：

| 链接方式 | 编译命令 | 结果 |
|----------|---------|------|
| 静态链接 cudart | `nvcc -cudart static` 或默认 | cudart 代码嵌入可执行文件 |
| 动态链接 cudart | `nvcc --cudart shared` | 可执行文件只引用 libcudart.so，运行时加载 |

**只有动态链接时，LD_PRELOAD 才能拦截成功。** 如果 cudart 代码是静态链接进可执行文件的，那符号已经在程序内部，LD_PRELOAD 无法替换。

验证链接方式：
```bash
ldd ./vector_add
# 如果看到 libcudart.so => ...，说明是动态链接
# 如果看不到，说明是静态链接
```

### 4.3 vGPU client 库的结构

```text
libcudart_proxy.so 内部：
├── 拦截的 CUDA Runtime 函数（导出符号）
│   ├── cudaMalloc()           → 发 gRPC: Malloc → 返回虚拟指针
│   ├── cudaFree()             → 发 gRPC: Free
│   ├── cudaMemcpy()           → 发 gRPC: Memcpy（H2D 带数据，D2H 收数据）
│   ├── cudaLaunchKernel()     → 发 gRPC: LaunchKernel
│   ├── __cudaRegisterFatBinary() → 发 gRPC: RegisterFatBinary
│   └── ...
│
├── gRPC client（非导出，内部使用）
│   ├── 连接 server（127.0.0.1:50051）
│   ├── 序列化请求（protobuf）
│   └── 反序列化响应
│
└── 本地状态（线程局部）
    ├── tls_last_error（线程局部，记录最后一个 CUDA 错误）
    ├── session_id（与 server 的会话 ID）
    └── function registry（host_fun_id → device_name 映射）
```

---

## 5. 关键知识点速查

| 概念 | 一句话解释 |
|------|-----------|
| ELF | Linux 可执行文件和 .so 的文件格式 |
| .so 文件 | Linux 的动态库，类似 Windows 的 .dll |
| 符号 | 函数名或变量名，.so 通过符号表告知"我有哪些函数" |
| 动态链接 | 程序运行时才加载 .so，按符号名找函数 |
| LD_PRELOAD | 环境变量，指定优先加载的 .so，实现函数拦截 |
| dlsym | 动态查找符号，`RTLD_NEXT` 跳过自己找下一个 |
| ldd | 命令，查看程序依赖哪些 .so |
| nm -D | 命令，查看 .so 导出了哪些符号 |
| `-fPIC` | 编译选项，生成"位置无关代码"，是 .so 的必要条件 |

---

## 6. 自己动手试试

### 6.1 观察真实 CUDA 程序的链接

```bash
# 看看 nvcc 编译出的程序依赖什么
nvcc --cudart shared tools/vector_add_smoke.cu -o /tmp/test_cuda
ldd /tmp/test_cuda
# 注意观察 libcudart.so 是否在列表中

# 看看 libcudart.so 导出了什么
nm -D /usr/lib/x86_64-linux-gnu/libcudart.so | head -30
# 你会看到一堆 cudaMalloc, cudaFree, cudaMemcpy...
```

### 6.2 写一个 LD_PRELOAD 小实验

写一个简单的 C 程序，用 LD_PRELOAD 拦截 `write` 系统调用包装函数（`write` 是 `printf` 底层使用的），把所有输出变成大写。这种小练习能帮你直观理解"拦截"是怎么回事。

---

## 7. 常见疑问

**Q: LD_PRELOAD 和 LD_LIBRARY_PATH 有什么区别？**

`LD_LIBRARY_PATH`：指定去哪里找 .so（搜索路径），不影响优先级。
`LD_PRELOAD`：强制在搜索任何东西之前先加载指定的 .so，并且它的符号优先级最高。

**Q: 如果 libcudart_proxy.so 没有实现所有 CUDA 函数怎么办？**

程序调用未实现的函数时会找不到符号，动态链接器报错。所以 proxy 库要么实现所有程序会用到的函数，要么用一个"全部转发的 fallback"机制。

**Q: 程序静态链接了 cudart 还能拦截吗？**

不能通过 LD_PRELOAD 拦截。需要其他方法（如编译时替换、二进制改写），这不在 vGPU 项目初版范围内。

---

> **上一份文档**：[GPU 与 CUDA 基础概念](basics_gpu_cuda.md)
> **下一份文档**：[计算机网络基础：TCP、RPC 与 gRPC](basics_network_rpc.md)
