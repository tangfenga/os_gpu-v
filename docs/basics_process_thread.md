# 进程与线程：并发编程基础

本文讲解进程和线程的概念，以及多线程编程中的同步问题。这些都是 vGPU server 端并发处理的核心基础。

---

## 1. 进程（Process）

### 1.1 进程是什么：一个"正在运行的程序"

你在终端输入 `./a.out` 后，操作系统做了这些事：

```text
1. 读取 a.out 可执行文件（ELF 格式）
2. 创建一个新的"进程"——就像填一张表格，记录这个程序的状态
3. 给进程分配一个独立的虚拟地址空间
4. 加载代码和数据到内存
5. 加载需要的 .so 动态库
6. 找到 main 函数的入口
7. 开始执行第一条指令
```

**进程 = 程序代码 + 数据 + 虚拟地址空间 + 操作系统跟踪的状态。**

### 1.2 每个进程有什么

| 资源 | 说明 |
|------|------|
| **PID** | 进程 ID，唯一标识，`ps aux` 可以看到 |
| **虚拟地址空间** | 代码段、数据段、堆、栈——独立的 |
| **文件描述符** | 打开的文件、socket 连接 |
| **工作目录** | `getcwd()` 返回的当前目录 |
| **环境变量** | `PATH`、`HOME`、`VGPU_SERVER` 等 |

### 1.3 进程之间的隔离

进程 A 和进程 B 之间天然隔离：
- A 不能读写 B 的内存（除非用特殊的共享内存机制）
- A 崩溃不会直接导致 B 崩溃（除非 B 依赖 A）
- A 的 `0x7fff1234` 和 B 的 `0x7fff1234` 是不同的物理地址

**这就是"客户机"和"宿主机"的基本隔离机制**——即使它们在同一个 WSL Ubuntu 里，client 进程和 server 进程也是独立的进程，有不同的虚拟地址空间，天然隔离开。

---

## 2. 线程（Thread）

### 2.1 线程是什么：进程内的"轻量级执行流"

一个进程可以有多个线程。所有线程共享进程的虚拟地址空间，但各自有自己的栈和寄存器。

```text
进程（一个程序实例）
├── 主线程（从 main 开始执行）
├── 线程 A（接收网络请求）
├── 线程 B（处理请求 1）
└── 线程 C（处理请求 2）

所有线程共享：
  - 代码段（同一份程序指令）
  - 数据段（全局变量）
  - 堆（malloc 的都在这里，所有线程可见）
  - 文件描述符

每个线程私有：
  - 自己的栈
  - 自己的寄存器（RSP、RIP 等）
  - 线程局部存储（TLS）
```

### 2.2 进程 vs 线程

| | 进程 | 线程 |
|---|---|---|
| 创建开销 | 高（复制整个地址空间） | 低（只分配栈） |
| 切换开销 | 高（切换地址空间） | 低（共享地址空间） |
| 通信方式 | 管道/socket/共享内存 | 直接读写共享变量 |
| 隔离性 | 强（一个崩了不影响） | 弱（一个线程崩了整个进程崩） |
| 地址空间 | 独立 | 共享 |

### 2.3 线程在 vGPU 项目中的角色

gRPC 框架内部使用线程池。每个 RPC 请求到达时，gRPC 分配一个线程来处理它：

```text
gRPC Server 线程模型（简化）：

请求 1："Malloc session=1 size=1MB"
请求 2："Malloc session=2 size=2MB"
请求 3："Memcpy session=1 H2D ..."

gRPC 线程池：
  线程 A → 处理请求 1 → cuMemAlloc(...)
  线程 B → 处理请求 2 → cuMemAlloc(...)
  线程 C → 处理请求 3 → cuMemcpyHtoD(...)

这些线程同时运行，都在 server 进程内，都共享 session map。
```

**这意味着多个线程可能同时读写 session map**（比如两个线程同时访问 `sessions_[1]`），这就引出了并发安全的问题。我们马上会讲到。

---

## 3. 多线程编程的核心问题：竞态条件（Race Condition）

### 3.1 一个致命场景

```cpp
// 全局计数器
int counter = 0;

// 线程 A 和线程 B 同时执行 counter++：
void increment() {
    counter++;  // 看似一条语句，实际是三步：
    //   temp = counter    (从内存读到寄存器)
    //   temp = temp + 1   (在寄存器中加 1)
    //   counter = temp    (写回内存)
}

// 时间线（两线程交错执行）：
// 时刻 1：线程 A 读 counter → temp_A = 0
// 时刻 2：线程 B 读 counter → temp_B = 0   ← 也读到 0！
// 时刻 3：线程 A temp_A = 0 + 1 = 1
// 时刻 4：线程 B temp_B = 0 + 1 = 1       ← 算出来也是 1！
// 时刻 5：线程 A counter = 1
// 时刻 6：线程 B counter = 1               ← 覆盖了！
// 结果：counter = 1，但预期是 2
```

两个线程各自做了一次 `counter++`，但最终结果只加了 1。这就是**竞态条件**——多个线程同时访问共享数据，结果取决于线程的"运气"（谁先执行完）。

### 3.2 竞态条件的特点

- **很难复现**：可能运行 10000 次都不触发，第 10001 次才崩溃
- **结果不确定**：取决于 CPU 调度的精确时刻
- **编译器优化可能放大问题**：编译器认为 `counter++` 是原子的（但不是），可能做更激进的优化让情况更糟

---

## 4. 互斥锁（Mutex）——解决竞争条件

### 4.1 锁的基本思想

```cpp
#include <mutex>

int counter = 0;
std::mutex mtx;  // 互斥锁

void increment() {
    mtx.lock();   // 获取锁（如果别人拿着，就等）
    counter++;    // ← 这段代码一次只有一个线程能执行
    mtx.unlock(); // 释放锁
}

// 等价写法（推荐，自动加锁/解锁）：
void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    // lock_guard：构造时自动 lock()，析构时自动 unlock()
    counter++;
}   // lock_guard 析构，自动 unlock
```

**锁就是"厕所门"的类比**：
- 进去（`lock`）：如果门开着，进去，关门上锁
- 其他人想进（`lock`）：门锁着，在外面排号
- 出来（`unlock`）：开锁开门，下一个排队的人进去

### 4.2 锁保护的不是代码，是数据

重要概念：锁保护的是**数据**，不是代码段。应该给每个共享数据配一个锁。

```cpp
// ✅ 正确的设计：每个 session 有自己的锁
struct Session {
    uint64_t id;
    std::mutex mu;              // 保护这个 session 的所有共享表
    std::unordered_map<uint64_t, DeviceAllocation> allocations;  // 受 mu 保护
    std::unordered_map<uint64_t, CUstream> streams;              // 受 mu 保护
};

// 访问 session 资源前加锁
void free_allocation(Session *session, uint64_t vptr) {
    std::lock_guard<std::mutex> lock(session->mu);  // 锁这个 session

    auto it = session->allocations.find(vptr);
    if (it == session->allocations.end()) {
        return;  // 不存在
    }

    cuMemFree(it->second.real_ptr);
    session->allocations.erase(it);
}  // lock_guard 析构，自动释放锁
```

### 4.3 项目中的锁设计（来自设计文档 §8.2）

```cpp
// 四种不同粒度的锁
std::mutex sessions_mu_;         // 保护全局的 session map（加了哪个 session）
                                 // 每个 session 内部的 mutex：保护一个 session 的资源表
std::mutex primary_ctx_mu_;      // 只保护 context 初始化（一次性的）
std::mutex log_mu_;              // 保护日志输出（同时只有一个线程写日志）
```

### 4.4 死锁（Deadlock）

```cpp
std::mutex A, B;

// 线程 1：
void thread1() {
    A.lock();   // 获取锁 A
    B.lock();   // 尝试获取锁 B（被线程 2 持有着，等待...）
    // ...
    B.unlock();
    A.unlock();
}

// 线程 2：
void thread2() {
    B.lock();   // 获取锁 B
    A.lock();   // 尝试获取锁 A（被线程 1 持有着，等待...）
    // ...
    A.unlock();
    B.unlock();
}

// 线程 1：拿着 A 等 B
// 线程 2：拿着 B 等 A
// → 两个都在等对方释放 → 死锁！
```

**避免死锁的规则**：所有线程以相同顺序获取锁。如果设计文档说先锁 `sessions_mu_` 再锁 `session->mu`，那就永远是这个顺序。

---

## 5. 线程局部存储（TLS）

### 5.1 每个线程私有的变量

```cpp
// thread_local 变量：每个线程有自己的独立副本
thread_local int last_error = 0;

void do_something() {
    last_error = 42;     // 只影响当前线程的副本
}

// 线程 A 调用 do_something → 线程 A 的 last_error = 42
// 线程 B 调用 do_something → 线程 B 的 last_error = 0（不变）
```

### 5.2 CUDA Runtime Error 就是 TLS

CUDA Runtime 的 last error 是线程局部状态，每个线程独立：

```cpp
// 这是 CUDA Runtime 的语义（简化）：
thread_local cudaError_t tls_last_error = cudaSuccess;

cudaError_t cudaGetLastError() {
    cudaError_t e = tls_last_error;  // 读当前线程的
    tls_last_error = cudaSuccess;    // 重置
    return e;
}

cudaError_t cudaPeekAtLastError() {
    return tls_last_error;           // 只读不重置
}

// 每个拦截 API 返回前都要更新
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    // ...
    cudaError_t err = ...; // 操作结果
    tls_last_error = err;  // 设置当前线程的 last error
    return err;
}
```

**为什么必须用 TLS？** 因为多个线程可能同时调 CUDA API。如果 last_error 是全局变量，线程 A 的 `cudaMalloc` 设置了错误，线程 B 的 `cudaGetLastError` 可能读到线程 A 的错。

---

## 6. 原子操作（Atomic）

竞态条件也可以用原子操作解决：

```cpp
#include <atomic>

// 不用 mutex，也能安全地做 counter++
std::atomic<int> counter(0);  // 原子变量

void increment() {
    counter++;  // 这个 ++ 是原子的——CPU 保证不会被打断
}

// 但原子变量只适合简单操作（++、--、赋值、CAS）
// 复杂操作（遍历 map、分配+插入）还是需要 mutex
```

在 vGPU 项目中，简单的计数器（如 `next_vptr_id`）可以用 `std::atomic<uint64_t>`。

---

## 7. `std::lock_guard` 和 `std::unique_lock`

```cpp
std::mutex mtx;

// lock_guard：轻量，不能手动解锁，离开作用域自动解锁
{
    std::lock_guard<std::mutex> lock(mtx);
    // ... 操作 ...
}  // 自动解锁

// unique_lock：更灵活，可以手动 unlock/lock
{
    std::unique_lock<std::mutex> lock(mtx);
    // ... 操作 ...
    lock.unlock();   // 可以提前解锁
    // 做不需要锁的事 ...
    lock.lock();     // 再次加锁
}  // 如果还锁着，自动解锁
```

**优先用 `lock_guard`**，够用了。只有需要中途解锁时才用 `unique_lock`。

---

## 8. 在 vGPU 项目中的实际应用

### 8.1 server 收到多请求时的时序

```text
时间 →
线程 A（处理 client 1 的 Malloc）：
  [lock session1.mu]
  cuMemAlloc(...)
  session1.allocations[vptr] = {...}
  [unlock session1.mu]
                            [lock session1.mu]
                            cuLaunchKernel(...)
                            [unlock session1.mu]

线程 B（处理 client 1 的 Memcpy）：
       [等待 session1.mu...]
       [lock session1.mu]
       cuMemcpyHtoD(...)
       [unlock session1.mu]
                                       [等待 session1.mu...]
                                       [lock session1.mu]
                                       查表 session1.allocations[vptr]
                                       [unlock session1.mu]

线程 C（处理 client 2 的 Malloc）：
  [lock session2.mu]         ← 不同锁！不冲突，真正的并行
  cuMemAlloc(...)
  session2.allocations[vptr] = {...}
  [unlock session2.mu]
```

不同 session 用不同的锁，所以 client 1 和 client 2 的操作可以真正并行。

### 8.2 CUDA Context 与线程：重要限制

CUDA Driver API 中，每个线程需要显式设置当前 Context：

```cpp
// 每个 gRPC 线程在调用 CUDA 前必须：
cuCtxSetCurrent(primary_ctx);  // 设置当前线程的 CUDA Context

// 然后才能：
cuMemAlloc(&ptr, size);
```

**为什么？** CUDA Context 是线程绑定的——每个线程执行 CUDA 操作前，必须告诉驱动"我要用哪个 Context"。如果线程之前设置过另一个 Context，cuMemAlloc 可能会去错误的地方分配。

因为 server 用的是全局一个 primary context，所以每个线程都调 `cuCtxSetCurrent(primary_ctx)` 即可。

### 8.3 推荐实现：设计文档 §8.2 的锁设计

```cpp
// 全局的 session map
std::unordered_map<uint64_t, Session*> sessions_;
std::mutex sessions_mu_;  // 保护 sessions_ 的插入/删除

// 查找 session（需要锁 sessions_mu_）
Session* find_session(uint64_t sid) {
    std::lock_guard<std::mutex> lock(sessions_mu_);
    auto it = sessions_.find(sid);
    if (it != sessions_.end()) {
        return it->second;  // 返回 session 指针
    }
    return nullptr;
}

// 操作 session 内的资源（不需要锁 sessions_mu_，但需要锁 session->mu）
grpc::Status Malloc(...) {
    Session *session = find_session(sid);  // 找 session 时锁了 sessions_mu_
    if (!session) {
        return grpc::Status(grpc::NOT_FOUND, "session not found");
    }

    cuCtxSetCurrent(primary_ctx);  // 设置 CUDA Context

    CUdeviceptr real_ptr;
    cuMemAlloc(&real_ptr, size);   // 吃时间的操作，不要在锁内！

    std::lock_guard<std::mutex> lock(session->mu);  // 只在改表时加锁
    uint64_t vptr = next_vptr_id_++;
    session->allocations[vptr] = {
        .vptr = vptr,
        .real_ptr = real_ptr,
        .size = size,
        .freed = false
    };

    reply->set_vptr(vptr);
    return grpc::Status::OK;
}
```

**要点**：CUDA 调用（吃时间的操作）在外面做，只在修改 session 内部表时加锁。

---

## 9. 速查表

| 概念 | 一句话解释 |
|------|-----------|
| 进程 | 一个正在运行的程序，有独立的虚拟地址空间 |
| 线程 | 进程内的执行流，共享地址空间，独立栈 |
| PID | 进程 ID，`ps aux` 或 `getpid()` 获取 |
| 竞态条件 | 多线程同时读写共享数据，结果不确定 |
| mutex | 互斥锁，一次只有一个线程能进入临界区 |
| lock_guard | RAII 封装，自动加锁/解锁 |
| 死锁 | 两个线程各自等对方释放锁，永远卡住 |
| thread_local | 每个线程独立副本的变量 |
| atomic | CPU 保证操作的原子性（不被中断） |
| TLS | Thread-Local Storage，线程局部存储 |

---

> **上一份文档**：[内存深入：栈、堆、虚拟内存](basics_memory_deep_dive.md)
> **下一份文档**：[Linux 开发环境与命令行基础](basics_linux_dev.md)
