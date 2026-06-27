# 计算机网络基础：TCP、RPC 与 gRPC

本文面向没有计算机网络基础的读者，从网络是什么讲起，逐步介绍 TCP、RPC 概念、gRPC 和 Protocol Buffers。这是理解 vGPU 项目 client-server 通信机制的必要知识。

---

## 1. 网络是什么

### 1.1 两台电脑怎么通信

想象两台电脑 A 和 B，你想让 A 发一句话给 B。

最简单的方式：用一根网线把两台电脑连起来，通过网线传输数据。这就是**物理层**。

但光有物理连接不够——你需要约定一些规则：
- 数据怎么打包？（一次发多少字节？）
- B 怎么知道 A 开始/结束发送了？（消息边界）
- 如果传输途中数据损坏了怎么办？（错误检测）
- 如果网络拥堵了怎么办？（流量控制）

这些规则就是**网络协议（Protocol）**。

### 1.2 网络分层的简化理解

网络被设计成分层的，每层负责不同的事情。你不需要记住所有层，只需要理解这个思路：

```text
应用层：    你的程序（浏览器 / CUDA client）
  ↕          "我要访问 bilibili.com"
传输层：    TCP / UDP
  ↕          "把数据切成段，保证不丢、不乱序"
网络层：    IP
  ↕          "找到目标电脑（IP 地址）"
链路层：    以太网 / WiFi
  ↕          "通过网线/无线信号传输比特"
```

**类比**：你寄快递
- 应用层 = 你写了信的内容
- 传输层 = 把信装进信封，写上编号（确保不丢失不重排）
- 网络层 = 写上收件人地址（IP 地址）
- 链路层 = 邮政卡车实际运送

### 1.3 IP 地址和端口

**IP 地址**：电脑在网络上的"门牌号"。比如 `192.168.1.100`、`10.0.0.1`。特殊地址 `127.0.0.1` 永远指向"本机"（自己这台电脑）。

**端口（Port）**：一台电脑上通常同时运行很多程序（浏览器、游戏、CUDA server...），端口号用来区分"数据该给哪个程序"。端口范围 0~65535。

**IP + 端口 = 唯一标识了"网络中哪台电脑上的哪个程序"**：
```text
127.0.0.1:50051   →   本机上的 vgpu_server（监听 50051 端口）
192.168.1.5:50051 →   局域网内另一台电脑上的 vgpu_server
```

### 1.4 Unix Domain Socket（Unix 域套接字）

除了通过 IP 地址通信，在同一台 Linux 电脑上，进程间还可以通过**Unix Domain Socket**（也叫 Unix Socket）通信。

| | TCP Socket | Unix Domain Socket |
|---|---|---|
| 寻址方式 | IP 地址 + 端口号（如 `127.0.0.1:50051`） | 文件路径（如 `/tmp/vgpu.sock`） |
| 范围 | 可以跨机器 | 只能同一台机器 |
| 性能 | 经过 TCP/IP 协议栈 | 不经过网络层，更快 |
| 用途 | 跨机器通信 | 本机进程间通信 |

在 vGPU 项目的初版测试中，client 和 server 都在同一台笔记本上，用 Unix Domain Socket 可以避免不必要的网络开销。

---

## 2. TCP：可靠的传输

### 2.1 TCP 的两个核心保证

TCP（Transmission Control Protocol，传输控制协议）是网络上最常用的协议。它做两件关键的事：

**1. 保证数据完整送达**

你发的每个数据包，接收方都会确认"收到了"。如果发送方没收到确认，会自动重发。

**2. 保证数据顺序正确**

数据可能在网络中走不同的路径，先发的包可能后到。TCP 给每个包编号，接收方按编号重新排列，保证你读到的数据顺序和发送顺序一致。

### 2.2 TCP 是"面向连接"的

TCP 在通信之前需要先"握手"建立连接（三次握手）。通信结束后需要关闭连接（四次挥手）。

```text
client                          server
  |                               |
  |  --- SYN (我想连接你) --------> |
  |  <-- SYN+ACK (好的，我同意) --- |
  |  --- ACK (收到，开始通信) ----> |
  |                               |
  |  === 连接建立，开始传数据 ==== |
  |                               |
  |  --- FIN (我发完了，拜拜) ----> |
  |  <-- ACK (知道了) ------------ |
  |  <-- FIN (我也发完了) --------- |
  |  --- ACK (拜拜) -------------> |
  |                               |
```

### 2.3 TCP 是"字节流"

TCP 没有"消息"的概念。它只负责把字节从 A 传到 B。如果你发了 100 字节，接收方可能一次收到 50 字节，再收到 50 字节；也可能一次收到 100 字节。你需要自己处理消息边界。

这就是为什么很多基于 TCP 的协议（包括 gRPC）都需要自己定义"一条消息从哪里开始、到哪里结束"。

---

## 3. 客户端-服务器模型

### 3.1 C/S 模型

网络编程中最常见的模式：

```text
客户机（Client）                          服务器（Server）
  启动                                      启动
    |                                         |
    | --- 1. 连接请求 -------------------->  |  一直在监听
    | <--  2. 连接建立 --------------------  |
    | --- 3. 发送请求 ("做XXX") -----------> |
    |                                     处理请求
    | <--  4. 返回响应 ("做完了，结果如下") - |
    | --- 5. 断开连接 --------------------> |
```

**关键点**：
- Server 先启动，一直在某个端口上**监听**（listen），等待 client 连接
- Client 主动发起连接（connect）
- 请求-响应的交互模式

### 3.2 在 vGPU 项目中

```text
Client（CUDA 程序）              Server（vgpu_server）
   调用 cudaMalloc                  一直在监听 0.0.0.0:50051
     |                                   |
     | --- RPC: Malloc(size=1MB) ------> |
     |                                   调用 cuMemAlloc 在真实 GPU 上分配
     | <-- RPC: vptr=0x00010001 -------- |
     |                                   |
  返回虚拟指针给用户程序                  |
```

---

## 4. RPC：远程过程调用

### 4.1 函数调用 vs 远程调用

**普通函数调用**（同一进程内）：

```cpp
// client.c
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(3, 5);  // 直接在当前进程中执行
    printf("%d\n", result);   // 8
}
```

执行 `add(3, 5)` 时：
- CPU 跳转到 `add` 的代码地址
- 在栈上分配参数 a=3, b=5
- 执行加法
- 返回结果
- 全部在同一个进程、同一台机器上完成

**RPC（Remote Procedure Call，远程过程调用）**：

```cpp
// client.cpp
int main() {
    // 看起来像普通函数调用，实际发给了远程 server
    int result = stub.add(3, 5);
    printf("%d\n", result);  // 还是 8，但计算发生在另一台机器上
}
```

执行 `stub.add(3, 5)` 时：
- 把参数 (3, 5) 序列化成网络消息
- 通过网络发给 server
- Server 收到消息，反序列化参数
- Server 真正执行 a+b
- Server 把结果 (8) 序列化发给 client
- Client 返回结果

**你调用函数的代码几乎不变，但实际计算发生在远程机器上。**

### 4.2 RPC 的核心思想：让远程调用看起来像本地调用

RPC 框架的目标是隐藏网络通信的复杂性，让你写代码时感觉像在调用本地函数：

```text
你的代码：
    result = server.add(3, 5)     ← 看起来像普通函数调用

RPC 框架帮你做的事（你看不到）：
    ┌─ 序列化参数 (3, 5)
    ├─ 建立 TCP 连接
    ├─ 发送消息
    ├─ 等待响应
    ├─ 接收消息
    ├─ 反序列化结果 (8)
    └─ 返回 result = 8
```

### 4.3 为什么用 RPC 而不是原始 Socket

你可以直接用 TCP socket 收发数据：

```cpp
// 原始方式：自己处理一切
send(sockfd, "MALLOC 1048576\n", 15, 0);   // 发请求
recv(sockfd, buffer, 1024, 0);              // 收回复
// 手动解析 buffer 中的字符串，转换类型，处理错误...
```

问题是：
- 你需要自己设计消息格式（怎么表示"这是一个 Malloc 请求"？参数怎么编码？）
- 不同编程语言的数据类型不一样（C/C++ 的 struct vs Python 的 dict）
- 需要处理粘包、半包、大小端、超时、重连……

**RPC 框架帮你处理所有这些底层细节**，你只需要定义"有哪些操作、每个操作的参数和返回值是什么"，框架自动生成序列化/反序列化/网络传输代码。

---

## 5. gRPC 和 Protocol Buffers

### 5.1 gRPC 是什么

gRPC（Google Remote Procedure Call，因为 g 是 Google 的首字母，但 gRPC 本身是开源项目）是一个高性能的 RPC 框架。它由 Google 开发并开源。

**gRPC 的核心特点**：

| 特点 | 说明 |
|------|------|
| 使用 Protobuf 序列化 | 二进制格式，比 JSON/XML 更小更快 |
| 基于 HTTP/2 | 支持多路复用、双向流等特性 |
| 支持多语言 | C++、Python、Go、Java 等 |
| 自动生成代码 | 写一个 .proto 文件，gRPC 自动生成 client 和 server 的 C++ 类 |
| 支持流式传输 | 可以发一连串数据，不用发完一条等响应再发下一条 |

### 5.2 Protocol Buffers（Protobuf）

Protobuf 是一种**数据序列化格式**，也是 gRPC 默认使用的格式。你需要写一个 `.proto` 文件来描述"消息长什么样"和"有哪些 RPC 操作"：

```proto
// 定义一个消息类型（类似 C 的 struct）
message MallocRequest {
  uint64 session_id = 1;   // =1 是字段编号，不是默认值！
  uint64 size = 2;
}

message MallocReply {
  int32 cuda_error = 1;
  string message = 2;
  uint64 vptr = 3;          // 返回的虚拟指针
}

// 定义一个服务（有哪些 RPC 操作）
service VgpuRuntime {
  rpc Malloc(MallocRequest) returns (MallocReply);
}
```

然后 gRPC 工具自动生成 C++ 代码。你可以在代码中像这样使用：

```cpp
// Client 侧
MallocRequest req;
req.set_session_id(sid);
req.set_size(1048576);

MallocReply reply;
stub->Malloc(&context, req, &reply);  // 看起来像普通函数调用！

// reply.vptr() 就是 server 返回的虚拟指针
uint64_t vptr = reply.vptr();
```

```cpp
// Server 侧
grpc::Status Malloc(ServerContext* context,
                    const MallocRequest* request,
                    MallocReply* reply) override {
    uint64_t sid = request->session_id();
    size_t size = request->size();

    // 在真实 GPU 上分配
    CUdeviceptr real_ptr;
    cuMemAlloc(&real_ptr, size);

    // 生成虚拟指针并返回
    uint64_t vptr = allocate_vptr(sid, real_ptr);
    reply->set_cuda_error(cudaSuccess);
    reply->set_vptr(vptr);
    return grpc::Status::OK;
}
```

### 5.3 为什么 vGPU 项目选择 gRPC + Protobuf

1. **`.proto` 文件本身就是接口文档**：看一眼 proto 就知道支持哪些 API、参数是什么
2. **自动生成代码**：不用手写序列化/反序列化/网络收发，减少 bug
3. **强类型**：每个字段有明确类型（uint64, int32, bytes...），编译时就能发现类型错误
4. **跨语言**：如果以后想用 Python 写测试脚本，不需要重新定义协议
5. **支持 streaming**：后续优化大数据传输时，gRPC 的流式传输可以减少内存拷贝

### 5.4 gRPC 的通信过程

```text
Client                               Server
  |                                     |
  | 1. 创建 gRPC channel                |
  |    (连接 127.0.0.1:50051)           |
  | ----------------------------------> |
  |                                     | 2. 监听 50051 端口
  |                                     |    (grpc::Server)
  |                                     |
  | 3. 调用 stub->Malloc(req)           |
  |    ┌ 把 req 序列化成 protobuf 二进制 |
  |    ├ 通过 HTTP/2 发送               |
  |    └─────────────────────────────────> 4. 收到 HTTP/2 请求
  |                                          ┌ 反序列化成 MallocRequest
  |                                          ├ 调用 handler 函数
  |                                          ├ 执行 cuMemAlloc
  |                                          ├ 创建 MallocReply
  |                                          ├ 序列化
  | 5. 收到响应 <─────────────────────────── ┘ HTTP/2 返回
  |    ┌ 反序列化成 MallocReply
  |    └ 返回给调用者
```

### 5.5 gRPC 中的同步 vs 异步

**同步（初版使用）**：client 发一个请求，阻塞等待，直到 server 返回才继续。代码简单，但效率低。

```cpp
grpc::ClientContext ctx;
MallocReply reply;
grpc::Status status = stub_->Malloc(&ctx, request, &reply);
// 上面这行会阻塞等待，直到收到 server 回复才继续
```

**异步**：client 发请求后立即返回，可以同时发多个请求，响应到了通过回调通知。代码复杂，但效率高。

初版用同步即可，容易理解和调试。

---

## 6. protobuf 基础语法速查

### 6.1 字段编号（Field Number）

```proto
message MallocRequest {
  uint64 session_id = 1;   // =1 是字段编号，不是默认值！
  uint64 size = 2;         // =2 也是编号
}
```

**字段编号是 protobuf 二进制格式中的标识**，不是默认值。编号 1-15 占用 1 字节空间，16-2047 占用 2 字节。常用字段用 1-15。

### 6.2 常用数据类型

| proto 类型 | C++ 类型 | 说明 |
|-----------|---------|------|
| `int32` | `int32_t` | 32 位有符号整数 |
| `uint64` | `uint64_t` | 64 位无符号整数 |
| `bool` | `bool` | 布尔值 |
| `string` | `std::string` | 文本字符串（UTF-8） |
| `bytes` | `std::string` | 任意二进制数据 |
| `float` | `float` | 32 位浮点数 |
| `repeated` | `std::vector` 或迭代器 | 数组/列表 |
| `enum` | 枚举类 | 枚举类型 |

### 6.3 bytes 字段在 vGPU 中的特殊角色

```proto
message MemcpyRequest {
  bytes host_data = 5;  // H2D 时携带要发送的数据
}
```

`bytes` 可以存任意二进制数据。在 vGPU 中：
- **H2D**：client 把要传给 GPU 的数据（1MB 的浮点数组）塞进 `host_data`，server 收到后写入显存
- **D2H**：server 把从 GPU 读回的数据塞进 `MemcpyReply.host_data`，client 收到后写入用户内存

初版这样用很方便，但数据太大时（几百 MB）protobuf 序列化会多复制一次内存。所以文档中说第二版要改成 streaming。

---

## 7. vGPU 项目中的网络数据流

以一个完整的 vector add 运行为例：

```text
Client（无 GPU）                          Server（有 GPU）
─────────────────                        ─────────────────
1. cudaMalloc(&d_A, 4MB)
   → RPC: Malloc(size=4MB)
                                         → cuMemAlloc 在 GPU 分配 4MB
                                         → 生成虚拟指针 vptr=0x00010001
   ← RPC: vptr=0x00010001

2. cudaMemcpy(d_A, h_A, 4MB, H2D)
   → RPC: Memcpy(H2D, dst=0x00010001, data=<4MB 内容>)
                                         → 把收到数据写入 GPU 的 0x00010001
   ← RPC: cudaSuccess

3. cudaLaunchKernel(func, grid, block, args, 0, stream)
   → RPC: LaunchKernel(name="vecAdd", grid=..., args=...)
   → (args 中的虚拟指针已替换成 vptr)
                                         → 查表把 vptr 转成真实 CUdeviceptr
                                         → cuLaunchKernel(...)
   ← RPC: cudaSuccess

4. cudaDeviceSynchronize()
   → RPC: DeviceSynchronize
                                         → cuCtxSynchronize()
   ← RPC: cudaSuccess

5. cudaMemcpy(h_C, d_C, 4MB, D2H)
   → RPC: Memcpy(D2H, src=0x00010003, count=4MB)
                                         → cuMemcpyDtoH 读回结果
   ← RPC: data=<4MB 结果>
   → client 把收到的数据写入 h_C

整个过程没有任何真实 CUDA 调用发生在 client 侧。
```

---

## 8. 关键概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| IP 地址 | 网络中电脑的门牌号 |
| 端口 | 一台电脑上区分不同程序的编号 |
| TCP | 保证数据完整、有序到达的传输协议 |
| 127.0.0.1 | 本机回环地址，永远指向自己的电脑 |
| Unix Domain Socket | 本机进程间通信方式，比 TCP 快 |
| C/S 模型 | Client 主动请求，Server 被动响应 |
| RPC | 远程过程调用，让远程函数调用像本地一样 |
| gRPC | Google 开源的高性能 RPC 框架 |
| Protobuf | 二进制序列化格式，gRPC 的默认编码 |
| .proto 文件 | 定义消息类型和服务接口的描述文件 |
| stub | gRPC 自动生成的 client 代理类 |
| streaming | gRPC 中持续发送/接收一系列消息的模式 |

---

> **上一份文档**：[操作系统基础：动态链接与 LD_PRELOAD](basics_os_dynamic_linking.md)
> **下一份文档**：[CUDA Runtime API 与 Driver API 对比](basics_cuda_api_comparison.md)
