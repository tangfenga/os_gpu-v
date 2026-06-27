# C++ 进阶：项目用到的 STL 和面向对象特性

C++ 是 C 的超集——绝大多数 C 代码可以直接在 C++ 中编译。但 C++ 多了很多东西。本文只讲 vGPU 项目代码中实际会出现的 C++ 特性，不追求全面覆盖。

---

## 1. 命名空间（namespace）——防止名字冲突

### 1.1 为什么需要命名空间

假设有两个库都定义了 `log` 函数，编译器不知道你调用的是哪个。

```cpp
// 库 A
namespace libA {
    void log(const char *msg) { /* ... */ }
}

// 库 B
namespace libB {
    void log(const char *msg) { /* ... */ }
}

// 使用
libA::log("hello");  // 明确调用库 A 的
libB::log("world");  // 明确调用库 B 的
```

### 1.2 项目中的实际使用

```cpp
// gRPC 的所有类都在 grpc 命名空间中
grpc::ServerBuilder builder;
grpc::Status status;

// protobuf 的消息类在 vgpu 命名空间中（package vgpu;）
vgpu::MallocRequest req;
vgpu::MallocReply reply;

// 标准库在 std 命名空间中
std::string name = "hello";
std::vector<int> numbers;
std::map<uint64_t, DeviceAllocation> table;
```

### 1.3 `using` 的用法

```cpp
// 方式 1：全局引入整个 std（方便但不推荐在大项目）
using namespace std;
string s = "hello";       // 不用写 std::
vector<int> v;

// 方式 2：只引入一个名字
using std::string;
using std::vector;

// 方式 3：不用 using，每次都写全名（最清晰，推荐）
std::string s = "hello";
```

---

## 2. `std::string`——真正的字符串

### 2.1 C 字符串 vs C++ 字符串

```cpp
// C 风格（项目中尽量避免）
char c_str[100];
strcpy(c_str, "hello");
strcat(c_str, " world");    // 容易溢出！
size_t len = strlen(c_str); // 每次调用都要遍历整个字符串

// C++ 风格（项目中使用这个）
std::string cpp_str = "hello";
cpp_str += " world";        // 自动扩容，不会溢出
size_t len = cpp_str.size(); // O(1) 操作，内部记录了长度
```

**C++ string 会自动管理内存**，你不需要手动 `malloc`/`free`。

### 2.2 常用操作

```cpp
std::string s1 = "hello";
std::string s2 = "world";

std::string s3 = s1 + " " + s2;  // "hello world"（运算符重载）
s1 += "!!!";                       // "hello!!!"

if (s1 == s2) { /* ... */ }       // 直接比较（对比 C 的 strcmp）
if (s1.empty()) { /* ... */ }     // 判断是否为空
s1.clear();                       // 清空

// 获取 C 风格字符串（调用某些 C API 时需要）
const char *c_str = s1.c_str();

// 查找子串
size_t pos = s1.find("lo");       // pos = 3
if (pos != std::string::npos) {   // npos 表示"没找到"
    // 找到了
}

// 取子串
std::string sub = s1.substr(0, 3);  // "hel"（位置0开始，取3个字符）
```

### 2.3 `std::string` 在项目中的使用

```cpp
// proto 中的 string 映射到 C++ std::string
message MallocReply {
  string message = 2;   // → std::string message
}

// client 代码
vgpu::MallocReply reply;
std::string error_msg = reply.message();  // getter 返回 std::string

// kernel 名称用 string 存储
std::string kernel_name = "vectorAdd";
```

---

## 3. `std::vector`——可变长数组

### 3.1 为什么不用 C 数组

```cpp
// C 数组：大小固定，不能自动扩容
int arr[100];
arr[101] = 42;  // ❌ 越界写，覆盖了别的数据！

// std::vector：动态数组，自动管理大小
std::vector<int> vec;
vec.push_back(10);       // 添加一个元素
vec.push_back(20);
vec.push_back(30);       // vec 现在有 3 个元素：{10, 20, 30}
vec[0] = 100;            // 像数组一样访问
size_t sz = vec.size();  // 获取当前元素数量：3
```

### 3.2 常用操作

```cpp
std::vector<int> v;

// 添加
v.push_back(42);          // 尾部添加
v.emplace_back(42);       // 更高效的尾部添加（直接在 vector 内部构造）

// 访问
int x = v[0];             // 像数组一样，不检查越界
int y = v.at(0);          // 带越界检查，越界会抛异常
int first = v.front();    // 第一个元素
int last = v.back();      // 最后一个元素

// 大小
size_t n = v.size();      // 元素个数
bool empty = v.empty();   // 是否为空
v.reserve(1000);          // 预留空间，减少后续添加时的重新分配

// 遍历
for (int x : v) {         // 范围 for 循环
    printf("%d\n", x);
}

for (size_t i = 0; i < v.size(); i++) {  // 传统 for
    printf("%d\n", v[i]);
}

// 清空
v.clear();                // 清空所有元素
```

### 3.3 `repeated` 字段在 proto 中映射为 vector

```proto
message LaunchKernelRequest {
  repeated KernelArg args = 8;
  //     ^^^^^^^^
  //     在生成的 C++ 代码中，args 是一个 std::vector<KernelArg>
}
```

C++ 中使用：
```cpp
vgpu::LaunchKernelRequest req;

// 添加一个参数
auto *arg = req.add_args();   // 在 vector 尾部添加一个元素，返回指针
arg->set_raw(data, size);     // 设置这个元素的内容

// 遍历所有参数
for (const auto &arg : req.args()) {
    const std::string &raw = arg.raw();
    // 处理 raw 数据...
}

// 获取参数个数
int n_args = req.args_size();  // 等价于 req.args().size()
```

---

## 4. `std::map` 和 `std::unordered_map`——字典/映射表

### 4.1 什么是 map

map 就是"字典"——你给一个 key，拿到对应的 value。类比英汉词典：key="apple"，value="苹果"。

```cpp
#include <map>
#include <string>

// 声明：从 string 到 int 的映射
std::map<std::string, int> ages;

// 添加
ages["张三"] = 20;
ages["李四"] = 22;

// 查找
int age = ages["张三"];        // 20（key 不存在时会自动创建，值为 0）
int age2 = ages.at("张三");    // 20（key 不存在会抛异常）
auto it = ages.find("王五");   // 返回迭代器，找不到返回 ages.end()

// 检查是否存在
if (ages.find("赵六") != ages.end()) {
    // "赵六" 存在
}
if (ages.count("张三") > 0) {
    // "张三" 存在
}

// 删除
ages.erase("张三");

// 遍历
for (auto &pair : ages) {
    std::string name = pair.first;   // key
    int age = pair.second;           // value
}
```

### 4.2 `std::map` vs `std::unordered_map`

```cpp
std::map<std::string, int> m;           // 基于红黑树，key 有序，操作 O(log n)
std::unordered_map<std::string, int> um; // 基于哈希表，key 无序，操作 O(1) 平均
```

大多数情况下用 `unordered_map`，更快。除非你需要 key 按顺序排列。

### 4.3 项目中的使用

这是 vGPU 项目中最核心的数据结构——所有虚拟资源的管理都靠 map：

```cpp
// server 侧 Session 结构（来自设计文档）
struct Session {
    // 虚拟指针 → 真实显存分配
    std::unordered_map<uint64_t, DeviceAllocation> allocations;
    // vptr      → {vptr, real_ptr, size, freed}

    // 虚拟 stream → 真实 CUstream
    std::unordered_map<uint64_t, CUstream> streams;

    // 虚拟 event → 真实 CUevent
    std::unordered_map<uint64_t, CUevent> events;

    // module handle → CUmodule
    std::unordered_map<uint64_t, CUmodule> modules;

    // host_fun_id → kernel 符号信息
    std::unordered_map<uint64_t, KernelSymbol> functions;
};
```

server 收到 Malloc RPC 时：
```cpp
grpc::Status MallocHandler(..., const MallocRequest *req, MallocReply *reply) {
    uint64_t sid = req->session_id();
    uint64_t size = req->size();

    Session *session = find_session(sid);

    CUdeviceptr real_ptr;
    cuMemAlloc(&real_ptr, size);         // 在真实 GPU 上分配

    uint64_t vptr = allocate_vptr();     // 生成虚拟指针

    // 记录到映射表 ← 这就是 map 的用途！
    session->allocations[vptr] = DeviceAllocation{
        .vptr = vptr,
        .real_ptr = real_ptr,
        .size = size,
        .freed = false
    };

    reply->set_vptr(vptr);               // 返回虚拟指针给 client
    return grpc::Status::OK;
}
```

---

## 5. 类（class）和对象——面向对象基础

### 5.1 从 struct 到 class

C 的 struct 只能存数据。C++ 的 class 可以把数据和操作数据的函数打包在一起。

```cpp
// C 风格（数据和操作分离）
struct Point {
    double x;
    double y;
};
double distance_from_origin(Point *p) {  // 操作是独立函数
    return sqrt(p->x * p->x + p->y * p->y);
}
Point p = {3.0, 4.0};
double d = distance_from_origin(&p);

// C++ 风格（数据和操作打包）
class Point {
public:                           // public: 对外可见
    double x;
    double y;

    double distance_from_origin() {  // 方法（成员函数）
        return sqrt(x * x + y * y);  // 直接访问自己的 x, y
    }
};
Point p{3.0, 4.0};
double d = p.distance_from_origin();  // 通过 . 调用方法
```

**class 和 struct 在 C++ 中几乎一样**，唯一区别：struct 默认 public，class 默认 private。

### 5.2 构造函数和析构函数

```cpp
class FileReader {
private:
    FILE *file_;

public:
    // 构造函数：对象创建时自动调用
    FileReader(const std::string &filename) {
        file_ = fopen(filename.c_str(), "r");
        printf("文件打开了\n");
    }

    // 析构函数：对象销毁时自动调用
    ~FileReader() {
        if (file_) {
            fclose(file_);
            printf("文件关闭了\n");
        }
    }

    // 普通方法
    std::string readLine() {
        char buf[1024];
        fgets(buf, sizeof(buf), file_);
        return std::string(buf);
    }
};

// 使用
{
    FileReader reader("data.txt");   // 构造时自动打开文件
    std::string line = reader.readLine();
    // ...
}  // 离开作用域，reader 析构，自动关闭文件
// 即使中间 return 或抛异常，析构函数也一定会被调用
```

这就是 **RAII（Resource Acquisition Is Initialization）**——资源在构造函数中获取，在析构函数中释放。项目中的 gRPC context、protobuf 对象都是这样的。

### 5.3 继承：gRPC 生成代码中的典型用法

```cpp
// gRPC 给你生成一个抽象基类
class VgpuRuntime::Service {
public:
    // 虚函数——需要你重写的函数
    virtual grpc::Status Malloc(
        grpc::ServerContext *context,
        const MallocRequest *request,
        MallocReply *response) = 0;   // = 0 表示"纯虚函数"，必须子类实现
};

// 你继承它，实现自己的版本
class MyVgpuService : public VgpuRuntime::Service {
public:
    grpc::Status Malloc(
        grpc::ServerContext *context,
        const MallocRequest *request,
        MallocReply *response) override  // override 表示"我在重写基类的函数"
    {
        // 你的 Malloc 实现
        uint64_t sid = request->session_id();
        // ... 实际逻辑 ...
        response->set_vptr(vptr);
        return grpc::Status::OK;
    }
};
```

**简单理解**：
- `Service` 是模板，定义了"应该有哪些函数"，但不知道具体怎么做
- 你的类继承它，提供每个函数的具体实现
- gRPC 框架调用你的函数时，用的是基类的接口，但实际执行的是你的代码（多态）

---

## 6. `auto`——让编译器推断类型

```cpp
// 不用 auto：写全类型
std::unordered_map<uint64_t, DeviceAllocation>::iterator it = allocations.find(vptr);

// 用 auto：
auto it = allocations.find(vptr);  // 编译器知道 it 是什么类型

// 常用场景：
auto reply = stub->Malloc(&context, req);  // reply 的类型是 MallocReply
auto &session = sessions[sid];             // session 是 Session& 引用
for (auto &[key, value] : map) { ... }    // 遍历 map 的结构化绑定
```

`auto` 不等于"弱类型"——类型在编译时就确定了，只是你不用手写。对于巨长的类型名（STL 迭代器），`auto` 极大提高可读性。

---

## 7. 迭代器（Iterator）——遍历容器的通用方式

```cpp
std::vector<int> v = {10, 20, 30, 40, 50};

// 方式 1：范围 for（最简洁，推荐）
for (int x : v) {
    printf("%d\n", x);
}

// 方式 2：传统下标
for (size_t i = 0; i < v.size(); i++) {
    printf("%d\n", v[i]);
}

// 方式 3：迭代器
for (auto it = v.begin(); it != v.end(); ++it) {
    printf("%d\n", *it);  // *it 取当前位置的元素
}
// it 就是一个"指向容器内部某个位置的指针"
// begin() 指向第一个元素，end() 指向末尾之后
// ++it 把迭代器移到下一个元素
```

**迭代器就是容器内部的"指针"**——`*it` 取元素，`++it` 移动到下一个。

在 vGPU 项目中遍历 map 经常用迭代器：

```cpp
// 遍历 session 的所有显存分配
for (auto &[vptr, alloc] : session->allocations) {
    printf("vptr=0x%lx real_ptr=0x%lx size=%zu\n",
           alloc.vptr, alloc.real_ptr, alloc.size);
}

// 查找某个 vptr
auto it = session->allocations.find(vptr);
if (it != session->allocations.end()) {
    // 找到了，it->second 是 DeviceAllocation
    CUdeviceptr real = it->second.real_ptr;
} else {
    // 没找到
}
```

---

## 8. 智能指针（Smart Pointer）——不用手动 delete

```cpp
// C 风格：malloc + free（容易泄漏）
void *ptr = malloc(1024);
// ... 如果中途 return，free 不会被调用
free(ptr);

// C++ 03 风格：new + delete（也容易泄漏）
int *p = new int(42);
// ... 如果抛异常，delete 不会被调用
delete p;

// C++ 11 风格：智能指针（自动释放）
#include <memory>

std::unique_ptr<int> p = std::make_unique<int>(42);
// p 离开作用域时自动 delete，绝不泄漏
// unique_ptr：独占所有权，不能复制

std::shared_ptr<Session> session = std::make_shared<Session>();
// shared_ptr：引用计数，最后一个引用消失时自动释放
```

在 gRPC 项目中，很多对象已经通过 gRPC 框架管理生命周期。但你自己的 Session/资源管理代码中，建议用智能指针避免手写 `delete`。

---

## 9. 模板（Template）——泛型编程

模板的基本思路：**写一次代码，适用多种类型**。

```cpp
// 不用模板：每种类型写一个函数
int max_int(int a, int b) { return a > b ? a : b; }
float max_float(float a, float b) { return a > b ? a : b; }
double max_double(double a, double b) { return a > b ? a : b; }
// 恶心的重复！

// 用模板：写一次，适用所有类型
template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}

int x = max<int>(3, 5);        // T = int
float y = max<float>(3.0, 5.0); // T = float
auto z = max(3, 5);            // 编译器自动推断 T = int
```

STL 容器本身就是模板：
```cpp
std::vector<int>     // T = int
std::vector<float>   // T = float
std::vector<std::string>  // T = std::string
// 都是同一个 std::vector<T> 模板，换了不同的 T
```

在 vGPU 代码中，你不需要写模板，但你会用到模板（`std::vector<uint64_t>`、`std::unordered_map<uint64_t, CUstream>` 等）。

---

## 10. 头文件和实现分离：C++ 惯例

```cpp
// ============== session.h ==============
#pragma once  // 或 #ifndef SESSION_H / #define SESSION_H / #endif

#include <cuda.h>
#include <unordered_map>
#include <string>

struct DeviceAllocation {
    uint64_t vptr;
    CUdeviceptr real_ptr;
    size_t size;
    bool freed;
};

class SessionManager {
public:
    SessionManager();
    ~SessionManager();

    uint64_t createSession(uint32_t pid, const std::string &name);
    bool destroySession(uint64_t sid);
    Session* findSession(uint64_t sid);

private:
    std::unordered_map<uint64_t, Session*> sessions_;
    std::mutex mutex_;
    uint64_t next_id_;
};

// ============== session.cpp ==============
#include "session.h"

SessionManager::SessionManager() : next_id_(1) {}
// 构造函数，初始化 next_id_ 为 1

SessionManager::~SessionManager() {
    // 析构函数，清理所有 session
    for (auto &[id, session] : sessions_) {
        delete session;
    }
}

uint64_t SessionManager::createSession(uint32_t pid, const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);  // 加锁（后面文档讲）

    auto *session = new Session{
        .id = next_id_,
        .client_pid = pid,
        .client_name = name
    };
    sessions_[next_id_] = session;
    return next_id_++;
}

// ============== main.cpp（使用者） ==============
#include "session.h"

int main() {
    SessionManager mgr;
    uint64_t sid = mgr.createSession(1234, "vector_add");
    // ...
}
```

---

## 11. 速查表

| C++ 概念 | 一句话解释 | 项目中使用 |
|----------|-----------|-----------|
| `std::string` | 自动管理内存的字符串 | kernel 名、错误消息、proto string 字段 |
| `std::vector<T>` | 可变长数组 | proto `repeated` 字段 |
| `std::map<K,V>` | 有序字典 | 资源映射表 |
| `std::unordered_map<K,V>` | 哈希字典（更快） | session 内部的 vptr→资源 映射 |
| `class` | 数据和操作的打包体 | server session 管理、gRPC service 实现 |
| `::` | 作用域分隔符 | `grpc::ServerBuilder`、`vgpu::MallocRequest` |
| `auto` | 让编译器推断类型 | 迭代器、复杂的 map 遍历 |
| `virtual` / `override` | 多态：子类重写基类行为 | gRPC service 子类实现 |
| `template<typename T>` | 泛型：一种代码适用多种类型 | STL 容器（你使用，不自己写） |
| RAII | 资源绑定对象生命周期 | gRPC Context、智能指针 |
| `new` / `delete` | C++ 的堆分配/释放 | 创建 Session 对象 |
| `nullptr` | C++ 的空指针（比 NULL 好） | 初始化指针 |

---

> **上一份文档**：[C 语言核心概念深入](basics_c_deep_dive.md)
> **下一份文档**：[内存深入：栈、堆、虚拟内存](basics_memory_deep_dive.md)
