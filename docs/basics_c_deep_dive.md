# C 语言核心概念深入——为你读懂项目代码做准备

本文假设你学过最基础的 C 语法（`int x = 3;`、`if/for/while`、函数定义），但不理解指针、内存、结构体等真正干活的东西。本文从零讲起，直击项目代码中用到的每一个 C/C++ 概念。

---

## 1. 指针：整个 C 语言的灵魂

### 1.1 指针到底是什么

记住一句话：**指针就是一个变量，它的值是"内存中某个格子的编号"。**

```c
int x = 42;        // x 是一个 int 变量，存的值是 42
int *p = &x;       // p 是一个指针变量，存的值是 x 的地址

// 假设编译器把 x 放在内存地址 0x1000 处：
//   x 的内容：[42]         ← 在地址 0x1000
//   p 的内容：[0x1000]     ← p 存在另一个地址，p 的值是 0x1000
```

图示——把内存想象成一排带门牌号的储物柜：

```text
地址:   0x1000    0x1004    0x1008    0x100C   ...   0x2000
       ┌────────┬────────┬────────┬────────┐      ┌────────┐
       │   42   │        │        │        │ ...  │0x1000  │
       └────────┴────────┴────────┴────────┘      └────────┘
          ↑                                          ↑
         x 的值                                    p 的值
       (int x=42)                             (int *p=&x)
```

- `x`：把 42 放进编号为 0x1000 的柜子里
- `p`：把编号"0x1000"放进编号为 0x2000 的柜子里

### 1.2 `*` 和 `&`：两个最关键的运算符

这两个运算符是理解指针的关键。它们的作用**完全相反**：

| 运算符 | 名字 | 作用 | 例子 |
|--------|------|------|------|
| `&` | 取地址 | 给我这个变量的门牌号 | `&x` → 0x1000 |
| `*` | 解引用 | 按这个门牌号，去柜子里拿东西 | `*p` → 42 |

```c
int x = 42;
int *p = &x;     // p = x 的地址

*p = 100;        // 按 p 存的地址找到柜子，往里面放 100
                 // 等价于 x = 100，因为 p 指向 x

printf("%d\n", x);  // 输出 100
```

**通俗理解**：
- `&x` = "x 住几号房？" → 0x1000
- `*p` = "去 p 写的那个房号，把里面的东西取出来"

### 1.3 指针的声明：`int *p` 到底是什么意思

```c
int *p;    // 读作："p 是一个指针，指向 int 类型的数据"
           // p 本身的值是一个地址
           // *p 是那个地址里存的 int 值

int **pp;  // "pp 是一个指针，指向一个指向 int 的指针"
           // pp → p → x(42)
```

很多人搞混的地方：声明里的 `*` 和表达式里的 `*` 是同一个字符但含义不同：

```c
int *p = &x;   // 这里的 * 是声明的一部分："p 是指针类型"
*p = 100;      // 这里的 * 是运算符："去 p 指向的地方"
```

### 1.4 NULL 指针

```c
int *p = NULL;  // p 不指向任何有效的地方
                // NULL 实际上是地址 0

// 解引用 NULL 指针是灾难：
*p = 42;  // ❌ 崩溃！"段错误（Segmentation Fault）"
          //    操作系统说：地址 0 不是你的，不能写
```

### 1.5 指针在项目中的实际使用

在 vGPU 项目中，指针无处不在。看这个例子：

```c
// cudaMalloc 的签名
cudaError_t cudaMalloc(void **devPtr, size_t size);
//                     ^^^^^^^^
//                     为什么是"指向指针的指针"？

// 因为 cudaMalloc 需要修改"调用者的指针变量"的值
// 调用方式：
float *d_A = NULL;                    // d_A 是一个指针
cudaMalloc((void**)&d_A, 1048576);   // 传 d_A 的地址进去
// 函数内部：
//   *devPtr = 某个新分配的地址
// 效果相当于 d_A = 某个新分配的地址
```

**为什么要 `void**`？** 因为 C 函数参数是"按值传递"的。如果你传 `d_A` 本身（它的值 NULL），函数只能拿到 NULL 这个值，改不了 `d_A` 变量本身。传 `&d_A`（d_A 变量的地址），函数就能通过 `*devPtr` 修改 d_A 的值。

### 1.6 `void*` 是什么

```c
void *ptr;  // "通用指针"，可以指向任何类型的数据
            // 不能直接用 *ptr（编译器不知道要取多少字节）
            // 需要先转换成具体类型的指针
```

在 CUDA 项目中大量使用：
```c
void *d_ptr;                    // 通用 device pointer
float *d_float = (float*)d_ptr; // 告诉编译器：把这里当 float 数组用
```

`(float*)d_ptr` 叫做**类型转换（cast）**——"我确定这个地址存的是 float，请按 float 来理解"。

### 1.7 指针与数组的关系

```c
int arr[5] = {10, 20, 30, 40, 50};

// arr 本身可以被当成指针用——它的值是数组第一个元素的地址
int *p = arr;      // 等价于 int *p = &arr[0];

// 访问数组元素的三种写法，完全等价：
arr[2];            // 30
*(arr + 2);        // 30
p[2];              // 30（p 指向 arr[0]）
```

`arr[i]` 的含义：从 arr 地址开始，往后跳 i 个元素（每个元素占 sizeof(int)=4 字节），然后读/写那个位置。

---

## 2. 内存中的数据类型：大小和对齐

### 2.1 `sizeof` 告诉你一个类型占多少字节

```c
sizeof(char)      // 1（总是 1）
sizeof(short)     // 2
sizeof(int)       // 4（大多数平台）
sizeof(long)      // 8（64 位 Linux）
sizeof(long long) // 8
sizeof(float)     // 4
sizeof(double)    // 8
sizeof(void*)     // 8（64 位系统）
```

**每个类型的 `sizeof` 是不固定的，取决于平台。** 32 位系统上指针是 4 字节，64 位是 8 字节。

### 2.2 内存对齐

```c
struct Example {
    char a;     // 1 字节
    int b;      // 4 字节
    char c;     // 1 字节
};
// sizeof(Example) = ? 你猜 6 字节？
// 实际 = 12 字节！
```

内存中的实际布局：
```text
偏移:  0   1   2   3   4   5   6   7   8   9   10  11
      ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
      │ a │空 │空 │空 │       b       │ c │空 │空 │空 │
      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

编译器插入**填充字节**来保证每个成员的地址是它大小的整数倍（`int` 必须放在 4 的倍数地址）。这是为了性能——CPU 读对齐的数据更快。

---

## 3. 结构体（struct）

### 3.1 struct 就是把多个变量打包在一起

```c
// 定义一个"学生"类型
struct Student {
    char name[32];   // 32 字节
    int age;         // 4 字节
    float score;     // 4 字节
};

// 使用
struct Student s1;
strcpy(s1.name, "张三");
s1.age = 20;
s1.score = 95.5;
```

struct 在内存中**连续存放**：
```text
name[0..31]                 age         score
┌──────────────────────...──┬──────────┬──────────┐
│   "张三\0"                 │    20    │   95.5   │
└──────────────────────...──┴──────────┴──────────┘
```

### 3.2 指向 struct 的指针和 `->` 运算符

```c
struct Student s1;
struct Student *p = &s1;

// 两种访问成员的方式：
s1.age = 20;       // 用 . 访问
p->age = 20;       // 用 -> 访问（指针用这个）
(*p).age = 20;     // 等价于 p->age，但不常用
```

`p->age` 就是 `(*p).age` 的简写。项目代码里大量使用 `->`。

### 3.3 项目中的 struct 例子

回顾设计文档中的 Session 结构：

```c
struct Session {
    uint64_t id;                                        // 8 字节
    uint32_t client_pid;                                // 4 字节
    std::string client_name;                            // C++ string
    uint64_t memory_limit;                              // 8 字节
    uint64_t memory_used;                               // 8 字节
    std::unordered_map<uint64_t, DeviceAllocation> allocations; // C++ map
    // ... 更多成员
};
```

这个 struct 把一个 session 的所有相关数据打包在一起。server 收到 RPC 请求时：

```c
Session *session = find_session(session_id);
// 现在可以通过 session->memory_used 知道这个 session 用了多少显存
// 通过 session->allocations 找到虚拟指针对应的真实指针
```

### 3.4 `typedef`：给类型起别名

```c
// 不用 typedef：每次都要写 struct
struct Student s1;

// 用 typedef：给 struct Student 起名叫 Student
typedef struct Student Student;
Student s1;  // 简洁

// 更常见的写法（定义 + typedef 一起做）：
typedef struct {
    int x;
    int y;
} Point;

Point p1;
p1.x = 10;
p1.y = 20;
```

---

## 4. 函数指针

### 4.1 函数也有地址

函数编译后变成一段机器码，存放在内存中。这段机器码有起始地址。**函数指针就是存"函数入口地址"的变量。**

```c
// 普通函数
int add(int a, int b) {
    return a + b;
}

// 声明一个函数指针
int (*func_ptr)(int, int);     // func_ptr 是一个指针，指向：
                               // "接收两个 int，返回一个 int" 的函数

func_ptr = &add;               // 或直接 func_ptr = add;
                               // 把 add 函数的地址赋给 func_ptr

int result = func_ptr(3, 5);   // 通过函数指针调用 → 8
                               // 等价于 add(3, 5)
```

### 4.2 函数指针在 CUDA 中的作用

`cudaLaunchKernel` 的第一个参数就是一个函数指针：

```c
cudaLaunchKernel(
    (const void*)func,   // ← func 是 kernel 函数的地址！
    grid, block, args, sharedMem, stream
);
```

CUDA Runtime 内部做了这样的事：
```c
// Runtime 维护一个映射表（简化版）
std::map<void*, std::string> host_func_to_name;
// key:   host 函数指针（func 的地址）
// value: GPU kernel 的真正名字（如 "vectorAdd"）

// 调用 cudaLaunchKernel 时：
std::string kernel_name = host_func_to_name[(void*)func];
// 然后调用 cuLaunchKernel(kernel_name, ...)
```

在 vGPU 项目中，client 拦截 `cudaLaunchKernel` 时：
```c
// client 侧
uint64_t host_fun_id = (uint64_t)func;  // 把函数地址转成整数
// 在本地映射表中查找 host_fun_id → kernel_name
// 把 kernel_name 发给 server
```

### 4.3 `reinterpret_cast` / C 风格类型转换

```c
void *func = ...;                // func 是一个函数指针

// 把函数指针当成整数用（C 风格）
uint64_t id = (uint64_t)func;

// C++ 风格（更明确地表达"我就是要重新解释这些位"）
uint64_t id = reinterpret_cast<uint64_t>(func);

// 反过来：把整数当成函数指针
void *recovered_func = (void*)id;
// 或 C++ 风格：
void *recovered_func = reinterpret_cast<void*>(id);
```

`reinterpret_cast` 的意思是：**不改变任何二进制位，只是换个方式解释这些位**。一块内存，当作 `float` 解释是 1.0，当作 `int` 解释是 1065353216，但底层的位是一样的。

---

## 5. 内存的"生命周期"

### 5.1 三种内存区域

```c
// ① 全局/静态区：程序启动时分配，结束时释放
int global_var = 100;       // 全局变量
static int static_var = 0;  // 静态变量

// ② 栈（Stack）：函数调用时分配，函数返回时自动释放
void foo() {
    int local_var = 42;     // 栈变量，foo 返回后就没了
    int arr[100];           // 也是栈变量
}
// arr 在这里已经不存在了！不要返回指向它的指针！

// ③ 堆（Heap）：手动分配，手动释放（不释放就泄漏）
void bar() {
    int *p = malloc(100 * sizeof(int));  // 在堆上分配 400 字节
    // 用 p...
    free(p);  // ❗必须手动释放，否则"内存泄漏"
    // p 在这里 free 了，不要再用 *p
}
```

| 区域 | 谁分配 | 谁释放 | 生命周期 | 大小限制 |
|------|--------|--------|---------|---------|
| 全局区 | 编译器 | 程序结束时 | 整个程序运行期 | 不限 |
| 栈 | 自动（函数调用时） | 自动（函数返回时） | 函数内 | 有限（通常 8MB） |
| 堆 | 手动（malloc/new） | 手动（free/delete） | 手动控制 | 总可用内存 |

### 5.2 最常见的指针错误

```c
// ❌ 错误 1：返回栈变量的地址
int* create_bad() {
    int x = 42;
    return &x;   // x 在函数返回后就没了！
}                // 返回的指针成了"野指针"

// ✅ 正确：在堆上分配
int* create_good() {
    int *p = malloc(sizeof(int));
    *p = 42;
    return p;    // 堆上的内存不会自动释放
}                // 调用者记着 free(p)

// ❌ 错误 2：忘记 free（内存泄漏）
void leak() {
    int *p = malloc(1000000);  // 分配 4MB
    // 忘了 free(p)！
}  // 函数返回后，这 4MB 就永远丢了（直到程序退出）

// ❌ 错误 3：free 后再用（悬垂指针）
void dangling() {
    int *p = malloc(sizeof(int));
    *p = 42;
    free(p);
    *p = 100;  // ❌ p 指向的内存已经还给系统了！
}

// ❌ 错误 4：重复 free
void double_free() {
    int *p = malloc(sizeof(int));
    free(p);
    free(p);  // ❌ 崩溃！
}
```

### 5.3 C 风格的字符串

```c
// C 没有真正的 "string" 类型。字符串就是字符数组，以 '\0' 结尾
char str[6] = {'H', 'e', 'l', 'l', 'o', '\0'};
char str2[] = "Hello";  // 等价，编译器自动加 '\0'

// 字符串操作需要函数：
strlen(str);          // 5（不包含 '\0'）
strcpy(dest, src);    // 复制
strcmp(a, b);         // 比较

// 常见错误：目标缓冲区太小
char dest[3];
strcpy(dest, "Hello");  // ❌ 溢出！dest 只有 3 字节，"Hello" 需要 6 字节
```

在 vGPU 项目的 proto 中，`string` 是 protobuf 管理的，它会帮你处理长度和内存。但在 C 代码中手动处理 fatbin 数据时，你需要明确知道数据的大小——这通常是从 `__cudaRegisterFatBinary` 的参数推断的。

---

## 6. C 程序的编译与链接深入

### 6.1 为什么有 .h 和 .c 之分

```c
// vector_add.h（头文件）
#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

// 声明：告诉编译器 "这个函数存在，在别处"
void vectorAdd(float *A, float *B, float *C, int N);

#endif

// vector_add.cu（实现文件）
#include "vector_add.h"

// 定义：函数的真正代码
void vectorAdd(float *A, float *B, float *C, int N) {
    // ... 实际代码
}

// main.cu（使用方）
#include "vector_add.h"
// 只需要看到声明就能编译
int main() {
    vectorAdd(A, B, C, N);  // 编译器知道有这个函数就行
}
```

**声明 vs 定义**：
- 声明 = "我告诉你这东西存在"（函数签名、变量类型）
- 定义 = "这是真正的代码/数据"（函数体、变量初始化）

### 6.2 `extern "C"` 是什么

你的 vGPU 拦截库中会看到这个：

```c
extern "C" {
    cudaError_t cudaMalloc(void **devPtr, size_t size) {
        // ... 拦截实现
    }
}
```

原因：C++ 编译器会修改函数名（Name Mangling，详见编译工具链文档）。`extern "C"` 告诉 C++ 编译器："这个函数用 C 的方式命名，不要改它的名字"。

因为你的拦截库要"冒充" `libcudart.so`，而 `libcudart.so` 里的 `cudaMalloc` 是用 C 的方式命名的（没有 mangling），所以你的版本也必须用 C 方式命名。否则动态链接器找不到你的 `cudaMalloc`。

### 6.3 `__attribute__((constructor))` 和 `__attribute__((destructor))`

```c
// 这个函数会在 main() 之前自动被调用
__attribute__((constructor))
void my_init() {
    printf("我在 main 之前执行！\n");
}

// 这个函数会在程序退出时自动被调用（或 dlclose 时）
__attribute__((destructor))
void my_cleanup() {
    printf("我在程序退出时执行！\n");
}
```

vGPU 设计文档中提到："不要在动态库 constructor 里做复杂网络连接"。这是因为 constructor 执行时，动态链接器可能还没初始化完全，gRPC 调用可能失败。所以文档推荐**懒初始化**：在第一个 CUDA API 被调用时才初始化 gRPC 连接。

---

## 7. 几个项目代码中会反复出现的模式

### 7.1 错误码返回模式

```c
// CUDA 风格：函数返回错误码，真实结果通过指针参数返回
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    // 成功：
    *devPtr = allocated_pointer;
    return cudaSuccess;

    // 失败：
    return cudaErrorMemoryAllocation;  // devPtr 不会被修改
}

// 调用方检查返回值
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    printf("分配失败！错误码: %d\n", err);
    // 错误处理...
}
```

### 7.2 handle 模式（用了很多）

```c
// "handle" = "一个不透明的 ID，用来在内部表中查找实际对象"
// client 拿到的是虚拟 handle，server 用这个 handle 查表找真实对象

// vGPU 中的例子：
// client 调用 cudaMalloc，server 返回 vptr=0x00010001
// client 把这个 vptr 当成 device pointer 用
// client 调用 cudaMemcpy(dst=vptr, ...) 时，把 vptr 发回给 server
// server 查表：vptr 0x00010001 → 真实 CUdeviceptr 0x7fabc0000000
```

这就像餐厅给你一个号码牌——你不需要知道你的菜在后厨哪个锅里，只需要拿着号码牌，叫号时凭号码取菜。

### 7.3 RAII 的简化理解（C++ 特有的）

```cpp
// C 方式：手动管理资源，容易忘记清理
void foo_c() {
    FILE *f = fopen("data.txt", "r");
    // ... 用 f ...
    fclose(f);  // 如果中间 return 了，fclose 可能被跳过 → 资源泄漏
}

// C++ RAII 方式：资源绑定到对象生命周期
void foo_cpp() {
    std::ifstream f("data.txt");  // 打开文件
    // ... 用 f ...
    // 函数返回时，f 的析构函数自动关闭文件
    // 即使中间 return 或抛异常，也会自动清理
}
```

vGPU 项目中使用 gRPC C++ API 时，`grpc::ClientContext`、protobuf 消息对象等都是 RAII 的——你不用手动管理它们的内存。

---

## 8. 速查表

| 语法 | 含义 | 例子 |
|------|------|------|
| `int *p` | p 是指向 int 的指针 | `int *p = &x;` |
| `*p` | 解引用：取 p 指向的东西 | `*p = 42;` |
| `&x` | 取地址：得到 x 的地址 | `int *p = &x;` |
| `p->member` | 等价于 `(*p).member` | `s->age = 20;` |
| `(Type)expr` | C 风格类型转换 | `(void*)d_ptr` |
| `sizeof(type)` | 该类型占多少字节 | `sizeof(int)` → 4 |
| `malloc(n)` | 在堆上分配 n 字节 | `malloc(100)` |
| `free(p)` | 释放 malloc 的内存 | `free(p)` |
| `void*` | 通用指针，不指定类型 | `void *ptr;` |
| `typedef` | 给类型起别名 | `typedef int Age;` |
| `extern "C"` | 用 C 方式命名，不要 C++ name mangling | 拦截 cudart 时用 |

---

> **下一份文档**：[C++ 进阶：项目用到的 STL 和面向对象特性](basics_cpp_for_project.md)
