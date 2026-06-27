# Linux 开发环境与命令行基础

本文教你用 Linux 命令行做开发。假设你以前只用过 Windows 的图形界面，对黑底白字的终端完全不熟。

---

## 1. 终端、Shell、命令行——别再搞混了

三个词经常被混用，但意思是不同的：

```text
┌─────────────────────────────────────────┐
│  终端模拟器（Terminal Emulator）          │  ← 你看到的那个黑窗口
│  Windows Terminal / gnome-terminal 等    │    负责显示文字和颜色
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Shell（壳）                        │  │  ← 窗口里等着你输入命令的程序
│  │  bash / zsh / fish                 │  │    解释你输入的文字，执行命令
│  │                                    │  │
│  │  $ ls -la       ← 你输入的命令     │  │
│  │  total 824      ← Shell 执行的输出 │  │
│  │  ...                               │  │
│  │  $ _           ← 光标闪烁，等你输入 │  │
│  └────────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

- **终端** = 显示窗口
- **Shell** = 解释命令的程序（bash 是最常见的）
- **命令行** = 你输入的一行命令

$ 叫**提示符（Prompt）**，表示 Shell 准备好接受你的命令了。

---

## 2. 文件系统：Linux 的目录结构

### 2.1 一切从根目录 `/` 开始

Windows 的盘符概念（C:\、D:\）在 Linux 中不存在。Linux 只有一个根目录 `/`，所有东西都在它下面。

```text
/                           ← 根目录，一切开始的地方
├── home/
│   └── hyj/                ← 你的家目录（~ 就代表这个目录）
│       └── code/
│           └── os_hyj/     ← 你当前的项目目录
│               ├── docs/
│               └── tools/
├── usr/
│   ├── bin/                ← 系统命令（ls, cp, gcc 等）
│   └── lib/                ← 系统库（libc.so 等）
│       └── x86_64-linux-gnu/
│           ├── libcudart.so.12
│           └── libcuda.so.1
├── tmp/                    ← 临时文件，重启清空
├── dev/                    ← 设备文件（/dev/dxg 就是 WSL 的 GPU 设备）
├── proc/                   ← 进程和系统信息（/proc/cpuinfo 等）
└── etc/                    ← 配置文件
```

### 2.2 路径：绝对路径 vs 相对路径

```bash
# 绝对路径（从 / 开始，完整写出）
/home/hyj/code/os_hyj/docs/final_implementation_plan.md

# 相对路径（从"当前你在的目录"开始）
docs/final_implementation_plan.md                # 当前在 os_hyj 目录
../another_project/file.txt                      # .. 表示上一级目录
./tools/vector_add_smoke.cu                      # . 表示当前目录
```

### 2.3 特殊目录符号

| 符号 | 含义 | 例子 |
|------|------|------|
| `~` | 你的家目录 | `cd ~` → `/home/hyj` |
| `.` | 当前目录 | `./a.out` → 当前目录下的 a.out |
| `..` | 上一级目录 | `cd ..` → 往上一级 |
| `-` | 上一次所在的目录 | `cd -` → 返回刚才的目录 |

---

## 3. 最常用的 15 个命令

### 3.1 文件和目录操作

```bash
# ls — 列出文件
ls                  # 列出当前目录
ls -l               # 长格式（权限、大小、时间）
ls -la              # 包括隐藏文件（以 . 开头的文件）
ls -R               # 递归列出所有子目录

# cd — 切换目录
cd /home/hyj/code   # 跳到指定目录
cd ..               # 往上一级
cd ~                # 回自己的家目录
cd -                # 回刚才的目录

# pwd — 显示当前所在目录（Print Working Directory）
pwd
# 输出: /home/hyj/code/os_hyj

# mkdir — 创建目录
mkdir build         # 创建 build 目录
mkdir -p a/b/c      # 递归创建（a 不存在时自动创建 a 和 a/b）

# cp — 复制
cp source.txt dest.txt         # 复制文件
cp -r dir1 dir2                # 复制目录（-r 递归）

# mv — 移动/重命名
mv old_name.txt new_name.txt   # 重命名
mv file.txt /tmp/              # 移动到 /tmp

# rm — 删除（危险！没有回收站！）
rm file.txt         # 删除文件
rm -r dir/          # 删除目录及其中所有内容
rm -rf dir/         # 强制删除，不确认（非常危险！慎用！）
# ⚠️ rm 是永久的，没有回收站。谨慎使用，尤其是 rm -rf
```

### 3.2 查看文件内容

```bash
# cat — 显示整个文件
cat file.txt

# less — 分页查看（q 退出，空格翻页，/搜索）
less huge_file.log

# head — 看前几行
head -20 file.txt   # 前 20 行

# tail — 看最后几行
tail -20 file.txt   # 最后 20 行
tail -f file.log    # 持续监控文件新增的内容（Ctrl+C 退出）
```

### 3.3 搜索和过滤

```bash
# grep — 搜索文本
grep "cudaMalloc" *.c              # 在所有 .c 文件中搜索 cudaMalloc
grep -r "error" .                  # 递归搜索当前目录下所有文件
grep -v "DEBUG" file.log          # 排除包含 DEBUG 的行
grep -E "Malloc|Free" file.log    # 支持正则表达式（Malloc 或 Free）

# find — 查找文件
find . -name "*.cpp"                 # 找所有 .cpp 文件
find . -name "*.o" -delete           # 找所有 .o 并删除
find . -type f                       # 列出所有普通文件
```

### 3.4 进程和系统

```bash
# ps — 查看进程
ps aux              # 所有进程
ps aux | grep vgpu  # 查找 vgpu 相关进程

# top / htop — 实时进程监控（q 退出）
top                 # 类似任务管理器
htop                # 更漂亮的版本

# kill — 终止进程
kill 1234           # 礼貌地请 PID=1234 退出
kill -9 1234        # 强制杀（不礼貌，但肯定死）

# df — 磁盘空间
df -h               # 人类可读格式（GB/MB）

# du — 目录大小
du -sh *            # 当前目录下每个东西的大小
```

### 3.5 权限

```bash
# 三种权限：r（读）=4, w（写）=2, x（执行）=1
# 三类用户：owner / group / others

# ls -l 输出解读：
# -rwxr-xr-x   owner   group   size   date   name
#  └┬┘└─┬─┘└─┬─┘
#   │   │    └─ others 可读可执行（r-x = 5）
#   │   └─ group 可读可执行（r-x = 5）
#   └─ owner 可读可写可执行（rwx = 7）

# chmod — 改权限
chmod +x script.sh        # 给所有人加执行权限
chmod 755 script.sh       # rwxr-xr-x
chmod 644 file.txt        # rw-r--r--（所有者可以读写，其他人只读）
```

---

## 4. 环境变量

### 4.1 什么是环境变量

环境变量是 Shell 中保存的"全局变量"，所有程序都能读取。

```bash
# 查看所有环境变量
env

# 查看某个环境变量
echo $HOME          # 输出: /home/hyj
echo $PATH          # 输出: /usr/local/bin:/usr/bin:/bin:...

# 临时设置（只对当前 Shell 生效）
export MY_VAR="hello"
echo $MY_VAR        # 输出: hello

# 永久设置（写入 ~/.bashrc，每次打开终端生效）
echo 'export MY_VAR="hello"' >> ~/.bashrc
source ~/.bashrc    # 立即生效
```

### 4.2 PATH 环境变量

当你在终端打 `ls`，Shell 怎么知道 `ls` 程序在哪里？它在 `PATH` 的目录列表里逐个找：

```bash
echo $PATH
# /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 打 ls → Shell 依次找：
# /usr/local/sbin/ls  → 没有
# /usr/local/bin/ls   → 没有
# /usr/sbin/ls        → 没有
# /usr/bin/ls         → 没有
# /sbin/ls            → 没有
# /bin/ls             → 找到了！执行！
```

### 4.3 项目中使用的重要环境变量

```bash
# vGPU 项目要用的
VGPU_SERVER=127.0.0.1:50051         # server 地址
VGPU_LOG_LEVEL=info                 # 日志级别
VGPU_SESSION_MEMORY_LIMIT_MB=1024   # 每 session 显存上限
VGPU_GRPC_MAX_MESSAGE_MB=256        # gRPC 最大消息大小

# 使用方式
export VGPU_SERVER=127.0.0.1:50051
LD_PRELOAD=./libcudart_proxy.so ./vector_add

# 或者一行搞定
VGPU_SERVER=127.0.0.1:50051 LD_PRELOAD=./libcudart_proxy.so ./vector_add
# (环境变量放在命令前面，只影响这一个命令)
```

### 4.4 LD_PRELOAD 就是环境变量

```bash
# LD_PRELOAD 是一个环境变量
# 动态链接器读取它，在加载其他 .so 之前先加载它指定的 .so
LD_PRELOAD=./libcudart_proxy.so ./vector_add

# 等价于
export LD_PRELOAD=./libcudart_proxy.so
./vector_add
```

---

## 5. 编译和构建

### 5.1 gcc 基础

```bash
# 编译单个文件
gcc hello.c -o hello          # 编译成可执行文件 hello
./hello                       # 运行

# 常用选项
gcc -c file.c -o file.o       # 只编译不链接（生成 .o）
gcc -g file.c -o file         # 带调试信息
gcc -O2 file.c -o file        # 优化编译（O0=不优化, O1, O2, O3）
gcc -Wall file.c -o file      # 开启所有警告（-Wall = warn all）
gcc -I/path/to/include file.c # 添加头文件搜索路径
gcc -L/path/to/lib file.c -l库名  # 添加库搜索路径和要链接的库

# nvcc（CUDA 编译器）
nvcc --cudart shared file.cu -o file   # 动态链接 cudart
nvcc -ptx file.cu -o file.ptx          # 只生成 PTX
nvcc -arch=sm_86 file.cu -o file       # 指定 GPU 架构
```

### 5.2 手动编译多个文件

```bash
# 方式 1：一次搞定
gcc main.c utils.c -o program

# 方式 2：分别编译，再链接（大型项目用这个）
gcc -c main.c -o main.o       # 只编译 main.c
gcc -c utils.c -o utils.o     # 只编译 utils.c
gcc main.o utils.o -o program # 链接所有 .o

# 为什么分步？因为改了 utils.c 后只需要：
gcc -c utils.c -o utils.o     # 重编译改了的文件
gcc main.o utils.o -o program # 重新链接（很快）
# 不用重编译 main.c！
```

### 5.3 CMake 基础

当文件多了，手动编译就烦了。CMake 让你写一个 `CMakeLists.txt` 描述"这个项目怎么编译"，然后自动生成 Makefile。

```bash
# CMake 的经典三步走
mkdir build && cd build   # 创建构建目录（分离源码和构建产物）
cmake ..                  # 读取 ../CMakeLists.txt，生成 Makefile
cmake --build . -j8       # 用 8 个线程并行编译

# 或者用 make
make -j8
```

### 5.4 管道和重定向

```bash
# 管道 |：把前一个命令的输出，作为后一个命令的输入
ps aux | grep vgpu        # 从所有进程中筛选 vgpu 相关
ls -l | wc -l             # 数有多少个文件
find . -name "*.cpp" | xargs grep "TODO"  # 在所有 .cpp 中搜索 TODO

# 重定向 >：把输出存到文件
./vector_add > output.txt          # 标准输出到文件
./vector_add 2> errors.txt         # 标准错误到文件
./vector_add > all.txt 2>&1        # 标准输出和错误都到同一个文件

# > 覆盖   >> 追加
echo "new line" >> log.txt          # 追加到文件末尾
```

### 5.5 最重要的：查看帮助

```bash
# man — 查看命令的详细手册（q 退出）
man gcc            # gcc 的完整文档
man pthread_mutex_lock  # pthread 的 mutex lock

# --help — 大多数命令支持
gcc --help

# tldr — 给出常用例子（需要安装，但非常好用）
sudo apt install tldr
tldr grep
tldr tar

# 查看命令位置
which gcc           # /usr/bin/gcc
which nvcc          # /usr/bin/nvcc
which cmake         # /usr/bin/cmake
```

---

## 6. VSCode / IDE 相关的命令行操作

### 6.1 在 VSCode 中打开目录

```bash
code .                   # 在当前目录打开 VSCode
code docs/design.md      # 打开指定文件
```

### 6.2 常用的 VSCode 终端技巧

- Ctrl+` (反引号) ：打开/关闭内置终端
- 在 VSCode 终端中你的项目目录就是工作区目录
- 编译后的错误，Ctrl+点击文件名可以直接跳到对应行

---

## 7. WSL 特有的命令

```bash
# 查看 WSL 版本
wsl --version         # 在 PowerShell 中执行
wsl -l -v             # 列出所有 WSL 发行版

# 在 WSL 中打开 Windows 文件
cd /mnt/c/Users/      # Windows 的 C 盘

# 在 Windows 中访问 WSL 文件
# 资源管理器地址栏输入：\\wsl$\Ubuntu\home\hyj\

# WSL 关机/重启
wsl --shutdown        # 在 PowerShell 中执行
```

---

## 8. 开发工作流示例

```bash
# 1. 进入项目目录
cd ~/code/os_hyj

# 2. 看看项目里有什么
ls -la
tree  # 如果装了的话

# 3. 编辑代码（用 VSCode）
code .

# 4. 编译（假设已经在 build 目录配置好了）
cd build
cmake --build . -j8

# 5. 如果编译出错，看错误信息：
# - "undefined reference to X" → 链接错误，缺少库
# - "fatal error: X.h: No such file" → 头文件找不到，检查 -I 路径
# - "error: expected ';' before" → 语法错误，看前面一行漏了分号

# 6. 编译成功，运行
./vector_add

# 7. 用 vGPU 方式运行
VGPU_SERVER=127.0.0.1:50051 \
LD_PRELOAD=./libcudart_proxy.so \
./vector_add

# 8. 检查结果
# 查看 server 日志
# 比较原生和 vGPU 输出是否一致

# 9. 改代码 → 回到步骤 3
```

---

## 9. 速查表：最常用的命令行快捷键

| 快捷键 | 作用 |
|--------|------|
| Ctrl+C | 终止当前运行的程序 |
| Ctrl+D | 发送 EOF（结束输入） |
| Ctrl+Z | 暂停当前程序，放到后台 |
| Ctrl+L | 清屏（和 clear 命令一样） |
| Ctrl+A | 光标跳到行首 |
| Ctrl+E | 光标跳到行尾 |
| Ctrl+U | 删除整行 |
| Tab | 自动补全文件名/命令 |
| ↑↓ | 历史命令 |
| Ctrl+R | 搜索历史命令 |
| `!!` | 重复上一条命令 |
| `!$` | 上一条命令的最后一个参数 |

---

> **上一篇文档**：[进程与线程：并发编程基础](basics_process_thread.md)
>
> 至此，全部基础文档完成。**建议阅读顺序**：
> 1. [计算机基础：从晶体管到程序运行](basics_computer_architecture.md)
> 2. [C 语言核心概念深入](basics_c_deep_dive.md)
> 3. [C++ 进阶：项目用到的 STL 和面向对象特性](basics_cpp_for_project.md)
> 4. [内存深入：栈、堆、虚拟内存](basics_memory_deep_dive.md)
> 5. [进程与线程：并发编程基础](basics_process_thread.md)
> 6. [Linux 开发环境与命令行基础](basics_linux_dev.md)（本文）
>
> 这六份基础文档 + 五份技术文档（GPU/CUDA、动态链接、网络 RPC、API 对比、编译工具链），共 11 份文档，覆盖了理解和实现 vGPU 项目所需的全部前置知识。
