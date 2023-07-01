# 四子棋大作业 Linux 版

本文档为面向 Linux 版的四子棋框架文档，详细描述了评测框架的使用以及策略程序的编译方法。

## 目录结构

```
Connect4
├── Compete        # 评测框架文件夹
├── Strategy          # 策略程序文件夹	
├── makefile	     # 用于快速上手的一键makefile
├── readme.md   # 说框架文档
└── so                      # 存放bug样例与你策略编译结果的文件夹
```

## 环境依赖

- make >= 3.8.2
- g++ >= 4.8.5

## 评测框架

评测框架位于 `Compete` 文件夹下，该文件夹内容如下

```bash
Compete
├── Compete.cpp
├── Compete.h
├── Data.h
├── Exception.hpp
├── Judge.cpp
├── Judge.h
├── Point.h
├── main.cpp
└── makefile
```

使用 `make` 编译得到可执行程序 `Compete`， `Compete` 程序的使用方法如下

```bash
./Compete	<A的so文件路径> <B的so文件路径>	<结果文件名>	<对抗轮数>
```

A 和 B 使用完全相同的初始棋盘对抗指定轮数，每轮两次，分别为 A 先手和 B 先手。

对局结果存放在<结果文件名>指定的文件中，每轮的结果存放格式为：

```
结果 A的时间(s) B的时间(s)	// A 先手时
结果 A的时间(s) B的时间(s)	// B 先手时
```

文件会最后给出总的结果统计情况，注意只有当程序的返回值为 0/1/2 时，时间才有意义。

程序的返回值意义如下：

- 0 : 平局
- 1 : A 胜出，结束
- 2 : B 胜出，结束
- 3 : A 出bug，结束
- 4 : A 做出了非法的落子，结束
- 5 : B 出 bug，结束
- 6 : B 做出了非法的落子，结束
- 7 : A 超时
- 8 : B 超时
- -1 : 载入文件 A 出错
- -2 : 载入文件 B 出错
- -3 : A 文件中无法找到需要的函数接口
- -4 : B 文件中无法找到需要的函数接口

**注意：**由于 `dlopen` 并不会搜索当前文件夹下的 so 文件，若要加载同文件夹下的 so 文件，请在路径前面加入 `./`，即使用 `./ai.so` 表示 so 文件路径。

## 编译策略程序

策略程序位于 `Strategy` 文件夹下，该文件夹内容如下

```
Strategy
├── Judge.cpp
├── Judge.h
├── Makefile
├── Point.h
├── Strategy.cpp
└── Strategy.h
```

其中 `Strategy.cpp` 是你需要编写的策略文件。策略编写完成后，执行

```bash
make so
```

这会生成 `../so/Strategy.so` 文件，可以直接被评测框架调用。

## 错误捕获

由于 Linux 下的 Access Violation 较为严格，故策略程序相较于其他平台更容易出现崩溃情况。由于框架本身的限制，我们无法完全保证策略程序的崩溃不影响框架的正常运行。

通过处理 Linux 的相关信号，框架能够对常见的导致程序崩溃的错误进行捕获。在捕获到错误时，框架会报告错误并以 `bug occurred` 对相应AI判负以结束此轮对局，可能的错误提示如下：

> \*\*CRITICAL\*\* error occurs when A getPoint: segment_fault_exception

这表明框架在调用策略 A 的 `getPoint` 函数时出现了内存访问冲突（SigSegV），常见的情况有数组越界、访问空/野指针等。

在 `./so` 文件夹中包含了四个错误样例

```
so
├── ClearPointDoubleFree.so
├── ClearPointSegmentFault.so
├── GetPointDoubleFree.so
└── GetPointSegmentFault.so
```

它们分别是在 `ClearPoint `函数与 `GetPoint` 函数内发生`SegmentFault`或 `DoubleFree` 错误的策略程序。你可以通过调用这些错误 AI 来校验你的电脑上是否能正常捕获这些信号，注意，由于系统环境的差异，这些错误未必能够成功捕获。在这种情况下，只能尽力**确保你的程序不会出现 BUG**。

## 快速上手

为了便于对相关操作不熟悉的同学快速上手，在 `Connect4` 文件夹内准备了 makefile 用于一键生成可执行文件与策略的编译结果。

你可以在编写策略前先在 `Connect4` 目录下用以下两行指令快速跑通程序

```bash
make
./Compete/Compete so/Strategy.so so/GetPointSegmentFault.so result.txt 1
```

期望的结果是，第一回合 A 因为 `made illegal step` 判负，第二回合 B 因为 `bug occurred` 判负。如果没有问题，则可以开始编写你的代码。

