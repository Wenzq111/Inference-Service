# AGENTS.md — Inference Service AI 协作规范

## 项目概述

- **项目名称**：高性能推理服务 (Inference Service)
- **项目目标**：构建一个支持多后端（ONNX Runtime / NCNN）的 C++ 推理服务，提供图像分类和目标检测的 HTTP API 接口。
- **语言标准**：C++17
- **构建系统**：CMake 3.15+
- **代码风格**：Google C++ Style Guide
- **核心依赖**：OpenCV 4.x、onnxruntime (>=1.15.0)、ncnn（最新版）
- **平台限制**：Ubuntu 20.04+ 或 Windows 10 (msvc)，专注于 CPU 推理，无需 GPU 开发。

## 目录结构约定

```
inference-service/
├── include/          # 公共头文件
├── src/              # 源文件（backend, preprocess, server）
├── tests/            # 单元测试 (Google Test)
├── benchmarks/       # 性能基准
├── models/           # 存放模型文件 (gitignored)
├── docs/             # 项目文档（本文件所在目录）
└── CMakeLists.txt
```

## AI 协作规范

1. 在编写任何非原子性操作的代码前，必须先输出实现计划，等待确认后才能编写代码。
2. 对于复杂任务，优先采用测试驱动开发（TDD）。
3. 生成的代码必须包含必要的错误处理逻辑（如空指针、边界条件、文件不存在等）。
4. 提交代码前，确保代码已通过 clang-format 格式化。
5. **安全规则**：执行 `git commit`、`git push` 或 `git merge` 等修改远程仓库的命令前，必须先将拟执行的命令展示给用户，并在获得明确确认后方可执行。严禁使用 `git push --force`。