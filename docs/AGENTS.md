# AGENTS.md — Inference Service AI 协作规范

## 项目概述

- **项目名称**：高性能推理服务 (Inference Service)
- **项目目标**：构建一个支持多后端（ONNX Runtime / NCNN）的 C++ 推理服务，提供图像分类和目标检测的 HTTP API 接口。
- **语言标准**：C++17
- **构建系统**：CMake 3.15+
- **代码风格**：Google C++ Style Guide
- **核心依赖**：OpenCV 4.x (core, imgproc, imgcodecs)、onnxruntime (>=1.15.0)、ncnn（最新版）
- **平台限制**：Ubuntu 20.04+ 或 Windows 10 (msvc)，专注于 CPU 推理，无需 GPU 开发。
- **构建脚本**：`build.sh`（全量重建）、`rebuild.sh`（增量编译）

## 目录结构约定

```
Inference-Service/
├── include/              # 公共头文件
│   ├── logger.h              # Logger 工具类
│   ├── timer.h               # Timer 性能测量类
│   ├── preprocess.h          # 图像预处理函数声明
│   └── inference_backend.h   # 推理后端抽象接口
├── src/                  # 源文件
│   ├── main.cpp          # 主入口
│   ├── utils/            # 工具类实现
│   │   ├── logger.cpp    # Logger 实现，带时间戳和级别前缀的分级日志输出
│   │   └── timer.cpp     # Timer 实现，基于 steady_clock 的高精度计时
│   ├── preprocess/       # 图像预处理实现
│   │   └── preprocess.cpp    # ResizeAndNorm、Letterbox 实现
│   ├── backend/          # 推理后端实现（待开发）
│   ├── postprocess/      # 后处理实现（待开发）
│   ├── detector/         # 目标检测器（待开发）
│   ├── llm/              # LLM 文本生成（待开发）
│   ├── pipeline/         # 批量预处理流水线（待开发）
│   └── server/           # REST API 服务（待开发）
├── tests/                # 单元测试 (Google Test，待开发)
├── benchmarks/           # 性能基准（待开发）
├── models/               # 存放模型文件 (gitignored)
├── docs/                 # 项目文档
│   ├── AGENTS.md         # AI 协作规范
│   ├── TASKS.md          # 开发任务清单
│   └── ARCHITECTURE.md   # 架构设计文档
├── build.sh              # 全量重建脚本
├── rebuild.sh            # 增量编译脚本
├── .vscode/              # VSCode 配置
├── CMakeLists.txt        # CMake 构建配置
└── README.md             # 项目说明
```

## AI 协作规范

1. 在编写任何非原子性操作的代码前，必须先输出实现计划，等待确认后才能编写代码。
2. 对于复杂任务，优先采用测试驱动开发（TDD）。
3. 生成的代码必须包含必要的错误处理逻辑（如空指针、边界条件、文件不存在等）。
4. 提交代码前，确保代码已通过 clang-format 格式化。
5. **安全规则**：执行 `git commit`、`git push` 或 `git merge` 等修改远程仓库的命令前，必须先将拟执行的命令展示给用户，并在获得明确确认后方可执行。严禁使用 `git push --force`。
6. 代码注释规范：每个函数/类/结构体必须有 `//` 注释说明用途。
7. 代码提交规范：每次提交前必须先编译验证代码能正常运行，commit 信息使用中文。

## 编译与验证

提交代码前必须通过编译验证：

- 全量重建：`./build.sh`
- 增量编译：`./rebuild.sh`

## 命名规范

- namespace: `inference`
- 类名: PascalCase（如 `Logger`, `Timer`, `ObjectDetector`）
- 函数名: PascalCase（如 `ResizeAndNorm`, `Letterbox`, `LoadModel`）
- 成员变量: snake_case 带下划线后缀（如 `level_`, `running_`, `start_time_`）
- 枚举: PascalCase（如 `LogLevel::Debug`）

## 模块进度

详见 [开发任务清单](TASKS.md)。