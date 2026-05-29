# Inference Service

高性能推理服务，支持多后端（ONNX Runtime / NCNN），提供图像分类和目标检测的 HTTP API 接口。

## 项目信息

- 语言标准：C++17
- 构建系统：CMake 3.15+
- 代码风格：Google C++ Style Guide
- 核心依赖：OpenCV 4.x、onnxruntime (>=1.15.0)、ncnn（最新版）
- 平台限制：Ubuntu 20.04+ 或 Windows 10 (msvc)，专注于 CPU 推理

## 目录结构

```
Inference-Service/
├── include/          # 公共头文件
├── src/              # 源文件
│   ├── utils/        # 工具类（Logger、Timer）
│   └── main.cpp      # 主入口
├── docs/             # 项目文档
├── CMakeLists.txt    # CMake 构建配置
└── README.md
```

## 编译与运行

```bash
cd Inference-Service
mkdir build && cd build
cmake ..
make
./inference_service
```

## 文档

- [AI 协作规范](docs/AGENTS.md)
- [开发任务清单](docs/TASKS.md)
- [架构设计文档](docs/ARCHITECTURE.md)