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
│   ├── preprocess/   # 图像预处理（ResizeAndNorm、Letterbox）
│   └── main.cpp      # 主入口
├── docs/             # 项目文档
├── build.sh          # 全量重建脚本
├── rebuild.sh        # 增量编译脚本
├── CMakeLists.txt    # CMake 构建配置
└── README.md
```

## 依赖安装

macOS: `brew install opencv`
Ubuntu: `sudo apt install libopencv-dev`
Windows: `vcpkg install opencv`

## 编译与运行

### 方式一：使用构建脚本（推荐）

```bash
# 全量重建（清理 build 目录后重新编译运行）
./build.sh

# 增量编译（保留 build 缓存，速度更快）
./rebuild.sh
```

### 方式二：手动编译

```bash
mkdir build && cd build
cmake ..
make
./inference_service
```

## 文档

- [AI 协作规范](docs/AGENTS.md)
- [开发任务清单](docs/TASKS.md)
- [架构设计文档](docs/ARCHITECTURE.md)