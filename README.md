# Inference Service

高性能推理服务，支持多后端（ONNX Runtime / NCNN），提供图像分类和目标检测的 HTTP API 接口。

## 项目信息

- 语言标准：C++17
- 构建系统：CMake 3.15+
- 代码风格：Google C++ Style Guide
- 核心依赖：OpenCV 4.x、onnxruntime (>=1.20.0)、ncnn（最新版）
- 平台兼容性：macOS（Apple Silicon M 系列）和 Linux（Ubuntu 20.04+），架构设计保持可扩展性以便未来兼容更多平台

## 目录结构

```
Inference-Service/
├── include/              # 公共头文件
│   ├── logger.h            # Logger 工具类
│   ├── timer.h             # Timer 性能测量类
│   ├── preprocess.h        # 图像预处理函数声明
│   ├── inference_backend.h # 推理后端抽象接口
│   └── onnx_backend.h      # ONNX Runtime 后端声明
├── src/                  # 源文件
│   ├── utils/              # 工具类（Logger、Timer）
│   ├── preprocess/         # 图像预处理（ResizeAndNorm、Letterbox）
│   ├── backend/            # 推理后端（OnnxBackend）
│   └── main.cpp            # 主入口
├── docs/                 # 项目文档
├── build.sh              # 全量重建脚本
├── rebuild.sh            # 增量编译脚本
├── CMakeLists.txt        # CMake 构建配置
└── README.md
```

## 依赖安装

### OpenCV
macOS: `brew install opencv`
Ubuntu: `sudo apt install libopencv-dev`

### ONNX Runtime
macOS: `brew install onnxruntime`
Ubuntu: 参考 [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) 下载预编译包，设置 `ONNXRUNTIME_ROOT` 指向解压路径

### NCNN
macOS: `brew install ncnn`
Ubuntu: 参考 [ncnn 编译指南](https://github.com/Tencent/ncnn#build) 从源码编译，设置 `NCNN_ROOT` 指向安装路径

> 自定义路径示例：`cmake -DONNXRUNTIME_ROOT=/path/to/ort -DNCNN_ROOT=/path/to/ncnn ..`

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