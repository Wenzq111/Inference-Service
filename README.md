# Inference Service

高性能推理服务，支持多后端（ONNX Runtime / NCNN），提供目标检测和 LLM 文本生成的 HTTP API 接口。

## 项目信息

- 语言标准：C++17
- 构建系统：CMake 3.15+
- 代码风格：Google C++ Style Guide
- 核心依赖：OpenCV 4.x、onnxruntime (>=1.20.0)、ncnn、llama.cpp、cpp-httplib
- 平台兼容性：macOS（Apple Silicon M 系列）和 Linux（Ubuntu 20.04+）

## 目录结构

```
Inference-Service/
├── include/              # 公共头文件
│   ├── logger.h            # Logger 工具类
│   ├── timer.h             # Timer 性能测量类
│   ├── preprocess.h        # 图像预处理函数声明
│   ├── inference_backend.h # 推理后端抽象接口
│   ├── onnx_backend.h      # ONNX Runtime 后端声明
│   ├── ncnn_backend.h      # NCNN 后端声明
│   ├── yolo_postprocess.h  # YOLO 后处理函数声明
│   ├── detector.h          # 目标检测器声明
│   ├── llm_generator.h     # LLM 文本生成器声明
│   ├── batch_preprocessor.h # 批量预处理器声明
│   └── http_server.h       # HTTP 服务声明
├── src/                  # 源文件
│   ├── utils/              # 工具类（Logger、Timer）
│   ├── preprocess/         # 图像预处理（ResizeAndNorm、Letterbox、MatToChw）
│   ├── backend/            # 推理后端（OnnxBackend、NcnnBackend）
│   ├── postprocess/        # YOLO 后处理（NMS、坐标缩放）
│   ├── detector/           # 目标检测器（ObjectDetector）
│   ├── llm/                # LLM 文本生成（LlamaGenerator）
│   ├── pipeline/           # 批量预处理（BatchPreprocessor）
│   ├── server/             # HTTP 服务（RunServer）
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

### llama.cpp

macOS: `brew install llama.cpp`
Ubuntu: 参考 [llama.cpp 编译指南](https://github.com/ggml-org/llama.cpp) 从源码编译，设置 `LLAMA_ROOT` 指向安装路径

### cpp-httplib

无需手动安装，CMake 通过 FetchContent 自动下载。

> 自定义路径示例：`cmake -DONNXRUNTIME_ROOT=/path/to/ort -DNCNN_ROOT=/path/to/ncnn -DLLAMA_ROOT=/path/to/llama.cpp ..`

## 编译与运行

### 方式一：使用构建脚本（推荐）

```bash
# 全量重建（清理 build 目录后重新编译）
./build.sh

# 增量编译（保留 build 缓存，速度更快）
./rebuild.sh
```

### 方式二：手动编译

```bash
mkdir build && cd build
cmake ..
make
```

### 启动服务

```bash
./build/inference_service
```

服务默认监听 `http://localhost:8080`，浏览器打开可查看 API 说明页面。

## HTTP API 使用

### 健康检查

```bash
curl http://localhost:8080/health
```

响应：

```json
{"status":"ok"}
```

### 目标检测

上传图像进行 YOLO 目标检测：

```bash
curl -X POST -F "image=@test.jpg" http://localhost:8080/detect
```

可选查询参数（`confidence` 和 `nms_threshold`）：

```bash
curl -X POST -F "image=@test.jpg" "http://localhost:8080/detect?confidence=0.5&nms_threshold=0.4"
```

成功响应：

```json
{
  "success": true,
  "detections": [
    {"bbox": [100.5, 200.3, 300.1, 400.7], "class_id": 0, "confidence": 0.95}
  ],
  "inference_time_ms": 23.5
}
```

错误响应：

| 状态码 | 含义 |
|---|---|
| 400 | 请求缺少 image 字段或图像解码失败 |
| 503 | 检测器模型未加载 |

### 文本生成

发送 prompt 进行 LLM 文本生成：

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"你好，请介绍一下自己","max_tokens":128,"temperature":0.7}' \
  http://localhost:8080/generate
```

请求参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| prompt | string | （必填） | 输入提示文本 |
| max_tokens | int | 512 | 最大生成 token 数 |
| temperature | float | 0.8 | 温度（0 = 贪心解码） |
| top_p | float | 0.95 | 核采样概率阈值 |

成功响应：

```json
{
  "success": true,
  "text": "我是一个AI助手...",
  "inference_time_ms": 150.3
}
```

错误响应：

| 状态码 | 含义 |
|---|---|
| 400 | 请求缺少 prompt 字段 |
| 503 | LLM 模型未加载 |

## 环境变量配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| INFERENCE_PORT | 8080 | HTTP 服务监听端口 |
| DETECTOR_MODEL | models/yolov5s.onnx | 检测器模型文件路径 |
| LLM_MODEL | models/llama.gguf | LLM 模型文件路径（GGUF 格式） |
| BACKEND_TYPE | onnx | 推理后端类型（onnx 或 ncnn） |
| INPUT_WIDTH | 640 | 检测器输入宽度 |
| INPUT_HEIGHT | 640 | 检测器输入高度 |

示例：

```bash
# 使用 NCNN 后端，端口 9090
INFERENCE_PORT=9090 BACKEND_TYPE=ncnn ./build/inference_service

# 指定模型路径
DETECTOR_MODEL=/data/models/yolov5s.onnx LLM_MODEL=/data/models/qwen.gguf ./build/inference_service
```

## 模型文件

将模型文件放置在 `models/` 目录下：

- `models/yolov5s.onnx` — YOLOv5s 目标检测模型（ONNX 格式，**必须为 float32 输入类型**）
- `models/llama.gguf` — LLM 文本生成模型（GGUF 格式）

模型文件不纳入版本控制，需自行下载或导出。

### 获取 YOLOv5s float32 ONNX 模型

> **注意**：GitHub releases 上的 `yolov5s.onnx` 为 float16 输入类型，与本项目不兼容（ONNX Runtime 后端要求 float32 输入）。请按以下方式导出 float32 版本：

```bash
# 1. 安装依赖
pip install torch onnx onnxruntime onnxslim

# 2. 克隆 YOLOv5 仓库
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 3. 下载 PyTorch 权重并导出 float32 ONNX
python export.py --weights yolov5s.pt --include onnx --simplify --opset 12

# 4. 将导出的模型复制到项目目录
cp yolov5s.onnx /path/to/Inference-Service/models/yolov5s.onnx
```

导出完成后可验证输入类型：

```python
from onnx import load
model = load('models/yolov5s.onnx')
for inp in model.graph.input:
    # elem_type=1 表示 float32，elem_type=10 表示 float16（不兼容）
    print(f'{inp.name}: type={inp.type.tensor_type.elem_type}')
```

### 获取 LLM 模型

推荐使用 [TinyLlama 1.1B Chat](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)（Q4_K_M 量化，约 638MB）：

```bash
# 从 HuggingFace 下载
wget -O models/llama.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

也可使用其他 GGUF 格式模型，通过环境变量 `LLM_MODEL` 指定路径。

## 文档

- [AI 协作规范](docs/AGENTS.md)
- [开发任务清单](docs/TASKS.md)
- [架构设计文档](docs/ARCHITECTURE.md)
- [项目交接文档](docs/HANDOVER.md)
- [llama.cpp API 指南](docs/LLAMA_CPP_GUIDE.md)
- [批量预处理器指南](docs/BATCH_PREPROCESSOR_GUIDE.md)
