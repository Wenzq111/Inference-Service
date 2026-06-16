# Inference Service 学习文档

本文档从工程实践角度讲解项目的核心设计思路、代码价值和可学习的技术要点。每节均引用项目中的真实代码进行讲解。

---

## 一、项目做了什么

Inference Service 是一个用 C++ 从零构建的**多后端 AI 推理服务**，提供两大能力：

1. **目标检测**（YOLOv5s）—— 上传图片，返回检测框坐标、类别和置信度
2. **文本生成**（TinyLlama）—— 发送 prompt，返回模型生成的文本

两大能力通过统一的 HTTP API 暴露，支持 ONNX Runtime 和 NCNN 两种推理后端自由切换。

整个项目从零构建了完整的推理链路：

```
HTTP 请求 → 图像预处理（Letterbox/Resize） → 推理后端（ONNX/NCNN） → 后处理（NMS/坐标映射） → JSON 响应
```

### 项目架构总览

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  HTTP Server │────→│  ObjectDetector  │────→│ InferenceBackend  │
│  (cpp-httplib)│     │  (组合器)         │     │   (抽象接口)       │
│  /detect     │     │  预处理+后处理     │     │  ┌──OnnxBackend   │
│  /generate   │     └──────────────────┘     │  └──NcnnBackend   │
│  /health     │     ┌──────────────────┐     └───────────────────┘
└──────────────┘     │  LlamaGenerator  │
                     │  (llama.cpp)      │
                     └──────────────────┘
```

---

## 二、项目的核心价值

### 2.1 完整的工程闭环

这不是一个 demo，而是一个**完整可运行的工程**：从 CMake 构建系统、跨平台依赖管理、单元测试、性能基准测试，到 HTTP 服务部署，每个环节都有对应的代码和文档。

### 2.2 真实的多后端适配难题

ONNX 和 NCNN 两个后端的模型格式、输出格式、依赖管理完全不同。项目中真实遇到了 NCNN 模型兼容性问题（onnx2ncnn 转换模型加载失败），并通过在代码中实现 YOLOv5 grid decoding 解决——这是实际工程中常见的"运行时补齐模型能力"的思路。

### 2.3 可复用的设计模式

PImpl、策略模式、生产者-消费者模式、RAII 资源管理——这些不是教科书示例，而是为解决真实问题引入的，每个模式都能在代码中看到它解决的具体问题。

---

## 三、亮点设计与代码讲解

### 3.1 策略模式 + 多态：InferenceBackend 抽象接口

**问题**：ONNX Runtime 和 NCNN 的 API 完全不同（前者用 `Ort::Session`，后者用 `ncnn::Net`），但上层检测器不关心底层用哪个框架。

**解法**：定义纯虚基类 `InferenceBackend`，上层只依赖接口，运行时通过环境变量选择具体实现：

```cpp
// include/inference_backend.h
class InferenceBackend {
public:
    virtual bool LoadModel(const std::string& model_path) = 0;
    virtual std::vector<std::vector<float>> Predict(
        const std::vector<std::vector<float>>& inputs) = 0;
    virtual std::vector<std::vector<int64_t>> GetInputShapes() const = 0;
    virtual std::vector<std::string> GetInputNames() const = 0;
    virtual std::vector<std::string> GetOutputNames() const = 0;
    virtual std::vector<std::vector<int64_t>> GetOutputShapes() const = 0;
    virtual ~InferenceBackend() = default;
};
```

在 HTTP 服务启动时，根据环境变量创建对应后端：

```cpp
// src/server/http_server.cpp
std::unique_ptr<InferenceBackend> backend;
if (config.backend_type == "ncnn") {
    auto ncnn_backend = std::make_unique<NcnnBackend>();
    ncnn_backend->SetInputSize(config.input_width, config.input_height);
    backend = std::move(ncnn_backend);
} else {
    backend = std::make_unique<OnnxBackend>();
}
```

**关键点**：`ObjectDetector` 的构造函数接收 `std::unique_ptr<InferenceBackend>`，内部只通过基类指针调用 `Predict()`——完全不关心底层是 ONNX 还是 NCNN。新增后端（如 TensorRT）只需继承 `InferenceBackend`，无需修改检测器代码。

---

### 3.2 PImpl 模式：隔离第三方依赖

**问题**：NCNN 和 ONNX Runtime 的头文件会污染公共头文件，导致使用方必须安装这些依赖才能编译。

**解法**：PImpl（Pointer to Implementation）模式，将所有第三方依赖的成员变量藏到 `.cpp` 文件中的 `Impl` 结构体里。

头文件只声明一个前向声明的 `Impl` 和 `unique_ptr`：

```cpp
// include/ncnn_backend.h
class NcnnBackend : public InferenceBackend {
public:
    NcnnBackend();
    ~NcnnBackend() override;
    bool LoadModel(const std::string& model_path) override;
    // ...
private:
    struct Impl;                      // 前向声明，不暴露内部结构
    std::unique_ptr<Impl> impl_;      // 只存指针，大小固定
};
```

实现在 `.cpp` 中定义，持有所有 NCNN 相关对象：

```cpp
// src/backend/ncnn_backend.cpp
#include <net.h>          // NCNN 头文件只在 .cpp 中引入
#include <gpu.h>

struct NcnnBackend::Impl {
    ncnn::Net net;                                 // NCNN 推理网络
    std::vector<std::string> input_names;          // 输入 blob 名称
    std::vector<std::string> output_names;         // 输出 blob 名称
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
    bool model_loaded = false;
    bool vulkan_enabled = false;
    int fallback_input_width = 640;
    int fallback_input_height = 640;
    int fallback_input_channels = 3;
};
```

**效果**：
- `ncnn_backend.h` **完全不包含** `<net.h>` 等NCNN头文件
- 使用 `NcnnBackend` 的代码（如 `http_server.cpp`）只需 `#include "ncnn_backend.h"`，不需要安装 NCNN
- 修改 `Impl` 内部结构不影响头文件，不触发下游重编译

**同一个模式也用于** `LlamaGenerator` 和 `BatchPreprocessor`——所有持有第三方库对象的类都使用 PImpl。

---

### 3.3 YOLOv5 Grid Decoding：在代码中补齐模型缺失的能力

**问题**：ONNX 版 YOLOv5s 模型内含 Detect 层，输出已经是解码后的 `[25200, 85]` 格式。但 NCNN 版模型（nihui/ncnn-assets）不包含 Detect 层，输出 3 个头的 raw logits——坐标是 grid 偏移量，宽高是 log 空间值，置信度是未激活的 logits。

**解法**：在 `NcnnBackend::Predict()` 中实现完整的 YOLOv5 grid decoding，将 raw logits 解码为与 ONNX 输出一致的格式：

```cpp
// src/backend/ncnn_backend.cpp（简化版，省略外围循环）
for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
        // 从 NCNN 通道分离存储中读取 85 个 raw logits
        float raw[85];
        for (int k = 0; k < 85; ++k) {
            int ch = a * 85 + k;
            raw[k] = data[ch * h * w + y * w + x];
        }

        // YOLOv5 坐标解码（grid decoding）
        // cx = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
        // cy = (sigmoid(ty) * 2 - 0.5 + grid_y) * stride
        // w  = (sigmoid(tw) * 2)^2 * anchor_w
        // h  = (sigmoid(th) * 2)^2 * anchor_h
        float sig_tx = 1.0f / (1.0f + std::exp(-raw[0]));
        float sig_ty = 1.0f / (1.0f + std::exp(-raw[1]));
        float sig_tw = 1.0f / (1.0f + std::exp(-raw[2]));
        float sig_th = 1.0f / (1.0f + std::exp(-raw[3]));

        float cx = (sig_tx * 2.0f - 0.5f + static_cast<float>(x)) * stride;
        float cy = (sig_ty * 2.0f - 0.5f + static_cast<float>(y)) * stride;
        float bw = (sig_tw * 2.0f) * (sig_tw * 2.0f) * anchor_w;
        float bh = (sig_th * 2.0f) * (sig_th * 2.0f) * anchor_h;

        // objectness + class scores sigmoid
        concatenated.push_back(1.0f / (1.0f + std::exp(-raw[4])));
        for (int k = 5; k < 85; ++k) {
            concatenated.push_back(1.0f / (1.0f + std::exp(-raw[k])));
        }
    }
}
```

**为什么这个设计有价值**：

1. **不是所有模型都能完美转换**：onnx2ncnn 转换的 NCNN 模型与 Homebrew NCNN 存在兼容性问题，而 nihui 的预转换模型可以加载但不包含 Detect 层。在代码中补齐解码逻辑，是实际工程中常见的妥协方案。

2. **解码位置选在后端层而非后处理层**：grid decoding 产出的 `[25200, 85]` 格式与 ONNX 输出完全一致，`ProcessYoloOutput` 无需任何修改——NCNN 的特殊性被封装在后端内部，不泄漏到上层。

3. **验证结果证明方案正确**：NCNN 与 ONNX 对同一张图的检测结果，高置信度框 bbox 差异 <2px，置信度差异 <0.02。

---

### 3.4 组合器模式：ObjectDetector 统一推理链路

**问题**：目标检测不只是"模型推理"，还需要预处理（Letterbox）、后处理（NMS、坐标映射）。如何让这些步骤协同工作？

**解法**：`ObjectDetector` 作为组合器，持有推理后端的所有权，在 `Detect()` 方法中串联完整链路：

```cpp
// src/detector/object_detector.cpp
std::vector<Detection> ObjectDetector::Detect(const cv::Mat& img,
                                                float confidence_threshold,
                                                float nms_threshold) {
    // 步骤1：Letterbox 预处理，保持宽高比缩放并填充灰色
    LetterboxResult lb = Letterbox(img, input_width_, input_height_);

    // 步骤2：BGR→RGB + CHW + 归一化
    std::vector<float> chw_data = MatToChw(lb.image, mean_, std_);

    // 步骤3：调用后端推理（多态调用，不关心具体后端）
    std::vector<std::vector<float>> inputs = {std::move(chw_data)};
    std::vector<std::vector<float>> outputs = backend_->Predict(inputs);

    // 步骤4：YOLO 后处理（NMS 过滤 + 坐标裁剪）
    std::vector<Detection> detections = ProcessYoloOutput(
        outputs[0], confidence_threshold, nms_threshold,
        input_width_, input_height_);

    // 步骤5：Letterbox 坐标映射回原图
    std::vector<Detection> original_dets = ScaleDetectionsToOriginal(
        detections, lb.x_offset, lb.y_offset, lb.scale,
        img.cols, img.rows);

    return original_dets;
}
```

**关键设计**：`backend_` 是 `std::unique_ptr<InferenceBackend>`，构造函数通过 `std::move` 接收所有权：

```cpp
ObjectDetector::ObjectDetector(std::unique_ptr<InferenceBackend> backend)
    : backend_(std::move(backend)) {}
```

这意味着：
- 后端的生命周期由检测器管理，不会出现悬空指针
- 后端在构造时就已加载模型（`LoadModel` 在创建检测器之前调用），检测器不需要处理"模型未加载"的状态
- 整个链路是单向数据流：图像 → 预处理 → 推理 → 后处理 → 结果，每步失败都返回空向量并记录日志

---

### 3.5 生产者-消费者模式：BatchPreprocessor 线程池

**问题**：批量图像预处理是 CPU 密集型任务，串行处理太慢，需要多线程并行。

**解法**：`BatchPreprocessor` 内部维护一个线程池和任务队列，采用经典的生产者-消费者模式：

```cpp
// src/pipeline/batch_preprocessor.cpp
struct BatchPreprocessor::Impl {
    std::queue<Task> queue_;                   // 任务队列
    std::mutex queue_mutex_;                   // 队列锁
    std::condition_variable queue_cv_;         // 队列条件变量（通知工作线程）
    std::condition_variable done_cv_;          // 完成条件变量（通知 WaitAll）
    std::vector<std::thread> workers_;         // 工作线程池
    std::atomic<bool> stopped_;                // 停止标志
    std::atomic<size_t> pending_count_;        // 待处理任务计数
};
```

工作线程主循环：

```cpp
void WorkerLoop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // 等待：队列非空 或 收到停止信号
            queue_cv_.wait(lock, [this] {
                return !queue_.empty() || stopped_.load();
            });
            // 停止且队列为空时退出
            if (stopped_.load() && queue_.empty()) return;
            task = std::move(queue_.front());
            queue_.pop();
        }

        // 执行预处理（不持锁，允许其他线程并发取任务）
        PreprocessResult result;
        // ... 预处理逻辑 ...

        // 调用回调通知完成
        if (callback_) callback_(task.index, result);

        // 减少待处理计数，通知 WaitAll
        pending_count_--;
        done_cv_.notify_all();
    }
}
```

**值得学习的细节**：

1. **锁粒度控制**：取任务时加锁，执行预处理时不持锁——多线程可以并行处理不同图像
2. **`condition_variable::wait` 带谓词**：避免虚假唤醒，只有队列非空或收到停止信号才继续
3. **`atomic` + `condition_variable` 配合**：`pending_count_` 用 `atomic` 保证无锁计数，`done_cv_` 用于 `WaitAll()` 阻塞等待
4. **RAII 析构**：析构函数调用 `Stop()`，`Stop()` 设置停止标志、通知所有线程、等待线程退出（`join`），确保不会泄漏线程

---

### 3.6 Letterbox 坐标映射：预处理与后处理的对称设计

**问题**：Letterbox 预处理会在图像周围填充灰色边距，检测框坐标是基于 640×640 模型输入空间的。如何准确映射回原始图像？

**解法**：预处理时记录 `scale`、`x_offset`、`y_offset`，后处理时反变换：

预处理（`Letterbox`）：
```
原图 ──× scale──→ 缩放图 ──+ offset──→ 640×640 Letterbox图
```

后处理（`ScaleDetectionsToOriginal`）：
```
640×640 坐标 ──- offset──→ 缩放图坐标 ──÷ scale──→ 原图坐标
```

代码实现：

```cpp
// src/postprocess/yolo_postprocess.cpp
// Letterbox 重载：减偏移 → 除缩放比 → 裁剪边界
std::vector<Detection> ScaleDetectionsToOriginal(
    const std::vector<Detection>& detections,
    int x_offset, int y_offset, float scale,
    int src_width, int src_height) {

    float inv_scale = 1.0f / scale;
    float offset_x = static_cast<float>(x_offset);
    float offset_y = static_cast<float>(y_offset);
    float max_x = static_cast<float>(src_width - 1);
    float max_y = static_cast<float>(src_height - 1);

    for (const auto& det : detections) {
        Detection scaled;
        scaled.x1 = std::max(0.0f, std::min((det.x1 - offset_x) * inv_scale, max_x));
        scaled.y1 = std::max(0.0f, std::min((det.y1 - offset_y) * inv_scale, max_y));
        scaled.x2 = std::max(0.0f, std::min((det.x2 - offset_x) * inv_scale, max_x));
        scaled.y2 = std::max(0.0f, std::min((det.y2 - offset_y) * inv_scale, max_y));
        // class_id 和 confidence 不变
    }
}
```

**设计亮点**：`ScaleDetectionsToOriginal` 提供了两个重载——一个用于简单线性缩放（`ResizeAndNorm`），一个用于 Letterbox（带偏移和缩放比）。上层代码根据预处理方式选择对应重载，不需要判断逻辑。

---

### 3.7 NMS 非极大值抑制：后处理核心算法

**问题**：YOLO 模型对同一个目标可能输出多个重叠的检测框，需要去除冗余。

**解法**：经典 NMS 算法——按置信度排序，保留最高置信度的框，抑制与它 IoU 过高的其他框：

```cpp
// src/postprocess/yolo_postprocess.cpp
static std::vector<Detection> Nms(
    const std::vector<Detection>& detections, float nms_threshold) {
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);  // 保留当前最高置信度框

        // 抑制与当前框 IoU 过高的后续框
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (ComputeIoU(detections[i], detections[j]) > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}
```

**可以学到的**：
- NMS 前必须按置信度降序排序——先处理最高置信度的框，再抑制低置信度的重复框
- 不同类别的框不应互相抑制（当前实现未按类别分组，是技术债，已记录在 HANDOVER.md 中）
- 当前 O(n²) 复杂度对小规模检测（几十到几百个框）足够，大规模场景可考虑 Soft-NMS 或分区索引

---

### 3.8 RAII 与资源管理：确保不泄漏

项目中大量使用 RAII（Resource Acquisition Is Initialization）模式管理资源：

**1. `std::unique_ptr` 管理后端对象的所有权**

```cpp
// 构造时接收所有权
ObjectDetector::ObjectDetector(std::unique_ptr<InferenceBackend> backend)
    : backend_(std::move(backend)) {}

// 析构时自动释放，无需手动 delete
```

**2. Vulkan GPU 实例的 RAII 管理**

```cpp
// src/backend/ncnn_backend.cpp
NcnnBackend::~NcnnBackend() {
    if (impl_->vulkan_enabled) {
        ncnn::destroy_gpu_instance();  // 只在启用时销毁
    }
    // impl_ 的 unique_ptr 析构自动释放 Net 等对象
}
```

**3. 线程池的 RAII 停止**

```cpp
// src/pipeline/batch_preprocessor.cpp
BatchPreprocessor::~BatchPreprocessor() {
    Stop();  // 析构时自动停止线程池，等待所有线程退出
}
```

**核心原则**：每个获取资源的操作（创建 GPU 实例、启动线程、加载模型）都在对应的析构函数中有匹配的释放操作。即使用户忘记调用 `Stop()`，析构函数也会兜底。

---

### 3.9 环境变量配置：零配置文件部署

**问题**：不同部署环境（开发/测试/生产）需要不同的端口、后端、模型路径，但不想引入配置文件依赖。

**解法**：全部通过环境变量配置，`main.cpp` 逐个读取并设置默认值：

```cpp
// src/main.cpp
ServerConfig config;
config.port = 8080;                          // 默认端口
config.detector_model_path = "models/yolov5s.onnx";  // 默认 ONNX 模型
config.backend_type = "onnx";                // 默认后端

const char* env_port = std::getenv("INFERENCE_PORT");
if (env_port) config.port = std::stoi(env_port);

const char* env_backend = std::getenv("BACKEND_TYPE");
if (env_backend) config.backend_type = env_backend;
```

切换后端只需一个环境变量：

```bash
# ONNX（默认）
./build/inference_service

# NCNN
BACKEND_TYPE=ncnn DETECTOR_MODEL=models/yolov5s.param ./build/inference_service
```

**优点**：无配置文件、Docker 友好（环境变量天然支持 `-e` 传入）、12-Factor App 合规。

---

### 3.10 跨平台兼容性处理

项目需要兼容 macOS（Apple Silicon）和 Linux，几个关键处理：

**1. CMake 依赖搜索路径**

```cmake
# CMakeLists.txt —— 支持用户指定路径、Homebrew、系统标准路径
set(_ORT_SEARCH_PATHS
    ${ONNXRUNTIME_ROOT}
    /opt/homebrew      # macOS Apple Silicon
    /usr/local         # macOS Intel / Linux
    /usr               # Linux
)
```

**2. Vulkan GPU 加速的条件启用**

```cpp
// src/backend/ncnn_backend.cpp
// macOS 上 Vulkan 不稳定，默认禁用；Linux 可通过环境变量启用
const char* env_vulkan = std::getenv("NCNN_VULKAN");
if (env_vulkan && std::string(env_vulkan) == "1" &&
    ncnn::get_gpu_count() > 0) {
    // 启用 Vulkan
}
```

**3. GTest ABI 兼容性**

macOS 上 anaconda3 的 GTest 动态库与 Apple Clang arm64 存在 ABI 不兼容，CMakeLists.txt 自动优先使用 Homebrew GTest：

```cmake
# CMakeLists.txt
if(NOT GTest_ROOT)
    if(APPLE AND EXISTS /opt/homebrew/lib/cmake/GTest)
        set(GTest_ROOT /opt/homebrew)
    endif()
endif()
```

---

## 四、可以学到的技术要点总结

| 技术要点 | 本项目中的应用 | 代码位置 |
|----------|--------------|----------|
| 策略模式 + 多态 | InferenceBackend 抽象接口，运行时选择后端 | `include/inference_backend.h` |
| PImpl 模式 | NcnnBackend、LlamaGenerator、BatchPreprocessor 隐藏第三方依赖 | `include/ncnn_backend.h` 等 |
| RAII 资源管理 | unique_ptr 管理后端所有权，析构函数释放 GPU/线程池 | `src/backend/ncnn_backend.cpp` |
| 生产者-消费者模式 | BatchPreprocessor 线程池 + 条件变量 | `src/pipeline/batch_preprocessor.cpp` |
| YOLOv5 Grid Decoding | NCNN 多头输出解码，补齐模型缺失的 Detect 层 | `src/backend/ncnn_backend.cpp:326-423` |
| Letterbox 坐标映射 | 预处理记录 offset/scale，后处理反变换回原图 | `src/postprocess/yolo_postprocess.cpp` |
| NMS 非极大值抑制 | 后处理核心算法，去除重叠检测框 | `src/postprocess/yolo_postprocess.cpp` |
| CMake 跨平台依赖管理 | 多路径搜索 + 用户自定义路径 + FetchContent | `CMakeLists.txt` |
| 环境变量配置 | 零配置文件部署，Docker 友好 | `src/main.cpp` |
| 条件编译与平台适配 | Vulkan 条件启用、GTest ABI 兼容、Metal 框架链接 | `CMakeLists.txt`、`src/backend/ncnn_backend.cpp` |

---

## 五、推荐阅读顺序

如果你是第一次接触本项目，建议按以下顺序阅读代码：

1. **`include/inference_backend.h`** —— 理解核心抽象接口
2. **`src/detector/object_detector.cpp`** —— 理解完整推理链路
3. **`include/ncnn_backend.h`** + **`src/backend/ncnn_backend.cpp`** —— 理解 PImpl 模式和 YOLOv5 grid decoding
4. **`src/postprocess/yolo_postprocess.cpp`** —— 理解 NMS 和坐标映射
5. **`src/pipeline/batch_preprocessor.cpp`** —— 理解线程池和条件变量
6. **`src/server/http_server.cpp`** —— 理解 HTTP API 和后端选择逻辑
7. **`CMakeLists.txt`** —— 理解跨平台构建和依赖管理
