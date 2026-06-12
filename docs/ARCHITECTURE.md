# 架构设计文档

创建日期：2026-05-30

## 1. 总体架构分层

### Layer 0：纯数据对象

无任何依赖，仅定义结构体和枚举。

| 模块 | 说明 |
|------|------|
| Detection | 目标检测结果结构体（bbox, label, confidence） |
| ClassificationResult | 图像分类结果结构体 |
| InferenceConfig | 推理配置结构体 |

**依赖方向**：无外部依赖，被所有上层模块使用。

### Layer 1-2：工具类与配置

| 层级 | 模块 | 说明 |
|------|------|------|
| Layer 1 | Logger | 日志工具类 |
| Layer 1 | Timer | 性能测量工具类 |
| Layer 2 | Preprocessor | 图像预处理（resize、归一化、letterbox、BGR/RGB 转换） |

**依赖方向**：Layer 1 → Layer 0；Layer 2 → Layer 1（使用 Logger/Timer）→ Layer 0

### Layer 3：核心业务逻辑

| 模块 | 说明 |
|------|------|
| InferenceBackend | 纯虚基类，定义推理后端抽象接口 |
| OnnxRuntimeBackend | ONNX Runtime 后端实现 |
| NcnnBackend | NCNN 后端实现 |
| ObjectDetector | 组合后端 + 后处理的目标检测器 |
| LlamaGenerator | LLM 文本生成模块 |

**依赖方向**：Layer 3 → Layer 2 → Layer 1 → Layer 0

### Layer 4：控制器与接口

| 模块 | 说明 |
|------|------|
| RestServer | HTTP REST API 服务 |
| CLI Entry | 命令行入口 |

**依赖方向**：Layer 4 → Layer 3 → Layer 2 → Layer 1 → Layer 0

## 2. 模块依赖关系图

```
M9 (REST API) ──→ M3 (ONNX Runtime Backend) ──→ M2 (InferenceBackend 抽象接口)
                ──→ M6 (ObjectDetector) ──→ M2 (InferenceBackend 抽象接口)
                                        ──→ M5 (YOLO 后处理) ──→ M1 (预处理)
M6 (ObjectDetector) ──→ M5 ──→ M1
M7 (LLM 文本生成) — 独立模块，依赖 Llama.cpp
M8 (批量预处理流水线) ──→ M1 (预处理)
```

## 3. 关键接口定义

### InferenceBackend 抽象类

```cpp
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual bool LoadModel(const std::string& model_path, const InferenceConfig& config) = 0;
    virtual bool Predict(const std::vector<float>& input, std::vector<float>& output) = 0;
    virtual std::vector<InputShape> GetInputShapes() const = 0;
};
```

### ObjectDetector 类

```cpp
class ObjectDetector {
public:
    bool Init(const std::string& model_path, const InferenceConfig& config);
    std::vector<Detection> Detect(const cv::Mat& image);
};
```

### LlamaGenerator 类

```cpp
class LlamaGenerator {
public:
    bool Load(const std::string& model_path);
    std::string Generate(const std::string& prompt, std::function<void(const std::string&)> callback = nullptr);
};
```

## 4. 设计决策记录

| 决策 | 原因 | 状态 |
|------|------|------|
| 为什么选择 ONNX Runtime 而非 LibTorch？ | ONNX Runtime 专注推理而非训练，体积更小、部署更轻量；支持多框架模型转换（PyTorch→ONNX、TF→ONNX）；CPU 推理性能优于 LibTorch；提供 int8 量化能力；无需 Python 依赖链 | 已完善 |
| 为什么预处理输出 CHW 顺序的 float 数组？ | ONNX Runtime 和 NCNN 均要求 CHW 输入格式；CHW 使每个通道数据连续排列，利于 CPU/GPU 向量化运算和缓存命中；与训练框架（PyTorch）的数据布局一致，避免推理时额外转换 | 已完善 |
| 为什么 NMS 放在后处理单独模块？ | NMS 是通用算法，与具体推理后端无关，独立模块可被不同检测器复用；不同检测任务可能需要不同 NMS 参数，独立模块便于参数化配置；便于单独测试和优化 NMS 性能 | 已完善 |
| 为什么 Logger 使用静态方法而非单例？ | Logger 无状态依赖，静态方法更简洁，避免单例的线程安全和生命周期管理问题 | 已完善 |
| 预处理函数为什么要加 target_w/target_h 边界检查？ | OpenCV 的 resize 对非法尺寸会产生未定义行为或崩溃，防御性检查避免下游错误 | 已完善 |
| InferenceBackend 接口中为何增加 GetInputNames/GetOutputNames？ | ONNX Runtime 和 NCNN 都依赖输入/输出节点名称来绑定数据，后端实现必须暴露这些名称供上层调用 | 已完善 |
| 为什么项目要求跨平台兼容？ | macOS（Apple Silicon）和 Linux 是主流部署平台；条件编译隔离平台特有代码（如 CoreML EP 仅 Apple Silicon）保证其他平台零影响；CMake 多路径查找和 `ONNXRUNTIME_ROOT` 变量支持自定义安装位置；架构保持可扩展性便于未来添加 Windows、Android 等平台支持 | 已完善 |

## 5. 性能优化方向

| 方向 | 说明 | 状态 |
|------|------|------|
| int8 量化 | [待补充] | 待完善 |
| 多线程预处理流水线 | [待补充] | 待完善 |
| Winograd 卷积加速（NCNN） | [待补充] | 待完善 |

## 6. 已知限制与未来扩展

- 当前仅验证 macOS (Apple Silicon) 和 Linux (Ubuntu 20.04+) 编译运行，Windows 支持待验证
- 平台特有功能（CoreML EP）通过条件编译隔离，未来可类似方式添加 CUDA EP（Linux/Windows）、NNAPI EP（Android）等
- `localtime_r` 在 Windows 上不兼容，后续需封装跨平台时间函数
- CMake 依赖查找已支持多平台路径和用户自定义路径，可随平台扩展继续添加搜索路径