# 项目交接文档 (HANDOVER.md)

创建日期：2026-05-30
最后更新：2026-06-16

## 项目整体状态

- 已完成模块：M0（项目骨架）、M1（图像预处理）、M2（推理后端抽象接口）、M3（ONNX Runtime 后端）、M4（NCNN 后端）、M5（YOLO 后处理）、M6（目标检测器）、M7（LLM 文本生成模块）、M8（批量预处理流水线）、M9（REST API 服务）、M10（单元测试集）、M11（基准测试）
- 代码量估算：约 5100 行（不含注释和文档）
- 当前可编译运行，启动 HTTP 服务监听 8080 端口，提供 /detect、/generate、/health 端点

## 目录结构

```
Inference-Service/
├── include/
│   ├── logger.h
│   ├── timer.h
│   ├── preprocess.h
│   ├── inference_backend.h
│   ├── onnx_backend.h
│   └── ncnn_backend.h
│   ├── yolo_postprocess.h
│   └── detector.h
│   ├── llm_generator.h
│   └── batch_preprocessor.h
│   └── http_server.h
├── src/
│   ├── main.cpp
│   ├── utils/
│   │   ├── logger.cpp
│   │   └── timer.cpp
│   ├── preprocess/
│   │   └── preprocess.cpp
│   ├── backend/
│   │   ├── onnx_backend.cpp
│   │   └── ncnn_backend.cpp
│   ├── postprocess/
│   │   └── yolo_postprocess.cpp
│   ├── detector/
│   │   └── object_detector.cpp
│   ├── llm/
│   │   └── llm_generator.cpp
│   ├── pipeline/
│   │   └── batch_preprocessor.cpp
│   ├── server/
│   │   └── http_server.cpp
├── tests/
│   ├── test_logger.cpp
│   ├── test_timer.cpp
│   ├── test_preprocess.cpp
│   ├── test_yolo_postprocess.cpp
│   ├── test_batch_preprocessor.cpp
│   └── test_object_detector.cpp
├── benchmarks/
│   ├── performance_benchmark.cpp
│   └── benchmark_utils.h
├── docs/
│   ├── AGENTS.md
│   ├── TASKS.md
│   ├── ARCHITECTURE.md
│   ├── HANDOVER.md
│   ├── API_VERIFICATION.md
│   ├── LLAMA_CPP_GUIDE.md
│   └── BATCH_PREPROCESSOR_GUIDE.md
├── models/               # 模型文件 (gitignored)
├── build.sh
├── rebuild.sh
├── .vscode/
├── CMakeLists.txt
└── README.md
```

## 关键代码文件列表

| 文件 | 模块 | 说明 |
|------|------|------|
| `include/logger.h` | M0 | Logger 类声明 |
| `include/timer.h` | M0 | Timer 类声明 |
| `include/preprocess.h` | M1 | ResizeAndNorm、MatToChw、Letterbox 函数声明，LetterboxResult 结构体 |
| `include/inference_backend.h` | M2 | InferenceBackend 纯虚基类声明 |
| `include/onnx_backend.h` | M3 | OnnxBackend 类声明（PImpl 模式） |
| `include/ncnn_backend.h` | M4 | NcnnBackend 类声明（PImpl 模式） |
| `src/utils/logger.cpp` | M0 | Logger 实现（时间戳+级别前缀） |
| `src/utils/timer.cpp` | M0 | Timer 实现（steady_clock） |
| `src/preprocess/preprocess.cpp` | M1 | ResizeAndNorm、MatToChw、Letterbox 实现 |
| `src/backend/onnx_backend.cpp` | M3 | OnnxBackend 实现（PImpl，Ort::Session 封装，CoreML EP 使用 AppendExecutionProvider 新 API） |
| `src/backend/ncnn_backend.cpp` | M4 | NcnnBackend 实现（PImpl，ncnn::Net 封装，Vulkan GPU 自动加速） |
| `include/yolo_postprocess.h` | M5 | Detection 结构体 + ProcessYoloOutput / ScaleDetectionsToOriginal 函数声明 |
| `src/postprocess/yolo_postprocess.cpp` | M5 | YOLO 后处理实现（NMS、bbox 解码、置信度过滤、坐标缩放） |
| `include/detector.h` | M6 | ObjectDetector 类声明（组合后端+预处理+后处理） |
| `src/detector/object_detector.cpp` | M6 | ObjectDetector 实现（Detect: Letterbox→MatToChw→Predict→ProcessYoloOutput→ScaleDetectionsToOriginal） |
| `include/llm_generator.h` | M7 | LlamaGenerator 类声明（PImpl 模式），GenerationConfig 结构体 |
| `src/llm/llm_generator.cpp` | M7 | LlamaGenerator 实现（PImpl，llama_model_load_from_file + llama_init_from_model + sampler chain + Metal GPU 加速 + Chat Template + KV Cache 清空） |
| `include/batch_preprocessor.h` | M8 | BatchPreprocessor 类声明（PImpl 模式），PreprocessMode 枚举，PreprocessResult/PreprocessCallback 类型 |
| `src/pipeline/batch_preprocessor.cpp` | M8 | BatchPreprocessor 实现（PImpl，线程池 + 任务队列 + condition_variable + Resize/Letterbox 双模式） |
| `include/http_server.h` | M9 | ServerConfig 结构体，RunServer 函数声明 |
| `benchmarks/benchmark_utils.h` | M11 | 基准测试辅助结构体（BenchmarkResult）+ 统计计算 + Markdown 表格 + CSV 输出 |
| `benchmarks/performance_benchmark.cpp` | M11 | 基准测试主程序（命令行参数解析 + ONNX/NCNN 多后端测试 + 预热 + 批量 + Markdown/CSV 输出） |
| `src/server/http_server.cpp` | M9 | HTTP 服务实现（cpp-httplib，/detect 支持 multipart 文件上传 + JSON image_path 路径输入，/generate 支持 Chat Template，/health，互斥锁保护检测/生成） |
| `src/main.cpp` | M0+M9 | 主入口（读取环境变量配置 → 调用 RunServer 阻塞运行） |
| `tests/test_logger.cpp` | M10 | Logger 单元测试（8 个测试用例：级别过滤、空消息、长消息） |
| `tests/test_timer.cpp` | M10 | Timer 单元测试（12 个测试用例：启停、计时精度、重置） |
| `tests/test_preprocess.cpp` | M10 | 预处理单元测试（17 个测试用例：MatToChw、ResizeAndNorm、Letterbox） |
| `tests/test_yolo_postprocess.cpp` | M10 | YOLO 后处理单元测试（25 个测试用例：NMS、坐标缩放、边界裁剪） |
| `tests/test_batch_preprocessor.cpp` | M10 | 批量预处理单元测试（10 个测试用例：双模式、WaitAll、Stop、回调） |
| `tests/test_object_detector.cpp` | M10 | 目标检测器单元测试（10 个测试用例：MockBackend、端到端流程） |

## 依赖项

| 依赖 | 版本 | 状态 |
|------|------|------|
| OpenCV | 4.13.0 (Homebrew) | ✅ 已安装并链接（core, imgproc, imgcodecs） |
| onnxruntime | 1.26.0 (Homebrew) | ✅ 已安装并链接（via CMake imported target） |
| ncnn | 20260526 (Homebrew) | ✅ 已安装并链接 |
| Llama.cpp | 9620 (Homebrew) | ✅ 已安装并链接（via CMake config，含 ggml） |
| cpp-httplib | v0.18.0 (FetchContent) | ✅ CMake 自动下载并链接 |
| Google Test | 1.17.0 (Homebrew) | ✅ 已安装并链接（via CMake config，-DGTest_ROOT=/opt/homebrew） |
| Google Benchmark | - | ❌ 待安装（M11 需要） |

## 构建命令

```bash
# 全量重建（清理 build 后重新编译运行）
./build.sh

# 增量编译（保留 build 缓存，速度更快）
./rebuild.sh

# 手动编译（注意：需在项目根目录下运行，确保 models/ 相对路径正确）
mkdir build && cd build
cmake ..
make
cd ..
./build/inference_service
```

## 未完成任务清单

- [x] M3. ONNX Runtime 后端 — 依赖 M2, ONNX Runtime ✅ 已完成
- [x] M4. NCNN 后端 — 依赖 M2, NCNN ✅ 已完成
- [x] M5. YOLO 后处理 — 依赖 M1 ✅ 已完成
- [x] M6. 目标检测器 — 依赖 M2, M5 ✅ 已完成
- [x] M7. LLM 文本生成模块 — 依赖 Llama.cpp ✅ 已完成
- [x] M8. 批量预处理流水线 — 依赖 M1 ✅ 已完成
- [x] M9. REST API 服务 — 依赖 M3/M6/M7, cpp-httplib ✅ 已完成
- [x] M10. 单元测试集 — 依赖所有模块 ✅ 已完成（82 个测试用例，6 个测试文件）
- [x] M11. 基准测试 — 依赖 M3, M4, M6 ✅ 已完成（手写循环基准测试，支持多后端/批量/Markdown+CSV 输出）

## 编码约定

- 命名空间：`inference`
- 类名：PascalCase（如 `Logger`, `Timer`, `InferenceBackend`, `OnnxBackend`）
- 函数名：PascalCase（如 `ResizeAndNorm`, `LoadModel`, `GetInputShapes`）
- 成员变量：snake_case 带下划线后缀（如 `level_`, `running_`, `start_time_`, `impl_`）
- 枚举：PascalCase（如 `LogLevel::Debug`）
- 注释：`//` 单行注释，每个函数/类/结构体必须有注释
- 代码风格：Google C++ Style Guide
- commit 信息：中文
- **跨平台规则**：代码须兼容 macOS（Apple Silicon）和 Linux；平台特有功能通过条件编译隔离；CMake 依赖查找须支持多平台路径和用户自定义路径

## 已知技术债

| 问题 | 影响 | 建议修复方式 |
|------|------|-------------|
| ResizeAndNorm 三重循环性能低 | 大尺寸图像预处理耗时较长 | 改用 OpenCV split + SIMD 或并行遍历 |
| localtime_r 跨平台不兼容 | Windows 上编译会失败 | 封装跨平台时间函数，Windows 用 localtime_s |
| Logger 未支持文件输出 | 日志只能输出到 stdout | 后续添加 FileLogger 或可配置输出目标 |
| InferenceBackend 使用 vector<vector<float>> | 批量推理时内存拷贝开销大 | 改用连续内存 + 形状信息方案 |
| Timer 不支持多次分段计时 | 只能测量单段耗时 | 扩展 LapTimer 支持多段计时 |
| OnnxBackend Predict 拷贝输入数据 | CreateTensor 要求非 const 指针，需额外拷贝 | 改用 Ort::Value::CreateTensor 分配器模式，避免用户数据拷贝 |
| NMS 算法 O(n²) 复杂度 | 检测框数量多时后处理耗时较长 | 改用 Soft-NMS（降低重叠框置信度而非直接剔除）或分区索引（按类别/空间区域分组后分别 NMS） |
| ProcessYoloOutput 硬编码 85 元素 | 仅支持 YOLOv5/v8 的 80 类输出，其他变体（如 4 类、20 类）不兼容 | 将 kBoxElementCount 和 kClassCount 改为函数参数，支持不同 YOLO 变体 |
| M10 单元测试 GTest ABI 不兼容 | anaconda3 的 GTest 1.11.0 与 arm64 编译器 ABI 不匹配，链接失败 | CMake 配置 `-DGTest_ROOT=/opt/homebrew -DGTest_DIR=/opt/homebrew/lib/cmake/GTest` 强制使用 Homebrew GTest 1.17.0 |

## 已修复问题

| 问题 | 修复方式 | 验证结果 |
|------|----------|----------|
| ONNX float16 输入类型不匹配 | 使用 YOLOv5 export.py 重新导出 float32 ONNX 模型 | ✅ bus.jpg 检测到 5 目标，zidane.jpg 检测到 4 目标 |
| LLM KV Cache 未清空导致第二次调用失败 | 在 Generate/GenerateStream 开头添加 `llama_memory_clear(llama_get_memory(ctx), true)` | ✅ 连续多次调用均正常，无 inconsistent sequence positions 错误 |
| LLM 缺少 Chat Template 导致重复生成或直接 EOS | 在 Generate/GenerateStream 中调用 `llama_chat_apply_template` 包装 prompt 为对话格式 | ✅ 模型正确回答问题（2+2=4, Paris, Shakespeare），不再重复 prompt |
| 构建脚本工作目录导致模型路径找不到 | build.sh/rebuild.sh 运行前 `cd ..` 回项目根目录 | ✅ 服务启动时 ONNX 和 LLM 模型均正常加载 |
| /detect 不支持图片路径输入 | 新增 JSON `image_path` 字段支持，优先 multipart 文件上传 | ✅ 两种输入方式均正常工作 |

## 下一步行动建议

1. **实现 M11：基准测试** — 使用 Google Benchmark 测试不同后端、批次大小、量化模式的延迟和吞吐量