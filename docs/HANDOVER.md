# 项目交接文档 (HANDOVER.md)

创建日期：2026-05-30
最后更新：2026-06-07

## 项目整体状态

- 已完成模块：M0（项目骨架）、M1（图像预处理）、M2（推理后端抽象接口）、M3（ONNX Runtime 后端）、M4（NCNN 后端）、M5（YOLO 后处理）
- 代码量估算：约 1500 行（不含注释和文档）
- 当前可编译运行，输出 Logger、Timer、OnnxBackend 和 NcnnBackend 创建演示日志

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
│   └── yolo_postprocess.h
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
│   ├── detector/         （待开发）
│   ├── llm/              （待开发）
│   ├── pipeline/         （待开发）
│   └── server/           （待开发）
├── docs/
│   ├── AGENTS.md
│   ├── TASKS.md
│   ├── ARCHITECTURE.md
│   └── HANDOVER.md
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
| `include/preprocess.h` | M1 | ResizeAndNorm、Letterbox 函数声明 |
| `include/inference_backend.h` | M2 | InferenceBackend 纯虚基类声明 |
| `include/onnx_backend.h` | M3 | OnnxBackend 类声明（PImpl 模式） |
| `include/ncnn_backend.h` | M4 | NcnnBackend 类声明（PImpl 模式） |
| `src/utils/logger.cpp` | M0 | Logger 实现（时间戳+级别前缀） |
| `src/utils/timer.cpp` | M0 | Timer 实现（steady_clock） |
| `src/preprocess/preprocess.cpp` | M1 | ResizeAndNorm、Letterbox 实现 |
| `src/backend/onnx_backend.cpp` | M3 | OnnxBackend 实现（PImpl，Ort::Session 封装，CoreML EP 使用 AppendExecutionProvider 新 API） |
| `src/backend/ncnn_backend.cpp` | M4 | NcnnBackend 实现（PImpl，ncnn::Net 封装，Vulkan GPU 自动加速） |
| `include/yolo_postprocess.h` | M5 | Detection 结构体 + ProcessYoloOutput / ScaleDetectionsToOriginal 函数声明 |
| `src/postprocess/yolo_postprocess.cpp` | M5 | YOLO 后处理实现（NMS、bbox 解码、置信度过滤、坐标缩放） |
| `src/main.cpp` | M0+M3+M4 | 主入口（Logger+Timer+OnnxBackend+NcnnBackend 演示） |

## 依赖项

| 依赖 | 版本 | 状态 |
|------|------|------|
| OpenCV | 4.13.0 (Homebrew) | ✅ 已安装并链接（core, imgproc, imgcodecs） |
| onnxruntime | 1.26.0 (Homebrew) | ✅ 已安装并链接（via CMake imported target） |
| ncnn | 20260526 (Homebrew) | ✅ 已安装并链接 |
| Llama.cpp | - | ❌ 待安装（M7 需要） |
| cpp-httplib | - | ❌ 待安装（M9 需要） |
| Google Test | - | ❌ 待安装（M10 需要） |
| Google Benchmark | - | ❌ 待安装（M11 需要） |

## 构建命令

```bash
# 全量重建（清理 build 后重新编译运行）
./build.sh

# 增量编译（保留 build 缓存，速度更快）
./rebuild.sh

# 手动编译
mkdir build && cd build
cmake ..
make
./inference_service
```

## 未完成任务清单

- [x] M3. ONNX Runtime 后端 — 依赖 M2, ONNX Runtime ✅ 已完成
- [x] M4. NCNN 后端 — 依赖 M2, NCNN ✅ 已完成
- [x] M5. YOLO 后处理 — 依赖 M1 ✅ 已完成
- [ ] M6. 目标检测器 — 依赖 M2, M5
- [ ] M7. LLM 文本生成模块 — 依赖 Llama.cpp
- [ ] M8. 批量预处理流水线 — 依赖 M1
- [ ] M9. REST API 服务 — 依赖 M3/M6/M7, cpp-httplib
- [ ] M10. 单元测试集 — 依赖所有模块
- [ ] M11. 基准测试 — 依赖 M3, M4, M6

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

## 下一步行动建议

1. **实现 M6：目标检测器** — 组合后端 + M5 后处理，提供 Detect(cv::Mat) 接口
2. 更新 TASKS.md 和 HANDOVER.md