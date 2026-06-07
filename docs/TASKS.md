# 开发任务清单

完成任务后，请将 `- [ ]` 改为 `- [x]` 以标记已完成。

> 代码行数仅为大致参考，可根据实际需求适当调整，以优先完成项目功能和可扩展性为第一优先级。

- [x] M0. 项目骨架 — 200行 — 无依赖 — 创建 CMakeLists.txt、目录结构、Logger 工具类、Timer 性能测量类
- [x] M1. 图像预处理库 — 600行 — 依赖 OpenCV — 实现图像读取、resize、归一化、letterbox、BGR/RGB 转换，输出 CHW 顺序的 float 数组
- [x] M2. 推理后端抽象接口 — 150行 — 无依赖 — 定义纯虚基类 InferenceBackend，包含 LoadModel、Predict、GetInputShapes 方法
- [x] M3. ONNX Runtime 后端 — 500行 — 依赖 M2, ONNX Runtime — 实现 InferenceBackend 接口，封装 ONNX Runtime 推理，支持动态输入形状和 int8 量化选项
- [ ] M4. NCNN 后端 — 500行 — 依赖 M2, NCNN — 实现 InferenceBackend 接口，封装 NCNN 推理，开启 fp16 和 Winograd 优化
- [ ] M5. YOLO 后处理 — 400行 — 依赖 M1 — 实现 NMS（非极大值抑制）、边界框解码、置信度过滤，输出 Detection 结构体
- [ ] M6. 目标检测器 — 300行 — 依赖 M2, M5 — 组合后端和后处理，提供 Detect(cv::Mat) 接口，内部自动调用预处理和后处理
- [ ] M7. LLM 文本生成模块 — 600行 — 依赖 Llama.cpp — 封装 Llama.cpp，提供 Load 和 Generate（支持流式回调）接口，与检测模块独立
- [ ] M8. 批量预处理流水线 — 500行 — 依赖 M1 — 实现生产者-消费者队列，多线程并行预处理图像，支持批次等待和超时
- [ ] M9. REST API 服务 — 500行 — 依赖 M3/M6/M7, cpp-httplib — 实现 HTTP 服务，提供 /classify、/detect、/health 端点，支持并发请求
- [ ] M10. 单元测试集 — 1500行 — 依赖所有模块 — 使用 Google Test 为每个模块编写单元测试，覆盖主要功能和边界条件
- [ ] M11. 基准测试 — 800行 — 依赖 M3, M4, M6 — 使用 Google Benchmark 测试不同后端、批次大小、量化模式的延迟和吞吐量

## 模块进度

| 模块 | 状态 | 关键文件 |
|------|------|----------|
| M0 项目骨架 | ✅ 完成 | logger.h/cpp, timer.h/cpp |
| M1 图像预处理 | ✅ 完成 | preprocess.h/cpp |
| M2 推理后端抽象接口 | ✅ 完成 | inference_backend.h |
| M3 ONNX Runtime 后端 | ✅ 完成 | onnx_backend.h, onnx_backend.cpp |
| M4 NCNN 后端 | 待开发 | - |
| M5 YOLO 后处理 | 待开发 | - |
| M6 目标检测器 | 待开发 | - |
| M7 LLM 文本生成模块 | 待开发 | - |
| M8 批量预处理流水线 | 待开发 | - |
| M9 REST API 服务 | 待开发 | - |
| M10 单元测试集 | 待开发 | - |
| M11 基准测试 | 待开发 | - |