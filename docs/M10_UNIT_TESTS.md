# M10 单元测试文档

创建日期：2026-06-16
测试框架：Google Test 1.17.0（Homebrew）
测试结果：6 个测试文件，82 个测试用例，全部通过，总耗时约 3.28 秒

---

## 运行测试

### 方式一：全量重建 + 测试

```bash
cd /path/to/Inference-Service
rm -rf build && mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j8
ctest --output-on-failure
```

### 方式二：增量编译 + 测试

```bash
cd /path/to/Inference-Service
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j8
ctest --output-on-failure
```

### 方式三：运行单个测试

```bash
cd build
./test_logger
./test_timer
./test_preprocess
./test_yolo_postprocess
./test_batch_preprocessor
./test_object_detector
```

### 方式四：使用 ctest 过滤

```bash
cd build
# 运行名称匹配的测试
ctest -R preprocess --output-on-failure
# 显示详细输出
ctest -V
```

### 关闭测试编译

```bash
cmake .. -DBUILD_TESTS=OFF
```

---

## 测试文件概览

| 测试文件 | 模块 | 用例数 | 测试目标 |
|----------|------|--------|----------|
| test_logger.cpp | M0 Logger | 8 | 日志级别过滤、边界输入 |
| test_timer.cpp | M0 Timer | 12 | 计时精度、状态转换、边界 |
| test_preprocess.cpp | M1 预处理 | 17 | MatToChw/ResizeAndNorm/Letterbox |
| test_yolo_postprocess.cpp | M5 YOLO 后处理 | 25 | NMS、坐标缩放、边界裁剪 |
| test_batch_preprocessor.cpp | M8 批量预处理 | 10 | 多线程预处理、回调、Stop |
| test_object_detector.cpp | M6 检测器 | 10 | MockBackend 端到端流程 |

---

## 1. test_logger.cpp — Logger 测试（8 个用例）

| 用例 | 验证内容 |
|------|----------|
| DefaultLevelIsInfo | 默认日志级别为 Info，Info/Warning/Error 正常输出 |
| SetLevelChangesFilter | SetLevel 切换级别后，低级别日志被过滤 |
| LogLevelOrdering | 枚举值递增：Debug < Info < Warning < Error |
| DebugLevelPassesAll | Debug 级别下所有日志均可输出 |
| SetLevelIdempotent | 重复设置同一级别不出错 |
| EmptyMessage | 空字符串日志不崩溃 |
| LongMessage | 10000 字符长日志不崩溃 |
| RestoreDefaultLevel | 恢复默认 Info 级别后日志正常 |

---

## 2. test_timer.cpp — Timer 测试（12 个用例）

| 用例 | 验证内容 |
|------|----------|
| InitialNotRunning | 初始状态非运行中 |
| StartSetsRunning | Start 后 IsRunning=true |
| StopClearsRunning | Stop 后 IsRunning=false |
| ElapsedMillisecondsApprox | sleep 100ms 后计时在 50-200ms 范围内 |
| ElapsedSecondsApprox | 秒级计时与毫秒级一致 |
| ElapsedWhileRunning | 运行中获取耗时（无需 Stop） |
| ResetClearsState | Reset 后状态清空 |
| MultipleStartStopCycles | 5 次 Start/Stop 循环均正常计时 |
| StopWithoutStart | 未 Start 就 Stop 不崩溃 |
| ElapsedWithoutStart | 未 Start 就获取耗时值不崩溃 |
| ResetAndRestart | Reset 后重新计时正常 |
| SecondsMillisecondsConsistency | 秒和毫秒单位一致性验证 |

---

## 3. test_preprocess.cpp — 图像预处理测试（17 个用例）

### MatToChw（5 个）

| 用例 | 验证内容 |
|------|----------|
| SolidColorNormalization | 纯蓝 BGR→RGB + 归一化到 [0,1] 正确 |
| CustomMeanAndStd | 自定义均值/标准差归一化 |
| EmptyImageReturnsEmpty | 空图像返回空向量 |
| OutputSizeCorrect | 输出尺寸 = 3×W×H |
| BgrToRgbConversion | BGR 黄色 (0,255,255) → RGB (255,255,0) |

### ResizeAndNorm（4 个）

| 用例 | 验证内容 |
|------|----------|
| OutputSizeAfterResize | 缩放后输出尺寸正确 |
| EmptyImageReturnsEmpty | 空图像返回空 |
| InvalidTargetSizeReturnsEmpty | 目标宽/高 ≤0 返回空 |
| NormalizedRange | 归一化值在 [0,1] 范围内 |

### Letterbox（8 个）

| 用例 | 验证内容 |
|------|----------|
| SquareImageScaleOne | 正方形图像 scale=1.0，无偏移 |
| AspectRatioPreserved | 800×400 → 640×640，scale=0.8，y_offset=160 |
| OutputSizeEqualsTarget | 输出尺寸始终等于目标尺寸 |
| EmptyImageReturnsEmpty | 空图像返回空 Mat |
| InvalidTargetSizeReturnsEmpty | 目标尺寸 ≤0 返回空 |
| TallImageOffset | 400×800 窄高图，x_offset=160 |
| FillColor | 自定义填充色正常 |
| OnePixelImage | 1×1 图像不崩溃 |

---

## 4. test_yolo_postprocess.cpp — YOLO 后处理测试（25 个用例）

### ProcessYoloOutput（12 个）

| 用例 | 验证内容 |
|------|----------|
| EmptyOutput | 空输出返回空 |
| InvalidOutputSize | 非 85 倍数返回空 |
| SingleHighConfidenceBox | 单框：置信度=obj_conf×class_score，坐标转换，类别 ID |
| LowConfidenceFiltered | 低置信度框被过滤 |
| LowObjConfFiltered | obj_conf 低于阈值直接跳过 |
| NegativeThresholdUsesDefault | 负数阈值使用默认值 |
| BoxClippedToImageBounds | 左上角坐标裁剪到 ≥0 |
| BoxRightBottomClipped | 右下角坐标裁剪到 ≤640 |
| NmsSuppressesOverlapping | NMS 抑制高 IoU 重叠框 |
| NonOverlappingBoxesKept | 不重叠框均保留 |
| DifferentClassesNotSuppressed | 不同类别重叠框行为 |
| ResultsSortedByConfidence | 结果按置信度降序排列 |

### ScaleDetectionsToOriginal 线性缩放（5 个）

| 用例 | 验证内容 |
|------|----------|
| LinearScaling | 640→1280 坐标按比例缩放 |
| LinearEmptyInput | 空输入返回空 |
| LinearInvalidTargetSize | 无效目标尺寸返回空 |
| LinearClippedToOriginalBounds | 缩放后坐标裁剪到原图边界 |
| LinearPreservesClassAndConfidence | 类别 ID 和置信度不变 |

### ScaleDetectionsToOriginal Letterbox（8 个）

| 用例 | 验证内容 |
|------|----------|
| LetterboxScaling | 减偏移 ÷scale 正确映射 |
| LetterboxEmptyInput | 空输入返回空 |
| LetterboxInvalidScale | scale ≤0 返回空 |
| LetterboxInvalidOffset | 负偏移返回空 |
| LetterboxInvalidSrcDimensions | 原图宽/高 ≤0 返回空 |
| LetterboxClippedToOriginalBounds | 映射后坐标裁剪到原图边界 |
| LetterboxPreservesClassAndConfidence | 类别和置信度不变 |
| BoxInPaddingRegionClipped | 填充区域检测框坐标裁剪到 0 |

---

## 5. test_batch_preprocessor.cpp — 批量预处理测试（10 个用例）

| 用例 | 验证内容 |
|------|----------|
| ResizeModeSingleImage | Resize 模式：tensor 大小正确，Letterbox 元数据为 0 |
| ResizeModeMultipleImages | Resize 模式：10 张图并行处理全部完成 |
| LetterboxModeReturnsMetadata | Letterbox 模式：scale=0.8, y_offset=160 |
| WaitAllBlocksUntilComplete | WaitAll 阻塞直到 5 个任务完成 |
| SubmitAfterStopReturnsFalse | Stop 后 Submit 返回 false |
| MultipleStopCalls | 多次 Stop 不崩溃 |
| EmptyImageCallbackStillCalled | 空图像回调仍触发，tensor 为空 |
| SingleWorker | 单线程模式正常工作 |
| CallbackIndexCorrect | 回调 index 正确传递（多线程排序后验证） |
| ZeroWorkersAdjustedToOne | 0 线程自动调整为 1 |

---

## 6. test_object_detector.cpp — ObjectDetector 测试（10 个用例）

使用 **MockBackend** 模拟推理后端，无需真实模型。

| 用例 | 验证内容 |
|------|----------|
| EmptyImageReturnsEmpty | 空图像返回空结果 |
| DetectWithMockOutput | Mock 高置信度框：检测到 ≥1 目标 |
| LowConfidenceOutputReturnsEmpty | Mock 低置信度框：结果为空 |
| EmptyInferenceOutputReturnsEmpty | Mock 空输出：结果为空 |
| SetInputSize | 修改输入尺寸不崩溃 |
| SetNormalizeParams | 修改归一化参数不崩溃 |
| DetectMultipleBoxes | Mock 两个框：检测到 ≥2 目标 |
| NonSquareImage | 1920×1080 非正方形图像正常 |
| SmallImage | 32×32 小图像正常 |
| CustomThresholds | 低阈值检测到、高阈值检测不到 |

---

## 未覆盖的模块

| 模块 | 未测试原因 | 建议测试方式 |
|------|-----------|-------------|
| OnnxBackend（M3） | 需要真实 ONNX 模型 + GPU | 集成测试，启动服务后 curl 验证 |
| NcnnBackend（M4） | 需要真实 NCNN 模型 + GPU | 集成测试，启动服务后 curl 验证 |
| LlamaGenerator（M7） | 需要真实 GGUF 模型（638MB） | 集成测试，启动服务后 curl 验证 |
| HTTP Server（M9） | 需要启动网络服务 | 集成测试，参考 `docs/API_VERIFICATION.md` |

---

## MockBackend 设计

`test_object_detector.cpp` 中的 MockBackend 继承 `InferenceBackend` 纯虚基类，通过预设 `mock_output_` 字段模拟推理结果：

```cpp
class MockBackend : public InferenceBackend {
public:
    std::vector<std::vector<float>> mock_output_;

    bool LoadModel(const std::string& model_path) override { return true; }
    std::vector<std::vector<float>> Predict(...) override { return mock_output_; }
    // ... 其他接口返回预设形状和名称
};
```

优势：
- 无需真实模型文件和 GPU 资源
- 可精确控制推理输出，验证检测管线各环节
- 测试运行快速（毫秒级）

---

## GTest 配置注意事项

macOS 上 CMake 可能优先找到 anaconda3 的 GTest 1.11.0（动态库），与 Apple Clang arm64 存在 ABI 不兼容导致链接失败。项目已在 `CMakeLists.txt` 中自动设置 `GTest_ROOT=/opt/homebrew`，优先使用 Homebrew GTest 1.17.0（静态库）。

如需手动指定：

```bash
cmake .. -DGTest_ROOT=/opt/homebrew -DGTest_DIR=/opt/homebrew/lib/cmake/GTest
```
