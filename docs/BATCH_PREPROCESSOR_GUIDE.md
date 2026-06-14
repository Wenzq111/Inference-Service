# M8 批量预处理流水线 — 模块介绍文档

创建日期：2026-06-14

## 概述

M8 实现了异步批量图像预处理能力，通过**线程池 + 生产者-消费者队列**模型，让多张图像的预处理在多个工作线程中并行执行，提高整体吞吐量。M8 支持 **Resize**（直接缩放）和 **Letterbox**（保持宽高比）两种预处理模式，依赖 M1 的预处理函数，不依赖推理后端。

### 为什么需要 M8

在单线程模式下，预处理 10 张 640×640 图像需要串行执行 10 次 ResizeAndNorm：

```
主线程: [图1预处理] → [图2预处理] → ... → [图10预处理]  总耗时 ≈ 10 × 单张耗时
```

M8 将这些任务分发到 4 个工作线程并行执行：

```
线程1: [图1] → [图5] → [图9]
线程2: [图2] → [图6] → [图10]
线程3: [图3] → [图7]
线程4: [图4] → [图8]
总耗时 ≈ 3 × 单张耗时（约 3 倍加速）
```

---

## 类与类型定义

### PreprocessMode

```cpp
enum class PreprocessMode {
    Resize,      // 直接缩放到目标尺寸（拉伸，不保持宽高比）
    Letterbox    // 保持宽高比缩放，不足部分填充灰色
};
```

预处理模式枚举，决定工作线程中调用哪种 M1 预处理函数：

| 模式 | 内部调用 | 适用场景 |
|---|---|---|
| `Resize` | `ResizeAndNorm` | 图像分类等对宽高比不敏感的任务 |
| `Letterbox` | `Letterbox` + `MatToChw` | YOLO 目标检测等需要保持宽高比的任务 |

### PreprocessResult

```cpp
struct PreprocessResult {
    std::vector<float> tensor;  // CHW 顺序的浮点张量（已归一化），失败时为空
    float scale = 0.0f;         // Letterbox 缩放比例（Resize 模式下为 0.0f）
    int x_offset = 0;           // Letterbox x 偏移量（Resize 模式下为 0）
    int y_offset = 0;           // Letterbox y 偏移量（Resize 模式下为 0）
};
```

预处理结果结构体，包含浮点张量和 Letterbox 元数据：

- `tensor`：CHW 顺序、已归一化的浮点张量，与 M1 的 `ResizeAndNorm` / `MatToChw` 返回值类型一致。预处理失败时为空向量
- `scale`、`x_offset`、`y_offset`：Letterbox 模式下的坐标映射参数，用于将检测框从模型输入尺寸映射回原图（与 M1 的 `LetterboxResult` 对应）。Resize 模式下这些字段为默认值 0

### PreprocessCallback

```cpp
using PreprocessCallback = std::function<void(size_t index, const PreprocessResult& result)>;
```

完成回调类型。每个图像预处理完成后被调用：
- `index`：用户提交时指定的标识，用于区分不同图像
- `result`：预处理结果，包含 tensor 和 Letterbox 元数据；失败时 `result.tensor` 为空

### BatchPreprocessor

主类，使用 PImpl 模式隐藏线程池和队列细节。头文件无需引入 `<thread>`、`<mutex>` 等。

---

## 公共方法详解

### 构造函数

```cpp
explicit BatchPreprocessor(size_t num_workers = 4);
```

创建线程池并启动 `num_workers` 个工作线程。默认 4 个工作线程。每个线程立即进入等待状态，等待任务入队。

**内部流程**：
1. 初始化参数默认值（640×640，mean={0,0,0}，std={1,1,1}）
2. 启动 N 个工作线程，每个线程执行 `WorkerLoop()`
3. 工作线程阻塞在 `condition_variable::wait()`，等待队列有任务

### 析构函数

```cpp
~BatchPreprocessor();
```

自动调用 `Stop()`，通知所有工作线程退出并 join。

### Submit（cv::Mat 版本）

```cpp
bool Submit(size_t index, const cv::Mat& img);
```

提交一张内存中的图像进行异步预处理。

- `index`：用户自定义标识，回调时原样返回
- `img`：BGR 格式的 cv::Mat（浅拷贝，引用计数安全）
- 返回 `true`：成功入队；`false`：已调用 Stop()，拒绝提交

**与 M1 的关系**：入队时不执行预处理，仅保存图像引用。预处理在工作线程中根据 `PreprocessMode` 调用 M1 的 `ResizeAndNorm` 或 `Letterbox+MatToChw`。

### Submit（文件路径版本）

```cpp
bool Submit(size_t index, const std::string& image_path);
```

提交一张图像文件路径进行异步预处理。工作线程中通过 `cv::imread` 读取，再根据 `PreprocessMode` 调用对应的预处理函数。

- 读取失败时回调中 `result.tensor` 为空，并记录 `Logger::Error`

### WaitAll

```cpp
void WaitAll();
```

阻塞当前线程，直到所有已提交任务完成（`pending_count_ == 0`）。

**内部机制**：通过 `condition_variable done_cv_` 等待，每完成一个任务触发 `notify_all()`。

**典型用法**：
```cpp
bp.Submit(0, img0);
bp.Submit(1, img1);
bp.Submit(2, img2);
bp.WaitAll();  // 阻塞直到 3 张图都处理完
```

### SetPreprocessParams

```cpp
void SetPreprocessParams(int target_w, int target_h,
                         PreprocessMode mode = PreprocessMode::Resize,
                         const std::vector<float>& mean = {0.0f, 0.0f, 0.0f},
                         const std::vector<float>& std = {1.0f, 1.0f, 1.0f});
```

设置预处理参数：

| 参数 | 作用 | 对应 M1 调用 |
|---|---|---|
| `target_w` | 目标宽度 | `ResizeAndNorm(img, target_w, ...)` 或 `Letterbox(img, target_w, ...)` |
| `target_h` | 目标高度 | `ResizeAndNorm(img, ..., target_h, ...)` 或 `Letterbox(img, ..., target_h)` |
| `mode` | 预处理模式 | `Resize` → `ResizeAndNorm`；`Letterbox` → `Letterbox` + `MatToChw` |
| `mean` | 各通道均值 | `ResizeAndNorm(img, ..., mean, ...)` 或 `MatToChw(lb.image, mean, ...)` |
| `std` | 各通道标准差 | `ResizeAndNorm(img, ..., ..., std)` 或 `MatToChw(lb.image, ..., std)` |

**线程安全**：通过 `params_mutex_` 保护，防止设置参数与工作线程读取竞争。

### SetCallback

```cpp
void SetCallback(PreprocessCallback callback);
```

设置完成回调。每个图像预处理完成后在工作线程中调用此回调。

**注意**：回调在**工作线程**中执行，如果回调中访问共享状态，用户需自行加锁。

### Stop

```cpp
void Stop();
```

停止线程池：
1. 设置 `stopped_ = true`
2. 唤醒所有等待中的工作线程
3. Join 所有工作线程
4. 清空线程列表

**调用后**：`Submit()` 返回 false，已入队但未执行的任务会被丢弃。如需等所有任务完成再停止，应先 `WaitAll()` 再 `Stop()`。

---

## 内部实现

### PImpl 结构

```
BatchPreprocessor
    └── Impl
        ├── target_w_, target_h_, mode_, mean_, std_  ← 预处理参数
        ├── params_mutex_                       ← 参数读写锁
        ├── callback_                           ← 完成回调
        ├── callback_mutex_                     ← 回调读写锁
        ├── queue_<Task>                        ← 任务队列
        ├── queue_mutex_                        ← 队列锁
        ├── queue_cv_                           ← 新任务通知
        ├── done_cv_                            ← 完成通知
        ├── workers_<thread>                    ← 工作线程池
        ├── stopped_                            ← 停止标志（atomic）
        └── pending_count_                      ← 待处理计数（atomic）
```

### WorkerLoop 工作流程

```
while (true) {
    1. 加锁等待 queue_cv_
       条件: 队列非空 || stopped
    2. stopped 且队列为空 → 退出线程
    3. 取出一个 Task，解锁
    4. 读取预处理参数（加 params_mutex_），含 mode
    5. 若 use_path → cv::imread 读取图像
    6. 根据 mode 执行预处理:
       - Resize 模式:  ResizeAndNorm(img, tw, th, mean, std)
       - Letterbox 模式: Letterbox(img, tw, th) → MatToChw(lb.image, mean, std)
                         填充 result.scale/x_offset/y_offset
    7. 调用 callback_(index, result)
    8. pending_count_-- → done_cv_.notify_all()
}
```

### 线程安全设计

| 共享资源 | 保护方式 | 读方 | 写方 |
|---|---|---|---|
| 任务队列 | `queue_mutex_` | Worker 取任务 | Submit 入队 |
| 预处理参数 | `params_mutex_` | Worker 读参数 | SetPreprocessParams |
| 回调函数 | `callback_mutex_` | Worker 调用 | SetCallback |
| pending_count_ | `std::atomic` | WaitAll 检查 | Worker 递减 |
| stopped_ | `std::atomic` | Worker/Submit 检查 | Stop 设置 |

---

## 与其他模块的关系

```
M0 Logger ───────────→ M8 日志输出
M0 Timer ────────────→ (可选，M8 演示中可统计耗时)
M1 ResizeAndNorm ────→ M8 Resize 模式的核心预处理逻辑（Worker 中调用）
M1 Letterbox ────────→ M8 Letterbox 模式中保持宽高比缩放
M1 MatToChw ─────────→ M8 Letterbox 模式中 BGR→RGB + 归一化 + CHW 转换（也可被 ResizeAndNorm 间接调用）

M8 不依赖:
  M2 InferenceBackend  ← M8 只做预处理，不做推理
  M3 OnnxBackend
  M4 NcnnBackend
  M5 YOLO 后处理
  M6 ObjectDetector
  M7 LlamaGenerator
```

**M8 在整体流水线中的位置**：

```
原始图像 → [M8 BatchPreprocessor] → CHW 张量 → [M2/M3/M4 推理后端] → 原始输出 → [M5 后处理] → 最终结果
              ↑ 并行预处理                    ↑ 串行/批量推理              ↑ NMS/解码
```

M8 的输出（`PreprocessResult.tensor`）可直接作为 `InferenceBackend::Predict()` 的输入；Letterbox 模式下的 `scale/x_offset/y_offset` 可传给 M5 的 `ScaleDetectionsToOriginal`，实现**预处理-推理-后处理**的完整流水线。

---

## 使用示例

### Resize 模式（图像分类等场景）

```cpp
inference::BatchPreprocessor bp(4);

// 设置 Resize 模式预处理参数
bp.SetPreprocessParams(640, 640, inference::PreprocessMode::Resize,
                       {0.0f, 0.0f, 0.0f}, {255.0f, 255.0f, 255.0f});

bp.SetCallback([](size_t idx, const inference::PreprocessResult& result) {
    if (result.tensor.empty()) {
        // 预处理失败
    } else {
        // result.tensor 大小 = 3 * 640 * 640 = 1228800
        // result.scale = 0.0f, x_offset = 0, y_offset = 0（Resize 模式无坐标映射）
        // 可送入 InferenceBackend::Predict()
    }
});

bp.Submit(0, "images/cat.jpg");
bp.Submit(1, "images/dog.jpg");
bp.WaitAll();
```

### Letterbox 模式（YOLO 目标检测等场景）

```cpp
inference::BatchPreprocessor bp(4);

// 设置 Letterbox 模式预处理参数
bp.SetPreprocessParams(640, 640, inference::PreprocessMode::Letterbox,
                       {0.0f, 0.0f, 0.0f}, {255.0f, 255.0f, 255.0f});

bp.SetCallback([](size_t idx, const inference::PreprocessResult& result) {
    if (result.tensor.empty()) {
        // 预处理失败
    } else {
        // result.tensor 大小 = 3 * 640 * 640 = 1228800
        // result.scale / x_offset / y_offset 可传给 ScaleDetectionsToOriginal
        // 将检测框坐标从 640×640 映射回原图
    }
});

bp.Submit(0, "images/cat.jpg");
bp.Submit(1, "images/dog.jpg");
cv::Mat img = cv::imread("images/bird.jpg");
bp.Submit(2, img);
bp.WaitAll();
```
