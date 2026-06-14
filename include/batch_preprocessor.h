#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace inference {

// 预处理模式
enum class PreprocessMode {
    // 直接缩放到目标尺寸（拉伸，不保持宽高比）
    Resize,
    // 保持宽高比缩放，不足部分填充灰色（Letterbox）
    Letterbox
};

// 预处理结果：包含浮点张量和 Letterbox 元数据
struct PreprocessResult {
    // CHW 顺序的浮点张量（已归一化），失败时为空
    std::vector<float> tensor;
    // Letterbox 缩放比例（Resize 模式下为 0.0f）
    float scale = 0.0f;
    // Letterbox x 偏移量（Resize 模式下为 0）
    int x_offset = 0;
    // Letterbox y 偏移量（Resize 模式下为 0）
    int y_offset = 0;
};

// 预处理回调：参数为原始图像索引和预处理结果
// 当某张图像预处理完成时调用，若失败则张量为空
using PreprocessCallback = std::function<void(size_t index, const PreprocessResult& result)>;

// 批量异步预处理器
// 内部维护一个线程池和任务队列，支持多张图像并行预处理
class BatchPreprocessor {
public:
    // 构造：指定最大并行任务数（线程池大小）
    explicit BatchPreprocessor(size_t num_workers = 4);

    // 析构函数，停止线程池并释放资源
    ~BatchPreprocessor();

    // 禁止拷贝和赋值
    BatchPreprocessor(const BatchPreprocessor&) = delete;
    BatchPreprocessor& operator=(const BatchPreprocessor&) = delete;

    // 提交一张图像（通过 cv::Mat）进行预处理
    // index: 用户自定义标识（用于回调中识别）
    // img: 原始 BGR 图像
    // 返回: 是否成功提交到队列
    bool Submit(size_t index, const cv::Mat& img);

    // 提交一张图像（通过文件路径）
    // index: 用户自定义标识
    // image_path: 图像文件路径
    // 返回: 是否成功提交到队列
    bool Submit(size_t index, const std::string& image_path);

    // 等待所有已提交的任务完成（阻塞）
    void WaitAll();

    // 设置预处理参数（目标尺寸、模式、均值、标准差）
    // mode: PreprocessMode::Resize（直接缩放）或 PreprocessMode::Letterbox（保持宽高比）
    void SetPreprocessParams(int target_w, int target_h,
                             PreprocessMode mode = PreprocessMode::Resize,
                             const std::vector<float>& mean = {0.0f, 0.0f, 0.0f},
                             const std::vector<float>& std = {255.0f, 255.0f, 255.0f});

    // 设置完成回调（每完成一张图像的处理即调用）
    void SetCallback(PreprocessCallback callback);

    // 停止所有工作线程并清空队列（不再接受新任务）
    void Stop();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

}  // namespace inference
