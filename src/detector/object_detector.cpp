#include "detector.h"
#include "logger.h"
#include "preprocess.h"
#include "timer.h"
#include "yolo_postprocess.h"

namespace inference {

// 构造函数，接收推理后端的所有权
// backend: 已加载模型的推理后端，通过 InferenceBackend 基类实现多态
ObjectDetector::ObjectDetector(std::unique_ptr<InferenceBackend> backend)
    : backend_(std::move(backend)) {}

// 对单张图像执行目标检测
// 完整流程：Letterbox → MatToChw → Predict → ProcessYoloOutput → ScaleDetectionsToOriginal
// 每一步失败均返回空向量并记录错误日志
std::vector<Detection> ObjectDetector::Detect(const cv::Mat& img,
                                               float confidence_threshold,
                                               float nms_threshold) {
    if (img.empty()) {
        Logger::Error("ObjectDetector::Detect: input image is empty");
        return {};
    }

    if (!backend_) {
        Logger::Error("ObjectDetector::Detect: backend is null");
        return {};
    }

    Timer timer;
    timer.Start();

    // 步骤1：Letterbox 预处理，保持宽高比缩放并填充灰色
    // LetterboxResult 包含预处理图像和坐标映射参数（scale, x_offset, y_offset）
    LetterboxResult lb = Letterbox(img, input_width_, input_height_);
    if (lb.image.empty()) {
        Logger::Error("ObjectDetector::Detect: Letterbox preprocessing failed");
        return {};
    }

    // 步骤2：将 Letterbox 输出转为 CHW float 数组（BGR→RGB + 减均值除方差归一化）
    // 使用 mean_ 和 std_ 参数，默认 {0,0,0}/{255,255,255} 匹配 YOLOv5/v8 标准预处理
    std::vector<float> chw_data = MatToChw(lb.image, mean_, std_);
    if (chw_data.empty()) {
        Logger::Error("ObjectDetector::Detect: MatToChw conversion failed");
        return {};
    }

    // 步骤3：包装为 batch=1 的输入张量，调用后端推理
    std::vector<std::vector<float>> inputs = {std::move(chw_data)};
    std::vector<std::vector<float>> outputs = backend_->Predict(inputs);
    if (outputs.empty() || outputs[0].empty()) {
        Logger::Error("ObjectDetector::Detect: inference returned empty output");
        return {};
    }

    // 步骤4：YOLO 后处理，解析输出为检测框并执行 NMS
    // 检测框坐标基于模型输入尺寸（input_width_ × input_height_）
    std::vector<Detection> detections = ProcessYoloOutput(
        outputs[0], confidence_threshold, nms_threshold,
        input_width_, input_height_);

    // 步骤5：使用 Letterbox 重载将坐标从模型输入尺寸映射回原图
    // 先减偏移量，再除缩放比，正确处理 Letterbox 填充区域
    std::vector<Detection> original_dets = ScaleDetectionsToOriginal(
        detections, lb.x_offset, lb.y_offset, lb.scale,
        img.cols, img.rows);

    timer.Stop();
    Logger::Info("ObjectDetector::Detect: detected " +
                 std::to_string(original_dets.size()) +
                 " objects in " +
                 std::to_string(timer.ElapsedMilliseconds()) + " ms");

    return original_dets;
}

// 设置模型输入尺寸，默认 640×640（与 YOLOv5/v8 训练时一致）
void ObjectDetector::SetInputSize(int width, int height) {
    input_width_ = width;
    input_height_ = height;
}

// 设置归一化参数（减均值除方差）
// mean: 各通道均值，默认 {0,0,0}
// std: 各通道标准差，默认 {255,255,255}（即除以 255 将像素归一化到 [0,1]）
void ObjectDetector::SetNormalizeParams(const std::vector<float>& mean,
                                         const std::vector<float>& std) {
    mean_ = mean;
    std_ = std;
}

}  // namespace inference
