#pragma once

#include "inference_backend.h"
#include "yolo_postprocess.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

namespace inference {

// 目标检测器，组合预处理、推理后端和后处理为统一的检测接口
// 对外提供 Detect(cv::Mat) 方法，输入原始图像，输出基于原图坐标的检测框
// 通过 InferenceBackend 基类支持 ONNX Runtime 和 NCNN 两种后端
class ObjectDetector {
public:
    // 构造函数，接收推理后端的所有权
    // backend: 已加载模型的推理后端（OnnxBackend 或 NcnnBackend）
    explicit ObjectDetector(std::unique_ptr<InferenceBackend> backend);

    // 对单张图像执行目标检测
    // 流程：Letterbox 预处理 → MatToChw → 后端推理 → ProcessYoloOutput → 坐标映射回原图
    // img: 原始 BGR 图像
    // confidence_threshold: 置信度阈值，低于此值的框直接过滤（默认 0.25）
    // nms_threshold: NMS 的 IoU 阈值（默认 0.45）
    // 返回: 检测框列表，坐标基于原图尺寸
    std::vector<Detection> Detect(const cv::Mat& img,
                                   float confidence_threshold = 0.25f,
                                   float nms_threshold = 0.45f);

    // 设置模型输入尺寸，默认 640×640（与 YOLOv5/v8 训练时一致）
    void SetInputSize(int width, int height);

    // 设置归一化参数（减均值除方差）
    // mean: 各通道均值，默认 {0,0,0}（YOLOv5/v8 标准）
    // std: 各通道标准差，默认 {255,255,255}（即除以 255 将像素归一化到 [0,1]）
    void SetNormalizeParams(const std::vector<float>& mean = {0.0f, 0.0f, 0.0f},
                            const std::vector<float>& std = {255.0f, 255.0f, 255.0f});

private:
    // 推理后端，拥有所有权
    std::unique_ptr<InferenceBackend> backend_;
    // 模型输入宽度
    int input_width_ = 640;
    // 模型输入高度
    int input_height_ = 640;
    // 归一化均值，默认 {0,0,0}
    std::vector<float> mean_ = {0.0f, 0.0f, 0.0f};
    // 归一化标准差，默认 {255,255,255}（YOLOv5/v8 标准预处理）
    std::vector<float> std_ = {255.0f, 255.0f, 255.0f};
};

}  // namespace inference
