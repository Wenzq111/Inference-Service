#pragma once

#include <cstdint>
#include <vector>

namespace inference {

// 目标检测结果结构体，存储单个检测框的信息
// x1,y1: 左上角绝对坐标；x2,y2: 右下角绝对坐标
// 坐标基于模型输入尺寸，需通过 ScaleDetectionsToOriginal 缩放回原图
struct Detection {
    // 检测框左上角 x 坐标（绝对坐标）
    float x1;
    // 检测框左上角 y 坐标（绝对坐标）
    float y1;
    // 检测框右下角 x 坐标（绝对坐标）
    float x2;
    // 检测框右下角 y 坐标（绝对坐标）
    float y2;
    // 检测到的目标类别 ID（0 起始）
    int class_id;
    // 检测置信度（obj_conf * max_class_score）
    float confidence;
};

// 对单张图像的原始模型输出进行后处理，返回经过 NMS 过滤后的检测框列表
// output: 模型原始输出，形状为 [num_boxes, 85] 的扁平向量（按行排列）
//         85 = cx, cy, w, h, obj_conf, class_scores[0..79]
// confidence_threshold: 置信度阈值，低于此值的框直接过滤；负数时使用默认值 0.25
// nms_threshold: NMS 的 IoU 阈值；负数时使用默认值 0.45
// image_width, image_height: 模型输入图像尺寸，用于中心宽高→绝对坐标转换和边界裁剪
// 返回: 经过 NMS 过滤后的 Detection 列表，坐标基于模型输入尺寸
std::vector<Detection> ProcessYoloOutput(const std::vector<float>& output,
                                         float confidence_threshold,
                                         float nms_threshold,
                                         int image_width,
                                         int image_height);

// 将 Detection 坐标从模型输入尺寸缩放到原始图像尺寸（简单线性缩放）
// 适用于 ResizeAndNorm 等无偏移预处理方式
// detections: 基于模型输入尺寸的检测结果
// src_width, src_height: 原始图像尺寸
// target_width, target_height: 模型输入尺寸（与 ProcessYoloOutput 的 image_width/height 一致）
// 返回: 缩放后的 Detection 列表，坐标基于原始图像尺寸，不修改原 detections
std::vector<Detection> ScaleDetectionsToOriginal(
    const std::vector<Detection>& detections,
    int src_width, int src_height,
    int target_width, int target_height);

// 将 Detection 坐标从模型输入尺寸缩放到原始图像尺寸（Letterbox 模式）
// 适用于 M1 Letterbox 预处理产生的带偏移和灰色填充的图像
// 先减去 Letterbox 居中偏移量，再除以缩放比例，最后裁剪到原图边界
// detections: 基于模型输入尺寸的检测结果
// x_offset, y_offset: Letterbox 预处理时图像居中放置的偏移量（像素）
// scale: Letterbox 预处理时的缩放比例
// src_width, src_height: 原始图像尺寸
// 返回: 缩放后的 Detection 列表，坐标基于原始图像尺寸，不修改原 detections
std::vector<Detection> ScaleDetectionsToOriginal(
    const std::vector<Detection>& detections,
    int x_offset, int y_offset,
    float scale,
    int src_width, int src_height);

}  // namespace inference
