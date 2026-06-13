#include "yolo_postprocess.h"
#include "logger.h"

#include <algorithm>

namespace inference {

// YOLO 每个检测框输出的元素数：cx, cy, w, h, obj_conf, class_scores[0..79]
static constexpr int kBoxElementCount = 85;

// obj_conf 在 box 输出中的索引
static constexpr int kObjConfIndex = 4;

// class_scores 在 box 输出中的起始索引
static constexpr int kClassScoreStart = 5;

// 类别数量 = 总元素数 - 坐标数(cx,cy,w,h) - obj_conf
static constexpr int kClassCount = kBoxElementCount - kClassScoreStart;

// 默认置信度阈值
static constexpr float kDefaultConfidenceThreshold = 0.25f;

// 默认 NMS IoU 阈值
static constexpr float kDefaultNmsThreshold = 0.45f;

// 计算两个检测框的交并比（Intersection over Union）
// a, b: 待计算的两个 Detection
// 返回: IoU 值，范围 [0.0, 1.0]；无交集时返回 0.0
static float ComputeIoU(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_width = std::max(0.0f, inter_x2 - inter_x1);
    float inter_height = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_width * inter_height;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

// 非极大值抑制（NMS），过滤重叠度过高的检测框
// 算法复杂度 O(n²)，当前阶段可接受
// 后续优化方向：Soft-NMS（降低而非直接剔除重叠框的置信度）、
// 分区索引（按类别或空间区域分组后再做 NMS，减少比较次数）
// detections: 已按 confidence 降序排序的检测框列表
// nms_threshold: IoU 阈值，超过此值的重叠框被剔除
// 返回: 经过 NMS 过滤后的检测框列表
static std::vector<Detection> Nms(
    const std::vector<Detection>& detections,
    float nms_threshold) {
    std::vector<Detection> result;
    // 标记每个框是否被抑制
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        result.push_back(detections[i]);

        // 将与当前框 IoU 过高的后续框标记为抑制
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            float iou = ComputeIoU(detections[i], detections[j]);
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// 对单张图像的原始模型输出进行后处理，返回经过 NMS 过滤后的检测框列表
// 流程：
//   1. 校验输出数据大小是否为 85 的倍数
//   2. 修正阈值参数（负数时使用默认值）
//   3. 遍历每个 box，解析 cx,cy,w,h,obj_conf,class_scores
//   4. 计算最终置信度 = obj_conf * max_class_score，低于阈值跳过
//   5. 中心宽高转换为左上右下绝对坐标，裁剪到图像边界
//   6. 按 confidence 降序排序
//   7. 执行 NMS 过滤
// 返回: 经过 NMS 过滤后的 Detection 列表
std::vector<Detection> ProcessYoloOutput(const std::vector<float>& output,
                                         float confidence_threshold,
                                         float nms_threshold,
                                         int image_width,
                                         int image_height) {
    // 校验输出数据大小
    if (output.empty()) {
        return {};
    }

    if (output.size() % kBoxElementCount != 0) {
        Logger::Warning(
            "ProcessYoloOutput: output size " +
            std::to_string(output.size()) +
            " is not a multiple of " + std::to_string(kBoxElementCount));
        return {};
    }

    // 修正阈值参数
    float conf_thresh = confidence_threshold;
    if (conf_thresh < 0.0f) {
        conf_thresh = kDefaultConfidenceThreshold;
        Logger::Warning(
            "ProcessYoloOutput: negative confidence_threshold, using default " +
            std::to_string(kDefaultConfidenceThreshold));
    }

    float nms_thresh = nms_threshold;
    if (nms_thresh < 0.0f) {
        nms_thresh = kDefaultNmsThreshold;
        Logger::Warning(
            "ProcessYoloOutput: negative nms_threshold, using default " +
            std::to_string(kDefaultNmsThreshold));
    }

    int num_boxes = static_cast<int>(output.size() / kBoxElementCount);
    std::vector<Detection> candidates;

    for (int i = 0; i < num_boxes; ++i) {
        const float* box = output.data() + i * kBoxElementCount;

        float obj_conf = box[kObjConfIndex];

        // 跳过目标置信度过低的框，提前过滤减少后续计算
        if (obj_conf < conf_thresh) {
            continue;
        }

        // 在 class_scores 中找最大值及对应类别
        int best_class_id = 0;
        float max_class_score = box[kClassScoreStart];
        for (int c = 1; c < kClassCount; ++c) {
            if (box[kClassScoreStart + c] > max_class_score) {
                max_class_score = box[kClassScoreStart + c];
                best_class_id = c;
            }
        }

        // 最终置信度 = 目标置信度 × 最大类别分数
        float confidence = obj_conf * max_class_score;
        if (confidence < conf_thresh) {
            continue;
        }

        // 中心宽高 → 左上右下绝对坐标
        float cx = box[0];
        float cy = box[1];
        float w = box[2];
        float h = box[3];

        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // 裁剪到图像边界 [0, image_width/height]
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(image_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(image_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(image_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(image_height)));

        candidates.push_back({x1, y1, x2, y2, best_class_id, confidence});
    }

    // 按 confidence 降序排序，NMS 依赖此顺序
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // 执行 NMS 过滤
    return Nms(candidates, nms_thresh);
}

// 将 Detection 坐标从模型输入尺寸缩放到原始图像尺寸
// 对每个检测框乘以缩放因子并裁剪到原图边界，不修改原 detections
// 返回: 缩放后的 Detection 列表，坐标基于原始图像尺寸
std::vector<Detection> ScaleDetectionsToOriginal(
    const std::vector<Detection>& detections,
    int src_width, int src_height,
    int target_width, int target_height) {
    if (detections.empty()) {
        return {};
    }

    if (target_width <= 0 || target_height <= 0) {
        Logger::Warning(
            "ScaleDetectionsToOriginal: invalid target dimensions " +
            std::to_string(target_width) + "x" + std::to_string(target_height));
        return {};
    }

    float scale_x = static_cast<float>(src_width) / static_cast<float>(target_width);
    float scale_y = static_cast<float>(src_height) / static_cast<float>(target_height);

    std::vector<Detection> result;
    result.reserve(detections.size());

    float max_x = static_cast<float>(src_width - 1);
    float max_y = static_cast<float>(src_height - 1);

    for (const auto& det : detections) {
        Detection scaled;
        scaled.x1 = std::max(0.0f, std::min(det.x1 * scale_x, max_x));
        scaled.y1 = std::max(0.0f, std::min(det.y1 * scale_y, max_y));
        scaled.x2 = std::max(0.0f, std::min(det.x2 * scale_x, max_x));
        scaled.y2 = std::max(0.0f, std::min(det.y2 * scale_y, max_y));
        scaled.class_id = det.class_id;
        scaled.confidence = det.confidence;
        result.push_back(scaled);
    }

    return result;
}

}  // namespace inference
