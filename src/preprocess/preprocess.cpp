#include "preprocess.h"
#include "logger.h"

namespace inference {

// 将 cv::Mat 转为 CHW 顺序 float 数组，BGR→RGB，减均值除方差
// 核心转换逻辑：颜色空间转换 → 浮点化 → 通道分离 → 归一化
// 被 ResizeAndNorm 和 Letterbox+MatToChw 流程共用，避免代码重复
std::vector<float> MatToChw(const cv::Mat& img,
                             const std::vector<float>& mean,
                             const std::vector<float>& std) {
    if (img.empty()) {
        Logger::Error("MatToChw: input image is empty");
        return {};
    }

    std::vector<float> mean_val = mean;
    std::vector<float> std_val = std;
    if (mean_val.size() != 3) {
        Logger::Warning("MatToChw: mean size is not 3, using default {0,0,0}");
        mean_val = {0.0f, 0.0f, 0.0f};
    }
    if (std_val.size() != 3) {
        Logger::Warning("MatToChw: std size is not 3, using default {1,1,1}");
        std_val = {1.0f, 1.0f, 1.0f};
    }

    // BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    // 转为 float 类型
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3);

    // 按 CHW 顺序输出，减均值除方差
    int w = float_img.cols;
    int h = float_img.rows;
    int channel_size = w * h;
    std::vector<float> output(3 * channel_size);

    for (int c = 0; c < 3; ++c) {
        for (int row = 0; row < h; ++row) {
            for (int col = 0; col < w; ++col) {
                float val = float_img.at<cv::Vec3f>(row, col)[c];
                output[c * channel_size + row * w + col] =
                    (val - mean_val[c]) / std_val[c];
            }
        }
    }

    return output;
}

// 将 BGR 图像转为 RGB，缩放到指定尺寸，减均值除方差，输出 CHW 顺序 float 数组
// 流程：resize → MatToChw（BGR→RGB + 归一化 + CHW 排列）
std::vector<float> ResizeAndNorm(const cv::Mat& img, int target_w, int target_h,
                                  const std::vector<float>& mean,
                                  const std::vector<float>& std) {
    if (img.empty()) {
        Logger::Error("ResizeAndNorm: input image is empty");
        return {};
    }

    if (target_w <= 0 || target_h <= 0) {
        Logger::Error("ResizeAndNorm: target size must be positive, got (" +
                      std::to_string(target_w) + ", " + std::to_string(target_h) + ")");
        return {};
    }

    // 缩放到目标尺寸
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, target_h));

    // 调用 MatToChw 完成 BGR→RGB、归一化、CHW 排列
    return MatToChw(resized, mean, std);
}

// 保持宽高比缩放图像，不足部分用 fill_color 填充，返回 LetterboxResult
// LetterboxResult 包含预处理后的图像和坐标映射参数（scale, x_offset, y_offset）
// M6 可通过 scale/x_offset/y_offset 调用 M5 的 ScaleDetectionsToOriginal Letterbox 重载
// 将检测框坐标从模型输入尺寸正确映射回原图
LetterboxResult Letterbox(const cv::Mat& img, int target_w, int target_h,
                           const cv::Scalar& fill_color) {
    if (img.empty()) {
        Logger::Error("Letterbox: input image is empty");
        return {cv::Mat(), 0.0f, 0, 0};
    }

    if (target_w <= 0 || target_h <= 0) {
        Logger::Error("Letterbox: target size must be positive, got (" +
                      std::to_string(target_w) + ", " + std::to_string(target_h) + ")");
        return {cv::Mat(), 0.0f, 0, 0};
    }

    int src_w = img.cols;
    int src_h = img.rows;

    // 计算保持宽高比的缩放比例
    float scale = std::min(static_cast<float>(target_w) / src_w,
                           static_cast<float>(target_h) / src_h);

    // 防御性保护：确保缩放后尺寸至少为 1，避免 cv::resize 崩溃
    int new_w = std::max(1, static_cast<int>(src_w * scale));
    int new_h = std::max(1, static_cast<int>(src_h * scale));

    // 缩放图像
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    // 创建目标尺寸的 Mat，填充背景色
    cv::Mat result(target_h, target_w, img.type(), fill_color);

    // 将缩放后的图像居中放置
    int x_offset = (target_w - new_w) / 2;
    int y_offset = (target_h - new_h) / 2;

    resized.copyTo(result(cv::Rect(x_offset, y_offset, new_w, new_h)));

    return {result, scale, x_offset, y_offset};
}

}  // namespace inference
