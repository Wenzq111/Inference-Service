#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace inference {

// Letterbox 预处理结果，包含输出图像和坐标映射参数
// M6 目标检测器可通过 scale/x_offset/y_offset 将检测框坐标映射回原图
struct LetterboxResult {
    // 预处理后的图像（BGR 格式，尺寸为 target_w × target_h）
    cv::Mat image;
    // 缩放比例 = min(target_w/src_w, target_h/src_h)
    float scale;
    // 居中放置的 x 偏移量（像素）
    int x_offset;
    // 居中放置的 y 偏移量（像素）
    int y_offset;
};

// 将 cv::Mat 转为 CHW 顺序 float 数组，BGR→RGB，减均值除方差
// 用于 Letterbox 后的进一步转换，或任何需要将 cv::Mat 送入模型的场景
// img: 输入图像（BGR 格式）
// mean: 各通道均值，默认 {0,0,0}
// std: 各通道标准差，默认 {1,1,1}
// 返回: CHW 顺序的 float 数组，大小为 3 * img.cols * img.rows
std::vector<float> MatToChw(const cv::Mat& img,
                             const std::vector<float>& mean = {0.0f, 0.0f, 0.0f},
                             const std::vector<float>& std = {1.0f, 1.0f, 1.0f});

// 将 BGR 图像转为 RGB，缩放到指定尺寸，减均值除方差，输出 CHW 顺序 float 数组
// 内部先 resize 再调用 MatToChw 完成转换
std::vector<float> ResizeAndNorm(const cv::Mat& img, int target_w, int target_h,
                                  const std::vector<float>& mean,
                                  const std::vector<float>& std);

// 保持宽高比缩放图像，不足部分用 fill_color 填充，返回 LetterboxResult
// LetterboxResult 包含预处理后的图像和坐标映射参数（scale, x_offset, y_offset）
// 后续可通过 MatToChw 将 image 转为模型输入的 CHW float 数组
LetterboxResult Letterbox(const cv::Mat& img, int target_w, int target_h,
                           const cv::Scalar& fill_color = {114, 114, 114});

}  // namespace inference
