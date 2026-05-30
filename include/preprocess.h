#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace inference {

// 将 BGR 图像转为 RGB，缩放到指定尺寸，减均值除方差，输出 CHW 顺序 float 数组
std::vector<float> ResizeAndNorm(const cv::Mat& img, int target_w, int target_h,
                                  const std::vector<float>& mean,
                                  const std::vector<float>& std);

// 保持宽高比缩放图像，不足部分用 fill_color 填充，输出与目标尺寸一致的 Mat
cv::Mat Letterbox(const cv::Mat& img, int target_w, int target_h,
                  const cv::Scalar& fill_color = {114, 114, 114});

}  // namespace inference