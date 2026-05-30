#include "preprocess.h"
#include "logger.h"

namespace inference {

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

    std::vector<float> mean_val = mean;
    std::vector<float> std_val = std;
    if (mean_val.size() != 3) {
        Logger::Warning("ResizeAndNorm: mean size is not 3, using default {0,0,0}");
        mean_val = {0.0f, 0.0f, 0.0f};
    }
    if (std_val.size() != 3) {
        Logger::Warning("ResizeAndNorm: std size is not 3, using default {1,1,1}");
        std_val = {1.0f, 1.0f, 1.0f};
    }

    // BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    // 缩放到目标尺寸
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(target_w, target_h));

    // 转为 float 类型
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3);

    // 按 CHW 顺序输出，减均值除方差
    std::vector<float> output(3 * target_w * target_h);
    int channel_size = target_w * target_h;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_h; ++h) {
            for (int w = 0; w < target_w; ++w) {
                float val = float_img.at<cv::Vec3f>(h, w)[c];
                output[c * channel_size + h * target_w + w] =
                    (val - mean_val[c]) / std_val[c];
            }
        }
    }

    return output;
}

cv::Mat Letterbox(const cv::Mat& img, int target_w, int target_h,
                  const cv::Scalar& fill_color) {
    if (img.empty()) {
        Logger::Error("Letterbox: input image is empty");
        return cv::Mat();
    }

    if (target_w <= 0 || target_h <= 0) {
        Logger::Error("Letterbox: target size must be positive, got (" +
                      std::to_string(target_w) + ", " + std::to_string(target_h) + ")");
        return cv::Mat();
    }

    int src_w = img.cols;
    int src_h = img.rows;

    // 计算保持宽高比的缩放比例
    float scale = std::min(static_cast<float>(target_w) / src_w,
                           static_cast<float>(target_h) / src_h);

    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);

    // 缩放图像
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    // 创建目标尺寸的 Mat，填充背景色
    cv::Mat result(target_h, target_w, img.type(), fill_color);

    // 将缩放后的图像居中放置
    int x_offset = (target_w - new_w) / 2;
    int y_offset = (target_h - new_h) / 2;

    resized.copyTo(result(cv::Rect(x_offset, y_offset, new_w, new_h)));

    return result;
}

}  // namespace inference