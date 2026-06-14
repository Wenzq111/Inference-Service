#include <gtest/gtest.h>

#include "preprocess.h"

#include <cmath>
#include <numeric>

using namespace inference;

// 辅助函数：创建指定尺寸和颜色的 BGR 图像
static cv::Mat MakeSolidImage(int width, int height, cv::Scalar color) {
    return cv::Mat(height, width, CV_8UC3, color);
}

// ============ MatToChw 测试 ============

// 测试 MatToChw 对纯色图像的归一化结果
TEST(MatToChwTest, SolidColorNormalization) {
    // 创建 2x2 纯蓝色 BGR 图像 (255, 0, 0)
    cv::Mat img = MakeSolidImage(2, 2, cv::Scalar(255, 0, 0));
    auto result = MatToChw(img, {0, 0, 0}, {255, 255, 255});

    // 输出大小 = 3 * 2 * 2 = 12
    ASSERT_EQ(result.size(), 12u);

    // BGR (255,0,0) → RGB 后 R=0, G=0, B=255
    // CHW 顺序：[R通道(4个), G通道(4个), B通道(4个)]
    // R 通道 (index 0-3): 0/255 = 0.0f
    // G 通道 (index 4-7): 0/255 = 0.0f
    // B 通道 (index 8-11): 255/255 = 1.0f
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);       // R
        EXPECT_FLOAT_EQ(result[4 + i], 0.0f);   // G
        EXPECT_FLOAT_EQ(result[8 + i], 1.0f);   // B
    }
}

// 测试 MatToChw 自定义均值和标准差
TEST(MatToChwTest, CustomMeanAndStd) {
    // 创建 2x2 纯灰色 BGR 图像 (128, 128, 128)
    cv::Mat img = MakeSolidImage(2, 2, cv::Scalar(128, 128, 128));
    auto result = MatToChw(img, {128, 128, 128}, {1, 1, 1});

    // RGB 都是 128，减均值 128 后为 0，除以 1 后为 0
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], 0.0f, 1e-5f);
    }
}

// 测试 MatToChw 空图像返回空向量
TEST(MatToChwTest, EmptyImageReturnsEmpty) {
    cv::Mat empty;
    auto result = MatToChw(empty);
    EXPECT_TRUE(result.empty());
}

// 测试 MatToChw 输出尺寸正确
TEST(MatToChwTest, OutputSizeCorrect) {
    cv::Mat img = MakeSolidImage(10, 8, cv::Scalar(0, 0, 0));
    auto result = MatToChw(img);
    // 3 channels * 10 * 8 = 240
    EXPECT_EQ(result.size(), 240u);
}

// 测试 MatToChw BGR 到 RGB 转换正确
TEST(MatToChwTest, BgrToRgbConversion) {
    // 创建 1x1 纯黄色 BGR (0, 255, 255)
    cv::Mat img = MakeSolidImage(1, 1, cv::Scalar(0, 255, 255));
    auto result = MatToChw(img, {0, 0, 0}, {1, 1, 1});

    // RGB 应为 (255, 255, 0)
    // CHW: R=255, G=255, B=0
    EXPECT_FLOAT_EQ(result[0], 255.0f);   // R
    EXPECT_FLOAT_EQ(result[1], 255.0f);   // G
    EXPECT_FLOAT_EQ(result[2], 0.0f);     // B
}

// ============ ResizeAndNorm 测试 ============

// 测试 ResizeAndNorm 正确缩放尺寸
TEST(ResizeAndNormTest, OutputSizeAfterResize) {
    cv::Mat img = MakeSolidImage(100, 200, cv::Scalar(128, 128, 128));
    auto result = ResizeAndNorm(img, 32, 16, {0, 0, 0}, {255, 255, 255});
    // 3 * 32 * 16 = 1536
    EXPECT_EQ(result.size(), 1536u);
}

// 测试 ResizeAndNorm 空图像返回空向量
TEST(ResizeAndNormTest, EmptyImageReturnsEmpty) {
    cv::Mat empty;
    auto result = ResizeAndNorm(empty, 32, 32, {0, 0, 0}, {1, 1, 1});
    EXPECT_TRUE(result.empty());
}

// 测试 ResizeAndNorm 无效目标尺寸返回空向量
TEST(ResizeAndNormTest, InvalidTargetSizeReturnsEmpty) {
    cv::Mat img = MakeSolidImage(10, 10, cv::Scalar(0, 0, 0));
    EXPECT_TRUE(ResizeAndNorm(img, 0, 10, {0, 0, 0}, {1, 1, 1}).empty());
    EXPECT_TRUE(ResizeAndNorm(img, 10, 0, {0, 0, 0}, {1, 1, 1}).empty());
    EXPECT_TRUE(ResizeAndNorm(img, -1, 10, {0, 0, 0}, {1, 1, 1}).empty());
}

// 测试 ResizeAndNorm 归一化结果在 [0,1] 范围内
TEST(ResizeAndNormTest, NormalizedRange) {
    cv::Mat img = MakeSolidImage(50, 50, cv::Scalar(200, 100, 50));
    auto result = ResizeAndNorm(img, 32, 32, {0, 0, 0}, {255, 255, 255});

    for (float val : result) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f + 1e-5f);
    }
}

// ============ Letterbox 测试 ============

// 测试 Letterbox 对正方形图像返回 scale=1.0
TEST(LetterboxTest, SquareImageScaleOne) {
    cv::Mat img = MakeSolidImage(640, 640, cv::Scalar(0, 0, 0));
    auto result = Letterbox(img, 640, 640);

    EXPECT_FLOAT_EQ(result.scale, 1.0f);
    EXPECT_EQ(result.x_offset, 0);
    EXPECT_EQ(result.y_offset, 0);
    EXPECT_EQ(result.image.cols, 640);
    EXPECT_EQ(result.image.rows, 640);
}

// 测试 Letterbox 保持宽高比缩放
TEST(LetterboxTest, AspectRatioPreserved) {
    // 800x400 图像 letterbox 到 640x640
    cv::Mat img = MakeSolidImage(800, 400, cv::Scalar(128, 128, 128));
    auto result = Letterbox(img, 640, 640);

    // scale = min(640/800, 640/400) = min(0.8, 1.6) = 0.8
    EXPECT_FLOAT_EQ(result.scale, 0.8f);

    // 缩放后尺寸: 800*0.8=640, 400*0.8=320
    // x_offset = (640-640)/2 = 0
    // y_offset = (640-320)/2 = 160
    EXPECT_EQ(result.x_offset, 0);
    EXPECT_EQ(result.y_offset, 160);
}

// 测试 Letterbox 输出图像尺寸等于目标尺寸
TEST(LetterboxTest, OutputSizeEqualsTarget) {
    cv::Mat img = MakeSolidImage(1920, 1080, cv::Scalar(0, 0, 0));
    auto result = Letterbox(img, 640, 640);

    EXPECT_EQ(result.image.cols, 640);
    EXPECT_EQ(result.image.rows, 640);
}

// 测试 Letterbox 空图像返回空 Mat
TEST(LetterboxTest, EmptyImageReturnsEmpty) {
    cv::Mat empty;
    auto result = Letterbox(empty, 640, 640);
    EXPECT_TRUE(result.image.empty());
}

// 测试 Letterbox 无效目标尺寸返回空 Mat
TEST(LetterboxTest, InvalidTargetSizeReturnsEmpty) {
    cv::Mat img = MakeSolidImage(10, 10, cv::Scalar(0, 0, 0));
    auto result = Letterbox(img, 0, 640);
    EXPECT_TRUE(result.image.empty());

    result = Letterbox(img, 640, 0);
    EXPECT_TRUE(result.image.empty());
}

// 测试 Letterbox 窄高图像的偏移量
TEST(LetterboxTest, TallImageOffset) {
    // 400x800 图像 letterbox 到 640x640
    cv::Mat img = MakeSolidImage(400, 800, cv::Scalar(128, 128, 128));
    auto result = Letterbox(img, 640, 640);

    // scale = min(640/400, 640/800) = min(1.6, 0.8) = 0.8
    EXPECT_FLOAT_EQ(result.scale, 0.8f);

    // 缩放后尺寸: 400*0.8=320, 800*0.8=640
    // x_offset = (640-320)/2 = 160
    // y_offset = (640-640)/2 = 0
    EXPECT_EQ(result.x_offset, 160);
    EXPECT_EQ(result.y_offset, 0);
}

// 测试 Letterbox 填充区域使用指定颜色
TEST(LetterboxTest, FillColor) {
    cv::Mat img = MakeSolidImage(100, 100, cv::Scalar(0, 0, 255));
    auto result = Letterbox(img, 200, 200, cv::Scalar(114, 114, 114));

    // 图像被缩放到 200x200 居中（scale=1.0，实际缩放到 200x200，无偏移...）
    // 实际 scale = min(200/100, 200/100) = 2.0
    // 缩放后 200x200，无偏移
    EXPECT_EQ(result.image.cols, 200);
    EXPECT_EQ(result.image.rows, 200);
}

// 测试 Letterbox 对 1x1 图像不会崩溃
TEST(LetterboxTest, OnePixelImage) {
    cv::Mat img = MakeSolidImage(1, 1, cv::Scalar(128, 128, 128));
    auto result = Letterbox(img, 640, 640);

    EXPECT_FALSE(result.image.empty());
    EXPECT_GT(result.scale, 0.0f);
}
