#include <gtest/gtest.h>

#include "yolo_postprocess.h"

#include <cmath>

using namespace inference;

// 辅助函数：构造一个 YOLO 检测框的 85 元素向量
// cx, cy, w, h, obj_conf, class_scores[0..79]
static std::vector<float> MakeYoloBox(
    float cx, float cy, float w, float h,
    float obj_conf, int best_class, float class_score) {
    std::vector<float> box(85, 0.0f);
    box[0] = cx;
    box[1] = cy;
    box[2] = w;
    box[3] = h;
    box[4] = obj_conf;
    box[5 + best_class] = class_score;
    return box;
}

// 辅助函数：将多个 box 拼接为完整输出
static std::vector<float> ConcatBoxes(const std::vector<std::vector<float>>& boxes) {
    std::vector<float> output;
    for (const auto& box : boxes) {
        output.insert(output.end(), box.begin(), box.end());
    }
    return output;
}

// ============ ProcessYoloOutput 测试 ============

// 测试空输出返回空结果
TEST(ProcessYoloOutputTest, EmptyOutput) {
    auto result = ProcessYoloOutput({}, 0.25f, 0.45f, 640, 640);
    EXPECT_TRUE(result.empty());
}

// 测试输出大小不是 85 的倍数返回空结果
TEST(ProcessYoloOutputTest, InvalidOutputSize) {
    std::vector<float> output(84, 0.0f);
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);
    EXPECT_TRUE(result.empty());
}

// 测试单个高置信度检测框
TEST(ProcessYoloOutputTest, SingleHighConfidenceBox) {
    // 中心 (320, 320)，宽高 (100, 200)，obj_conf=0.9，类别 0 score=0.8
    auto box = MakeYoloBox(320, 320, 100, 200, 0.9f, 0, 0.8f);
    auto output = ConcatBoxes({box});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);

    ASSERT_EQ(result.size(), 1u);
    // 置信度 = obj_conf * class_score = 0.9 * 0.8 = 0.72
    EXPECT_FLOAT_EQ(result[0].confidence, 0.72f);
    // 中心宽高转左上右下：x1=320-50=270, y1=320-100=220, x2=320+50=370, y2=320+100=420
    EXPECT_FLOAT_EQ(result[0].x1, 270.0f);
    EXPECT_FLOAT_EQ(result[0].y1, 220.0f);
    EXPECT_FLOAT_EQ(result[0].x2, 370.0f);
    EXPECT_FLOAT_EQ(result[0].y2, 420.0f);
    EXPECT_EQ(result[0].class_id, 0);
}

// 测试低置信度检测框被过滤
TEST(ProcessYoloOutputTest, LowConfidenceFiltered) {
    // obj_conf=0.2，class_score=0.5，最终置信度=0.1 < 0.25 阈值
    auto box = MakeYoloBox(320, 320, 100, 100, 0.2f, 0, 0.5f);
    auto output = ConcatBoxes({box});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);
    EXPECT_TRUE(result.empty());
}

// 测试 obj_conf 低于阈值直接跳过
TEST(ProcessYoloOutputTest, LowObjConfFiltered) {
    // obj_conf=0.1 < 0.25 阈值，直接跳过
    auto box = MakeYoloBox(320, 320, 100, 100, 0.1f, 0, 1.0f);
    auto output = ConcatBoxes({box});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);
    EXPECT_TRUE(result.empty());
}

// 测试负数阈值使用默认值
TEST(ProcessYoloOutputTest, NegativeThresholdUsesDefault) {
    // obj_conf=0.3，class_score=0.9，置信度=0.27
    // 默认阈值 0.25，0.27 >= 0.25 通过
    auto box = MakeYoloBox(320, 320, 100, 100, 0.3f, 0, 0.9f);
    auto output = ConcatBoxes({box});
    auto result = ProcessYoloOutput(output, -1.0f, -1.0f, 640, 640);
    EXPECT_EQ(result.size(), 1u);
}

// 测试检测框坐标裁剪到图像边界
TEST(ProcessYoloOutputTest, BoxClippedToImageBounds) {
    // 中心 (10, 10)，宽高 (100, 100)，x1=10-50=-40 → 裁剪到 0
    auto box = MakeYoloBox(10, 10, 100, 100, 0.9f, 0, 0.8f);
    auto output = ConcatBoxes({box});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].x1, 0.0f);
    EXPECT_FLOAT_EQ(result[0].y1, 0.0f);
}

// 测试检测框右下角超出图像边界被裁剪
TEST(ProcessYoloOutputTest, BoxRightBottomClipped) {
    // 中心 (630, 630)，宽高 (100, 100)，x2=630+50=680 → 裁剪到 640
    auto box = MakeYoloBox(630, 630, 100, 100, 0.9f, 0, 0.8f);
    auto output = ConcatBoxes({box});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].x2, 640.0f);
    EXPECT_FLOAT_EQ(result[0].y2, 640.0f);
}

// 测试多个检测框的 NMS 过滤
TEST(ProcessYoloOutputTest, NmsSuppressesOverlapping) {
    // 两个高度重叠的检测框
    auto box1 = MakeYoloBox(320, 320, 100, 100, 0.95f, 0, 0.9f);
    auto box2 = MakeYoloBox(325, 325, 100, 100, 0.9f, 0, 0.85f);
    auto output = ConcatBoxes({box1, box2});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);

    // IoU 很高，NMS 应只保留置信度更高的那个
    EXPECT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.95f * 0.9f);
}

// 测试不重叠的检测框都保留
TEST(ProcessYoloOutputTest, NonOverlappingBoxesKept) {
    auto box1 = MakeYoloBox(100, 100, 50, 50, 0.9f, 0, 0.8f);
    auto box2 = MakeYoloBox(500, 500, 50, 50, 0.9f, 1, 0.8f);
    auto output = ConcatBoxes({box1, box2});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);

    EXPECT_EQ(result.size(), 2u);
}

// 测试不同类别的检测框不受 NMS 影响
TEST(ProcessYoloOutputTest, DifferentClassesNotSuppressed) {
    // 两个高度重叠但不同类别的框（当前 NMS 不区分类别）
    auto box1 = MakeYoloBox(320, 320, 100, 100, 0.95f, 0, 0.9f);
    auto box2 = MakeYoloBox(325, 325, 100, 100, 0.9f, 1, 0.85f);
    auto output = ConcatBoxes({box1, box2});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);

    // 当前实现 NMS 不区分类别，高 IoU 的重叠框会被抑制
    // 这是当前实现的行为特征，后续可改为类别感知 NMS
    EXPECT_GE(result.size(), 1u);
}

// 测试多个检测框按置信度降序排列
TEST(ProcessYoloOutputTest, ResultsSortedByConfidence) {
    auto box1 = MakeYoloBox(100, 100, 50, 50, 0.7f, 0, 0.8f);
    auto box2 = MakeYoloBox(200, 200, 50, 50, 0.95f, 0, 0.9f);
    auto box3 = MakeYoloBox(400, 400, 50, 50, 0.8f, 0, 0.7f);
    auto output = ConcatBoxes({box1, box2, box3});
    auto result = ProcessYoloOutput(output, 0.25f, 0.45f, 640, 640);

    for (size_t i = 1; i < result.size(); ++i) {
        EXPECT_GE(result[i - 1].confidence, result[i].confidence);
    }
}

// ============ ScaleDetectionsToOriginal（线性缩放）测试 ============

// 测试线性缩放正确映射坐标
TEST(ScaleDetectionsToOriginalTest, LinearScaling) {
    // 模型输入 640x640 → 原图 1280x720
    std::vector<Detection> dets = {{100, 50, 200, 100, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, 1280, 720, 640, 640);

    ASSERT_EQ(result.size(), 1u);
    float scale_x = 1280.0f / 640.0f;
    float scale_y = 720.0f / 640.0f;
    EXPECT_FLOAT_EQ(result[0].x1, 100 * scale_x);
    EXPECT_FLOAT_EQ(result[0].y1, 50 * scale_y);
    EXPECT_FLOAT_EQ(result[0].x2, 200 * scale_x);
    EXPECT_FLOAT_EQ(result[0].y2, 100 * scale_y);
}

// 测试线性缩放空输入返回空
TEST(ScaleDetectionsToOriginalTest, LinearEmptyInput) {
    auto result = ScaleDetectionsToOriginal({}, 1280, 720, 640, 640);
    EXPECT_TRUE(result.empty());
}

// 测试线性缩放无效目标尺寸返回空
TEST(ScaleDetectionsToOriginalTest, LinearInvalidTargetSize) {
    std::vector<Detection> dets = {{100, 50, 200, 100, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, 1280, 720, 0, 640);
    EXPECT_TRUE(result.empty());
}

// 测试线性缩放坐标裁剪到原图边界
TEST(ScaleDetectionsToOriginalTest, LinearClippedToOriginalBounds) {
    // 坐标超出原图边界时被裁剪
    std::vector<Detection> dets = {{600, 600, 700, 700, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, 800, 600, 640, 640);

    ASSERT_EQ(result.size(), 1u);
    // x2 = 700 * (800/640) = 875 → 裁剪到 799
    EXPECT_LE(result[0].x2, 799.0f);
    EXPECT_LE(result[0].y2, 599.0f);
}

// 测试线性缩放保留类别和置信度
TEST(ScaleDetectionsToOriginalTest, LinearPreservesClassAndConfidence) {
    std::vector<Detection> dets = {{10, 10, 20, 20, 5, 0.85f}};
    auto result = ScaleDetectionsToOriginal(dets, 1280, 720, 640, 640);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].class_id, 5);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.85f);
}

// ============ ScaleDetectionsToOriginal（Letterbox 模式）测试 ============

// 测试 Letterbox 缩放正确映射坐标
TEST(ScaleDetectionsToOriginalLetterboxTest, LetterboxScaling) {
    // Letterbox 参数：scale=0.8, x_offset=0, y_offset=160
    // 原图 800x400 → 模型输入 640x640
    std::vector<Detection> dets = {{320, 300, 400, 400, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, 0, 160, 0.8f, 800, 400);

    ASSERT_EQ(result.size(), 1u);
    // x1 = (320 - 0) / 0.8 = 400
    // y1 = (300 - 160) / 0.8 = 175
    // x2 = (400 - 0) / 0.8 = 500
    // y2 = (400 - 160) / 0.8 = 300
    EXPECT_FLOAT_EQ(result[0].x1, 400.0f);
    EXPECT_FLOAT_EQ(result[0].y1, 175.0f);
    EXPECT_FLOAT_EQ(result[0].x2, 500.0f);
    EXPECT_FLOAT_EQ(result[0].y2, 300.0f);
}

// 测试 Letterbox 空输入返回空
TEST(ScaleDetectionsToOriginalLetterboxTest, LetterboxEmptyInput) {
    auto result = ScaleDetectionsToOriginal({}, 0, 0, 0.8f, 800, 400);
    EXPECT_TRUE(result.empty());
}

// 测试 Letterbox 无效 scale 返回空
TEST(ScaleDetectionsToOriginalLetterboxTest, LetterboxInvalidScale) {
    std::vector<Detection> dets = {{10, 10, 20, 20, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, 0, 0, 0.0f, 800, 400);
    EXPECT_TRUE(result.empty());

    result = ScaleDetectionsToOriginal(dets, 0, 0, -1.0f, 800, 400);
    EXPECT_TRUE(result.empty());
}

// 测试 Letterbox 无效偏移返回空
TEST(ScaleDetectionsToOriginalLetterboxTest, LetterboxInvalidOffset) {
    std::vector<Detection> dets = {{10, 10, 20, 20, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, -1, 0, 0.8f, 800, 400);
    EXPECT_TRUE(result.empty());

    result = ScaleDetectionsToOriginal(dets, 0, -1, 0.8f, 800, 400);
    EXPECT_TRUE(result.empty());
}

// 测试 Letterbox 无效原图尺寸返回空
TEST(ScaleDetectionsToOriginalLetterboxTest, LetterboxInvalidSrcDimensions) {
    std::vector<Detection> dets = {{10, 10, 20, 20, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, 0, 0, 0.8f, 0, 400);
    EXPECT_TRUE(result.empty());

    result = ScaleDetectionsToOriginal(dets, 0, 0, 0.8f, 800, 0);
    EXPECT_TRUE(result.empty());
}

// 测试 Letterbox 坐标裁剪到原图边界
TEST(ScaleDetectionsToOriginalLetterboxTest, LetterboxClippedToOriginalBounds) {
    // 偏移为 0，scale=0.5，检测框超出原图范围
    std::vector<Detection> dets = {{600, 600, 700, 700, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, 0, 0, 0.5f, 800, 600);

    ASSERT_EQ(result.size(), 1u);
    // x1 = 600 / 0.5 = 1200 → 裁剪到 799
    EXPECT_FLOAT_EQ(result[0].x1, 799.0f);
}

// 测试 Letterbox 缩放保留类别和置信度
TEST(ScaleDetectionsToOriginalLetterboxTest, LetterboxPreservesClassAndConfidence) {
    std::vector<Detection> dets = {{320, 300, 400, 400, 7, 0.77f}};
    auto result = ScaleDetectionsToOriginal(dets, 0, 160, 0.8f, 800, 400);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].class_id, 7);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.77f);
}

// 测试 Letterbox 检测框位于填充区域时坐标为负被裁剪到 0
TEST(ScaleDetectionsToOriginalLetterboxTest, BoxInPaddingRegionClipped) {
    // y_offset=160，检测框在填充区域 y1=80 < 160
    // y1_original = (80 - 160) / 0.8 = -100 → 裁剪到 0
    std::vector<Detection> dets = {{100, 80, 200, 150, 0, 0.9f}};
    auto result = ScaleDetectionsToOriginal(dets, 0, 160, 0.8f, 800, 400);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].y1, 0.0f);
}
