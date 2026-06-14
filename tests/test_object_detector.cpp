#include <gtest/gtest.h>

#include "detector.h"
#include "inference_backend.h"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace inference;

// Mock 推理后端，用于测试 ObjectDetector 的端到端流程
// 通过返回预设的输出向量来模拟推理结果
class MockBackend : public InferenceBackend {
public:
    // 预设的模型输入形状
    std::vector<std::vector<int64_t>> input_shapes_ = {{1, 3, 640, 640}};
    // 预设的模型输入名称
    std::vector<std::string> input_names_ = {"images"};
    // 预设的模型输出形状
    std::vector<std::vector<int64_t>> output_shapes_ = {{1, 25200, 85}};
    // 预设的模型输出名称
    std::vector<std::string> output_names_ = {"output"};
    // 预设的推理输出（由测试用例设置）
    std::vector<std::vector<float>> mock_output_;
    // LoadModel 调用标记
    bool load_called_ = false;

    bool LoadModel(const std::string& model_path) override {
        load_called_ = true;
        return true;
    }

    std::vector<std::vector<float>> Predict(
        const std::vector<std::vector<float>>& inputs) override {
        return mock_output_;
    }

    std::vector<std::vector<int64_t>> GetInputShapes() const override {
        return input_shapes_;
    }

    std::vector<std::string> GetInputNames() const override {
        return input_names_;
    }

    std::vector<std::string> GetOutputNames() const override {
        return output_names_;
    }

    std::vector<std::vector<int64_t>> GetOutputShapes() const override {
        return output_shapes_;
    }
};

// 辅助函数：构造一个 YOLO 检测框的 85 元素向量
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

// 辅助函数：创建指定尺寸的纯色图像
static cv::Mat MakeSolidImage(int width, int height, cv::Scalar color) {
    return cv::Mat(height, width, CV_8UC3, color);
}

// ============ ObjectDetector 测试 ============

// 测试 Detect 空图像返回空结果
TEST(ObjectDetectorTest, EmptyImageReturnsEmpty) {
    auto mock = std::make_unique<MockBackend>();
    mock->mock_output_ = {};

    ObjectDetector detector(std::move(mock));
    cv::Mat empty;
    auto result = detector.Detect(empty);
    EXPECT_TRUE(result.empty());
}

// 测试 Detect 正常输入返回检测结果
TEST(ObjectDetectorTest, DetectWithMockOutput) {
    auto mock = std::make_unique<MockBackend>();

    // 构造一个高置信度检测框
    auto box = MakeYoloBox(320, 320, 100, 100, 0.95f, 0, 0.9f);
    mock->mock_output_ = {box};

    ObjectDetector detector(std::move(mock));
    cv::Mat img = MakeSolidImage(640, 640, cv::Scalar(128, 128, 128));
    auto result = detector.Detect(img);

    // 应至少检测到 1 个目标
    EXPECT_GE(result.size(), 1u);
    // 置信度应大于 0
    if (!result.empty()) {
        EXPECT_GT(result[0].confidence, 0.0f);
    }
}

// 测试 Detect 低置信度输出返回空结果
TEST(ObjectDetectorTest, LowConfidenceOutputReturnsEmpty) {
    auto mock = std::make_unique<MockBackend>();

    // 构造一个低置信度检测框
    auto box = MakeYoloBox(320, 320, 100, 100, 0.1f, 0, 0.1f);
    mock->mock_output_ = {box};

    ObjectDetector detector(std::move(mock));
    cv::Mat img = MakeSolidImage(640, 640, cv::Scalar(0, 0, 0));
    auto result = detector.Detect(img, 0.25f, 0.45f);
    EXPECT_TRUE(result.empty());
}

// 测试 Detect 空推理输出返回空结果
TEST(ObjectDetectorTest, EmptyInferenceOutputReturnsEmpty) {
    auto mock = std::make_unique<MockBackend>();
    mock->mock_output_ = {};

    ObjectDetector detector(std::move(mock));
    cv::Mat img = MakeSolidImage(640, 640, cv::Scalar(0, 0, 0));
    auto result = detector.Detect(img);
    EXPECT_TRUE(result.empty());
}

// 测试 SetInputSize 修改输入尺寸
TEST(ObjectDetectorTest, SetInputSize) {
    auto mock = std::make_unique<MockBackend>();
    mock->mock_output_ = {};

    ObjectDetector detector(std::move(mock));
    detector.SetInputSize(320, 320);

    // 验证不崩溃即可（内部使用新尺寸做 Letterbox）
    cv::Mat img = MakeSolidImage(640, 480, cv::Scalar(0, 0, 0));
    EXPECT_NO_THROW(detector.Detect(img));
}

// 测试 SetNormalizeParams 修改归一化参数
TEST(ObjectDetectorTest, SetNormalizeParams) {
    auto mock = std::make_unique<MockBackend>();
    mock->mock_output_ = {};

    ObjectDetector detector(std::move(mock));
    detector.SetNormalizeParams({0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f});

    cv::Mat img = MakeSolidImage(100, 100, cv::Scalar(128, 128, 128));
    EXPECT_NO_THROW(detector.Detect(img));
}

// 测试 Detect 多个检测框
TEST(ObjectDetectorTest, DetectMultipleBoxes) {
    auto mock = std::make_unique<MockBackend>();

    // 构造两个不重叠的高置信度检测框
    auto box1 = MakeYoloBox(100, 100, 50, 50, 0.95f, 0, 0.9f);
    auto box2 = MakeYoloBox(500, 500, 50, 50, 0.9f, 1, 0.85f);
    std::vector<float> output;
    output.insert(output.end(), box1.begin(), box1.end());
    output.insert(output.end(), box2.begin(), box2.end());
    mock->mock_output_ = {output};

    ObjectDetector detector(std::move(mock));
    cv::Mat img = MakeSolidImage(640, 640, cv::Scalar(128, 128, 128));
    auto result = detector.Detect(img);

    EXPECT_GE(result.size(), 2u);
}

// 测试 Detect 对非正方形图像也能工作
TEST(ObjectDetectorTest, NonSquareImage) {
    auto mock = std::make_unique<MockBackend>();
    mock->mock_output_ = {};

    ObjectDetector detector(std::move(mock));
    cv::Mat img = MakeSolidImage(1920, 1080, cv::Scalar(128, 128, 128));
    EXPECT_NO_THROW(detector.Detect(img));
}

// 测试 Detect 对小图像也能工作
TEST(ObjectDetectorTest, SmallImage) {
    auto mock = std::make_unique<MockBackend>();
    mock->mock_output_ = {};

    ObjectDetector detector(std::move(mock));
    cv::Mat img = MakeSolidImage(32, 32, cv::Scalar(128, 128, 128));
    EXPECT_NO_THROW(detector.Detect(img));
}

// 测试 Detect 自定义置信度和 NMS 阈值
TEST(ObjectDetectorTest, CustomThresholds) {
    auto mock = std::make_unique<MockBackend>();

    // 中等置信度框
    auto box = MakeYoloBox(320, 320, 100, 100, 0.5f, 0, 0.6f);
    mock->mock_output_ = {box};

    ObjectDetector detector(std::move(mock));
    cv::Mat img = MakeSolidImage(640, 640, cv::Scalar(128, 128, 128));

    // 低阈值应该检测到
    auto result_low = detector.Detect(img, 0.1f, 0.45f);
    EXPECT_GE(result_low.size(), 1u);

    // 高阈值应该检测不到（置信度 = 0.5 * 0.6 = 0.3）
    auto result_high = detector.Detect(img, 0.5f, 0.45f);
    EXPECT_TRUE(result_high.empty());
}
