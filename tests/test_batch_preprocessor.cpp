#include <gtest/gtest.h>

#include "batch_preprocessor.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

using namespace inference;

// 辅助函数：创建指定尺寸的纯色图像
static cv::Mat MakeSolidImage(int width, int height, cv::Scalar color) {
    return cv::Mat(height, width, CV_8UC3, color);
}

// ============ BatchPreprocessor Resize 模式测试 ============

// 测试 Resize 模式下提交单张图像并获取结果
TEST(BatchPreprocessorTest, ResizeModeSingleImage) {
    BatchPreprocessor bp(2);
    bp.SetPreprocessParams(32, 32, PreprocessMode::Resize,
                           {0, 0, 0}, {255, 255, 255});

    std::vector<PreprocessResult> results(1);
    std::mutex mtx;

    bp.SetCallback([&](size_t index, const PreprocessResult& result) {
        std::lock_guard<std::mutex> lock(mtx);
        results[index] = result;
    });

    cv::Mat img = MakeSolidImage(100, 100, cv::Scalar(128, 128, 128));
    EXPECT_TRUE(bp.Submit(0, img));
    bp.WaitAll();

    // Resize 模式下 tensor 大小 = 3 * 32 * 32 = 3072
    EXPECT_EQ(results[0].tensor.size(), 3072u);
    // Resize 模式下 Letterbox 元数据为默认值
    EXPECT_FLOAT_EQ(results[0].scale, 0.0f);
    EXPECT_EQ(results[0].x_offset, 0);
    EXPECT_EQ(results[0].y_offset, 0);

    bp.Stop();
}

// 测试 Resize 模式下提交多张图像
TEST(BatchPreprocessorTest, ResizeModeMultipleImages) {
    BatchPreprocessor bp(4);
    bp.SetPreprocessParams(64, 64, PreprocessMode::Resize,
                           {0, 0, 0}, {255, 255, 255});

    const size_t kCount = 10;
    std::atomic<size_t> completed{0};
    std::vector<PreprocessResult> results(kCount);
    std::mutex mtx;

    bp.SetCallback([&](size_t index, const PreprocessResult& result) {
        std::lock_guard<std::mutex> lock(mtx);
        results[index] = result;
        completed++;
    });

    for (size_t i = 0; i < kCount; ++i) {
        cv::Mat img = MakeSolidImage(50 + i * 10, 50 + i * 10,
                                      cv::Scalar(i * 25, i * 25, i * 25));
        EXPECT_TRUE(bp.Submit(i, img));
    }

    bp.WaitAll();
    EXPECT_EQ(completed.load(), kCount);

    for (size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(results[i].tensor.size(), 3u * 64 * 64);
    }

    bp.Stop();
}

// ============ BatchPreprocessor Letterbox 模式测试 ============

// 测试 Letterbox 模式下返回缩放元数据
TEST(BatchPreprocessorTest, LetterboxModeReturnsMetadata) {
    BatchPreprocessor bp(2);
    bp.SetPreprocessParams(640, 640, PreprocessMode::Letterbox,
                           {0, 0, 0}, {255, 255, 255});

    std::vector<PreprocessResult> results(1);
    std::mutex mtx;

    bp.SetCallback([&](size_t index, const PreprocessResult& result) {
        std::lock_guard<std::mutex> lock(mtx);
        results[index] = result;
    });

    // 800x400 图像 Letterbox 到 640x640
    cv::Mat img = MakeSolidImage(800, 400, cv::Scalar(128, 128, 128));
    EXPECT_TRUE(bp.Submit(0, img));
    bp.WaitAll();

    // tensor 大小 = 3 * 640 * 640
    EXPECT_EQ(results[0].tensor.size(), 3u * 640 * 640);
    // scale = min(640/800, 640/400) = 0.8
    EXPECT_FLOAT_EQ(results[0].scale, 0.8f);
    EXPECT_EQ(results[0].x_offset, 0);
    EXPECT_EQ(results[0].y_offset, 160);

    bp.Stop();
}

// ============ BatchPreprocessor WaitAll 测试 ============

// 测试 WaitAll 阻塞直到所有任务完成
TEST(BatchPreprocessorTest, WaitAllBlocksUntilComplete) {
    BatchPreprocessor bp(2);
    bp.SetPreprocessParams(32, 32, PreprocessMode::Resize);

    std::atomic<size_t> completed{0};
    bp.SetCallback([&](size_t, const PreprocessResult&) {
        completed++;
    });

    for (size_t i = 0; i < 5; ++i) {
        cv::Mat img = MakeSolidImage(10, 10, cv::Scalar(0, 0, 0));
        bp.Submit(i, img);
    }

    bp.WaitAll();
    EXPECT_EQ(completed.load(), 5u);

    bp.Stop();
}

// ============ BatchPreprocessor Stop 测试 ============

// 测试 Stop 后 Submit 返回 false
TEST(BatchPreprocessorTest, SubmitAfterStopReturnsFalse) {
    BatchPreprocessor bp(1);
    bp.Stop();

    cv::Mat img = MakeSolidImage(10, 10, cv::Scalar(0, 0, 0));
    EXPECT_FALSE(bp.Submit(0, img));
}

// 测试多次调用 Stop 不会崩溃
TEST(BatchPreprocessorTest, MultipleStopCalls) {
    BatchPreprocessor bp(2);
    bp.Stop();
    EXPECT_NO_THROW(bp.Stop());
}

// ============ BatchPreprocessor 边界条件测试 ============

// 测试提交空图像时回调仍被调用（tensor 为空）
TEST(BatchPreprocessorTest, EmptyImageCallbackStillCalled) {
    BatchPreprocessor bp(1);
    bp.SetPreprocessParams(32, 32, PreprocessMode::Resize);

    std::vector<PreprocessResult> results(1);
    std::mutex mtx;

    bp.SetCallback([&](size_t index, const PreprocessResult& result) {
        std::lock_guard<std::mutex> lock(mtx);
        results[index] = result;
    });

    cv::Mat empty;
    EXPECT_TRUE(bp.Submit(0, empty));
    bp.WaitAll();

    // 空图像预处理失败，tensor 为空
    EXPECT_TRUE(results[0].tensor.empty());

    bp.Stop();
}

// 测试单线程工作模式
TEST(BatchPreprocessorTest, SingleWorker) {
    BatchPreprocessor bp(1);
    bp.SetPreprocessParams(16, 16, PreprocessMode::Resize);

    std::atomic<size_t> completed{0};
    bp.SetCallback([&](size_t, const PreprocessResult&) {
        completed++;
    });

    for (size_t i = 0; i < 3; ++i) {
        cv::Mat img = MakeSolidImage(10, 10, cv::Scalar(i * 80, 0, 0));
        bp.Submit(i, img);
    }

    bp.WaitAll();
    EXPECT_EQ(completed.load(), 3u);

    bp.Stop();
}

// 测试回调中 index 正确传递
TEST(BatchPreprocessorTest, CallbackIndexCorrect) {
    BatchPreprocessor bp(2);
    bp.SetPreprocessParams(8, 8, PreprocessMode::Resize);

    std::vector<size_t> indices;
    std::mutex mtx;

    bp.SetCallback([&](size_t index, const PreprocessResult&) {
        std::lock_guard<std::mutex> lock(mtx);
        indices.push_back(index);
    });

    bp.Submit(42, MakeSolidImage(10, 10, cv::Scalar(0, 0, 0)));
    bp.Submit(99, MakeSolidImage(10, 10, cv::Scalar(0, 0, 0)));
    bp.WaitAll();

    EXPECT_EQ(indices.size(), 2u);
    // 排序后检查（因为多线程完成顺序不确定）
    std::sort(indices.begin(), indices.end());
    EXPECT_EQ(indices[0], 42u);
    EXPECT_EQ(indices[1], 99u);

    bp.Stop();
}

// 测试零线程数时自动调整为 1
TEST(BatchPreprocessorTest, ZeroWorkersAdjustedToOne) {
    BatchPreprocessor bp(0);
    bp.SetPreprocessParams(8, 8, PreprocessMode::Resize);

    std::atomic<size_t> completed{0};
    bp.SetCallback([&](size_t, const PreprocessResult&) {
        completed++;
    });

    bp.Submit(0, MakeSolidImage(10, 10, cv::Scalar(0, 0, 0)));
    bp.WaitAll();

    EXPECT_EQ(completed.load(), 1u);
    bp.Stop();
}
