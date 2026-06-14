#include "logger.h"
#include "timer.h"
#include "onnx_backend.h"
#include "ncnn_backend.h"
#include "detector.h"
#include "llm_generator.h"
#include "batch_preprocessor.h"

#include <memory>
#include <thread>
#include <chrono>

#include <opencv2/opencv.hpp>

using namespace inference;

// 程序入口，演示 Logger、Timer、OnnxBackend、NcnnBackend 和 ObjectDetector 功能
int main() {
    // 设置日志过滤级别为 Debug，确保所有级别日志均可输出
    Logger::SetLevel(LogLevel::Debug);

    Logger::Debug("This is a debug message");
    Logger::Info("Hello");
    Logger::Warning("This is a warning");
    Logger::Error("This is an error");

    // 演示 Timer 计时功能
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    timer.Stop();

    Logger::Info("Timer elapsed: " + std::to_string(timer.ElapsedMilliseconds()) + " ms");

    // 演示 ONNX Runtime 后端创建
    auto onnx_backend = std::make_unique<OnnxBackend>();
    Logger::Info("ONNX Runtime backend created successfully");

    // 演示 NCNN 后端创建
    auto ncnn_backend = std::make_unique<NcnnBackend>();
    Logger::Info("NCNN backend created successfully");

    // 演示 ObjectDetector 创建并尝试加载模型
    // 使用独立的 OnnxBackend 实例，避免与上面的创建演示冲突
    auto onnx_backend_det = std::make_unique<OnnxBackend>();
    if (onnx_backend_det->LoadModel("models/yolov5s.onnx")) {
        auto detector = std::make_unique<ObjectDetector>(std::move(onnx_backend_det));
        detector->SetInputSize(640, 640);
        Logger::Info("ObjectDetector created successfully with ONNX backend");

        cv::Mat test_img = cv::imread("models/test.jpg");
        if (!test_img.empty()) {
            auto dets = detector->Detect(test_img);
            Logger::Info("Detected " + std::to_string(dets.size()) + " objects");
        } else {
            Logger::Info("No test image found, skipping detection demo");
        }
    } else {
        Logger::Warning("Failed to load ONNX model, detection demo skipped");
    }

    // 演示 LlamaGenerator 文本生成
    auto llm = std::make_unique<LlamaGenerator>();
    if (llm->LoadModel("models/llama.gguf")) {
        Logger::Info("LlamaGenerator model loaded, context_size=" +
                     std::to_string(llm->GetContextSize()));

        // 非流式生成
        GenerationConfig config;
        config.max_tokens = 128;
        config.temperature = 0.7f;
        std::string reply = llm->Generate("Hello, how are you?", config);
        Logger::Info("LLM reply: " + reply);

        // 流式生成
        Logger::Info("LLM stream: ");
        llm->GenerateStream("Tell me a short joke.",
            [](const std::string& piece) {
                std::cout << piece << std::flush;
            }, config);
        std::cout << std::endl;
    } else {
        Logger::Warning("Failed to load LLM model, LLM demo skipped");
    }

    // M8: 批量异步预处理演示 — Resize 模式
    BatchPreprocessor bp_resize(4);
    bp_resize.SetPreprocessParams(640, 640, PreprocessMode::Resize,
                                  {0.0f, 0.0f, 0.0f}, {255.0f, 255.0f, 255.0f});
    bp_resize.SetCallback([](size_t idx, const PreprocessResult& result) {
        if (result.tensor.empty()) {
            Logger::Error("BatchPreprocessor(Resize): failed index " + std::to_string(idx));
        } else {
            Logger::Info("BatchPreprocessor(Resize): done index " + std::to_string(idx) +
                         ", tensor_size=" + std::to_string(result.tensor.size()));
        }
    });
    bp_resize.Submit(0, "models/test.jpg");
    bp_resize.Submit(1, "models/test2.jpg");
    bp_resize.WaitAll();
    Logger::Info("BatchPreprocessor(Resize): all tasks completed");

    // M8: 批量异步预处理演示 — Letterbox 模式
    BatchPreprocessor bp_letterbox(4);
    bp_letterbox.SetPreprocessParams(640, 640, PreprocessMode::Letterbox,
                                     {0.0f, 0.0f, 0.0f}, {255.0f, 255.0f, 255.0f});
    bp_letterbox.SetCallback([](size_t idx, const PreprocessResult& result) {
        if (result.tensor.empty()) {
            Logger::Error("BatchPreprocessor(Letterbox): failed index " + std::to_string(idx));
        } else {
            Logger::Info("BatchPreprocessor(Letterbox): done index " + std::to_string(idx) +
                         ", tensor_size=" + std::to_string(result.tensor.size()) +
                         ", scale=" + std::to_string(result.scale) +
                         ", x_offset=" + std::to_string(result.x_offset) +
                         ", y_offset=" + std::to_string(result.y_offset));
        }
    });
    bp_letterbox.Submit(0, "models/test.jpg");
    bp_letterbox.Submit(1, "models/test2.jpg");
    bp_letterbox.WaitAll();
    Logger::Info("BatchPreprocessor(Letterbox): all tasks completed");

    return 0;
}
