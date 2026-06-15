#include "benchmark_utils.h"
#include "detector.h"
#include "logger.h"
#include "ncnn_backend.h"
#include "onnx_backend.h"
#include "timer.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace inference;

// 基准测试配置
struct BenchmarkConfig {
    // 后端选择："onnx"、"ncnn"、"all"
    std::string backend = "all";
    // 测试图片路径
    std::string image_path = "models/test.jpg";
    // 模型路径前缀（ONNX 追加 .onnx，NCNN 追加 .param/.bin）
    std::string model_prefix = "models/yolov5s";
    // 批量大小列表
    std::vector<int> batch_sizes = {1, 4, 8};
    // 正式测试次数
    int iterations = 50;
    // 预热次数
    int warmup = 10;
    // CSV 输出路径（空表示不保存）
    std::string csv_output;
    // 模型输入尺寸
    int input_width = 640;
    int input_height = 640;
};

// 解析逗号分隔的整数列表
static std::vector<int> ParseIntList(const std::string& s) {
    std::vector<int> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stoi(token));
    }
    return result;
}

// 解析命令行参数
// 支持格式：--key=value 或 --key value
static BenchmarkConfig ParseArgs(int argc, char* argv[]) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');
        std::string key, value;
        if (eq != std::string::npos) {
            key = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        } else {
            key = arg;
            if (i + 1 < argc) {
                value = argv[++i];
            }
        }

        if (key == "--backend") {
            config.backend = value;
        } else if (key == "--image") {
            config.image_path = value;
        } else if (key == "--model") {
            config.model_prefix = value;
        } else if (key == "--batch-sizes") {
            config.batch_sizes = ParseIntList(value);
        } else if (key == "--iterations") {
            config.iterations = std::stoi(value);
        } else if (key == "--warmup") {
            config.warmup = std::stoi(value);
        } else if (key == "--output") {
            config.csv_output = value;
        } else if (key == "--input-width") {
            config.input_width = std::stoi(value);
        } else if (key == "--input-height") {
            config.input_height = std::stoi(value);
        } else if (key == "--help" || key == "-h") {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("Options:\n");
            printf("  --backend=onnx|ncnn|all   Backend to test (default: all)\n");
            printf("  --image=PATH              Test image path (default: models/test.jpg)\n");
            printf("  --model=PREFIX            Model path prefix (default: models/yolov5s)\n");
            printf("  --batch-sizes=1,4,8       Comma-separated batch sizes (default: 1,4,8)\n");
            printf("  --iterations=N            Test iterations (default: 50)\n");
            printf("  --warmup=N                Warmup iterations (default: 10)\n");
            printf("  --output=PATH             Save results to CSV file\n");
            printf("  --input-width=N           Model input width (default: 640)\n");
            printf("  --input-height=N          Model input height (default: 640)\n");
            exit(0);
        }
    }
    return config;
}

// 加载测试图像
// 优先从路径读取，失败时用固定种子生成随机噪声图
static cv::Mat LoadTestImage(const std::string& path, int width, int height) {
    if (std::filesystem::exists(path)) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (!img.empty()) {
            printf("Loaded test image: %s (%dx%d)\n", path.c_str(), img.cols, img.rows);
            return img;
        }
    }
    printf("Image not found or unreadable: %s, generating random %dx%d noise image\n",
           path.c_str(), width, height);
    cv::Mat noise(height, width, CV_8UC3);
    cv::randu(noise, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    return noise;
}

// 对给定检测器和配置运行基准测试（检测器已加载模型，不重复创建）
// detector: 已初始化的检测器
// backend_name: 显示名称
// img: 测试图像
// config: 测试配置
// batch_size: 本次测试的批量大小
// 返回: 统计结果
static BenchmarkResult RunBenchmark(
    ObjectDetector& detector,
    const std::string& backend_name,
    const cv::Mat& img,
    const BenchmarkConfig& config,
    int batch_size) {

    // 预热
    for (int i = 0; i < config.warmup; ++i) {
        detector.Detect(img);
    }

    // 正式测试
    std::vector<double> latencies;
    latencies.reserve(static_cast<size_t>(config.iterations));

    for (int i = 0; i < config.iterations; ++i) {
        Timer timer;
        timer.Start();

        // 批量测试：循环 batch_size 次调用 Detect
        for (int b = 0; b < batch_size; ++b) {
            detector.Detect(img);
        }

        timer.Stop();
        latencies.push_back(timer.ElapsedMilliseconds());
    }

    return ComputeStats(latencies, backend_name,
                        config.input_width, config.input_height, batch_size);
}

// 测试 ONNX 后端
static void BenchmarkOnnx(const cv::Mat& img, const BenchmarkConfig& config,
                           std::vector<BenchmarkResult>& results) {
    std::string onnx_path = config.model_prefix + ".onnx";
    if (!std::filesystem::exists(onnx_path)) {
        printf("[SKIP] ONNX model not found: %s\n", onnx_path.c_str());
        return;
    }

    printf("\n[Benchmark] ONNX Backend (loading model...)\n");
    auto backend = std::make_unique<OnnxBackend>();
    if (!backend->LoadModel(onnx_path)) {
        printf("[SKIP] ONNX model failed to load: %s\n", onnx_path.c_str());
        return;
    }

    // 确定显示名称
    std::string name = "onnx (CoreML)";
#if !defined(__APPLE__) || !defined(__arm64__)
    name = "onnx (CPU)";
#endif
    printf("[Benchmark] ONNX backend ready: %s\n", name.c_str());

    // 只加载一次模型，创建一个检测器，对每个 batch_size 运行测试
    ObjectDetector detector(std::move(backend));
    detector.SetInputSize(config.input_width, config.input_height);

    for (int batch : config.batch_sizes) {
        printf("  Running batch_size=%d, iterations=%d, warmup=%d ...\n",
               batch, config.iterations, config.warmup);
        results.push_back(RunBenchmark(detector, name, img, config, batch));
    }
}

// 测试 NCNN 后端
static void BenchmarkNcnn(const cv::Mat& img, const BenchmarkConfig& config,
                           std::vector<BenchmarkResult>& results) {
    std::string param_path = config.model_prefix + ".param";
    std::string bin_path = config.model_prefix + ".bin";
    if (!std::filesystem::exists(param_path) || !std::filesystem::exists(bin_path)) {
        printf("[SKIP] NCNN model not found: %s / %s\n", param_path.c_str(), bin_path.c_str());
        printf("  Hint: convert ONNX to NCNN with: onnx2ncnn %s.onnx %s.param %s.bin\n",
               config.model_prefix.c_str(), config.model_prefix.c_str(),
               config.model_prefix.c_str());
        return;
    }

    printf("\n[Benchmark] NCNN Backend (loading model...)\n");
    auto backend = std::make_unique<NcnnBackend>();
    if (!backend->LoadModel(param_path)) {
        printf("[SKIP] NCNN model failed to load: %s\n", param_path.c_str());
        return;
    }

    std::string name = "ncnn (Vulkan)";
    printf("[Benchmark] NCNN backend ready: %s\n", name.c_str());

    ObjectDetector detector(std::move(backend));
    detector.SetInputSize(config.input_width, config.input_height);

    for (int batch : config.batch_sizes) {
        printf("  Running batch_size=%d, iterations=%d, warmup=%d ...\n",
               batch, config.iterations, config.warmup);
        results.push_back(RunBenchmark(detector, name, img, config, batch));
    }
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config = ParseArgs(argc, argv);

    printf("=== Inference Service Performance Benchmark ===\n");
    printf("Config:\n");
    printf("  Backend:      %s\n", config.backend.c_str());
    printf("  Image:        %s\n", config.image_path.c_str());
    printf("  Model prefix: %s\n", config.model_prefix.c_str());
    printf("  Batch sizes:  ");
    for (size_t i = 0; i < config.batch_sizes.size(); ++i) {
        if (i > 0) printf(", ");
        printf("%d", config.batch_sizes[i]);
    }
    printf("\n");
    printf("  Iterations:   %d\n", config.iterations);
    printf("  Warmup:       %d\n", config.warmup);
    printf("  Input size:   %dx%d\n", config.input_width, config.input_height);
    if (!config.csv_output.empty()) {
        printf("  CSV output:   %s\n", config.csv_output.c_str());
    }

    // 加载测试图像
    cv::Mat img = LoadTestImage(config.image_path, config.input_width, config.input_height);

    // 运行基准测试
    std::vector<BenchmarkResult> results;

    if (config.backend == "onnx" || config.backend == "all") {
        BenchmarkOnnx(img, config, results);
    }
    if (config.backend == "ncnn" || config.backend == "all") {
        BenchmarkNcnn(img, config, results);
    }

    // 输出结果
    if (results.empty()) {
        printf("\nNo benchmarks completed. Check model files and backend availability.\n");
        return 1;
    }

    PrintMarkdownTable(results);

    if (!config.csv_output.empty()) {
        SaveCsvFile(config.csv_output, results);
    }

    return 0;
}
