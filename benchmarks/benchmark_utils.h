#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace inference {

// 单次基准测试的统计结果
struct BenchmarkResult {
    // 后端名称（如 "onnx (CoreML)"、"ncnn (Vulkan)"、"ncnn (CPU)"）
    std::string backend_name;
    // 模型输入宽度
    int input_width;
    // 模型输入高度
    int input_height;
    // 批量大小
    int batch_size;
    // 平均延迟（ms）
    double avg_latency_ms;
    // 延迟标准差（ms）
    double std_dev_ms;
    // 最小延迟（ms）
    double min_latency_ms;
    // 最大延迟（ms）
    double max_latency_ms;
    // 吞吐量（FPS）
    double fps;
};

// 从延迟向量计算统计结果
// latencies: 每次迭代的延迟（ms）
// backend_name: 后端名称
// input_w: 模型输入宽度
// input_h: 模型输入高度
// batch_size: 批量大小
// 返回: 填充好的 BenchmarkResult
inline BenchmarkResult ComputeStats(
    const std::vector<double>& latencies,
    const std::string& backend_name,
    int input_w, int input_h, int batch_size) {
    BenchmarkResult result;
    result.backend_name = backend_name;
    result.input_width = input_w;
    result.input_height = input_h;
    result.batch_size = batch_size;

    if (latencies.empty()) {
        result.avg_latency_ms = 0.0;
        result.std_dev_ms = 0.0;
        result.min_latency_ms = 0.0;
        result.max_latency_ms = 0.0;
        result.fps = 0.0;
        return result;
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    result.avg_latency_ms = sum / static_cast<double>(latencies.size());

    double sq_sum = 0.0;
    for (double v : latencies) {
        sq_sum += (v - result.avg_latency_ms) * (v - result.avg_latency_ms);
    }
    result.std_dev_ms = std::sqrt(sq_sum / static_cast<double>(latencies.size()));

    result.min_latency_ms = *std::min_element(latencies.begin(), latencies.end());
    result.max_latency_ms = *std::max_element(latencies.begin(), latencies.end());

    // FPS = batch_size / 平均耗时(秒)
    if (result.avg_latency_ms > 0.0) {
        result.fps = static_cast<double>(batch_size) / (result.avg_latency_ms / 1000.0);
    } else {
        result.fps = 0.0;
    }

    return result;
}

// 将结果以 Markdown 表格形式打印到 stdout
inline void PrintMarkdownTable(const std::vector<BenchmarkResult>& results) {
    printf("\n");
    printf("| Backend | Input Size | Batch | Avg Latency (ms) | Std Dev (ms) | Min (ms) | Max (ms) | FPS |\n");
    printf("|---------|-----------|-------|-------------------|--------------|----------|----------|-----|\n");
    for (const auto& r : results) {
        printf("| %s | %dx%d | %d | %.2f | %.2f | %.2f | %.2f | %.1f |\n",
               r.backend_name.c_str(),
               r.input_width, r.input_height,
               r.batch_size,
               r.avg_latency_ms, r.std_dev_ms,
               r.min_latency_ms, r.max_latency_ms,
               r.fps);
    }
    printf("\n");
}

// 将结果保存为 CSV 文件
inline void SaveCsvFile(const std::string& path,
                        const std::vector<BenchmarkResult>& results) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        printf("Error: cannot open CSV file for writing: %s\n", path.c_str());
        return;
    }
    ofs << "backend,input_width,input_height,batch_size,avg_latency_ms,std_dev_ms,"
           "min_latency_ms,max_latency_ms,fps\n";
    for (const auto& r : results) {
        ofs << r.backend_name << ","
            << r.input_width << ","
            << r.input_height << ","
            << r.batch_size << ","
            << r.avg_latency_ms << ","
            << r.std_dev_ms << ","
            << r.min_latency_ms << ","
            << r.max_latency_ms << ","
            << r.fps << "\n";
    }
    ofs.close();
    printf("Results saved to %s\n", path.c_str());
}

}  // namespace inference
