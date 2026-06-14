#pragma once

#include <string>

namespace inference {

// HTTP 服务配置，控制端口、模型路径和后端类型
struct ServerConfig {
    // 监听端口
    int port = 8080;
    // 检测器模型路径（ONNX 或 NCNN 格式）
    std::string detector_model_path = "models/yolov5s.onnx";
    // LLM 模型路径（GGUF 格式）
    std::string llm_model_path = "models/llama.gguf";
    // 推理后端类型："onnx" 或 "ncnn"
    std::string backend_type = "onnx";
    // 检测器输入宽度
    int input_width = 640;
    // 检测器输入高度
    int input_height = 640;
};

// 启动 HTTP 服务，阻塞运行
// 内部创建推理后端、检测器和 LLM 生成器，设置路由
// config: 服务配置（端口、模型路径等）
void RunServer(const ServerConfig& config);

}  // namespace inference
