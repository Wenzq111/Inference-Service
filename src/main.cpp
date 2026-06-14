#include "logger.h"
#include "http_server.h"

#include <cstdlib>
#include <string>

using namespace inference;

int main() {
    Logger::SetLevel(LogLevel::Info);

    // 从环境变量读取配置，提供默认值
    ServerConfig config;
    config.port = 8080;
    config.detector_model_path = "models/yolov5s.onnx";
    config.llm_model_path = "models/llama.gguf";
    config.backend_type = "onnx";
    config.input_width = 640;
    config.input_height = 640;

    const char* env_port = std::getenv("INFERENCE_PORT");
    if (env_port) {
        config.port = std::stoi(env_port);
    }
    const char* env_det_model = std::getenv("DETECTOR_MODEL");
    if (env_det_model) {
        config.detector_model_path = env_det_model;
    }
    const char* env_llm_model = std::getenv("LLM_MODEL");
    if (env_llm_model) {
        config.llm_model_path = env_llm_model;
    }
    const char* env_backend = std::getenv("BACKEND_TYPE");
    if (env_backend) {
        config.backend_type = env_backend;
    }
    const char* env_input_w = std::getenv("INPUT_WIDTH");
    if (env_input_w) {
        config.input_width = std::stoi(env_input_w);
    }
    const char* env_input_h = std::getenv("INPUT_HEIGHT");
    if (env_input_h) {
        config.input_height = std::stoi(env_input_h);
    }

    Logger::Info("Main: config port=" + std::to_string(config.port) +
                 ", backend=" + config.backend_type +
                 ", detector_model=" + config.detector_model_path +
                 ", llm_model=" + config.llm_model_path +
                 ", input_size=" + std::to_string(config.input_width) +
                 "x" + std::to_string(config.input_height));

    RunServer(config);
    return 0;
}
