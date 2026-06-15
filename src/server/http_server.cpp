#include "http_server.h"
#include "detector.h"
#include "llm_generator.h"
#include "logger.h"
#include "timer.h"
#include "onnx_backend.h"
#include "ncnn_backend.h"

#include <httplib.h>

#include <mutex>
#include <sstream>
#include <cstdlib>
#include <memory>

#include <opencv2/opencv.hpp>

namespace inference {

// 从请求体中提取 JSON 字符串中的指定字段值
// 返回空字符串表示字段不存在
static std::string GetJsonValue(const std::string& json_str, const std::string& key) {
    // 简单 JSON 解析：查找 "key": value 模式
    std::string search = "\"" + key + "\"";
    size_t pos = json_str.find(search);
    if (pos == std::string::npos) {
        return "";
    }
    // 跳过 key 和冒号
    pos = json_str.find(':', pos + search.size());
    if (pos == std::string::npos) {
        return "";
    }
    pos++;
    // 跳过空白
    while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' ||
           json_str[pos] == '\n' || json_str[pos] == '\r')) {
        pos++;
    }
    // 判断值类型
    if (pos >= json_str.size()) {
        return "";
    }
    if (json_str[pos] == '"') {
        // 字符串值
        size_t start = pos + 1;
        size_t end = json_str.find('"', start);
        if (end == std::string::npos) {
            return "";
        }
        return json_str.substr(start, end - start);
    } else {
        // 数字值
        size_t start = pos;
        size_t end = start;
        while (end < json_str.size() && json_str[end] != ',' && json_str[end] != '}' &&
               json_str[end] != ' ' && json_str[end] != '\n' && json_str[end] != '\r') {
            end++;
        }
        return json_str.substr(start, end - start);
    }
}

// 将 Detection 向量转为 JSON 字符串
static std::string DetectionsToJson(const std::vector<Detection>& dets, double inference_ms) {
    std::ostringstream ss;
    ss << "{\"success\":true,\"detections\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "{\"bbox\":[" << dets[i].x1 << "," << dets[i].y1 << ","
           << dets[i].x2 << "," << dets[i].y2 << "],"
           << "\"class_id\":" << dets[i].class_id << ","
           << "\"confidence\":" << dets[i].confidence << "}";
    }
    ss << "],\"inference_time_ms\":" << inference_ms << "}";
    return ss.str();
}

// 构造错误 JSON 响应
static std::string ErrorJson(const std::string& msg) {
    return "{\"success\":false,\"error\":\"" + msg + "\"}";
}

// 启动 HTTP 服务，阻塞运行
void RunServer(const ServerConfig& config) {
    // 创建推理后端
    std::unique_ptr<InferenceBackend> backend;
    if (config.backend_type == "ncnn") {
        backend = std::make_unique<NcnnBackend>();
        Logger::Info("RunServer: using NCNN backend");
    } else {
        backend = std::make_unique<OnnxBackend>();
        Logger::Info("RunServer: using ONNX Runtime backend");
    }

    // 创建目标检测器
    std::unique_ptr<ObjectDetector> detector;
    if (backend->LoadModel(config.detector_model_path)) {
        detector = std::make_unique<ObjectDetector>(std::move(backend));
        detector->SetInputSize(config.input_width, config.input_height);
        Logger::Info("RunServer: detector loaded from " + config.detector_model_path);
    } else {
        Logger::Warning("RunServer: failed to load detector model from " +
                        config.detector_model_path);
    }

    // 创建 LLM 生成器（可选）
    std::unique_ptr<LlamaGenerator> llm;
    if (!config.llm_model_path.empty()) {
        llm = std::make_unique<LlamaGenerator>();
        if (llm->LoadModel(config.llm_model_path)) {
            Logger::Info("RunServer: LLM loaded from " + config.llm_model_path);
        } else {
            Logger::Warning("RunServer: failed to load LLM model from " +
                            config.llm_model_path);
            llm.reset();
        }
    }

    // 线程安全互斥锁
    std::mutex detect_mutex;
    std::mutex llm_mutex;

    // 创建 HTTP 服务
    httplib::Server svr;

    // GET / API 说明页面
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            "<html><head><title>Inference Service API</title></head><body>"
            "<h1>Inference Service API</h1>"
            "<h2>Endpoints</h2>"
            "<h3>GET /health</h3>"
            "<p>Health check. Response: <code>{\"status\":\"ok\"}</code></p>"
            "<h3>POST /detect</h3>"
            "<p>Object detection. Upload image via multipart/form-data field <code>image</code>.</p>"
            "<p>Query params: <code>confidence</code> (float, default 0.25), "
            "<code>nms_threshold</code> (float, default 0.45)</p>"
            "<p>Example: <code>curl -X POST -F \"image=@test.jpg\" http://localhost:8080/detect</code></p>"
            "<h3>POST /generate</h3>"
            "<p>Text generation. Send JSON with <code>prompt</code> (required), "
            "<code>max_tokens</code> (int, default 512), "
            "<code>temperature</code> (float, default 0.8), "
            "<code>top_p</code> (float, default 0.95)</p>"
            "<p>Example: <code>curl -X POST -H \"Content-Type: application/json\" "
            "-d '{\"prompt\":\"Hello\"}' http://localhost:8080/generate</code></p>"
            "</body></html>",
            "text/html");
    });

    // GET /health 健康检查
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // POST /detect 目标检测
    // 支持两种输入方式：
    //   1. multipart/form-data 上传图片文件（字段名 image）
    //   2. JSON body 指定图片路径（字段名 image_path）
    svr.Post("/detect", [&detector, &detect_mutex](const httplib::Request& req,
                                                     httplib::Response& res) {
        res.set_header("Content-Type", "application/json");

        // 检查检测器是否可用
        if (!detector) {
            res.status = 503;
            res.set_content(ErrorJson("Detector not available"), "application/json");
            return;
        }

        cv::Mat img;

        // 优先方式1：检查是否有 multipart 文件字段 image
        if (req.has_file("image")) {
            auto file = req.get_file_value("image");
            std::vector<uint8_t> img_data(file.content.begin(), file.content.end());
            img = cv::imdecode(img_data, cv::IMREAD_COLOR);
            if (img.empty()) {
                res.status = 400;
                res.set_content(ErrorJson("Failed to decode image"), "application/json");
                return;
            }
        } else {
            // 方式2：尝试从 JSON body 解析 image_path
            const std::string& body = req.body;
            std::string image_path = GetJsonValue(body, "image_path");
            if (image_path.empty()) {
                res.status = 400;
                res.set_content(ErrorJson("No image or image_path field in request"), "application/json");
                return;
            }
            img = cv::imread(image_path);
            if (img.empty()) {
                res.status = 400;
                res.set_content(ErrorJson("Failed to read image from path: " + image_path), "application/json");
                return;
            }
        }

        // 解析可选参数
        float confidence = 0.25f;
        float nms_threshold = 0.45f;
        if (req.has_param("confidence")) {
            confidence = std::stof(req.get_param_value("confidence"));
        }
        if (req.has_param("nms_threshold")) {
            nms_threshold = std::stof(req.get_param_value("nms_threshold"));
        }

        // 加锁执行检测
        Timer timer;
        timer.Start();
        std::vector<Detection> dets;
        {
            std::lock_guard<std::mutex> lock(detect_mutex);
            dets = detector->Detect(img, confidence, nms_threshold);
        }
        timer.Stop();

        Logger::Info("POST /detect: " + std::to_string(dets.size()) +
                     " detections, " + std::to_string(timer.ElapsedMilliseconds()) + " ms");

        res.set_content(DetectionsToJson(dets, timer.ElapsedMilliseconds()),
                        "application/json");
    });

    // POST /generate 文本生成
    svr.Post("/generate", [&llm, &llm_mutex](const httplib::Request& req,
                                                httplib::Response& res) {
        res.set_header("Content-Type", "application/json");

        // 检查 LLM 是否可用
        if (!llm) {
            res.status = 503;
            res.set_content(ErrorJson("LLM model not available"), "application/json");
            return;
        }

        // 解析请求 JSON
        const std::string& body = req.body;
        std::string prompt = GetJsonValue(body, "prompt");
        if (prompt.empty()) {
            res.status = 400;
            res.set_content(ErrorJson("Missing 'prompt' field"), "application/json");
            return;
        }

        // 构建 GenerationConfig，使用默认值，请求中有则覆盖
        GenerationConfig gen_config;
        std::string max_tokens_str = GetJsonValue(body, "max_tokens");
        if (!max_tokens_str.empty()) {
            gen_config.max_tokens = std::stoi(max_tokens_str);
        }
        std::string temperature_str = GetJsonValue(body, "temperature");
        if (!temperature_str.empty()) {
            gen_config.temperature = std::stof(temperature_str);
        }
        std::string top_p_str = GetJsonValue(body, "top_p");
        if (!top_p_str.empty()) {
            gen_config.top_p = std::stof(top_p_str);
        }

        // 加锁执行生成
        Timer timer;
        timer.Start();
        std::string result;
        {
            std::lock_guard<std::mutex> lock(llm_mutex);
            result = llm->Generate(prompt, gen_config);
        }
        timer.Stop();

        Logger::Info("POST /generate: " + std::to_string(result.size()) +
                     " chars, " + std::to_string(timer.ElapsedMilliseconds()) + " ms");

        // 构造响应 JSON（转义 result 中的特殊字符）
        std::string escaped;
        escaped.reserve(result.size());
        for (char c : result) {
            switch (c) {
                case '"':  escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\n': escaped += "\\n";  break;
                case '\r': escaped += "\\r";  break;
                case '\t': escaped += "\\t";  break;
                default:   escaped += c;      break;
            }
        }

        std::ostringstream ss;
        ss << "{\"success\":true,\"text\":\"" << escaped
           << "\",\"inference_time_ms\":" << timer.ElapsedMilliseconds() << "}";
        res.set_content(ss.str(), "application/json");
    });

    // 启动服务
    Logger::Info("RunServer: starting on port " + std::to_string(config.port));
    if (!svr.listen("0.0.0.0", config.port)) {
        Logger::Error("RunServer: failed to listen on port " + std::to_string(config.port));
    }
}

}  // namespace inference
