#pragma once

#include "inference_backend.h"

#include <memory>

namespace inference {

// ONNX Runtime 推理后端，继承 InferenceBackend 抽象接口
// 使用 PImpl 模式隐藏 ONNX Runtime 依赖，头文件无需引入 onnxruntime 头文件
class OnnxBackend : public InferenceBackend {
public:
    OnnxBackend();
    ~OnnxBackend() override;

    bool LoadModel(const std::string& model_path) override;
    std::vector<std::vector<float>> Predict(
        const std::vector<std::vector<float>>& inputs) override;
    std::vector<std::vector<int64_t>> GetInputShapes() const override;
    std::vector<std::string> GetInputNames() const override;
    std::vector<std::string> GetOutputNames() const override;
    std::vector<std::vector<int64_t>> GetOutputShapes() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace inference