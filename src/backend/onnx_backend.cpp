#include "onnx_backend.h"
#include "logger.h"
#include "timer.h"

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <stdexcept>

namespace inference {

// PImpl 内部结构，持有所有 ONNX Runtime 相关对象和缓存元信息
struct OnnxBackend::Impl {
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;

    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;

    bool model_loaded = false;

    Impl()
        : env(ORT_LOGGING_LEVEL_WARNING, "inference") {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    // 从已加载的 session 中提取输入/输出名称和形状
    void ExtractModelInfo() {
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session->GetInputCount();
        input_names.clear();
        input_name_ptrs.clear();
        input_shapes.clear();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_names.emplace_back(name.get());
            input_name_ptrs.emplace_back(input_names.back().c_str());
            auto type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_shapes.emplace_back(tensor_info.GetShape());
        }

        size_t num_outputs = session->GetOutputCount();
        output_names.clear();
        output_name_ptrs.clear();
        output_shapes.clear();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names.emplace_back(name.get());
            output_name_ptrs.emplace_back(output_names.back().c_str());
            auto type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_shapes.emplace_back(tensor_info.GetShape());
        }
    }
};

OnnxBackend::OnnxBackend()
    : impl_(std::make_unique<Impl>()) {}

OnnxBackend::~OnnxBackend() = default;

bool OnnxBackend::LoadModel(const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        Logger::Error("OnnxBackend::LoadModel: model file not found: " + model_path);
        return false;
    }

    try {
        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, model_path.c_str(), impl_->session_options);
        impl_->ExtractModelInfo();
        impl_->model_loaded = true;

        // 构建输入信息字符串
        std::string input_info = "inputs=[";
        for (size_t i = 0; i < impl_->input_names.size(); ++i) {
            input_info += "name:" + impl_->input_names[i] + ",shape:{";
            for (size_t j = 0; j < impl_->input_shapes[i].size(); ++j) {
                input_info += std::to_string(impl_->input_shapes[i][j]);
                if (j + 1 < impl_->input_shapes[i].size()) input_info += ",";
            }
            input_info += "}";
            if (i + 1 < impl_->input_names.size()) input_info += ",";
        }
        input_info += "]";

        // 构建输出信息字符串
        std::string output_info = "outputs=[";
        for (size_t i = 0; i < impl_->output_names.size(); ++i) {
            output_info += "name:" + impl_->output_names[i] + ",shape:{";
            for (size_t j = 0; j < impl_->output_shapes[i].size(); ++j) {
                output_info += std::to_string(impl_->output_shapes[i][j]);
                if (j + 1 < impl_->output_shapes[i].size()) output_info += ",";
            }
            output_info += "}";
            if (i + 1 < impl_->output_names.size()) output_info += ",";
        }
        output_info += "]";

        Logger::Info("OnnxBackend::LoadModel: model loaded from " + model_path
                     + ", " + input_info + ", " + output_info);
    } catch (const Ort::Exception& e) {
        Logger::Error("OnnxBackend::LoadModel: ONNX Runtime error: " +
                      std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        Logger::Error("OnnxBackend::LoadModel: error: " + std::string(e.what()));
        return false;
    }

    return true;
}

std::vector<std::vector<float>> OnnxBackend::Predict(
    const std::vector<std::vector<float>>& inputs) {
    if (!impl_->model_loaded) {
        Logger::Error("OnnxBackend::Predict: model not loaded");
        return {};
    }

    if (inputs.empty()) {
        Logger::Error("OnnxBackend::Predict: inputs is empty");
        return {};
    }

    // 检查输入数量与模型输入数量一致
    if (inputs.size() != impl_->input_shapes.size()) {
        Logger::Error("OnnxBackend::Predict: input count mismatch, expected "
                      + std::to_string(impl_->input_shapes.size())
                      + ", got " + std::to_string(inputs.size()));
        return {};
    }

    Timer timer;
    timer.Start();

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& shape = impl_->input_shapes[i];
            // 将 shape 中 -1（动态维度）替换为实际值
            std::vector<int64_t> actual_shape = shape;
            int64_t remaining = static_cast<int64_t>(inputs[i].size());
            for (auto& dim : actual_shape) {
                if (dim == -1) {
                    dim = remaining;
                    remaining = 1;
                } else {
                    remaining /= dim;
                }
            }

            // 检查数据大小与形状是否匹配
            if (remaining != 1) {
                Logger::Error("OnnxBackend::Predict: input shape mismatch for input "
                              + std::to_string(i) + ", data size "
                              + std::to_string(inputs[i].size())
                              + " does not match shape");
                return {};
            }

            // ONNX Runtime 不会修改输入数据，使用 const_cast 避免拷贝
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(inputs[i].data()),
                inputs[i].size(),
                actual_shape.data(), actual_shape.size()));
        }

        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_name_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            impl_->output_name_ptrs.data(),
            impl_->output_name_ptrs.size());

        timer.Stop();
        Logger::Info("OnnxBackend::Predict: inference completed in "
                     + std::to_string(timer.ElapsedMilliseconds()) + " ms");

        std::vector<std::vector<float>> results;
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& tensor = output_tensors[i];
            auto* data = tensor.GetTensorData<float>();
            auto info = tensor.GetTensorTypeAndShapeInfo();
            size_t element_count = info.GetElementCount();
            results.emplace_back(data, data + element_count);
        }

        return results;

    } catch (const Ort::Exception& e) {
        Logger::Error("OnnxBackend::Predict: ONNX Runtime error: " +
                      std::string(e.what()));
        return {};
    } catch (const std::exception& e) {
        Logger::Error("OnnxBackend::Predict: error: " + std::string(e.what()));
        return {};
    }
}

std::vector<std::vector<int64_t>> OnnxBackend::GetInputShapes() const {
    return impl_->input_shapes;
}

std::vector<std::string> OnnxBackend::GetInputNames() const {
    return impl_->input_names;
}

std::vector<std::string> OnnxBackend::GetOutputNames() const {
    return impl_->output_names;
}

std::vector<std::vector<int64_t>> OnnxBackend::GetOutputShapes() const {
    return impl_->output_shapes;
}

}  // namespace inference