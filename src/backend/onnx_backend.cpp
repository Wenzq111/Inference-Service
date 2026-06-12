#include "onnx_backend.h"
#include "logger.h"
#include "timer.h"

#include <onnxruntime_cxx_api.h>

#include <unordered_map>

#include <filesystem>
#include <stdexcept>

namespace inference {

// PImpl 内部结构，持有所有 ONNX Runtime 相关对象和缓存元信息
struct OnnxBackend::Impl {
    // ONNX Runtime 运行环境，全局唯一，管理日志输出和会话生命周期，构造时指定日志级别为 Warning
    Ort::Env env;
    // 会话配置选项，控制线程数、图优化级别等运行参数，构造时设置单线程和全级别优化
    Ort::SessionOptions session_options;
    // ONNX Runtime 推理会话，LoadModel 时创建，持有模型权重和执行图，RAII 管理资源释放
    std::unique_ptr<Ort::Session> session;

    // 模型输入节点名称列表，LoadModel 时从 session 提取，与 input_name_ptrs 和 input_shapes 顺序一致
    std::vector<std::string> input_names;
    // 模型输出节点名称列表，LoadModel 时从 session 提取，与 output_name_ptrs 和 output_shapes 顺序一致
    std::vector<std::string> output_names;
    // 模型输入张量形状列表，每个元素是一个维度向量如 {1,3,224,224}，动态维度用 -1 表示
    std::vector<std::vector<int64_t>> input_shapes;
    // 模型输出张量形状列表，每个元素是一个维度向量如 {1,1000}，动态维度用 -1 表示
    std::vector<std::vector<int64_t>> output_shapes;

    // 输入节点名称的 C 字符串指针列表，指向 input_names 中的 string 内部缓冲区，供 session.Run() 使用
    std::vector<const char*> input_name_ptrs;
    // 输出节点名称的 C 字符串指针列表，指向 output_names 中的 string 内部缓冲区，供 session.Run() 使用
    std::vector<const char*> output_name_ptrs;

    // 标记模型是否已成功加载，Predict 等方法的前置条件，LoadModel 成功后置为 true
    bool model_loaded = false;

    // 初始化 ONNX Runtime 环境和会话选项
    // env: 日志级别为 Warning，仅输出警告和错误信息
    // session_options: IntraOp 线程数设为 1（单线程推理），图优化级别设为 ORT_ENABLE_ALL（启用所有可用优化）
    // Apple Silicon 上额外尝试注册 CoreML EP，使用官方推荐的新 API（AppendExecutionProvider），
    // 旧 API OrtSessionOptionsAppendExecutionProvider_CoreML 已在 ONNX Runtime 1.20.0 中弃用
    // 失败时静默回退到 CPU
    Impl()
        : env(ORT_LOGGING_LEVEL_WARNING, "inference") {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

#if defined(__APPLE__) && defined(__arm64__)
        // 使用官方推荐的新 API：SessionOptions::AppendExecutionProvider("CoreML", options)
        // 新 API 通过字符串键值对配置，更灵活且与其他 EP 注册方式一致
        try {
            std::unordered_map<std::string, std::string> coreml_options;
            // MLComputeUnits=ALL: 让 CoreML 自动选择最优硬件（ANE > GPU > CPU）
            coreml_options["MLComputeUnits"] = "ALL";
            session_options.AppendExecutionProvider("CoreML", coreml_options);
            Logger::Info("CoreML Execution Provider enabled on Apple Silicon");
        } catch (...) {
            Logger::Warning("CoreML Execution Provider not available, falling back to CPU");
        }
#endif
    }

    // 从已加载的 Ort::Session 中提取并缓存模型元信息
    // 使用 Ort::AllocatorWithDefaultOptions 分配器获取节点名称
    // 遍历所有输入/输出节点，依次提取：
    //   1. 节点名称（AllocatedStringPtr 转 std::string）
    //   2. 名称对应的 C 字符串指针（指向 string 内部缓冲区，须在 string 生命周期内使用）
    //   3. 节点的 TensorTypeInfo 中的形状信息（含动态维度 -1）
    // 提取后清空旧缓存，保证 LoadModel 多次调用时数据一致性
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

// 构造 OnnxBackend 对象，创建 Impl 实例初始化 ONNX Runtime 环境
// 此时模型尚未加载，需后续调用 LoadModel 加载具体模型
OnnxBackend::OnnxBackend()
    : impl_(std::make_unique<Impl>()) {}

// 析构函数，由 unique_ptr<Impl> 自动释放 ONNX Runtime 资源
// Impl 内的 Ort::Session、Ort::Env 等均为 RAII 对象，无需手动清理
OnnxBackend::~OnnxBackend() = default;

// 加载 ONNX 模型文件并初始化推理会话
// model_path: 模型文件路径（.onnx 格式）
// 流程：
//   1. 检查模型文件是否存在，不存在则返回 false
//   2. 创建 Ort::Session 并加载模型，提取输入/输出元信息
//   3. 构建包含输入/输出名称和形状的详细日志
// 异常处理：捕获 Ort::Exception 和 std::exception，转为 Logger::Error 并返回 false
// 返回: 加载成功返回 true，文件不存在或 ONNX Runtime 异常返回 false
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

// 执行推理，将预处理后的数据送入模型并获取输出
// inputs: 输入张量列表，每个元素是一个 flatten 为一维的 float 数组（CHW 顺序）
//         数量必须与模型输入节点数量一致
// 流程：
//   1. 前置检查：模型已加载、inputs 不为空、输入数量匹配
//   2. 动态形状解析：将 -1 维度替换为实际值，校验数据大小与形状整除一致性
//   3. 创建 Ort::Value 输入张量（使用 const_cast 避免数据拷贝）
//   4. 调用 session.Run() 执行推理
//   5. 从输出 Ort::Value 中拷贝数据到 vector<float>
//   6. Timer 记录推理耗时并输出日志
// 异常处理：捕获 Ort::Exception 和 std::exception，转为 Logger::Error 并返回空 vector
// 返回: 推理成功返回输出张量列表，每个输出为 flatten 的一维 float 数组；
//       前置检查失败或推理异常返回空 vector
std::vector<std::vector<float>> OnnxBackend::Predict(
    const std::vector<std::vector<float>>& inputs) {
    // 前置检查1：模型必须已通过 LoadModel 加载，否则 Predict 无法构造输入张量和调用 session.Run()
    // 前置检查1：模型必须已通过 LoadModel 加载，否则无法构造输入张量和调用 session.Run()
    if (!impl_->model_loaded) {
        Logger::Error("OnnxBackend::Predict: model not loaded");
        return {};
    }
    // 前置检查2：inputs 不能为空，空输入无法构造 Ort::Value 张量
    if (inputs.empty()) {
        Logger::Error("OnnxBackend::Predict: inputs is empty");
        return {};
    }
    // 前置检查3：输入数量必须与模型输入节点数量一致
    // ONNX Runtime 要求每个输入节点都有对应数据，多余或不足都会导致 Run() 失败
    if (inputs.size() != impl_->input_shapes.size()) {
        Logger::Error("OnnxBackend::Predict: input count mismatch, expected "
                      + std::to_string(impl_->input_shapes.size())
                      + ", got " + std::to_string(inputs.size()));
        return {};
    }

    Timer timer;
    timer.Start();

    try {
        // ---- 推理核心流程开始 ----
        // 步骤1：创建 CPU 内存信息对象
        // OrtArenaAllocator 使用 ONNX Runtime 内部内存池（arena），分配效率高于直接 malloc
        // OrtMemTypeDefault 表示使用默认内存类型（非 GPU、非 pinned）
        // 该 memory_info 将用于所有输入张量的数据绑定
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault);

        // 步骤2：为每个模型输入创建 Ort::Value 张量对象
        std::vector<Ort::Value> input_tensors;
        // 遍历每个输入，将 flatten 的 float 数组包装为 ONNX Runtime 可识别的张量
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto& shape = impl_->input_shapes[i];
            // 将 shape 中 -1（动态维度）替换为实际值
            // ONNX 模型的输入形状可能包含 -1（如 {−1,3,224,224} 表示 batch_size 动态），实际推理时需根据输入数据大小推断具体值
            // remaining 从总元素数开始，逐维除以已知维度，遇到 -1 时将该维度设为 remaining 并将 remaining 重置为 1
            // 最终 remaining 应等于 1 表示数据量与形状完全匹配
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

            // ONNX Runtime 的 CreateTensor 要求 T*（非 const）指针，但 Run() 实际不会修改输入数据
            // 使用 const_cast 去除 const 限制，直接引用 inputs 中的数据，避免整个输入向量的深拷贝
            // 这是安全的：ONNX Runtime 只读取输入张量数据用于推理计算
            input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(inputs[i].data()),
                inputs[i].size(),
                actual_shape.data(), actual_shape.size()));
        }

        // 步骤3：执行推理
        // session.Run() 是 ONNX Runtime 的核心推理调用，参数含义：
        //   - Ort::RunOptions{nullptr}: 使用默认运行选项（无日志输出、无超时限制）
        //   - input_name_ptrs: 输入节点名称的 C 字符串指针数组，告诉 ONNX Runtime 每个输入张量对应哪个输入节点
        //   - input_tensors: 输入张量数组，与 input_name_ptrs 顺序一一对应
        //   - input_tensors.size(): 输入张量数量
        //   - output_name_ptrs: 输出节点名称数组，指定需要获取哪些输出节点的结果
        //   - output_name_ptrs.size(): 输出节点数量
        // Run() 内部执行流程：将输入数据送入模型执行图 → 各算子依次计算 → 将结果写入输出张量
        // 在 Apple Silicon 上，如果 CoreML EP 已注册，ONNX Runtime 会将兼容的算子调度到 CoreML（ANE/GPU/CPU），
        // 不兼容的算子自动回退到 CPU EP，用户无需手动处理
        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_name_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            impl_->output_name_ptrs.data(),
            impl_->output_name_ptrs.size());

        // 步骤4：记录推理耗时
        // Timer 基于 steady_clock 测量从 Start 到 Stop 的时间间隔
        // 该耗时包含张量创建、session.Run() 执行和数据拷贝的总时间
        timer.Stop();
        Logger::Info("OnnxBackend::Predict: inference completed in "
                     + std::to_string(timer.ElapsedMilliseconds()) + " ms");

        // 步骤5：从输出 Ort::Value 张量中提取推理结果数据
        // output_tensors 中的每个 Ort::Value 包含一个输出张量，数据存储在 ONNX Runtime 内部缓冲区
        // GetTensorData<float>() 返回指向内部缓冲区的 const float* 指针
        // GetTensorTypeAndShapeInfo().GetElementCount() 返回张量中总元素个数
        // 使用指针范围构造 vector<float> 将数据从 ONNX Runtime 缓冲区拷贝到用户拥有的 vector 中
        // 此拷贝不可避免：ONNX Runtime 的内部缓冲区在 session 生命周期内有效，用户需自行持有结果数据
        std::vector<std::vector<float>> results;
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto& tensor = output_tensors[i];
            auto* data = tensor.GetTensorData<float>();
            auto info = tensor.GetTensorTypeAndShapeInfo();
            size_t element_count = info.GetElementCount();
            results.emplace_back(data, data + element_count);
        }

        // 返回推理结果，每个输出张量已 flatten 为一维 float 数组
        // 调用方可通过 GetOutputShapes() 获取形状信息来解析多维数据
        return results;

    } catch (const Ort::Exception& e) {
        // 捕获 ONNX Runtime 特有异常：包括模型执行错误、张量类型不匹配、EP 调度失败等
        // Ort::Exception 包含 ONNX 错误码和详细描述
        Logger::Error("OnnxBackend::Predict: ONNX Runtime error: " +
                      std::string(e.what()));
        return {};
    } catch (const std::exception& e) {
        // 捕获通用异常：兜底处理非 ONNX Runtime 的异常（如内存分配失败、标准库错误等）
        Logger::Error("OnnxBackend::Predict: error: " + std::string(e.what()));
        return {};
    }
}

// 返回模型输入张量形状列表，顺序与 GetInputNames() 一致
std::vector<std::vector<int64_t>> OnnxBackend::GetInputShapes() const {
    return impl_->input_shapes;
}

// 返回模型输入节点名称列表，顺序与 GetInputShapes() 一致
std::vector<std::string> OnnxBackend::GetInputNames() const {
    return impl_->input_names;
}

// 返回模型输出节点名称列表，顺序与 GetOutputShapes() 一致
std::vector<std::string> OnnxBackend::GetOutputNames() const {
    return impl_->output_names;
}

// 返回模型输出张量形状列表，顺序与 GetOutputNames() 一致
std::vector<std::vector<int64_t>> OnnxBackend::GetOutputShapes() const {
    return impl_->output_shapes;
}

}  // namespace inference