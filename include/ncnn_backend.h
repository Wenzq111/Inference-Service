#pragma once

#include "inference_backend.h"

#include <memory>

namespace inference {

// NCNN 推理后端，继承 InferenceBackend 抽象接口
// 使用 PImpl 模式隐藏 NCNN 依赖，头文件无需引入 ncnn 头文件
// 支持 Vulkan GPU 自动加速（macOS Metal / Linux Vulkan），无可用 GPU 时回退 CPU
class NcnnBackend : public InferenceBackend {
public:
    // 构造 NcnnBackend 对象
    NcnnBackend();

    // 析构函数，释放 NCNN 资源，若启用了 Vulkan 则销毁 GPU 实例
    ~NcnnBackend() override;

    // 加载 NCNN 模型（.param + .bin）
    // model_path: .param 文件路径，或不含后缀的基础名（自动追加 .param/.bin）
    // 自动推导 .bin 路径，检查文件存在性
    // 自动检测 Vulkan GPU，可用时启用 GPU 加速
    // 开启 fp16、Winograd 等优化选项
    bool LoadModel(const std::string& model_path) override;

    // 执行批量推理
    // inputs: 输入张量列表，每个元素是 CHW 顺序的一维 float 数组
    // 首次调用后缓存输出形状到 output_shapes_
    std::vector<std::vector<float>> Predict(
        const std::vector<std::vector<float>>& inputs) override;

    // 返回模型输入张量形状列表，LoadModel 时从 blob.shape 解析
    std::vector<std::vector<int64_t>> GetInputShapes() const override;

    // 返回模型输入节点名称列表
    std::vector<std::string> GetInputNames() const override;

    // 返回模型输出节点名称列表
    std::vector<std::string> GetOutputNames() const override;

    // 返回模型输出张量形状列表，首次 Predict 后缓存，此前为空
    std::vector<std::vector<int64_t>> GetOutputShapes() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace inference
