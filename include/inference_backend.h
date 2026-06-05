#pragma once

#include <string>
#include <vector>

namespace inference {

// 推理后端抽象接口，定义统一的模型加载和推理方法
// 具体后端（ONNX Runtime、NCNN）需继承此类并实现所有纯虚函数
// 注意：当前接口使用 std::vector<std::vector<float>> 传递批量数据，
// 后续可考虑用 std::vector<float> + 形状信息替代以提升效率
class InferenceBackend {
public:
    // 加载推理模型
    // model_path: 模型文件路径（如 .onnx 或 .param/.bin）
    // 返回: 加载成功返回 true，失败返回 false
    virtual bool LoadModel(const std::string& model_path) = 0;

    // 执行批量推理
    // inputs: 一个 batch 的输入张量，每个张量已 flatten 为一维 float 数组
    // 返回: 一个 batch 的输出张量，每个输出也是一维 float 数组
    virtual std::vector<std::vector<float>> Predict(
        const std::vector<std::vector<float>>& inputs) = 0;

    // 获取模型所有输入张量的形状
    // 返回: 每个输入张量的形状，例如 {{1,3,224,224}} 表示一个 batch=1、3通道、224x224 的输入
    //       返回顺序与 GetInputNames() 一致
    virtual std::vector<std::vector<int64_t>> GetInputShapes() const = 0;

    // 获取模型所有输入节点的名称列表
    // 返回: 输入节点名称，顺序与 GetInputShapes() 返回的形状对应
    virtual std::vector<std::string> GetInputNames() const = 0;

    // 获取模型所有输出节点的名称列表
    // 返回: 输出节点名称
    virtual std::vector<std::string> GetOutputNames() const = 0;

    // 获取模型所有输出张量的形状
    // 返回: 每个输出张量的形状，例如 {{1,1000}} 表示一个 batch=1、1000类别的输出
    //       返回顺序与 GetOutputNames() 一致
    virtual std::vector<std::vector<int64_t>> GetOutputShapes() const = 0;

    // 虚析构函数，确保派生类对象通过基类指针删除时正确释放资源
    virtual ~InferenceBackend() = default;
};

}  // namespace inference