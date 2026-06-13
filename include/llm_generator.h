#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace inference {

// LLM 生成参数，控制文本生成的行为
struct GenerationConfig {
    // 最大生成 token 数
    int max_tokens = 512;
    // 温度（>0 增加随机性，=0 贪心解码）
    float temperature = 0.8f;
    // 核采样概率阈值
    float top_p = 0.95f;
    // 重复惩罚系数（>1 惩罚重复，=1 不惩罚）
    float repeat_penalty = 1.1f;
    // 随机种子（-1 表示随机）
    int seed = -1;
};

// LLM 文本生成器，封装 llama.cpp 提供文本生成能力
// 使用 PImpl 模式隐藏 llama.h 依赖，头文件无需引入 llama.cpp 头文件
// 支持 GGUF 格式模型，Apple Silicon 上自动启用 Metal GPU 加速
class LlamaGenerator {
public:
    // 构造 LlamaGenerator 对象
    LlamaGenerator();

    // 析构函数，释放 llama.cpp 资源（模型、上下文、后端）
    ~LlamaGenerator();

    // 禁止拷贝和赋值（llama.cpp 资源不可共享）
    LlamaGenerator(const LlamaGenerator&) = delete;
    LlamaGenerator& operator=(const LlamaGenerator&) = delete;

    // 加载 GGUF 模型文件
    // model_path: GGUF 模型文件路径
    // 流程：llama_backend_init → llama_model_load_from_file → llama_init_from_model
    // Apple Silicon 上自动设置 n_gpu_layers=-1 启用 Metal 加速
    // 返回: 加载成功返回 true，文件不存在或加载失败返回 false
    bool LoadModel(const std::string& model_path);

    // 生成文本（非流式）
    // prompt: 输入提示文本
    // config: 生成参数（温度、top_p、重复惩罚等）
    // 返回: 生成的文本（不含 prompt）
    std::string Generate(const std::string& prompt,
                         const GenerationConfig& config = GenerationConfig());

    // 生成文本（流式，每生成一个 token 调用一次 callback）
    // prompt: 输入提示文本
    // callback: 每生成一个 token 时调用的回调函数，参数为 token 对应的文本片段
    // config: 生成参数
    void GenerateStream(const std::string& prompt,
                        std::function<void(const std::string&)> callback,
                        const GenerationConfig& config = GenerationConfig());

    // 获取上下文长度（即模型可处理的最大 token 数）
    int GetContextSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

}  // namespace inference
