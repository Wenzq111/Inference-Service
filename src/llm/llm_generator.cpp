#include "llm_generator.h"
#include "logger.h"
#include "timer.h"

#include <llama.h>

#include <filesystem>
#include <thread>
#include <vector>

namespace inference {

// PImpl 内部结构，持有所有 llama.cpp 相关对象
struct LlamaGenerator::Impl {
    // llama.cpp 模型对象，LoadModel 时创建，析构时 llama_model_free
    llama_model* model = nullptr;
    // llama.cpp 推理上下文，LoadModel 时创建，析构时 llama_free
    llama_context* ctx = nullptr;
    // 模型词表指针，从 model 获取，无需手动释放
    const llama_vocab* vocab = nullptr;
    // 标记模型是否已成功加载
    bool model_loaded = false;
    // 标记 llama 后端是否已初始化，析构时用于判断是否需要 llama_backend_free
    bool backend_initialized = false;

    // 构建采样器链，根据 GenerationConfig 配置采样参数
    // 返回: 配置好的 sampler chain，调用方负责 llama_sampler_free
    llama_sampler* BuildSamplerChain(const GenerationConfig& config) const {
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler* chain = llama_sampler_chain_init(sparams);

        // 重复惩罚（penalty_last_n=-1 表示使用整个上下文）
        if (config.repeat_penalty != 1.0f) {
            llama_sampler_chain_add(
                chain,
                llama_sampler_init_penalties(
                    -1, config.repeat_penalty, 0.0f, 0.0f));
        }

        // Top-K 预过滤（加速 Top-P 采样）
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));

        // Top-P 核采样
        if (config.top_p < 1.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(config.top_p, 1));
        }

        // 温度采样
        if (config.temperature > 0.0f) {
            llama_sampler_chain_add(
                chain, llama_sampler_init_temp(config.temperature));
            // 温度 >0 时使用分布采样（随机性）
            uint32_t seed = config.seed >= 0
                                ? static_cast<uint32_t>(config.seed)
                                : LLAMA_DEFAULT_SEED;
            llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
        } else {
            // 温度 =0 时使用贪心解码（确定性）
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        }

        return chain;
    }

    // 将 token 转为文本片段
    // vocab: 模型词表
    // token: 待转换的 token id
    // 返回: token 对应的 UTF-8 文本
    static std::string TokenToText(const llama_vocab* vocab, llama_token token) {
        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            // 缓冲区不足，动态分配
            std::vector<char> large_buf(static_cast<size_t>(-n));
            llama_token_to_piece(vocab, token, large_buf.data(),
                                 static_cast<int32_t>(large_buf.size()), 0, true);
            return std::string(large_buf.data(),
                               static_cast<size_t>(-n));
        }
        return std::string(buf, static_cast<size_t>(n));
    }

    // 将 prompt 包装为对话模板格式
    // 使用 llama_chat_apply_template 自动读取模型自带的 chat_template
    // 返回: 包装后的完整字符串，如 "<s> </think>\nWhat is 2+2?</s>  user\n"
    std::string ApplyChatTemplate(const std::string& prompt) const {
        if (!model) {
            return prompt;
        }

        llama_chat_message msg;
        msg.role = "user";
        msg.content = prompt.c_str();

        int32_t len = llama_chat_apply_template(
            nullptr, &msg, 1, true, nullptr, 0);
        if (len <= 0) {
            Logger::Warning("Impl::ApplyChatTemplate: template not available, using raw prompt");
            return prompt;
        }

        std::vector<char> buf(static_cast<size_t>(len) + 1);
        len = llama_chat_apply_template(
            nullptr, &msg, 1, true, buf.data(), static_cast<int32_t>(buf.size()));
        if (len <= 0) {
            Logger::Warning("Impl::ApplyChatTemplate: template application failed, using raw prompt");
            return prompt;
        }

        return std::string(buf.data(), static_cast<size_t>(len));
    }

    // 释放所有 llama.cpp 资源
    void Cleanup() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        vocab = nullptr;
        model_loaded = false;

        if (backend_initialized) {
            llama_backend_free();
            backend_initialized = false;
        }
    }
};

// 构造 LlamaGenerator 对象，创建 Impl 实例
LlamaGenerator::LlamaGenerator()
    : pImpl_(std::make_unique<Impl>()) {}

// 析构函数，释放所有 llama.cpp 资源
LlamaGenerator::~LlamaGenerator() {
    pImpl_->Cleanup();
}

// 加载 GGUF 模型文件并初始化推理上下文
// 流程：
//   1. 首次调用时 llama_backend_init()
//   2. 检查文件存在性
//   3. 配置模型参数（n_gpu_layers: Apple Silicon=-1 启用 Metal，其他=0 纯 CPU）
//   4. llama_model_load_from_file 加载模型
//   5. 获取词表指针
//   6. 配置上下文参数（n_ctx=2048, n_threads=硬件并发数）
//   7. llama_init_from_model 创建推理上下文
// 返回: 加载成功返回 true，失败返回 false
bool LlamaGenerator::LoadModel(const std::string& model_path) {
    if (pImpl_->model_loaded) {
        Logger::Warning("LlamaGenerator::LoadModel: model already loaded, cleanup first");
        pImpl_->Cleanup();
    }

    // 初始化 llama 后端（全局一次）
    if (!pImpl_->backend_initialized) {
        llama_backend_init();
        pImpl_->backend_initialized = true;
        Logger::Info("LlamaGenerator::LoadModel: llama backend initialized");
    }

    // 检查文件存在性
    if (!std::filesystem::exists(model_path)) {
        Logger::Error("LlamaGenerator::LoadModel: model file not found: " + model_path);
        return false;
    }

    // 配置模型参数
    auto mparams = llama_model_default_params();
#if defined(__APPLE__) && defined(__arm64__)
    // Apple Silicon 上自动启用 Metal GPU 加速，-1 表示所有层都放到 GPU
    mparams.n_gpu_layers = -1;
    Logger::Info("LlamaGenerator::LoadModel: Metal GPU acceleration enabled (n_gpu_layers=-1)");
#else
    mparams.n_gpu_layers = 0;
    Logger::Info("LlamaGenerator::LoadModel: using CPU inference");
#endif

    // 加载模型
    pImpl_->model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!pImpl_->model) {
        Logger::Error("LlamaGenerator::LoadModel: failed to load model from " + model_path);
        return false;
    }

    // 获取词表
    pImpl_->vocab = llama_model_get_vocab(pImpl_->model);
    if (!pImpl_->vocab) {
        Logger::Error("LlamaGenerator::LoadModel: failed to get vocab from model");
        llama_model_free(pImpl_->model);
        pImpl_->model = nullptr;
        return false;
    }

    // 配置上下文参数
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    int32_t n_threads = static_cast<int32_t>(std::thread::hardware_concurrency());
    if (n_threads <= 0) {
        n_threads = 4;
    }
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    // 创建推理上下文
    pImpl_->ctx = llama_init_from_model(pImpl_->model, cparams);
    if (!pImpl_->ctx) {
        Logger::Error("LlamaGenerator::LoadModel: failed to create context");
        llama_model_free(pImpl_->model);
        pImpl_->model = nullptr;
        pImpl_->vocab = nullptr;
        return false;
    }

    pImpl_->model_loaded = true;
    Logger::Info("LlamaGenerator::LoadModel: model loaded from " + model_path +
                 ", context_size=" + std::to_string(llama_n_ctx(pImpl_->ctx)) +
                 ", n_threads=" + std::to_string(n_threads));

    return true;
}

// 生成文本（非流式）
// 流程：
//   1. 前置检查（模型已加载、prompt 非空）
//   2. 分词：llama_tokenize 将 prompt 转为 token ids
//   3. 构建采样器链（penalties → top-k → top-p → temp → dist/greedy）
//   4. 分配 batch，处理 prompt tokens（llama_decode）
//   5. 循环生成：llama_decode → llama_sampler_sample → llama_token_to_piece
//   6. 遇到 EOS 或达到 max_tokens 时停止
//   7. 释放 sampler 和 batch
// 返回: 生成的文本（不含 prompt）
std::string LlamaGenerator::Generate(const std::string& prompt,
                                      const GenerationConfig& config) {
    if (!pImpl_->model_loaded) {
        Logger::Error("LlamaGenerator::Generate: model not loaded");
        return "";
    }

    if (prompt.empty()) {
        Logger::Error("LlamaGenerator::Generate: prompt is empty");
        return "";
    }

    // 清空 KV Cache，确保每次 Generate 调用从干净状态开始
    // 不清空时，上一次生成的 token 仍留在 KV Cache 中，导致 llama_decode 报
    // "inconsistent sequence positions" 错误（新 prompt 从 pos=0 开始但 Cache 中有旧数据）
    // llama.cpp 9620 API: llama_get_memory 获取内存对象，llama_memory_clear(data=true) 清空所有序列数据
    llama_memory_clear(llama_get_memory(pImpl_->ctx), true);

    Timer timer;
    timer.Start();

    // 应用对话模板，将 prompt 包装为模型期望的对话格式
    std::string formatted = pImpl_->ApplyChatTemplate(prompt);

    // 分词：将模板包装后的文本转为 token ids
    int32_t n_tokens_max = static_cast<int32_t>(llama_n_ctx(pImpl_->ctx));
    std::vector<llama_token> tokens(static_cast<size_t>(n_tokens_max));
    int32_t n_tokens = llama_tokenize(
        pImpl_->vocab, formatted.c_str(), static_cast<int32_t>(formatted.size()),
        tokens.data(), n_tokens_max, true, false);
    if (n_tokens < 0) {
        Logger::Error("LlamaGenerator::Generate: tokenization failed, prompt too long");
        return "";
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    // 构建采样器链
    llama_sampler* sampler = pImpl_->BuildSamplerChain(config);

    // 分配 batch
    int32_t max_batch = std::max(n_tokens, 1);
    llama_batch batch = llama_batch_init(max_batch, 0, 1);

    // 处理 prompt tokens
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = n_tokens;
    // 只有最后一个 token 需要输出 logits 用于采样
    if (n_tokens > 0) {
        batch.logits[n_tokens - 1] = true;
    }

    int32_t ret = llama_decode(pImpl_->ctx, batch);
    if (ret != 0) {
        Logger::Error("LlamaGenerator::Generate: llama_decode failed for prompt with code " +
                      std::to_string(ret));
        llama_sampler_free(sampler);
        llama_batch_free(batch);
        return "";
    }

    // 循环生成
    std::string result;
    int32_t n_cur = n_tokens;
    llama_token eos_token = llama_vocab_eos(pImpl_->vocab);

    for (int i = 0; i < config.max_tokens; ++i) {
        // 采样下一个 token
        llama_token new_token = llama_sampler_sample(sampler, pImpl_->ctx, -1);

        // 检查是否到达结束标记
        if (new_token == eos_token) {
            break;
        }

        // 将 token 转为文本
        result += Impl::TokenToText(pImpl_->vocab, new_token);

        // 将新 token 送入 batch 进行下一轮推理
        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        ret = llama_decode(pImpl_->ctx, batch);
        if (ret != 0) {
            Logger::Error("LlamaGenerator::Generate: llama_decode failed during generation");
            break;
        }

        n_cur++;

        // 超出上下文长度时停止
        if (n_cur >= static_cast<int32_t>(llama_n_ctx(pImpl_->ctx))) {
            Logger::Warning("LlamaGenerator::Generate: context length exceeded");
            break;
        }
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    timer.Stop();
    Logger::Info("LlamaGenerator::Generate: generated " +
                 std::to_string(result.size()) + " chars in " +
                 std::to_string(timer.ElapsedMilliseconds()) + " ms");

    return result;
}

// 生成文本（流式），每生成一个 token 立即通过 callback 输出
// 逻辑与 Generate 相同，区别在于每个 token 立即回调而非最后拼接
void LlamaGenerator::GenerateStream(const std::string& prompt,
                                     std::function<void(const std::string&)> callback,
                                     const GenerationConfig& config) {
    if (!pImpl_->model_loaded) {
        Logger::Error("LlamaGenerator::GenerateStream: model not loaded");
        return;
    }

    if (prompt.empty()) {
        Logger::Error("LlamaGenerator::GenerateStream: prompt is empty");
        return;
    }

    if (!callback) {
        Logger::Error("LlamaGenerator::GenerateStream: callback is null");
        return;
    }

    // 清空 KV Cache，确保每次 GenerateStream 调用从干净状态开始
    llama_memory_clear(llama_get_memory(pImpl_->ctx), true);

    Timer timer;
    timer.Start();

    // 应用对话模板
    std::string formatted = pImpl_->ApplyChatTemplate(prompt);

    // 分词
    int32_t n_tokens_max = static_cast<int32_t>(llama_n_ctx(pImpl_->ctx));
    std::vector<llama_token> tokens(static_cast<size_t>(n_tokens_max));
    int32_t n_tokens = llama_tokenize(
        pImpl_->vocab, formatted.c_str(), static_cast<int32_t>(formatted.size()),
        tokens.data(), n_tokens_max, true, false);
    if (n_tokens < 0) {
        Logger::Error("LlamaGenerator::GenerateStream: tokenization failed");
        return;
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    // 构建采样器链
    llama_sampler* sampler = pImpl_->BuildSamplerChain(config);

    // 分配 batch
    int32_t max_batch = std::max(n_tokens, 1);
    llama_batch batch = llama_batch_init(max_batch, 0, 1);

    // 处理 prompt tokens
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = n_tokens;
    if (n_tokens > 0) {
        batch.logits[n_tokens - 1] = true;
    }

    int32_t ret = llama_decode(pImpl_->ctx, batch);
    if (ret != 0) {
        Logger::Error("LlamaGenerator::GenerateStream: llama_decode failed for prompt");
        llama_sampler_free(sampler);
        llama_batch_free(batch);
        return;
    }

    // 循环生成，每生成一个 token 立即回调
    int32_t n_cur = n_tokens;
    llama_token eos_token = llama_vocab_eos(pImpl_->vocab);

    for (int i = 0; i < config.max_tokens; ++i) {
        llama_token new_token = llama_sampler_sample(sampler, pImpl_->ctx, -1);

        if (new_token == eos_token) {
            break;
        }

        // 立即回调输出当前 token 的文本
        std::string piece = Impl::TokenToText(pImpl_->vocab, new_token);
        callback(piece);

        // 将新 token 送入 batch
        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        ret = llama_decode(pImpl_->ctx, batch);
        if (ret != 0) {
            Logger::Error("LlamaGenerator::GenerateStream: llama_decode failed during generation");
            break;
        }

        n_cur++;

        if (n_cur >= static_cast<int32_t>(llama_n_ctx(pImpl_->ctx))) {
            Logger::Warning("LlamaGenerator::GenerateStream: context length exceeded");
            break;
        }
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    timer.Stop();
    Logger::Info("LlamaGenerator::GenerateStream: completed in " +
                 std::to_string(timer.ElapsedMilliseconds()) + " ms");
}

// 获取上下文长度
int LlamaGenerator::GetContextSize() const {
    if (!pImpl_->model_loaded || !pImpl_->ctx) {
        return 0;
    }
    return static_cast<int>(llama_n_ctx(pImpl_->ctx));
}

}  // namespace inference
