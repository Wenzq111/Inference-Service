# llama.cpp API 详细指南

> 本文档对应项目 M7（LLM 文本生成模块），基于 llama.cpp 版本 9620，详细讲解 `LlamaGenerator` 中使用的每一个 llama.cpp API 函数，并按推理流程逐步代码走读。

创建日期：2026-06-14

---

## 目录

1. [概述与架构](#1-概述与架构)
2. [后端初始化与释放](#2-后端初始化与释放)
3. [模型加载](#3-模型加载)
4. [词表操作](#4-词表操作)
5. [上下文创建](#5-上下文创建)
6. [分词（Tokenization）](#6-分词tokenization)
7. [Batch 管理](#7-batch-管理)
8. [推理（Decode）](#8-推理decode)
9. [采样器链（Sampler Chain）](#9-采样器链sampler-chain)
10. [Token 转文本](#10-token-转文本)
11. [完整推理流程代码走读](#11-完整推理流程代码走读)

---

## 1. 概述与架构

### 1.1 llama.cpp 简介

llama.cpp 是一个高性能 LLM 推理库，支持 GGUF 格式模型，提供：

- **多种量化格式**：Q4_0、Q5_1、Q8_0、FP16 等
- **GPU 加速**：Apple Metal、CUDA、Vulkan、SYCL
- **采样器链**：灵活的采样策略组合（Top-K、Top-P、Temperature、重复惩罚等）
- **批量推理**：支持多 token 并行处理

### 1.2 M7 架构

```
LlamaGenerator (公共接口)
    └── Impl (PImpl 内部结构)
        ├── llama_model*      — 模型对象
        ├── llama_context*    — 推理上下文
        ├── const llama_vocab* — 词表指针
        └── BuildSamplerChain() — 构建采样器链
```

### 1.3 API 版本说明

本文档基于 llama.cpp **b9620** 版本。关键 API 变更：

| 旧 API（已废弃） | 新 API（本项目使用） | 说明 |
|---|---|---|
| `llama_load_model_from_file` | `llama_model_load_from_file` | 模型加载函数重命名 |
| `llama_new_context_with_model` | `llama_init_from_model` | 上下文创建函数重命名 |
| `llama_sample_*` 系列 | `llama_sampler_*` 系列 | 采样 API 从独立函数改为链式架构 |
| `llama_eval` | `llama_decode` + `llama_batch` | 推理 API 改为 batch 模式 |

---

## 2. 后端初始化与释放

### 2.1 llama_backend_init

```c
void llama_backend_init(void);
```

**功能**：初始化 llama.cpp 后端，必须在任何其他 llama.cpp API 之前调用。在 Apple Silicon 上，此函数会自动检测并注册 Metal GPU 后端。

**在项目中的使用**（`src/llm/llm_generator.cpp:129`）：

```cpp
if (!pImpl_->backend_initialized) {
    llama_backend_init();
    pImpl_->backend_initialized = true;
}
```

**要点**：
- 全局只需调用一次，重复调用会导致未定义行为
- 使用 `backend_initialized` 标志防止重复初始化
- 必须在 `llama_model_load_from_file` 之前调用

### 2.2 llama_backend_free

```c
void llama_backend_free(void);
```

**功能**：释放后端资源，在所有 llama.cpp 对象（model、context）释放后调用。

**在项目中的使用**（`src/llm/llm_generator.cpp:96`）：

```cpp
if (backend_initialized) {
    llama_backend_free();
    backend_initialized = false;
}
```

**要点**：
- 必须在 `llama_model_free` 和 `llama_free` 之后调用
- 释放后不应再使用任何 llama.cpp API
- 使用 `backend_initialized` 标志确保只释放一次

### 2.3 生命周期顺序

```
llama_backend_init()              ← 最先
  → llama_model_load_from_file()
    → llama_init_from_model()
      → llama_decode() / llama_sampler_*()
        → llama_sampler_free()
      → llama_free()              ← 先释放 context
    → llama_model_free()          ← 再释放 model
llama_backend_free()              ← 最后
```

---

## 3. 模型加载

### 3.1 llama_model_default_params

```c
struct llama_model_params llama_model_default_params(void);
```

**功能**：获取模型参数的默认值，返回 `llama_model_params` 结构体。

**关键字段**：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `n_gpu_layers` | int32_t | 0 | 放到 GPU 的层数，-1 表示全部 |
| `split_mode` | enum | LLAMA_SPLIT_MODE_LAYER | 多 GPU 分割模式 |
| `main_gpu` | int32_t | 0 | 主 GPU 索引 |
| `tensor_buft_overrides` | ptr | NULL | 自定义 buffer 类型覆盖 |
| `kv_overrides` | ptr | NULL | 模型 KV 覆盖 |

**在项目中的使用**（`src/llm/llm_generator.cpp:141-149`）：

```cpp
auto mparams = llama_model_default_params();
#if defined(__APPLE__) && defined(__arm64__)
mparams.n_gpu_layers = -1;  // Apple Silicon: 所有层放 Metal GPU
#else
mparams.n_gpu_layers = 0;   // 其他平台: 纯 CPU
#endif
```

**要点**：
- `n_gpu_layers = -1`：在 Apple Silicon 上将所有 transformer 层卸载到 Metal GPU，利用统一内存架构零拷贝访问模型权重
- `n_gpu_layers = 0`：纯 CPU 推理，兼容性最好
- 条件编译 `__APPLE__ && __arm64__` 确保只在 Apple Silicon 上启用 Metal

### 3.2 llama_model_load_from_file

```c
struct llama_model * llama_model_load_from_file(
    const char * path_model,
    struct llama_model_params params
);
```

**功能**：从 GGUF 文件加载模型到内存。

**参数**：
- `path_model`：GGUF 模型文件路径
- `params`：模型参数（GPU 层数等）

**返回值**：
- 成功：指向 `llama_model` 的指针
- 失败：`NULL`

**在项目中的使用**（`src/llm/llm_generator.cpp:152`）：

```cpp
pImpl_->model = llama_model_load_from_file(model_path.c_str(), mparams);
if (!pImpl_->model) {
    Logger::Error("LlamaGenerator::LoadModel: failed to load model from " + model_path);
    return false;
}
```

**要点**：
- 加载前应检查文件是否存在（项目通过 `std::filesystem::exists` 检查）
- 加载失败返回 NULL，必须检查返回值
- GGUF 文件包含模型权重、词表和元信息

### 3.3 llama_model_free

```c
void llama_model_free(struct llama_model * model);
```

**功能**：释放模型对象占用的内存和 GPU 资源。

**在项目中的使用**（`src/llm/llm_generator.cpp:89`）：

```cpp
if (model) {
    llama_model_free(model);
    model = nullptr;
}
```

**要点**：
- 释放前必须先释放所有使用该 model 创建的 context
- 释放后将指针置 nullptr，防止悬空指针

---

## 4. 词表操作

### 4.1 llama_model_get_vocab

```c
const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
```

**功能**：从已加载的模型中获取词表指针。

**返回值**：
- 成功：指向 `llama_vocab` 的常量指针
- 失败（模型未加载）：`NULL`

**在项目中的使用**（`src/llm/llm_generator.cpp:159`）：

```cpp
pImpl_->vocab = llama_model_get_vocab(pImpl_->model);
if (!pImpl_->vocab) {
    Logger::Error("LlamaGenerator::LoadModel: failed to get vocab from model");
    llama_model_free(pImpl_->model);
    pImpl_->model = nullptr;
    return false;
}
```

**要点**：
- 词表指针的生命周期与 model 相同，model 释放后 vocab 不可用
- 不需要手动释放，随 model 一起释放
- 后续分词和 token 转文本都需要 vocab

### 4.2 llama_vocab_eos

```c
llama_token llama_vocab_eos(const struct llama_vocab * vocab);
```

**功能**：获取词表的 EOS（End of Sequence）token id。

**在项目中的使用**（`src/llm/llm_generator.cpp:266`）：

```cpp
llama_token eos_token = llama_vocab_eos(pImpl_->vocab);
// ...
if (new_token == eos_token) {
    break;  // 生成结束
}
```

**要点**：
- 不同模型的 EOS token id 可能不同（如 LLaMA 为 2，Qwen 为 151643）
- 采样到 EOS 时应停止生成
- 某些模型可能有多个结束标记（如 `<|im_end|>`），需特殊处理

---

## 5. 上下文创建

### 5.1 llama_context_default_params

```c
struct llama_context_params llama_context_default_params(void);
```

**功能**：获取上下文参数的默认值。

**关键字段**：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `n_ctx` | uint32_t | 512 | 上下文窗口大小（最大 token 数） |
| `n_batch` | uint32_t | 2048 | 批处理大小（prompt 阶段一次处理的 token 数） |
| `n_threads` | int32_t | -1 | 推理线程数（-1 = 自动） |
| `n_threads_batch` | int32_t | -1 | 批处理阶段线程数 |
| `rope_freq_base` | float | 0.0 | RoPE 基频（0 = 使用模型默认值） |
| `rope_freq_scale` | float | 0.0 | RoPE 缩放因子（0 = 使用模型默认值） |
| `embeddings` | bool | false | 是否计算 embedding |

**在项目中的使用**（`src/llm/llm_generator.cpp:168-176`）：

```cpp
auto cparams = llama_context_default_params();
cparams.n_ctx = 2048;
cparams.n_batch = 512;
int32_t n_threads = static_cast<int32_t>(std::thread::hardware_concurrency());
if (n_threads <= 0) {
    n_threads = 4;
}
cparams.n_threads = n_threads;
cparams.n_threads_batch = n_threads;
```

**要点**：
- `n_ctx = 2048`：上下文窗口大小，决定模型能处理的最大 token 数（含 prompt + 生成）
- `n_batch = 512`：prompt 处理时一次最多处理的 token 数，影响首 token 延迟
- `n_threads`：使用 `std::thread::hardware_concurrency()` 获取硬件并发数，fallback 到 4
- `n_threads` 和 `n_threads_batch` 设为相同值，避免不同阶段线程数变化

### 5.2 llama_init_from_model

```c
struct llama_context * llama_init_from_model(
    const struct llama_model * model,
    struct llama_context_params params
);
```

**功能**：基于已加载的模型创建推理上下文。

**参数**：
- `model`：已加载的模型指针（`llama_model_load_from_file` 返回值）
- `params`：上下文参数

**返回值**：
- 成功：指向 `llama_context` 的指针
- 失败：`NULL`

**在项目中的使用**（`src/llm/llm_generator.cpp:179`）：

```cpp
pImpl_->ctx = llama_init_from_model(pImpl_->model, cparams);
if (!pImpl_->ctx) {
    Logger::Error("LlamaGenerator::LoadModel: failed to create context");
    llama_model_free(pImpl_->model);
    pImpl_->model = nullptr;
    pImpl_->vocab = nullptr;
    return false;
}
```

**要点**：
- context 依赖于 model，model 被释放后 context 不可用
- context 创建时会分配 KV cache 内存，大小 = `n_ctx × n_layers × 2 × sizeof(float16)`
- 创建失败时需要清理已分配的 model

### 5.3 llama_free

```c
void llama_free(struct llama_context * ctx);
```

**功能**：释放推理上下文，包括 KV cache。

**在项目中的使用**（`src/llm/llm_generator.cpp:85`）：

```cpp
if (ctx) {
    llama_free(ctx);
    ctx = nullptr;
}
```

**要点**：
- 必须在 `llama_model_free` 之前释放 context
- 释放后 context 不再可用，必须置 nullptr

### 5.4 llama_n_ctx

```c
uint32_t llama_n_ctx(const struct llama_context * ctx);
```

**功能**：获取上下文的最大 token 容量。

**在项目中的使用**（`src/llm/llm_generator.cpp:190, 222, 297`）：

```cpp
// 日志输出
llama_n_ctx(pImpl_->ctx)

// 分词时确定最大 token 数
int32_t n_tokens_max = static_cast<int32_t>(llama_n_ctx(pImpl_->ctx));

// 生成时检查是否超出上下文长度
if (n_cur >= static_cast<int32_t>(llama_n_ctx(pImpl_->ctx))) {
    break;
}
```

**要点**：
- 返回值等于创建 context 时设置的 `n_ctx`
- 用于限制 prompt 长度和生成 token 数
- prompt tokens + generated tokens 之和不得超过此值

---

## 6. 分词（Tokenization）

### 6.1 llama_tokenize

```c
int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_max_tokens,
    bool add_bos,
    bool special_tokens
);
```

**功能**：将文本字符串转换为 token id 数组。

**参数**：
- `vocab`：词表指针
- `text`：待分词的文本
- `text_len`：文本长度
- `tokens`：输出 token id 数组（调用方分配）
- `n_max_tokens`：输出数组最大容量
- `add_bos`：是否添加 BOS（Beginning of Sequence）token
- `special_tokens`：是否允许特殊 token（如 `<|im_start|>`）

**返回值**：
- 成功：实际写入的 token 数量（≥ 0）
- 失败（缓冲区不足）：所需的负值（`-(n+1)`）

**在项目中的使用**（`src/llm/llm_generator.cpp:222-231`）：

```cpp
int32_t n_tokens_max = static_cast<int32_t>(llama_n_ctx(pImpl_->ctx));
std::vector<llama_token> tokens(static_cast<size_t>(n_tokens_max));
int32_t n_tokens = llama_tokenize(
    pImpl_->vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
    tokens.data(), n_tokens_max, true, false);
if (n_tokens < 0) {
    Logger::Error("LlamaGenerator::Generate: tokenization failed, prompt too long");
    return "";
}
tokens.resize(static_cast<size_t>(n_tokens));
```

**要点**：
- `add_bos = true`：自动在开头添加 BOS token，大多数对话模型需要
- `special_tokens = false`：不解析特殊 token，避免误将 prompt 中的 `<` 开头文本当作特殊标记
- 先用 `n_ctx` 大小分配缓冲区，保证大多数情况足够
- 返回负值表示缓冲区不足，负值的绝对值减 1 是所需大小
- 分词后 `resize` 裁掉未使用的空间

---

## 7. Batch 管理

### 7.1 llama_batch 结构体

```c
struct llama_batch {
    llama_token * token;       // token id 数组
    int32_t     * pos;         // 每个 token 的位置
    int32_t     * n_seq_id;    // 每个 token 的序列 id 数量
    llama_seq_id** seq_id;     // 每个 token 的序列 id 数组
    int8_t      * logits;      // 是否计算 logits（0/1）
    int32_t       n_tokens;    // 当前 batch 中的 token 数
};
```

**功能**：描述一次推理的输入批次，包含 token ids、位置、序列信息和 logits 标记。

**关键字段说明**：
- `token[i]`：第 i 个 token 的 id
- `pos[i]`：第 i 个 token 在序列中的位置（从 0 开始）
- `n_seq_id[i]`：第 i 个 token 属于多少个序列（通常为 1）
- `seq_id[i][j]`：第 i 个 token 的第 j 个序列 id（通常 `seq_id[i][0] = 0`）
- `logits[i]`：是否为第 i 个 token 计算 logits（只有需要采样的 token 才设为 true）
- `n_tokens`：本次 batch 中实际使用的 token 数量

### 7.2 llama_batch_init

```c
struct llama_batch llama_batch_init(
    int32_t n_tokens_alloc,
    int32_t embd,
    int32_t n_seq_max
);
```

**功能**：分配 batch 所需的内存。

**参数**：
- `n_tokens_alloc`：预分配的最大 token 数
- `embd`：是否使用 embedding 模式（0 = token id 模式，1 = embedding 模式）
- `n_seq_max`：每个 token 的最大序列数

**在项目中的使用**（`src/llm/llm_generator.cpp:238`）：

```cpp
int32_t max_batch = std::max(n_tokens, 1);
llama_batch batch = llama_batch_init(max_batch, 0, 1);
```

**要点**：
- `embd = 0`：使用 token id 模式（大多数场景）
- `n_seq_max = 1`：单序列生成（非并行对话）
- 分配大小取 `max(n_tokens, 1)`，确保至少有 1 个 token 的空间
- batch 用于 prompt 处理和逐 token 生成两个阶段

### 7.3 llama_batch_free

```c
void llama_batch_free(struct llama_batch batch);
```

**功能**：释放 batch 分配的内存。

**在项目中的使用**（`src/llm/llm_generator.cpp:304`）：

```cpp
llama_batch_free(batch);
```

**要点**：
- 必须在 `llama_decode` 完成后释放
- 每次推理循环结束后都应释放，或复用 batch 直到生成结束

### 7.4 Batch 填充方式

#### Prompt 阶段（批量填充）

```cpp
for (int32_t i = 0; i < n_tokens; ++i) {
    batch.token[i] = tokens[i];       // 第 i 个 prompt token
    batch.pos[i] = i;                 // 位置: 0, 1, 2, ...
    batch.n_seq_id[i] = 1;            // 属于 1 个序列
    batch.seq_id[i][0] = 0;           // 序列 id = 0
    batch.logits[i] = false;          // 不需要 logits
}
batch.n_tokens = n_tokens;
batch.logits[n_tokens - 1] = true;   // 只有最后一个 token 需要 logits
```

**说明**：
- Prompt 所有 token 一次性填入 batch
- 只有**最后一个 token** 需要计算 logits（用于采样新 token）
- 中间 token 的 logits 不需要，设为 `false` 可节省计算

#### 生成阶段（逐 token 填充）

```cpp
batch.n_tokens = 1;
batch.token[0] = new_token;          // 刚采样出的新 token
batch.pos[0] = n_cur;                // 当前位置
batch.n_seq_id[0] = 1;
batch.seq_id[0][0] = 0;
batch.logits[0] = true;              // 每个生成的 token 都需要 logits
```

**说明**：
- 每次只填入 1 个新 token
- `pos` 必须递增（`n_cur` 从 prompt 长度开始递增）
- 每个生成 token 都需要 logits 用于下一次采样

---

## 8. 推理（Decode）

### 8.1 llama_decode

```c
int32_t llama_decode(
    struct llama_context * ctx,
    struct llama_batch batch
);
```

**功能**：执行一次前向推理，将 batch 中的 token 送入模型计算。

**参数**：
- `ctx`：推理上下文
- `batch`：输入 batch（包含 token ids、位置、序列信息）

**返回值**：
- `0`：成功
- `1`：部分成功（需要继续调用，如 KV cache 满时换页）
- `< 0`：错误

**在项目中的使用**（`src/llm/llm_generator.cpp:254, 288`）：

```cpp
// Prompt 阶段
int32_t ret = llama_decode(pImpl_->ctx, batch);
if (ret != 0) {
    Logger::Error("LlamaGenerator::Generate: llama_decode failed for prompt with code " +
                  std::to_string(ret));
    llama_sampler_free(sampler);
    llama_batch_free(batch);
    return "";
}

// 生成阶段
ret = llama_decode(pImpl_->ctx, batch);
if (ret != 0) {
    Logger::Error("LlamaGenerator::Generate: llama_decode failed during generation");
    break;
}
```

**要点**：
- Prompt 阶段：一次 decode 处理所有 prompt tokens
- 生成阶段：每次 decode 只处理 1 个新 token
- 返回值 != 0 时应停止生成并清理资源
- decode 后 logits 存储在 context 内部，通过 `llama_sampler_sample` 访问

---

## 9. 采样器链（Sampler Chain）

### 9.1 采样器链架构

llama.cpp b9620 采用**链式采样器**架构，每个采样器是一个独立的过滤/变换步骤，按顺序串行执行：

```
logits → penalties → top-k → top-p → temperature → dist/greedy → token
```

### 9.2 llama_sampler_chain_default_params

```c
struct llama_sampler_chain_params llama_sampler_chain_default_params(void);
```

**功能**：获取采样器链参数的默认值。

**在项目中的使用**（`src/llm/llm_generator.cpp:28`）：

```cpp
auto sparams = llama_sampler_chain_default_params();
llama_sampler* chain = llama_sampler_chain_init(sparams);
```

### 9.3 llama_sampler_chain_init

```c
struct llama_sampler * llama_sampler_chain_init(
    struct llama_sampler_chain_params params
);
```

**功能**：创建一个空的采样器链。

**返回值**：采样器链指针，后续通过 `llama_sampler_chain_add` 添加采样器。

### 9.4 llama_sampler_chain_add

```c
void llama_sampler_chain_add(
    struct llama_sampler * chain,
    struct llama_sampler * sampler
);
```

**功能**：向采样器链末尾添加一个采样器。采样器按添加顺序执行。

**要点**：
- 采样器的添加顺序**非常关键**，不同顺序产生不同的采样结果
- 本项目的顺序：penalties → top-k → top-p → temperature → dist/greedy

### 9.5 llama_sampler_init_penalties

```c
struct llama_sampler * llama_sampler_init_penalties(
    int32_t penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present
);
```

**功能**：初始化重复惩罚采样器。

**参数**：
- `penalty_last_n`：考虑最近 N 个 token 进行惩罚（-1 = 使用整个上下文）
- `penalty_repeat`：重复惩罚系数（>1 惩罚重复，=1 不惩罚）
- `penalty_freq`：频率惩罚系数
- `penalty_present`：存在惩罚系数

**在项目中的使用**（`src/llm/llm_generator.cpp:33-37`）：

```cpp
if (config.repeat_penalty != 1.0f) {
    llama_sampler_chain_add(
        chain,
        llama_sampler_init_penalties(
            -1, config.repeat_penalty, 0.0f, 0.0f));
}
```

**要点**：
- 只启用重复惩罚（`penalty_repeat`），频率和存在惩罚设为 0
- `penalty_last_n = -1`：对整个上下文进行惩罚检查
- `repeat_penalty = 1.0` 时不添加此采样器（跳过惩罚步骤）

### 9.6 llama_sampler_init_top_k

```c
struct llama_sampler * llama_sampler_init_top_k(int32_t k);
```

**功能**：初始化 Top-K 采样器，只保留概率最高的 K 个 token。

**在项目中的使用**（`src/llm/llm_generator.cpp:40`）：

```cpp
llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
```

**要点**：
- `k = 40`：保留概率最高的 40 个 token
- Top-K 在 Top-P 之前执行，先粗筛再精筛，提高采样效率
- 无条件添加（不过滤任何条件）

### 9.7 llama_sampler_init_top_p

```c
struct llama_sampler * llama_sampler_init_top_p(
    float p,
    size_t min_keep
);
```

**功能**：初始化 Top-P（核采样）采样器，保留累计概率达到 P 的最小 token 集合。

**参数**：
- `p`：累计概率阈值（0~1）
- `min_keep`：最少保留的 token 数

**在项目中的使用**（`src/llm/llm_generator.cpp:44`）：

```cpp
if (config.top_p < 1.0f) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(config.top_p, 1));
}
```

**要点**：
- `top_p = 0.95`：保留累计概率达到 95% 的 token
- `min_keep = 1`：至少保留 1 个 token
- `top_p = 1.0` 时跳过此步骤（不过滤）

### 9.8 llama_sampler_init_temp

```c
struct llama_sampler * llama_sampler_init_temp(float t);
```

**功能**：初始化温度采样器，调整 logits 分布的尖锐程度。

**在项目中的使用**（`src/llm/llm_generator.cpp:49`）：

```cpp
if (config.temperature > 0.0f) {
    llama_sampler_chain_add(
        chain, llama_sampler_init_temp(config.temperature));
}
```

**要点**：
- `temperature > 1.0`：分布更平坦，输出更随机
- `temperature = 1.0`：不改变原始分布
- `0 < temperature < 1.0`：分布更尖锐，输出更确定
- `temperature = 0`：贪心解码（不使用温度采样器）

### 9.9 llama_sampler_init_dist

```c
struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
```

**功能**：初始化分布采样器，根据概率分布随机采样。

**在项目中的使用**（`src/llm/llm_generator.cpp:52-55`）：

```cpp
uint32_t seed = config.seed >= 0
                    ? static_cast<uint32_t>(config.seed)
                    : LLAMA_DEFAULT_SEED;
llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
```

**要点**：
- `seed >= 0`：固定种子，确保可复现
- `seed < 0`：使用 `LLAMA_DEFAULT_SEED`（随机种子）
- 与 `temperature` 配合使用：温度 > 0 时使用随机分布采样

### 9.10 llama_sampler_init_greedy

```c
struct llama_sampler * llama_sampler_init_greedy(void);
```

**功能**：初始化贪心采样器，始终选择概率最高的 token。

**在项目中的使用**（`src/llm/llm_generator.cpp:58`）：

```cpp
llama_sampler_chain_add(chain, llama_sampler_init_greedy());
```

**要点**：
- `temperature = 0` 时使用贪心解码
- 输出确定性：相同输入永远产生相同输出
- 不需要 seed

### 9.11 llama_sampler_sample

```c
llama_token llama_sampler_sample(
    struct llama_sampler * sampler,
    struct llama_context * ctx,
    int32_t idx
);
```

**功能**：从当前 logits 中采样一个 token。

**参数**：
- `sampler`：采样器链
- `ctx`：推理上下文（包含上次 decode 产生的 logits）
- `idx`：从哪个位置采样（-1 表示最后一个位置）

**返回值**：采样得到的 token id。

**在项目中的使用**（`src/llm/llm_generator.cpp:270`）：

```cpp
llama_token new_token = llama_sampler_sample(sampler, pImpl_->ctx, -1);
```

**要点**：
- `idx = -1`：对最后一个 token 的 logits 进行采样（标准做法）
- 必须在 `llama_decode` 之后调用
- 采样后 sampler 内部状态更新（用于重复惩罚等）

### 9.12 llama_sampler_free

```c
void llama_sampler_free(struct llama_sampler * sampler);
```

**功能**：释放采样器链及其所有子采样器。

**在项目中的使用**（`src/llm/llm_generator.cpp:303`）：

```cpp
llama_sampler_free(sampler);
```

**要点**：
- 一次调用释放整个链（包括所有子采样器）
- 必须在生成结束后释放
- 释放后 sampler 不可用

### 9.13 采样器链完整构建代码走读

```cpp
// src/llm/llm_generator.cpp:27-62
llama_sampler* BuildSamplerChain(const GenerationConfig& config) const {
    // 1. 创建空链
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(sparams);

    // 2. 重复惩罚（可选，repeat_penalty != 1.0 时启用）
    //    对已出现 token 的 logits 施加惩罚，减少重复
    if (config.repeat_penalty != 1.0f) {
        llama_sampler_chain_add(chain,
            llama_sampler_init_penalties(-1, config.repeat_penalty, 0.0f, 0.0f));
    }

    // 3. Top-K 预过滤（固定 k=40）
    //    只保留概率最高的 40 个 token，大幅减少候选集
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));

    // 4. Top-P 核采样（可选，top_p < 1.0 时启用）
    //    在 Top-K 候选集中进一步筛选，保留累计概率达 95% 的 token
    if (config.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(config.top_p, 1));
    }

    // 5. 温度 + 分布采样 / 贪心解码
    //    temperature > 0: 温度缩放 + 随机采样
    //    temperature = 0: 贪心选择最高概率 token
    if (config.temperature > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(config.temperature));
        uint32_t seed = config.seed >= 0
                            ? static_cast<uint32_t>(config.seed)
                            : LLAMA_DEFAULT_SEED;
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }

    return chain;
}
```

---

## 10. Token 转文本

### 10.1 llama_token_to_piece

```c
int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special
);
```

**功能**：将 token id 转换为对应的 UTF-8 文本片段。

**参数**：
- `vocab`：词表指针
- `token`：待转换的 token id
- `buf`：输出缓冲区
- `length`：缓冲区大小
- `lstrip`：从左侧跳过的字节数（通常为 0）
- `special`：是否允许解码特殊 token

**返回值**：
- 成功：写入的字节数（≥ 0）
- 失败（缓冲区不足）：所需的负值

**在项目中的使用**（`src/llm/llm_generator.cpp:68-80`）：

```cpp
static std::string TokenToText(const llama_vocab* vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n < 0) {
        // 缓冲区不足，动态分配
        std::vector<char> large_buf(static_cast<size_t>(-n));
        llama_token_to_piece(vocab, token, large_buf.data(),
                             static_cast<int32_t>(large_buf.size()), 0, true);
        return std::string(large_buf.data(), static_cast<size_t>(-n));
    }
    return std::string(buf, static_cast<size_t>(n));
}
```

**要点**：
- 先用 256 字节栈缓冲区尝试，大多数 token 不超过此大小
- 返回负值时动态分配所需大小的堆缓冲区
- `special = true`：允许解码特殊 token（如 `<s>`、`</s>`）
- `lstrip = 0`：不跳过任何字节
- 返回的字符串可能是不完整的 UTF-8 字符（如多字节中文字符被拆分到多个 token），需要流式场景下做缓冲

---

## 11. 完整推理流程代码走读

本节将 M7 的 `Generate` 方法按步骤拆解，展示完整的 llama.cpp 推理流程。

### 11.1 步骤 1：前置检查

```cpp
// src/llm/llm_generator.cpp:208-216
if (!pImpl_->model_loaded) {
    Logger::Error("LlamaGenerator::Generate: model not loaded");
    return "";
}
if (prompt.empty()) {
    Logger::Error("LlamaGenerator::Generate: prompt is empty");
    return "";
}
```

确保模型已加载且 prompt 非空。

### 11.2 步骤 2：分词

```cpp
// src/llm/llm_generator.cpp:222-231
int32_t n_tokens_max = static_cast<int32_t>(llama_n_ctx(pImpl_->ctx));
std::vector<llama_token> tokens(static_cast<size_t>(n_tokens_max));
int32_t n_tokens = llama_tokenize(
    pImpl_->vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
    tokens.data(), n_tokens_max, true, false);
if (n_tokens < 0) {
    Logger::Error("LlamaGenerator::Generate: tokenization failed, prompt too long");
    return "";
}
tokens.resize(static_cast<size_t>(n_tokens));
```

将 prompt 字符串转为 token id 数组。`add_bos=true` 自动添加 BOS token。

### 11.3 步骤 3：构建采样器链

```cpp
// src/llm/llm_generator.cpp:234
llama_sampler* sampler = pImpl_->BuildSamplerChain(config);
```

根据 `GenerationConfig` 构建采样器链：penalties → top-k(40) → top-p(0.95) → temp(0.8) → dist。

### 11.4 步骤 4：初始化 Batch

```cpp
// src/llm/llm_generator.cpp:237-238
int32_t max_batch = std::max(n_tokens, 1);
llama_batch batch = llama_batch_init(max_batch, 0, 1);
```

分配足以容纳所有 prompt tokens 的 batch。

### 11.5 步骤 5：填充 Prompt Batch 并 Decode

```cpp
// src/llm/llm_generator.cpp:241-261
for (int32_t i = 0; i < n_tokens; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = false;
}
batch.n_tokens = n_tokens;
if (n_tokens > 0) {
    batch.logits[n_tokens - 1] = true;  // 最后一个 token 需要 logits
}

int32_t ret = llama_decode(pImpl_->ctx, batch);
if (ret != 0) {
    // 错误处理：释放 sampler 和 batch
    return "";
}
```

将 prompt 所有 token 一次性送入模型。注意只有最后一个 token 需要 logits。

### 11.6 步骤 6：循环生成

```cpp
// src/llm/llm_generator.cpp:264-301
std::string result;
int32_t n_cur = n_tokens;                           // 当前位置 = prompt 长度
llama_token eos_token = llama_vocab_eos(pImpl_->vocab);

for (int i = 0; i < config.max_tokens; ++i) {
    // 6a. 采样下一个 token
    llama_token new_token = llama_sampler_sample(sampler, pImpl_->ctx, -1);

    // 6b. 检查 EOS
    if (new_token == eos_token) {
        break;
    }

    // 6c. 转为文本并追加结果
    result += Impl::TokenToText(pImpl_->vocab, new_token);

    // 6d. 将新 token 送入 batch 进行下一轮推理
    batch.n_tokens = 1;
    batch.token[0] = new_token;
    batch.pos[0] = n_cur;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;

    ret = llama_decode(pImpl_->ctx, batch);
    if (ret != 0) {
        break;
    }

    n_cur++;

    // 6e. 超出上下文长度时停止
    if (n_cur >= static_cast<int32_t>(llama_n_ctx(pImpl_->ctx))) {
        break;
    }
}
```

生成循环的每一步：
1. **采样**：从当前 logits 中采样一个 token
2. **EOS 检查**：遇到结束标记则停止
3. **文本转换**：将 token 转为 UTF-8 文本
4. **Decode**：将新 token 送入模型，为下一次采样准备 logits
5. **上下文溢出检查**：超过 `n_ctx` 时停止

### 11.7 步骤 7：资源释放

```cpp
// src/llm/llm_generator.cpp:303-304
llama_sampler_free(sampler);
llama_batch_free(batch);
```

释放采样器链和 batch 内存。

### 11.8 流式 vs 非流式

`GenerateStream` 与 `Generate` 的逻辑完全相同，唯一区别在于 token 处理方式：

| 步骤 | Generate（非流式） | GenerateStream（流式） |
|---|---|---|
| Token 文本处理 | `result += TokenToText(...)` | `callback(TokenToText(...))` |
| 返回值 | 拼接的完整字符串 | void（通过 callback 逐步输出） |
| 适用场景 | 批量处理、不需要实时输出 | 聊天交互、实时显示 |

### 11.9 资源生命周期图

```
LlamaGenerator::LoadModel()
│
├── llama_backend_init()                     ← 全局局初始化
├── llama_model_load_from_file()             ← 创建 model
├── llama_model_get_vocab()                  ← 获取 vocab（依赖 model）
├── llama_init_from_model()                  ← 创建 ctx（依赖 model）
│
LlamaGenerator::Generate() / GenerateStream()
│
├── llama_tokenize()                         ← 分词（依赖 vocab）
├── BuildSamplerChain()                      ← 创建 sampler
│   ├── llama_sampler_chain_init()
│   ├── llama_sampler_init_penalties()
│   ├── llama_sampler_init_top_k()
│   ├── llama_sampler_init_top_p()
│   ├── llama_sampler_init_temp()
│   └── llama_sampler_init_dist()/greedy()
├── llama_batch_init()                       ← 创建 batch
│
├── [Prompt Phase]
│   ├── 填充 batch (n tokens)
│   └── llama_decode()                       ← 首次推理
│
├── [Generation Loop]
│   ├── llama_sampler_sample()               ← 采样
│   ├── llama_token_to_piece()               ← token 转文本
│   ├── 填充 batch (1 token)
│   └── llama_decode()                       ← 逐 token 推理
│
├── llama_sampler_free()                     ← 释放 sampler
├── llama_batch_free()                       ← 释放 batch
│
LlamaGenerator::~LlamaGenerator()
│
├── llama_free()                             ← 释放 ctx
├── llama_model_free()                       ← 释放 model
└── llama_backend_free()                     ← 全局清理
```

---

## 附录 A：GenerationConfig 参数速查

| 参数 | 类型 | 默认值 | 对应采样器 | 说明 |
|---|---|---|---|---|
| `max_tokens` | int | 512 | - | 最大生成 token 数 |
| `temperature` | float | 0.8 | `llama_sampler_init_temp` | 温度（0 = 贪心） |
| `top_p` | float | 0.95 | `llama_sampler_init_top_p` | 核采样概率阈值 |
| `repeat_penalty` | float | 1.1 | `llama_sampler_init_penalties` | 重复惩罚系数 |
| `seed` | int | -1 | `llama_sampler_init_dist` | 随机种子（-1 = 随机） |

## 附录 B：llama.cpp API 错误处理总结

| API | 失败返回值 | 项目中的处理方式 |
|---|---|---|
| `llama_model_load_from_file` | NULL | Logger::Error + return false |
| `llama_model_get_vocab` | NULL | Logger::Error + llama_model_free + return false |
| `llama_init_from_model` | NULL | Logger::Error + llama_model_free + return false |
| `llama_tokenize` | 负值 | Logger::Error + return "" |
| `llama_decode` | != 0 | Logger::Error + break/return |

## 附录 C：Apple Silicon Metal GPU 说明

- `n_gpu_layers = -1`：将所有 transformer 层卸载到 Metal GPU
- Metal 使用统一内存架构，模型权重零拷贝访问
- `has_tensor = false` 在 M1 Max 上是正常的（MPS Graph Tensor API 仅 M3+/A17+ 支持，不影响 Metal compute 加速）
- GPU 加速效果：M1 Max 上约 3-5 倍提速（取决于模型大小和量化级别）
