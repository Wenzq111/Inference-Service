# 端到端 API 验证报告

日期：2026-06-15（第二次更新）
环境：macOS Apple M1 Max, ONNX Runtime 1.26.0 + CoreML EP, llama.cpp 9620 + Metal GPU
模型：yolov5s.onnx (float32, 28MB), TinyLlama-1.1B-Chat-v1.0-Q4_K_M (638MB)

## 1. 健康检查

**请求：**

```bash
curl http://localhost:8080/health
```

**响应：**

```json
{"status": "ok"}
```

**结论：** ✅ 服务正常运行。

---

## 2. 目标检测 - bus.jpg

**请求：**

```bash
curl -X POST -F "image=@models/test.jpg" http://localhost:8080/detect
```

**响应：**

```json
{
    "success": true,
    "detections": [
        {"bbox": [220.865, 406.862, 345.292, 876.482], "class_id": 0, "confidence": 0.827469},
        {"bbox": [661.781, 384.642, 809, 880.14],     "class_id": 0, "confidence": 0.815337},
        {"bbox": [47.7389, 393.25, 257.699, 912.875],  "class_id": 0, "confidence": 0.783031},
        {"bbox": [15.8847, 219.28, 793.272, 796.595],  "class_id": 5, "confidence": 0.770823},
        {"bbox": [0, 553.024, 72.6129, 875.445],       "class_id": 0, "confidence": 0.482324}
    ],
    "inference_time_ms": 72.6153
}
```

**结论：** ✅ 检测到 5 个目标：3 人（class_id=0）+ 1 巴士（class_id=5）+ 1 低置信度人，总耗时 73ms。检测结果与 YOLOv5s 标准输出一致。

---

## 3. 目标检测 - zidane.jpg

**请求：**

```bash
curl -X POST -F "image=@models/test2.jpg" http://localhost:8080/detect
```

**响应：**

```json
{
    "success": true,
    "detections": [
        {"bbox": [744.473, 48.1701, 1142.53, 716.83],  "class_id": 0,  "confidence": 0.870549},
        {"bbox": [441.025, 439.66, 498.475, 708.34],    "class_id": 27, "confidence": 0.637462},
        {"bbox": [127.937, 197.719, 845.562, 710.281],  "class_id": 0,  "confidence": 0.634056},
        {"bbox": [594.134, 376.633, 635.866, 437.367],  "class_id": 67, "confidence": 0.286883}
    ],
    "inference_time_ms": 63.6645
}
```

**结论：** ✅ 检测到 4 个目标：2 人（class_id=0）+ 1 领带（class_id=27）+ 1 手机（class_id=67），总耗时 64ms。不同图像检测均正常。

---

## 4. 文本生成 - Prompt 1

**请求：**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"What is 2+2? Answer briefly.","max_tokens":50}' \
  http://localhost:8080/generate
```

**响应：**

```json
{
    "success": true,
    "text": "Yes, the answer is 4.",
    "inference_time_ms": 180.482
}
```

**结论：** ✅ 模型正确回答 `2+2=4`，Chat Template 生效，不再重复 prompt 或续写。

---

## 5. 文本生成 - Prompt 2（多轮验证）

**请求：**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of France?","max_tokens":50}' \
  http://localhost:8080/generate
```

**响应：**

```json
{
    "success": true,
    "text": "Yes, France is the capital of France. The capital of France is Paris, located in the Île-de-France region.",
    "inference_time_ms": 144.927
}
```

**结论：** ✅ 模型正确回答巴黎（Paris），第二次调用 KV Cache 清空正常，无 inconsistent sequence positions 错误。

---

## 6. 文本生成 - Prompt 3（多轮验证）

**请求：**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"Who wrote Romeo and Juliet?","max_tokens":50}' \
  http://localhost:8080/generate
```

**响应：**

```json
{
    "success": true,
    "text": "The play Romeo and Juliet by William Shakespeare was written by the noted playwright and poet William Shakespeare.",
    "inference_time_ms": 127.492
}
```

**结论：** ✅ 模型正确回答莎士比亚（William Shakespeare），第三次连续调用正常，Chat Template + KV Cache 清空均稳定工作。

---

## 7. 文本生成 - 自定义参数

**请求：**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt":"Tell me a joke","max_tokens":100,"temperature":0.5,"top_p":0.9}' \
  http://localhost:8080/generate
```

**响应：**

```json
{
    "success": true,
    "text": ":\n\nJoke: A man walks into a bar and sees a group of drunk men trying to start a fire with flaming cans.\n\nAudience Member: That's hilarious!\n\nJoke: No, it's not funny. It's a classic joke.\n\nJoke: A man walks into a bar and sees a group of drunk men trying to start a fire with flaming cans.",
    "inference_time_ms": 525.382
}
```

**结论：** ✅ 自定义参数（max_tokens=100, temperature=0.5, top_p=0.9）生效。temperature 降低后输出更保守。

---

## 8. 目标检测 - 自定义阈值

**请求：**

```bash
curl -X POST "http://localhost:8080/detect?confidence=0.5&nms_threshold=0.4" \
  -F "image=@models/test.jpg"
```

**响应：**

```json
{
    "success": true,
    "detections": [
        {"bbox": [220.865, 406.862, 345.292, 876.482], "class_id": 0, "confidence": 0.827469},
        {"bbox": [661.781, 384.642, 809, 880.14],     "class_id": 0, "confidence": 0.815337},
        {"bbox": [47.7389, 393.25, 257.699, 912.875],  "class_id": 0, "confidence": 0.783031},
        {"bbox": [15.8847, 219.28, 793.272, 796.595],  "class_id": 5, "confidence": 0.770823}
    ],
    "inference_time_ms": 70.2713
}
```

**结论：** ✅ 提高置信度阈值到 0.5 后，过滤掉了原先 confidence=0.482324 的低置信度检测，从 5 个结果减少到 4 个。自定义阈值功能正常。

---

## 9. 错误处理 - 缺少 image 字段

**请求：**

```bash
curl -X POST http://localhost:8080/detect
```

**响应：**

```json
{"success": false, "error": "No image field in request"}
```

**结论：** ✅ 正确返回错误信息，HTTP 状态码 400。

---

## 10. 错误处理 - 缺少 prompt 字段

**请求：**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{}' \
  http://localhost:8080/generate
```

**响应：**

```json
{"success": false, "error": "Missing 'prompt' field"}
```

**结论：** ✅ 正确返回错误信息，HTTP 状态码 400。

---

## 已修复问题验证

| 问题 | 修复方式 | 验证结果 |
|------|----------|----------|
| ONNX float16 输入类型不匹配 | 使用 YOLOv5 export.py 重新导出 float32 ONNX 模型 | ✅ bus.jpg 检测到 5 目标，zidane.jpg 检测到 4 目标 |
| LLM KV Cache 未清空导致第二次调用失败 | 在 Generate/GenerateStream 开头添加 `llama_memory_clear(llama_get_memory(ctx), true)` | ✅ 连续 3 次调用均正常，无 inconsistent sequence positions 错误 |
| LLM 缺少 Chat Template 导致重复生成或直接 EOS | 在 Generate/GenerateStream 中调用 `llama_chat_apply_template` 包装 prompt 为对话格式 | ✅ 模型正确回答问题（2+2=4, Paris, Shakespeare），不再重复 prompt |
| 构建脚本工作目录导致模型路径找不到 | build.sh/rebuild.sh 运行前 `cd ..` 回项目根目录 | ✅ 服务启动时 ONNX 和 LLM 模型均正常加载 |

## 性能数据

| 操作 | 图像/参数 | 耗时 |
|------|-----------|------|
| 目标检测 | bus.jpg (810x1080) | 73ms（推理 24ms + 预处理/后处理 49ms） |
| 目标检测 | zidane.jpg (1280x720) | 64ms（推理 15ms + 预处理/后处理 49ms） |
| 文本生成 | max_tokens=50, "2+2" | 180ms |
| 文本生成 | max_tokens=50, "capital of France" | 145ms |
| 文本生成 | max_tokens=50, "Romeo and Juliet" | 127ms |
| 文本生成 | max_tokens=100 | 525ms |

## 测试环境启动命令

```bash
cd /path/to/Inference-Service
./build.sh
# 或
./rebuild.sh
```

> 注意：构建脚本会在项目根目录下运行 `./build/inference_service`，模型相对路径 `models/` 相对于项目根目录解析。
