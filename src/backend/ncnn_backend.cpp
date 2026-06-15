#include "ncnn_backend.h"
#include "logger.h"
#include "timer.h"

#include <net.h>
#include <gpu.h>

#include <filesystem>
#include <cmath>
#include <stdexcept>

namespace inference {

// PImpl 内部结构，持有所有 NCNN 相关对象和缓存元信息
struct NcnnBackend::Impl {
    // NCNN 推理网络对象，LoadModel 时加载 .param 和 .bin
    ncnn::Net net;

    // 模型输入 blob 名称列表，LoadModel 时从 net.input_names() 提取
    std::vector<std::string> input_names;
    // 模型输出 blob 名称列表，LoadModel 时从 net.output_names() 提取
    std::vector<std::string> output_names;
    // 模型输入张量形状列表，LoadModel 时从 net.blobs() 的 shape 字段解析
    // 维度映射：dims==3 → {1,c,h,w}，dims==2 → {1,h,w}，dims==1 → {1,w}，dims==0 → {1}
    std::vector<std::vector<int64_t>> input_shapes;
    // 模型输出张量形状列表，首次 Predict 后从 ncnn::Mat 维度缓存，此前为空
    std::vector<std::vector<int64_t>> output_shapes;

    // 标记模型是否已成功加载，Predict 等方法的前置条件
    bool model_loaded = false;
    // 标记是否启用了 Vulkan GPU 加速，析构时用于判断是否需要销毁 GPU 实例
    bool vulkan_enabled = false;

    // 模型输入尺寸回退值，当 NCNN 模型 .param 未包含 shape 信息时使用
    int fallback_input_width = 640;
    int fallback_input_height = 640;
    int fallback_input_channels = 3;

    // 根据输入路径推导 .param 和 .bin 的完整路径
    // 若 path 以 .param 结尾，则直接使用，.bin 替换后缀
    // 否则视为基础名，自动追加 .param 和 .bin 后缀
    static std::pair<std::string, std::string> ResolveModelPaths(
        const std::string& model_path) {
        std::string param_path = model_path;
        std::string bin_path;

        if (param_path.size() >= 6 &&
            param_path.substr(param_path.size() - 6) == ".param") {
            bin_path = param_path.substr(0, param_path.size() - 6) + ".bin";
        } else {
            param_path = model_path + ".param";
            bin_path = model_path + ".bin";
        }

        return {param_path, bin_path};
    }

    // 将 ncnn::Mat 的维度信息转换为 NCHW 形状的 int64_t 向量
    // NCNN Mat 维度：dims==3 时 {c, h, w}，dims==2 时 {h, w}，dims==1 时 {w}
    // 转换后对应 ONNX 惯例：{1, c, h, w} / {1, h, w} / {1, w}
    // dims==0 表示形状未知，返回 {1} 占位
    static std::vector<int64_t> MatShapeToNchw(const ncnn::Mat& shape) {
        if (shape.dims == 3) {
            return {1, static_cast<int64_t>(shape.c),
                    static_cast<int64_t>(shape.h),
                    static_cast<int64_t>(shape.w)};
        } else if (shape.dims == 2) {
            return {1, static_cast<int64_t>(shape.h),
                    static_cast<int64_t>(shape.w)};
        } else if (shape.dims == 1) {
            return {1, static_cast<int64_t>(shape.w)};
        }
        return {1};
    }
};

// 构造 NcnnBackend 对象，创建 Impl 实例
NcnnBackend::NcnnBackend()
    : impl_(std::make_unique<Impl>()) {}

// 析构函数，释放 NCNN 资源
// 若 Vulkan 已启用且 GPU 实例已创建，销毁 GPU 实例
NcnnBackend::~NcnnBackend() {
    if (impl_->vulkan_enabled) {
        ncnn::destroy_gpu_instance();
    }
}

// 加载 NCNN 模型文件并初始化推理网络
// model_path: .param 文件路径或不含后缀的基础名
// 流程：
//   1. 推导 .param 和 .bin 路径，检查文件存在性
//   2. 配置网络优化选项（fp16、Winograd、packing）
//   3. 检测 Vulkan GPU，可用时启用 GPU 加速
//   4. 加载模型参数和权重
//   5. 提取输入/输出 blob 名称和形状
//   6. 构建详细日志输出
// 返回: 加载成功返回 true，文件不存在或加载失败返回 false
bool NcnnBackend::LoadModel(const std::string& model_path) {
    auto [param_path, bin_path] = Impl::ResolveModelPaths(model_path);

    if (!std::filesystem::exists(param_path)) {
        Logger::Error("NcnnBackend::LoadModel: param file not found: " +
                      param_path);
        return false;
    }

    if (!std::filesystem::exists(bin_path)) {
        Logger::Error("NcnnBackend::LoadModel: bin file not found: " +
                      bin_path);
        return false;
    }

    // 配置优化选项，须在 load_param 之前设置
    impl_->net.opt.use_winograd_convolution = true;
    impl_->net.opt.use_fp16_packed = true;
    impl_->net.opt.use_fp16_storage = true;
    impl_->net.opt.use_fp16_arithmetic = true;
    impl_->net.opt.use_packing_layout = true;

    // Vulkan GPU 加速：macOS 上 Vulkan 内存分配不稳定，暂时禁用
    // Linux 上可通过环境变量 NCNN_VULKAN=1 启用
    const char* env_vulkan = std::getenv("NCNN_VULKAN");
    if (env_vulkan && std::string(env_vulkan) == "1" &&
        ncnn::get_gpu_count() > 0) {
        ncnn::create_gpu_instance();
        impl_->net.opt.use_vulkan_compute = true;
        impl_->net.set_vulkan_device(ncnn::get_default_gpu_index());
        impl_->vulkan_enabled = true;
        Logger::Info("NcnnBackend::LoadModel: Vulkan GPU acceleration enabled");
    } else {
        Logger::Info("NcnnBackend::LoadModel: using CPU (Vulkan disabled by default)");
    }

    // 加载模型参数（网络结构），返回 0 表示成功
    int ret = impl_->net.load_param(param_path.c_str());
    if (ret != 0) {
        Logger::Error("NcnnBackend::LoadModel: load_param failed with code " +
                      std::to_string(ret) + ", path: " + param_path);
        return false;
    }

    // 加载模型权重（二进制数据），返回 0 表示成功
    ret = impl_->net.load_model(bin_path.c_str());
    if (ret != 0) {
        Logger::Error("NcnnBackend::LoadModel: load_model failed with code " +
                      std::to_string(ret) + ", path: " + bin_path);
        return false;
    }

    // 提取输入 blob 名称，net.input_names() 返回 const char* vector
    impl_->input_names.clear();
    for (const char* name : impl_->net.input_names()) {
        impl_->input_names.emplace_back(name);
    }

    // 提取输出 blob 名称
    impl_->output_names.clear();
    for (const char* name : impl_->net.output_names()) {
        impl_->output_names.emplace_back(name);
    }

    // 提取输入 blob 形状，从 net.blobs() 的 shape 字段解析
    // 形状映射为 NCHW 惯例以与 InferenceBackend 接口保持一致
    impl_->input_shapes.clear();
    const auto& blobs = impl_->net.blobs();
    for (int idx : impl_->net.input_indexes()) {
        ncnn::Mat shape = blobs[idx].shape;
        auto nchw_shape = Impl::MatShapeToNchw(shape);
        if (shape.dims == 0) {
            Logger::Warning(
                "NcnnBackend::LoadModel: input blob '" +
                impl_->input_names[impl_->input_shapes.size()] +
                "' has unknown shape, using {1} as placeholder");
        }
        impl_->input_shapes.push_back(std::move(nchw_shape));
    }

    // 输出形状初始化为空，首次 Predict 后从推理结果 ncnn::Mat 缓存
    impl_->output_shapes.clear();
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
        output_info += "name:" + impl_->output_names[i];
        if (i + 1 < impl_->output_names.size()) output_info += ",";
    }
    output_info += "]";

    Logger::Info("NcnnBackend::LoadModel: model loaded from " + param_path +
                 ", " + input_info + ", " + output_info);

    return true;
}

// 执行推理，将预处理后的数据送入模型并获取输出
// inputs: 输入张量列表，每个元素是 CHW 顺序的一维 float 数组
//         数量必须与模型输入 blob 数量一致
// 流程：
//   1. 前置检查：模型已加载、inputs 不为空、输入数量匹配
//   2. 将 vector<float> 转换为 ncnn::Mat（外部数据，零拷贝）
//   3. 创建 Extractor，按 blob 名称绑定输入/输出
//   4. 从输出 ncnn::Mat 拷贝数据到 vector<float>
//   5. 首次推理时直接从输出 Mat 缓存形状，避免二次推理
//   6. Timer 记录推理耗时
// 返回: 推理成功返回输出张量列表；前置检查失败或推理异常返回空 vector
std::vector<std::vector<float>> NcnnBackend::Predict(
    const std::vector<std::vector<float>>& inputs) {
    if (!impl_->model_loaded) {
        Logger::Error("NcnnBackend::Predict: model not loaded");
        return {};
    }
    if (inputs.empty()) {
        Logger::Error("NcnnBackend::Predict: inputs is empty");
        return {};
    }
    if (inputs.size() != impl_->input_names.size()) {
        Logger::Error(
            "NcnnBackend::Predict: input count mismatch, expected " +
            std::to_string(impl_->input_names.size()) + ", got " +
            std::to_string(inputs.size()));
        return {};
    }

    Timer timer;
    timer.Start();

    try {
        ncnn::Extractor ex = impl_->net.create_extractor();

        // 为每个输入创建 ncnn::Mat 并绑定到 Extractor
        // 使用外部数据指针构造 Mat，避免数据拷贝
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& data = inputs[i];
            const auto& shape = impl_->input_shapes[i];
            ncnn::Mat mat;

            if (shape.size() == 4 && shape[1] > 0 && shape[2] > 0 &&
                shape[3] > 0) {
                // NCHW {1,c,h,w} → ncnn::Mat(w, h, c, data)
                mat = ncnn::Mat(static_cast<int>(shape[3]),
                                static_cast<int>(shape[2]),
                                static_cast<int>(shape[1]),
                                const_cast<float*>(data.data()));
            } else if (shape.size() == 3 && shape[1] > 0 && shape[2] > 0) {
                // {1,h,w} → ncnn::Mat(w, h, data)
                mat = ncnn::Mat(static_cast<int>(shape[2]),
                                static_cast<int>(shape[1]),
                                const_cast<float*>(data.data()));
            } else {
                // 形状未知时，使用 fallback 输入尺寸构造 3D Mat
                // 典型场景：nihui/ncnn-assets 的 YOLOv5s 模型输入 in0 无 shape 信息
                int c = impl_->fallback_input_channels;
                int h = impl_->fallback_input_height;
                int w = impl_->fallback_input_width;
                Logger::Warning(
                    "NcnnBackend::Predict: input '" + impl_->input_names[i] +
                    "' has unknown shape, using fallback " +
                    std::to_string(w) + "x" + std::to_string(h) + "x" +
                    std::to_string(c));
                mat = ncnn::Mat(w, h, c,
                                const_cast<float*>(data.data()));
            }

            ex.input(impl_->input_names[i].c_str(), mat);
        }

        // 提取输出，同时缓存输出形状（仅首次推理时）
        bool cache_shapes = impl_->output_shapes.empty();
        std::vector<std::vector<float>> results;

        for (size_t i = 0; i < impl_->output_names.size(); ++i) {
            ncnn::Mat out;
            int ret = ex.extract(impl_->output_names[i].c_str(), out);
            if (ret != 0) {
                Logger::Error(
                    "NcnnBackend::Predict: extract output '" +
                    impl_->output_names[i] + "' failed with code " +
                    std::to_string(ret));
                return {};
            }

            // 首次推理时直接从输出 Mat 缓存形状，无需二次推理
            if (cache_shapes) {
                impl_->output_shapes.push_back(Impl::MatShapeToNchw(out));
            }

            // 将 ncnn::Mat 数据拷贝到 vector<float>
            // ncnn::Mat 使用通道分离存储，dims==3 时需按通道遍历
            std::vector<float> output_data;
            if (out.dims == 3) {
                output_data.reserve(out.c * out.h * out.w);
                for (int c = 0; c < out.c; ++c) {
                    auto ch_mat = out.channel(c);
                    const float* ptr = (const float*)ch_mat.data;
                    output_data.insert(output_data.end(), ptr,
                                       ptr + out.h * out.w);
                }
            } else if (out.dims == 2) {
                output_data.resize(out.h * out.w);
                memcpy(output_data.data(), out.data,
                       out.h * out.w * sizeof(float));
            } else if (out.dims == 1) {
                output_data.resize(out.w);
                memcpy(output_data.data(), out.data,
                       out.w * sizeof(float));
            }

            results.push_back(std::move(output_data));
        }

        // NCNN YOLOv5 多头输出解码拼接：
        // nihui/ncnn-assets 的 YOLOv5s 有 3 个输出头 out0/out1/out2
        // 每个输出头 NCNN Mat shape: {255, H, W}（dims=3），其中 255 = 3*(80+5) = 3*85
        // NCNN 模型输出的是 raw logits（无 sigmoid、无坐标解码），
        // 需要执行完整的 YOLOv5 解码后拼接为 [25200, 85] 格式，与 ONNX 输出一致
        // 使用 output_names 数量判断是否为多头输出
        if (results.size() == 3 && impl_->output_names.size() == 3) {
            // 每个头的原始 NCNN 输出大小 = 255 * H * W
            // YOLOv5s 三个检测头空间尺寸: 80x80, 40x40, 20x20
            std::vector<int> head_sizes = {255 * 80 * 80, 255 * 40 * 40, 255 * 20 * 20};
            bool is_yolo_multihead = true;
            for (size_t i = 0; i < results.size(); ++i) {
                if (static_cast<int>(results[i].size()) != head_sizes[i]) {
                    is_yolo_multihead = false;
                    break;
                }
            }

            if (is_yolo_multihead) {
                // YOLOv5s anchors (COCO, 从 P3/P4/P5 依次排列)
                // P3/stride=8:  [10,13], [16,30], [33,23]
                // P4/stride=16: [30,61], [62,45], [59,119]
                // P5/stride=32: [116,90], [156,198], [373,326]
                static const float anchors[3][3][2] = {
                    {{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}},
                    {{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}},
                    {{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}
                };
                static const int strides[3] = {8, 16, 32};
                static const int head_hws[3][2] = {{80, 80}, {40, 40}, {20, 20}};

                std::vector<float> concatenated;
                concatenated.reserve(25200 * 85);

                for (size_t i = 0; i < results.size(); ++i) {
                    int h = head_hws[i][0];
                    int w = head_hws[i][1];
                    int stride = strides[i];
                    const float* data = results[i].data();

                    for (int a = 0; a < 3; ++a) {
                        float anchor_w = anchors[i][a][0];
                        float anchor_h = anchors[i][a][1];

                        for (int y = 0; y < h; ++y) {
                            for (int x = 0; x < w; ++x) {
                                // 从 NCNN 通道分离存储中读取 85 个 raw logits
                                // 通道布局: [a0_tx, a0_ty, a0_tw, a0_th, a0_obj,
                                //            a0_cls0..79, a1_tx, ...]
                                float raw[85];
                                for (int k = 0; k < 85; ++k) {
                                    int ch = a * 85 + k;
                                    raw[k] = data[ch * h * w + y * w + x];
                                }

                                // YOLOv5 坐标解码（grid decoding）
                                // cx = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
                                // cy = (sigmoid(ty) * 2 - 0.5 + grid_y) * stride
                                // w  = (sigmoid(tw) * 2)^2 * anchor_w
                                // h  = (sigmoid(th) * 2)^2 * anchor_h
                                float sig_tx = 1.0f / (1.0f + std::exp(-raw[0]));
                                float sig_ty = 1.0f / (1.0f + std::exp(-raw[1]));
                                float sig_tw = 1.0f / (1.0f + std::exp(-raw[2]));
                                float sig_th = 1.0f / (1.0f + std::exp(-raw[3]));

                                float cx = (sig_tx * 2.0f - 0.5f + static_cast<float>(x)) * stride;
                                float cy = (sig_ty * 2.0f - 0.5f + static_cast<float>(y)) * stride;
                                float bw = (sig_tw * 2.0f) * (sig_tw * 2.0f) * anchor_w;
                                float bh = (sig_th * 2.0f) * (sig_th * 2.0f) * anchor_h;

                                concatenated.push_back(cx);
                                concatenated.push_back(cy);
                                concatenated.push_back(bw);
                                concatenated.push_back(bh);

                                // objectness sigmoid
                                concatenated.push_back(
                                    1.0f / (1.0f + std::exp(-raw[4])));

                                // class scores sigmoid
                                for (int k = 5; k < 85; ++k) {
                                    concatenated.push_back(
                                        1.0f / (1.0f + std::exp(-raw[k])));
                                }
                            }
                        }
                    }
                }

                Logger::Info(
                    "NcnnBackend::Predict: YOLOv5 3-head decoded to [25200, 85]");

                results.clear();
                results.push_back(std::move(concatenated));
                impl_->output_shapes.clear();
                impl_->output_shapes.push_back({1, 25200, 85});
            }
        }

        if (cache_shapes && !impl_->output_shapes.empty()) {
            Logger::Info("NcnnBackend::Predict: output shapes cached");
        }

        timer.Stop();
        Logger::Info("NcnnBackend::Predict: inference completed in " +
                     std::to_string(timer.ElapsedMilliseconds()) + " ms");

        return results;

    } catch (const std::exception& e) {
        Logger::Error("NcnnBackend::Predict: error: " +
                      std::string(e.what()));
        return {};
    }
}

// 返回模型输入张量形状列表，LoadModel 时从 blob.shape 解析
std::vector<std::vector<int64_t>> NcnnBackend::GetInputShapes() const {
    return impl_->input_shapes;
}

// 返回模型输入节点名称列表
std::vector<std::string> NcnnBackend::GetInputNames() const {
    return impl_->input_names;
}

// 返回模型输出节点名称列表
std::vector<std::string> NcnnBackend::GetOutputNames() const {
    return impl_->output_names;
}

// 返回模型输出张量形状列表，首次 Predict 后缓存，此前为空
std::vector<std::vector<int64_t>> NcnnBackend::GetOutputShapes() const {
    return impl_->output_shapes;
}

// 设置模型输入尺寸，用于 NCNN 模型未包含 shape 信息时的回退
void NcnnBackend::SetInputSize(int width, int height, int channels) {
    impl_->fallback_input_width = width;
    impl_->fallback_input_height = height;
    impl_->fallback_input_channels = channels;
}

}  // namespace inference
