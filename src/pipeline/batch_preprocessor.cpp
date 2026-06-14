#include "batch_preprocessor.h"
#include "preprocess.h"
#include "logger.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace inference {

// PImpl 内部结构，持有线程池、任务队列和预处理参数
struct BatchPreprocessor::Impl {
    // 预处理参数
    int target_w_ = 640;
    int target_h_ = 640;
    PreprocessMode mode_ = PreprocessMode::Resize;
    std::vector<float> mean_ = {0.0f, 0.0f, 0.0f};
    std::vector<float> std_ = {1.0f, 1.0f, 1.0f};
    // 预处理参数互斥锁（参数在主线程设置，工作线程读取）
    std::mutex params_mutex_;

    // 完成回调
    PreprocessCallback callback_;
    // 回调互斥锁
    std::mutex callback_mutex_;

    // 任务定义：一张待预处理的图像
    struct Task {
        // 用户自定义标识
        size_t index;
        // 图像数据（use_path=false 时使用）
        cv::Mat img;
        // 图像路径（use_path=true 时使用）
        std::string path;
        // 是否通过路径读取图像
        bool use_path;
    };

    // 线程安全任务队列
    std::queue<Task> queue_;
    // 队列互斥锁
    std::mutex queue_mutex_;
    // 队列条件变量：通知工作线程有新任务
    std::condition_variable queue_cv_;
    // 完成条件变量：通知 WaitAll 任务已全部完成
    std::condition_variable done_cv_;

    // 工作线程池
    std::vector<std::thread> workers_;
    // 线程数量
    size_t num_workers_;

    // 是否停止接受新任务并终止工作线程
    std::atomic<bool> stopped_;
    // 待处理任务计数（队列中 + 正在执行的）
    std::atomic<size_t> pending_count_;

    // 工作线程主循环：从队列取任务并执行预处理
    void WorkerLoop() {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !queue_.empty() || stopped_.load();
                });

                // 停止且队列为空时退出线程
                if (stopped_.load() && queue_.empty()) {
                    return;
                }

                // 取出任务
                task = std::move(queue_.front());
                queue_.pop();
            }

            // 执行预处理（不持锁）
            PreprocessResult result;

            // 读取参数（加锁，防止与 SetPreprocessParams 竞争）
            int tw, th;
            PreprocessMode mode;
            std::vector<float> m, s;
            {
                std::lock_guard<std::mutex> lock(params_mutex_);
                tw = target_w_;
                th = target_h_;
                mode = mode_;
                m = mean_;
                s = std_;
            }

            // 获取图像
            cv::Mat img;
            if (task.use_path) {
                img = cv::imread(task.path);
                if (img.empty()) {
                    Logger::Error("BatchPreprocessor: failed to read image: " + task.path);
                }
            } else {
                img = task.img;
                if (img.empty()) {
                    Logger::Error("BatchPreprocessor: submitted image is empty, index=" +
                                  std::to_string(task.index));
                }
            }

            // 根据模式执行预处理
            if (!img.empty()) {
                if (mode == PreprocessMode::Resize) {
                    // 直接缩放 + 归一化 + CHW
                    result.tensor = ResizeAndNorm(img, tw, th, m, s);
                    if (result.tensor.empty()) {
                        Logger::Error("BatchPreprocessor: ResizeAndNorm failed, index=" +
                                      std::to_string(task.index));
                    }
                } else {
                    // Letterbox + MatToChw
                    auto lb = Letterbox(img, tw, th);
                    if (lb.image.empty()) {
                        Logger::Error("BatchPreprocessor: Letterbox failed, index=" +
                                      std::to_string(task.index));
                    } else {
                        result.tensor = MatToChw(lb.image, m, s);
                        result.scale = lb.scale;
                        result.x_offset = lb.x_offset;
                        result.y_offset = lb.y_offset;
                        if (result.tensor.empty()) {
                            Logger::Error("BatchPreprocessor: MatToChw failed, index=" +
                                          std::to_string(task.index));
                        }
                    }
                }
            }

            // 调用回调（加锁，防止与 SetCallback 竞争）
            {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (callback_) {
                    callback_(task.index, result);
                }
            }

            // 减少待处理计数，通知 WaitAll
            pending_count_--;
            done_cv_.notify_all();
        }
    }
};

// 构造 BatchPreprocessor，启动工作线程池
BatchPreprocessor::BatchPreprocessor(size_t num_workers)
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->num_workers_ = num_workers > 0 ? num_workers : 1;
    pImpl_->stopped_.store(false);
    pImpl_->pending_count_.store(0);

    // 启动工作线程
    for (size_t i = 0; i < pImpl_->num_workers_; ++i) {
        pImpl_->workers_.emplace_back(&Impl::WorkerLoop, pImpl_.get());
    }

    Logger::Info("BatchPreprocessor: started with " +
                 std::to_string(pImpl_->num_workers_) + " workers");
}

// 析构函数，停止线程池
BatchPreprocessor::~BatchPreprocessor() {
    Stop();
}

// 提交一张图像（通过 cv::Mat）进行预处理
bool BatchPreprocessor::Submit(size_t index, const cv::Mat& img) {
    if (pImpl_->stopped_.load()) {
        Logger::Warning("BatchPreprocessor::Submit: stopped, rejecting task index=" +
                        std::to_string(index));
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(pImpl_->queue_mutex_);
        pImpl_->queue_.push({index, img, "", false});
    }
    pImpl_->pending_count_++;
    pImpl_->queue_cv_.notify_one();

    return true;
}

// 提交一张图像（通过文件路径）
bool BatchPreprocessor::Submit(size_t index, const std::string& image_path) {
    if (pImpl_->stopped_.load()) {
        Logger::Warning("BatchPreprocessor::Submit: stopped, rejecting task index=" +
                        std::to_string(index));
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(pImpl_->queue_mutex_);
        pImpl_->queue_.push({index, cv::Mat(), image_path, true});
    }
    pImpl_->pending_count_++;
    pImpl_->queue_cv_.notify_one();

    return true;
}

// 等待所有已提交的任务完成（阻塞）
void BatchPreprocessor::WaitAll() {
    std::unique_lock<std::mutex> lock(pImpl_->queue_mutex_);
    pImpl_->done_cv_.wait(lock, [this] {
        return pImpl_->pending_count_.load() == 0;
    });
}

// 设置预处理参数
void BatchPreprocessor::SetPreprocessParams(int target_w, int target_h,
                                             PreprocessMode mode,
                                             const std::vector<float>& mean,
                                             const std::vector<float>& std) {
    std::lock_guard<std::mutex> lock(pImpl_->params_mutex_);
    pImpl_->target_w_ = target_w;
    pImpl_->target_h_ = target_h;
    pImpl_->mode_ = mode;
    pImpl_->mean_ = mean;
    pImpl_->std_ = std;
}

// 设置完成回调
void BatchPreprocessor::SetCallback(PreprocessCallback callback) {
    std::lock_guard<std::mutex> lock(pImpl_->callback_mutex_);
    pImpl_->callback_ = std::move(callback);
}

// 停止所有工作线程并清空队列
void BatchPreprocessor::Stop() {
    if (pImpl_->stopped_.load()) {
        return;
    }

    pImpl_->stopped_.store(true);
    pImpl_->queue_cv_.notify_all();

    // 等待所有工作线程退出
    for (auto& worker : pImpl_->workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    pImpl_->workers_.clear();

    Logger::Info("BatchPreprocessor: stopped");
}

}  // namespace inference
