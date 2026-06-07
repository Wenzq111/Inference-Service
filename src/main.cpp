#include "logger.h"
#include "timer.h"
#include "onnx_backend.h"

#include <memory>
#include <thread>
#include <chrono>

using namespace inference;

// 程序入口，演示 Logger、Timer 和 OnnxBackend 功能
int main() {
    // 设置日志过滤级别为 Debug，确保所有级别日志均可输出
    Logger::SetLevel(LogLevel::Debug);

    Logger::Debug("This is a debug message");
    Logger::Info("Hello");
    Logger::Warning("This is a warning");
    Logger::Error("This is an error");

    // 演示 Timer 计时功能
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    timer.Stop();

    Logger::Info("Timer elapsed: " + std::to_string(timer.ElapsedMilliseconds()) + " ms");

    // 演示 ONNX Runtime 后端创建
    auto backend = std::make_unique<OnnxBackend>();
    Logger::Info("ONNX Runtime backend created successfully");

    return 0;
}