#include "logger.h"
#include "timer.h"

#include <thread>
#include <chrono>

using namespace inference;

// 程序入口，演示 Logger 和 Timer 功能
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

    return 0;
}