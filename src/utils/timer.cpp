#include "timer.h"

namespace inference {

// 启动计时器，记录当前时刻为起始时间
void Timer::Start() {
    start_time_ = std::chrono::steady_clock::now();
    running_ = true;
}

// 停止计时器，记录当前时刻为结束时间
void Timer::Stop() {
    end_time_ = std::chrono::steady_clock::now();
    running_ = false;
}

// 重置计时器，清空所有时间点并标记为未运行
void Timer::Reset() {
    start_time_ = std::chrono::steady_clock::time_point{};
    end_time_ = std::chrono::steady_clock::time_point{};
    running_ = false;
}

// 计算已耗时毫秒数，若计时器仍在运行则使用当前时刻作为结束点
double Timer::ElapsedMilliseconds() const {
    auto end = running_ ? std::chrono::steady_clock::now() : end_time_;
    return std::chrono::duration<double, std::milli>(end - start_time_).count();
}

// 将毫秒耗时转换为秒数
double Timer::ElapsedSeconds() const {
    return ElapsedMilliseconds() / 1000.0;
}

// 返回计时器是否正在运行
bool Timer::IsRunning() const {
    return running_;
}

}  // namespace inference