#pragma once

#include <chrono>

namespace inference {

// 高精度计时器类，基于 steady_clock 实现，用于测量代码段执行耗时
class Timer {
public:
    // 启动计时器，记录起始时间点
    void Start();
    // 停止计时器，记录结束时间点
    void Stop();
    // 重置计时器，清空所有时间点并恢复为未运行状态
    void Reset();
    // 返回已耗时的毫秒数
    double ElapsedMilliseconds() const;
    // 返回已耗时的秒数
    double ElapsedSeconds() const;
    // 返回计时器是否正在运行
    bool IsRunning() const;

private:
    // 计时起始时间点
    std::chrono::steady_clock::time_point start_time_;
    // 计时结束时间点
    std::chrono::steady_clock::time_point end_time_;
    // 标记计时器是否正在运行
    bool running_ = false;
};

}  // namespace inference