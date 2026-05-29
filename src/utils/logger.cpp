#include "logger.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace inference {

// 默认日志过滤级别为 Info
LogLevel Logger::level_ = LogLevel::Info;

// 设置全局日志过滤级别
void Logger::SetLevel(LogLevel level) {
    level_ = level;
}

// 输出 Debug 级别日志
void Logger::Debug(const std::string& msg) {
    Log(LogLevel::Debug, msg);
}

// 输出 Info 级别日志
void Logger::Info(const std::string& msg) {
    Log(LogLevel::Info, msg);
}

// 输出 Warning 级别日志
void Logger::Warning(const std::string& msg) {
    Log(LogLevel::Warning, msg);
}

// 输出 Error 级别日志
void Logger::Error(const std::string& msg) {
    Log(LogLevel::Error, msg);
}

// 内部日志输出方法，低于当前过滤级别的日志会被跳过
void Logger::Log(LogLevel level, const std::string& msg) {
    if (level < level_) {
        return;
    }
    std::cout << "[" << GetTimestamp() << "] [" << LevelToString(level) << "] " << msg << std::endl;
}

// 将 LogLevel 枚举转换为可读字符串
std::string Logger::LevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:    return "DEBUG";
        case LogLevel::Info:     return "INFO";
        case LogLevel::Warning:  return "WARN";
        case LogLevel::Error:    return "ERROR";
        default:                 return "UNKNOWN";
    }
}

// 获取当前系统时间的格式化字符串
std::string Logger::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now{};
    localtime_r(&time_t_now, &tm_now);
    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

}  // namespace inference