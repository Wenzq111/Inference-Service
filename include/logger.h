#pragma once

#include <string>

namespace inference {

// 日志级别枚举，用于控制日志输出过滤
enum class LogLevel { Debug, Info, Warning, Error };

// 全局日志工具类，提供分级日志输出，所有方法为静态方法
class Logger {
public:
    // 设置全局日志过滤级别
    static void SetLevel(LogLevel level);
    // 输出 Debug 级别日志
    static void Debug(const std::string& msg);
    // 输出 Info 级别日志
    static void Info(const std::string& msg);
    // 输出 Warning 级别日志
    static void Warning(const std::string& msg);
    // 输出 Error 级别日志
    static void Error(const std::string& msg);

private:
    // 当前全局日志过滤级别，默认为 Info
    static LogLevel level_;
    // 内部日志输出方法，根据级别过滤并格式化输出
    static void Log(LogLevel level, const std::string& msg);
    // 将日志级别转换为可读字符串
    static std::string LevelToString(LogLevel level);
    // 获取当前系统时间的格式化字符串
    static std::string GetTimestamp();
};

}  // namespace inference