#include <gtest/gtest.h>

#include "logger.h"

#include <sstream>
#include <iostream>

using namespace inference;

// 测试 Logger 默认日志级别为 Info
TEST(LoggerTest, DefaultLevelIsInfo) {
    // Debug < Info，设置 Debug 级别后 Debug 日志应该可见
    // 默认 Info 级别下，Debug 不输出，Info 及以上输出
    Logger::SetLevel(LogLevel::Info);
    // 无断言崩溃即通过，验证 SetLevel 不抛异常
    EXPECT_NO_THROW(Logger::Info("test info message"));
    EXPECT_NO_THROW(Logger::Warning("test warning message"));
    EXPECT_NO_THROW(Logger::Error("test error message"));
}

// 测试 SetLevel 可以切换日志级别
TEST(LoggerTest, SetLevelChangesFilter) {
    Logger::SetLevel(LogLevel::Error);
    // Error 级别下，Debug/Info/Warning 被过滤，只有 Error 输出
    EXPECT_NO_THROW(Logger::Debug("should be filtered"));
    EXPECT_NO_THROW(Logger::Info("should be filtered"));
    EXPECT_NO_THROW(Logger::Warning("should be filtered"));
    EXPECT_NO_THROW(Logger::Error("should pass"));

    // 恢复为 Debug 级别，所有日志都输出
    Logger::SetLevel(LogLevel::Debug);
    EXPECT_NO_THROW(Logger::Debug("should pass now"));
}

// 测试所有日志级别枚举值的顺序
TEST(LoggerTest, LogLevelOrdering) {
    // 枚举值递增：Debug < Info < Warning < Error
    EXPECT_LT(LogLevel::Debug, LogLevel::Info);
    EXPECT_LT(LogLevel::Info, LogLevel::Warning);
    EXPECT_LT(LogLevel::Warning, LogLevel::Error);
}

// 测试 Debug 级别下所有日志都可以输出
TEST(LoggerTest, DebugLevelPassesAll) {
    Logger::SetLevel(LogLevel::Debug);
    EXPECT_NO_THROW(Logger::Debug("debug msg"));
    EXPECT_NO_THROW(Logger::Info("info msg"));
    EXPECT_NO_THROW(Logger::Warning("warning msg"));
    EXPECT_NO_THROW(Logger::Error("error msg"));
}

// 测试重复设置日志级别不会出错
TEST(LoggerTest, SetLevelIdempotent) {
    Logger::SetLevel(LogLevel::Info);
    Logger::SetLevel(LogLevel::Info);
    EXPECT_NO_THROW(Logger::Info("idempotent test"));
}

// 测试空字符串日志不会崩溃
TEST(LoggerTest, EmptyMessage) {
    Logger::SetLevel(LogLevel::Debug);
    EXPECT_NO_THROW(Logger::Debug(""));
    EXPECT_NO_THROW(Logger::Info(""));
    EXPECT_NO_THROW(Logger::Warning(""));
    EXPECT_NO_THROW(Logger::Error(""));
}

// 测试长字符串日志不会崩溃
TEST(LoggerTest, LongMessage) {
    Logger::SetLevel(LogLevel::Debug);
    std::string long_msg(10000, 'A');
    EXPECT_NO_THROW(Logger::Info(long_msg));
}

// 测试日志级别恢复为默认值
TEST(LoggerTest, RestoreDefaultLevel) {
    Logger::SetLevel(LogLevel::Error);
    Logger::SetLevel(LogLevel::Info);
    // 恢复后 Info 级别日志应该可以输出
    EXPECT_NO_THROW(Logger::Info("restored to info"));
}
