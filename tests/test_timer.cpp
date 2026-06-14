#include <gtest/gtest.h>

#include "timer.h"

#include <chrono>
#include <thread>

using namespace inference;

// 测试 Timer 初始状态不是运行中
TEST(TimerTest, InitialNotRunning) {
    Timer timer;
    EXPECT_FALSE(timer.IsRunning());
}

// 测试 Start 后 IsRunning 返回 true
TEST(TimerTest, StartSetsRunning) {
    Timer timer;
    timer.Start();
    EXPECT_TRUE(timer.IsRunning());
}

// 测试 Stop 后 IsRunning 返回 false
TEST(TimerTest, StopClearsRunning) {
    Timer timer;
    timer.Start();
    timer.Stop();
    EXPECT_FALSE(timer.IsRunning());
}

// 测试计时器测量大致正确的时间间隔（毫秒）
TEST(TimerTest, ElapsedMillisecondsApprox) {
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    timer.Stop();
    double elapsed = timer.ElapsedMilliseconds();
    // 允许 ±50ms 的误差，因为系统调度可能有偏差
    EXPECT_GE(elapsed, 50.0);
    EXPECT_LE(elapsed, 200.0);
}

// 测试计时器测量大致正确的时间间隔（秒）
TEST(TimerTest, ElapsedSecondsApprox) {
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    timer.Stop();
    double elapsed = timer.ElapsedSeconds();
    EXPECT_GE(elapsed, 0.05);
    EXPECT_LE(elapsed, 0.2);
}

// 测试运行中调用 ElapsedMilliseconds 返回当前耗时
TEST(TimerTest, ElapsedWhileRunning) {
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // 不调用 Stop，直接获取耗时
    double elapsed = timer.ElapsedMilliseconds();
    EXPECT_GE(elapsed, 10.0);
    EXPECT_TRUE(timer.IsRunning());
}

// 测试 Reset 后计时器状态清空
TEST(TimerTest, ResetClearsState) {
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.Stop();
    timer.Reset();
    EXPECT_FALSE(timer.IsRunning());
}

// 测试多次 Start/Stop 循环
TEST(TimerTest, MultipleStartStopCycles) {
    Timer timer;
    for (int i = 0; i < 5; ++i) {
        timer.Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        timer.Stop();
        double elapsed = timer.ElapsedMilliseconds();
        EXPECT_GT(elapsed, 0.0);
    }
}

// 测试未 Start 就调用 Stop 不会崩溃
TEST(TimerTest, StopWithoutStart) {
    Timer timer;
    EXPECT_NO_THROW(timer.Stop());
    EXPECT_FALSE(timer.IsRunning());
}

// 测试未 Start 就调用 ElapsedMilliseconds 返回值接近 0
TEST(TimerTest, ElapsedWithoutStart) {
    Timer timer;
    double elapsed = timer.ElapsedMilliseconds();
    // 未启动时，start_time_ 为默认值，elapsed 可能为负或接近 0
    // 主要验证不崩溃
    EXPECT_NO_THROW(timer.ElapsedMilliseconds());
}

// 测试 Reset 后重新计时
TEST(TimerTest, ResetAndRestart) {
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.Stop();

    timer.Reset();
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.Stop();

    double elapsed = timer.ElapsedMilliseconds();
    EXPECT_GE(elapsed, 1.0);
    EXPECT_LE(elapsed, 100.0);
}

// 测试 ElapsedSeconds 与 ElapsedMilliseconds 的一致性
TEST(TimerTest, SecondsMillisecondsConsistency) {
    Timer timer;
    timer.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.Stop();

    double ms = timer.ElapsedMilliseconds();
    double sec = timer.ElapsedSeconds();
    EXPECT_NEAR(sec, ms / 1000.0, 0.01);
}
