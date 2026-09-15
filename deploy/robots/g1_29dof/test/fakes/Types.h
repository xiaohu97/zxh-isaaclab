#pragma once

// Test-only replacement for G1 DDS types. No sockets, publishers, or robot I/O.
#include <array>
#include <memory>
#include <mutex>
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

struct TestMotor
{
    float q_ = 0, dq_ = 0, kp_ = 0, kd_ = 0, tau_ = 0;
    float& q() { return q_; }
    float q() const { return q_; }
    float& dq() { return dq_; }
    float dq() const { return dq_; }
    float& kp() { return kp_; }
    float& kd() { return kd_; }
    float& tau() { return tau_; }
};
struct TestImu
{
    std::array<float, 4> quat{1, 0, 0, 0};
    std::array<float, 3> gyro{};
    const auto& quaternion() const { return quat; }
    const auto& gyroscope() const { return gyro; }
};
struct TestStateMessage
{
    std::array<TestMotor, 29> motors;
    TestImu imu;
    const auto& motor_state() const { return motors; }
    const auto& imu_state() const { return imu; }
};
struct TestCommandMessage
{
    std::array<TestMotor, 29> motors;
    auto& motor_cmd() { return motors; }
};
struct LowState_t
{
    using SharedPtr = std::shared_ptr<LowState_t>;
    std::mutex mutex_;
    TestStateMessage msg_;
    unitree::common::UnitreeJoystick joystick;
    void update() {}
    bool isTimeout() const { return false; }
};
struct LowCmd_t
{
    TestCommandMessage msg_;
    std::array<float, 29> published{};
    void unlockAndPublish()
    {
        for (int m = 0; m < 29; ++m) published[m] = msg_.motors[m].q();
    }
};
