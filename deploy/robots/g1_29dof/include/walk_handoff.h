#pragma once

// G1 walk-entry primitives. No DDS, motor publishing, or background threads:
// the same code is exercised by the offline regression tests.
#include <array>
#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <eigen3/Eigen/Dense>

namespace g1
{
constexpr int joint_count = 29;
using JointVector = std::array<float, joint_count>;

struct ControlFrame
{
    double time = 0;  // steady-clock seconds
    JointVector q{}, dq{}, previous_target{};  // motor order, previous *sent* q
    Eigen::Quaternionf quaternion = Eigen::Quaternionf::Identity();
    Eigen::Vector3f angular_velocity = Eigen::Vector3f::Zero();
    std::array<float, 3> command{};  // already filtered joystick: ly, -lx, -rx

    bool valid() const
    {
        const auto finite = [](const auto& values) {
            return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
        };
        return std::isfinite(time) && finite(q) && finite(dq) && finite(previous_target)
            && finite(command) && quaternion.coeffs().allFinite()
            && quaternion.norm() > 1e-6f && angular_velocity.allFinite();
    }
};

class ControlHistory
{
public:
    void push(const ControlFrame& frame)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        frames_.push_back(frame);
        // 1 kHz control samples; retains 250 ms for a five-frame 50 Hz history.
        while (frames_.size() > 256) frames_.pop_front();
    }

    std::vector<ControlFrame> window(int count, double step_dt) const
    {
        if (count < 1 || step_dt <= 0) throw std::invalid_argument("Invalid history window");
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames_.empty()) return {};
        std::vector<ControlFrame> result;
        for (int i = count - 1; i >= 0; --i) {
            const double target_time = frames_.back().time - i * step_dt;
            auto selected = frames_.begin();
            for (auto it = frames_.begin(); it != frames_.end() && it->time <= target_time + 1e-9; ++it) {
                selected = it;
            }
            result.push_back(*selected);
        }
        return result;  // oldest to newest; pad with oldest if just started
    }

private:
    mutable std::mutex mutex_;
    std::deque<ControlFrame> frames_;
};

struct WalkEntryConfig
{
    double blend_time = 0.30;
    double zero_command_time = 0.40;
    double command_ramp_time = 0.50;
    double inference_timeout = 0.20;

    void validate() const
    {
        if (!std::isfinite(blend_time) || !std::isfinite(zero_command_time)
            || !std::isfinite(command_ramp_time) || !std::isfinite(inference_timeout)
            || blend_time <= 0 || zero_command_time < blend_time
            || command_ramp_time <= 0 || inference_timeout <= 0) {
            throw std::invalid_argument("Invalid walk_entry timing configuration");
        }
    }
};

// Complete target snapshots are exchanged under one mutex. The control thread
// owns sample(); the inference thread only publishes candidates. Entry time is
// distinct from first application time, so slow initial inference cannot skip
// the blend or expose a prior entry's target.
class WalkHandoff
{
public:
    explicit WalkHandoff(WalkEntryConfig cfg = {}) : cfg_(cfg) { cfg_.validate(); }

    void begin(const JointVector& previous_target, double time)
    {
        require_finite(previous_target);
        std::lock_guard<std::mutex> lock(mutex_);
        seed_ = applied_ = desired_ = previous_target;
        entered_at_ = last_result_at_ = time;
        first_applied_at_ = -1;
        ready_ = failed_ = false;
    }

    void publish(const JointVector& target, double time)
    {
        require_finite(target);
        std::lock_guard<std::mutex> lock(mutex_);
        desired_ = target;
        last_result_at_ = time;
        ready_ = true;
    }

    JointVector sample(double time)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_ || fault_locked(time)) return applied_;
        if (first_applied_at_ < 0) first_applied_at_ = time;
        const float weight = smooth((time - first_applied_at_) / cfg_.blend_time);
        for (int i = 0; i < joint_count; ++i) {
            applied_[i] = seed_[i] + weight * (desired_[i] - seed_[i]);
        }
        return applied_;
    }

    float command_gain(double time) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (first_applied_at_ < 0 || fault_locked(time)) return 0;
        return smooth((time - first_applied_at_ - cfg_.zero_command_time) / cfg_.command_ramp_time);
    }

    bool fault(double time) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return fault_locked(time);
    }

    struct Snapshot {
        JointVector candidate{}, applied{};
        double result_age = 0;
        bool ready = false, failed = false;
    };

    Snapshot snapshot(double time) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return {desired_, applied_, time - (ready_ ? last_result_at_ : entered_at_), ready_, failed_};
    }

    void fail()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
    }

private:
    static float smooth(double x)
    {
        x = std::clamp(x, 0.0, 1.0);
        return static_cast<float>(x * x * x * (x * (6 * x - 15) + 10));
    }
    static void require_finite(const JointVector& values)
    {
        for (float v : values) if (!std::isfinite(v)) throw std::invalid_argument("Non-finite walk target");
    }
    bool fault_locked(double time) const
    {
        return failed_ || time - (ready_ ? last_result_at_ : entered_at_) > cfg_.inference_timeout;
    }

    WalkEntryConfig cfg_;
    mutable std::mutex mutex_;
    JointVector seed_{}, applied_{}, desired_{};
    double entered_at_ = 0, last_result_at_ = 0, first_applied_at_ = -1;
    bool ready_ = false, failed_ = false;
};
}  // namespace g1
