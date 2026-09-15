#include "State_Walk.h"
#include <chrono>

namespace
{
using Clock = std::chrono::steady_clock;
double seconds_now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
g1::ControlHistory control_history;
}

void State_Walk::record_control_frame()
{
    g1::ControlFrame frame;
    frame.time = seconds_now();
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto& state = lowstate->msg_;
        for (int m = 0; m < g1::joint_count; ++m) {
            frame.q[m] = state.motor_state()[m].q();
            frame.dq[m] = state.motor_state()[m].dq();
        }
        const auto& quat = state.imu_state().quaternion();
        frame.quaternion = Eigen::Quaternionf(quat[0], quat[1], quat[2], quat[3]);
        for (int j = 0; j < 3; ++j) frame.angular_velocity[j] = state.imu_state().gyroscope()[j];
    }
    // Both joystick updates and lowcmd writes belong to this control thread.
    frame.command = {lowstate->joystick.ly(), -lowstate->joystick.lx(), -lowstate->joystick.rx()};
    for (int m = 0; m < g1::joint_count; ++m) frame.previous_target[m] = lowcmd->msg_.motor_cmd()[m].q();
    control_history.push(frame);
}

State_Walk::State_Walk(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    const auto cfg = param::config["FSM"][state_string];
    const auto entry = cfg["walk_entry"];
    if (entry) {
        entry_cfg_.blend_time = entry["blend_time_s"].as<double>(0.30);
        entry_cfg_.zero_command_time = entry["zero_command_time_s"].as<double>(0.40);
        entry_cfg_.command_ramp_time = entry["command_ramp_time_s"].as<double>(0.50);
        entry_cfg_.inference_timeout = entry["inference_timeout_s"].as<double>(0.20);
    }
    entry_cfg_.validate();
    handoff_ = std::make_unique<g1::WalkHandoff>(entry_cfg_);
    const auto dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    policy_ = std::make_unique<g1::WalkPolicy>(YAML::LoadFile(dir / "params/deploy.yaml"));
    policy_->load((dir / "exported/policy.onnx").string());

    const auto trace_cfg = cfg["walk_trace"];
    if (trace_cfg && trace_cfg["enabled"].as<bool>(false)) {
        std::filesystem::path directory = trace_cfg["directory"].as<std::string>("log/walk_handoff");
        if (directory.is_relative()) directory = param::proj_dir / directory;
        trace_duration_ = trace_cfg["duration_s"].as<double>(2.0);
        if (!std::isfinite(trace_duration_) || trace_duration_ <= 0 || trace_duration_ > 10) {
            throw std::invalid_argument("walk_trace duration must be in (0, 10] seconds");
        }
        trace_ = std::make_unique<g1::WalkTrace>(directory);
    }

    // Record the configured Passive condition separately from automatic faults.
    if (cfg["transitions"] && cfg["transitions"]["Passive"]) {
        const auto expression = cfg["transitions"]["Passive"].as<std::string>();
        // The base constructor appends configured transitions before the
        // lowstate watchdog. Wrap the existing Passive predicate in place.
        for (auto& check : registered_checks) {
            if (check.second != FSMStringMap.right.at("Passive")) continue;
            const auto requested = check.first;
            check.first = [requested, expression] {
                const bool triggered = requested();
                if (triggered) spdlog::warn("Velocity Passive reason=configured_transition condition={}", expression);
                return triggered;
            };
            break;
        }
    }

    // Check faults before user transitions; never let a failed/stale inference
    // jump to another active policy. Existing emergency/lowstate checks remain.
    registered_checks.insert(registered_checks.begin(), {
        [this]() {
            const double now = seconds_now();
            const auto status = handoff_->snapshot(now);
            const auto frames = control_history.window(1, policy_->step_dt());
            unsigned reason = 0;
            if (status.failed) reason |= 1;
            if (status.result_age > entry_cfg_.inference_timeout) reason |= status.ready ? 4 : 2;
            if (frames.empty()) reason |= 8;
            double tilt = -1, gyro_norm = -1, frame_age = -1;
            if (!frames.empty()) {
                const auto& f = frames.back();
                frame_age = now - f.time;
                if (!f.valid()) reason |= 16;
                if (frame_age > entry_cfg_.inference_timeout) reason |= 32;
                if (f.valid()) {
                    const Eigen::Vector3f gravity = f.quaternion.normalized().conjugate() * Eigen::Vector3f(0, 0, -1);
                    tilt = std::acos(std::clamp(-gravity.z(), -1.0f, 1.0f));
                    gyro_norm = f.angular_velocity.norm();
                    if (tilt > 1.0) reason |= 64;
                }
            }
            if (lowstate->isTimeout()) reason |= 128;
            if (reason && !fault_logged_) {
                std::string names;
                for (const auto& item : std::vector<std::pair<unsigned, const char*>>{
                    {1,"inference_failed"},{2,"first_result_timeout"},{4,"result_timeout"},{8,"no_frame"},
                    {16,"invalid_frame"},{32,"frame_timeout"},{64,"orientation"},{128,"lowstate_timeout"}}) {
                    if (reason & item.first) { if (!names.empty()) names += '|'; names += item.second; }
                }
                spdlog::error("Velocity Passive reason={} mask={} elapsed_ms={:.1f} tilt_rad={:.3f} gyro_norm={:.3f} result_age_ms={:.1f} frame_age_ms={:.1f} last_inference_ms={:.2f}",
                              names, reason, (now-entry_started_at_)*1000, tilt, gyro_norm, status.result_age*1000, frame_age*1000, inference_ms_.load());
                if (!frames.empty() && tracing_) record_trace(3, reason, now, frames.back(), status.applied);
                fault_logged_ = true;
            }
            return reason != 0;
        }, FSMStringMap.right.at("Passive")});
}

void State_Walk::enter()
{
    // Called after the previous state's worker has joined. The q values here
    // are the last targets actually published by that state, in motor order.
    record_control_frame();
    const auto frame = control_history.window(1, policy_->step_dt()).back();
    float max_leg_speed = 0, max_leg_error = 0;
    for (int m = 0; m < 12; ++m) {
        max_leg_speed = std::max(max_leg_speed, std::fabs(frame.dq[m]));
        max_leg_error = std::max(max_leg_error, std::fabs(frame.previous_target[m] - frame.q[m]));
    }
    const Eigen::Vector3f gravity = frame.quaternion.normalized().conjugate() * Eigen::Vector3f(0, 0, -1);
    const float tilt = std::acos(std::clamp(-gravity.z(), -1.0f, 1.0f));
    spdlog::info("Velocity entry measured: pelvis_tilt={:.3f} rad, gyro=[{:.3f},{:.3f},{:.3f}] rad/s, max_leg_dq={:.3f} rad/s, max_leg_target_error={:.3f} rad, waist_pitch_q={:.3f}, waist_pitch_last_target={:.3f}",
                 tilt, frame.angular_velocity.x(), frame.angular_velocity.y(), frame.angular_velocity.z(),
                 max_leg_speed, max_leg_error, frame.q[14], frame.previous_target[14]);
    entry_started_at_ = seconds_now();
    handoff_->begin(frame.previous_target, entry_started_at_);
    previous_applied_ = frame.previous_target;
    peak_target_step_ = 0;
    inference_ms_ = 0;
    tracing_ = static_cast<bool>(trace_);
    next_trace_at_ = entry_started_at_ + .01;
    if (tracing_) record_trace(1, 0, entry_started_at_, frame, previous_applied_);
    fault_logged_ = false;
    policy_->resume_inference();
    for (int m = 0; m < g1::joint_count; ++m) {
        auto& cmd = lowcmd->msg_.motor_cmd()[m];
        cmd.kp() = policy_->stiffness()[m];
        cmd.kd() = policy_->damping()[m];
        cmd.dq() = cmd.tau() = 0;
    }
    spdlog::info("Velocity entry: hold last sent target until inference ready; blend {} s, zero command {} s, ramp {} s",
                 entry_cfg_.blend_time, entry_cfg_.zero_command_time, entry_cfg_.command_ramp_time);
    running_ = true;
    policy_thread_ = std::thread([this] {
        try {
            auto history = control_history.window(5, policy_->step_dt());
            if (history.empty()) throw std::runtime_error("No walk entry history");
            auto current = history.back();
            history.pop_back();
            policy_->reset(history);
            const auto dt = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(policy_->step_dt()));
            auto next = Clock::now();
            bool first = true;
            while (running_) {
                const double now = seconds_now();
                if (now - current.time > entry_cfg_.inference_timeout) throw std::runtime_error("Stale walk sensor frame");
                const double inference_start = seconds_now();
                const auto target = policy_->step(current, handoff_->command_gain(now));
                inference_ms_ = (seconds_now() - inference_start) * 1000;
                if (seconds_now() - current.time > entry_cfg_.inference_timeout) {
                    throw std::runtime_error("Walk inference exceeded the input age limit");
                }
                handoff_->publish(target, seconds_now());
                if (first) {
                    float delta = 0, leg_delta = 0, kp_delta = 0;
                    int worst_motor = 0;
                    for (int m = 0; m < g1::joint_count; ++m) {
                        const float change = std::fabs(target[m] - current.previous_target[m]);
                        if (change > delta) { delta = change; worst_motor = m; }
                        if (m < 12) leg_delta = std::max(leg_delta, change);
                        kp_delta = std::max(kp_delta, policy_->stiffness()[m] * change);
                    }
                    spdlog::info("Velocity first inference ready: candidate max |q_target - last_sent| = {:.4f} rad at motor {}, max_leg_delta={:.4f} rad, max_kp_delta={:.2f} Nm, waist_pitch_candidate={:.3f} rad (before blend; not applied torque)",
                                 delta, worst_motor, leg_delta, kp_delta, target[14]);
                    first = false;
                }
                next += dt;
                if (next < Clock::now()) next = Clock::now() + dt;  // skip missed deadlines; do not burst
                std::unique_lock<std::mutex> lock(wake_mutex_);
                if (wake_.wait_until(lock, next, [this] { return !running_.load(); })) break;
                lock.unlock();
                current = control_history.window(1, policy_->step_dt()).back();
            }
        } catch (const std::exception& e) {
            if (running_) {  // normal state exit also cancels an active Run
                handoff_->fail();
                spdlog::error("Velocity inference failed: {}", e.what());
            }
        }
    });
}

void State_Walk::run()
{
    const auto target = handoff_->sample(seconds_now());
    for (int m = 0; m < g1::joint_count; ++m) {
        lowcmd->msg_.motor_cmd()[m].q() = target[m];
        if (tracing_) peak_target_step_ = std::max(peak_target_step_, std::fabs(target[m]-previous_applied_[m]));
    }
    if (!tracing_) return;
    previous_applied_ = target;
    const double now = seconds_now();
    if (now >= next_trace_at_) {
        const auto frames = control_history.window(1, policy_->step_dt());
        if (frames.empty()) return;
        const bool end = now - entry_started_at_ >= trace_duration_;
        record_trace(end ? 2 : 0, 0, now, frames.back(), target);
        next_trace_at_ = now + .01;
        peak_target_step_ = 0;
        if (end) tracing_ = false;
    }
}

void State_Walk::exit()
{
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        running_ = false;
    }
    if (policy_) policy_->cancel_inference();
    wake_.notify_all();
    if (policy_thread_.joinable()) policy_thread_.join();
    if (tracing_) {
        const double now = seconds_now();
        const auto frames = control_history.window(1, policy_->step_dt());
        if (!frames.empty()) record_trace(2, 0, now, frames.back(), previous_applied_);
        tracing_ = false;
    }
}

void State_Walk::record_trace(int event, unsigned reason, double now,
                             const g1::ControlFrame& frame, const g1::JointVector& applied)
{
    if (!trace_) return;
    const auto status = handoff_->snapshot(now);
    g1::WalkTraceRow row;
    row.event = event; row.reason = reason; row.time = now; row.elapsed = now - entry_started_at_;
    row.frame = frame; row.applied = applied; row.candidate = status.candidate;
    row.result_age = status.result_age; row.ready = status.ready;
    row.command_gain = handoff_->command_gain(now); row.inference_ms = inference_ms_.load();
    row.peak_target_step = peak_target_step_;
    trace_->push(row);
}
