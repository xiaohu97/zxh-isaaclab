#pragma once

#include "FSM/FSMState.h"
#include "walk_policy.h"
#include "walk_trace.h"
#include <atomic>
#include <condition_variable>
#include <thread>

// Registered as Velocity in config.yaml; preserves the existing FSM name and
// joystick bindings while keeping this handoff local to G1-29dof.
class State_Walk : public FSMState
{
public:
    State_Walk(int state_mode, std::string state_string);
    ~State_Walk() { exit(); }
    void enter() override;
    void run() override;
    void exit() override;
    static void record_control_frame();

private:
    g1::WalkEntryConfig entry_cfg_;
    std::unique_ptr<g1::WalkPolicy> policy_;
    std::unique_ptr<g1::WalkHandoff> handoff_;
    std::thread policy_thread_;
    std::atomic<bool> running_{false};
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    bool fault_logged_ = false;
    std::unique_ptr<g1::WalkTrace> trace_;
    double trace_duration_ = 2.0, entry_started_at_ = 0, next_trace_at_ = 0;
    bool tracing_ = false;
    float peak_target_step_ = 0;
    g1::JointVector previous_applied_{};
    std::atomic<double> inference_ms_{0};
    void record_trace(int event, unsigned reason, double now,
                      const g1::ControlFrame& frame, const g1::JointVector& applied);
};

REGISTER_FSM(State_Walk)
