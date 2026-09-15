#pragma once

#include "walk_handoff.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <thread>
#include <spdlog/spdlog.h>

namespace g1 {
struct WalkTraceRow {
    int event = 0;  // 0 sample, 1 begin, 2 end, 3 fault
    unsigned reason = 0;
    double elapsed = 0, time = 0, result_age = 0, command_gain = 0, inference_ms = 0, peak_target_step = 0;
    bool ready = false;
    ControlFrame frame;
    JointVector candidate{}, applied{};
};

// One producer (FSM thread), one consumer (file writer). Control never waits
// for disk I/O or allocates memory to append a row. If full, drop a trace row.
class WalkTrace {
public:
    explicit WalkTrace(std::filesystem::path directory) : directory_(std::move(directory)), worker_([this]{ write_loop(); }) {}
    ~WalkTrace() { stopping_ = true; wake_.notify_one(); if (worker_.joinable()) worker_.join(); }
    void push(const WalkTraceRow& row) {
        const auto w = written_.load(std::memory_order_relaxed);
        if (w - read_.load(std::memory_order_acquire) >= capacity) { ++dropped_; return; }
        rows_[w % capacity] = row;
        written_.store(w + 1, std::memory_order_release);
        wake_.notify_one();
    }
    unsigned long dropped() const { return dropped_.load(); }
private:
    static constexpr unsigned long capacity = 512;
    std::filesystem::path directory_;
    std::array<WalkTraceRow, capacity> rows_{};
    std::atomic<unsigned long> written_{0}, read_{0}, dropped_{0};
    std::atomic<bool> stopping_{false};
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    void write_loop() {
        try {
            std::filesystem::create_directories(directory_);
            std::ofstream out;
            out.exceptions(std::ios::badbit | std::ios::failbit);
            unsigned long entry = 0;
            while (!stopping_.load() || read_.load() != written_.load()) {
                auto r = read_.load(std::memory_order_relaxed);
                if (r == written_.load(std::memory_order_acquire)) {
                    std::unique_lock<std::mutex> lock(wake_mutex_);
                    wake_.wait_for(lock, std::chrono::milliseconds(50));
                    continue;
                }
                const auto row = rows_[r % capacity];
                read_.store(r + 1, std::memory_order_release);
                if (row.event == 1) {
                    if (out.is_open()) out.close();
                    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                    const auto path = directory_ / ("walk_" + std::to_string(stamp) + "_" + std::to_string(++entry) + ".csv");
                    out.open(path);
                    if (!out) throw std::runtime_error("Cannot open walk trace: " + path.string());
                    out << std::setprecision(9) << "elapsed_s,steady_s,event,reason_mask,result_age_s,frame_age_s,command_gain,inference_ms,peak_control_target_step_rad,ready,qw,qx,qy,qz,gx,gy,gz,cmd_x,cmd_y,cmd_yaw";
                    for (const char* group : {"q", "dq", "previous_target", "candidate", "applied"})
                        for (int m = 0; m < joint_count; ++m) out << ',' << group << m;
                    out << '\n';
                    spdlog::info("Velocity trace: {}", path.string());
                }
                if (!out.is_open()) continue;
                out << row.elapsed << ',' << row.time << ',' << row.event << ',' << row.reason << ',' << row.result_age << ',' << row.time - row.frame.time << ',' << row.command_gain << ',' << row.inference_ms << ',' << row.peak_target_step << ',' << row.ready;
                const auto& q = row.frame.quaternion;
                out << ',' << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z();
                for (int j=0;j<3;++j) out << ',' << row.frame.angular_velocity[j];
                for (float v : row.frame.command) out << ',' << v;
                for (const auto* a : {&row.frame.q, &row.frame.dq, &row.frame.previous_target, &row.candidate, &row.applied})
                    for (float v : *a) out << ',' << v;
                out << '\n';
                if (row.event == 2 || row.event == 3) out.flush();
                if (row.event == 2) {
                    out.close();
                    if (dropped_.load()) spdlog::warn("Velocity trace dropped {} rows (control continues independently)", dropped_.load());
                }
            }
        } catch (const std::exception& e) { spdlog::error("Velocity trace writer stopped: {}", e.what()); }
    }
};
}
