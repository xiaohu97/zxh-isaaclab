#include "State_Walk.h"
#include <iostream>
#include <sstream>
#include <spdlog/sinks/ostream_sink.h>

std::unique_ptr<LowCmd_t> FSMState::lowcmd;
std::shared_ptr<LowState_t> FSMState::lowstate;
std::shared_ptr<Keyboard> FSMState::keyboard;

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv)
{
    try {
        if (argc != 2) throw std::runtime_error("usage: test_state_walk <g1_project_directory>");
        param::proj_dir = std::filesystem::absolute(argv[1]);
        param::config = YAML::LoadFile(param::proj_dir / "config/config.yaml");
        param::config["FSM"]["Velocity"]["walk_trace"]["enabled"] = false; // CSV writer has its own isolated test
        require(param::config["FSM"]["_"]["Velocity"]["type"].as<std::string>() == "Walk", "Velocity not wired to State_Walk");
        for (const auto& item : param::config["FSM"]["_"]) {
            FSMStringMap.insert({item.second["id"].as<int>(), item.first.as<std::string>()});
        }
        FSMState::lowcmd = std::make_unique<LowCmd_t>();
        FSMState::lowstate = std::make_shared<LowState_t>();
        FSMState::control_observer = State_Walk::record_control_frame;
        const auto cfg = YAML::LoadFile(param::proj_dir / "config/policy/velocity/params/deploy.yaml");
        const auto ids = cfg["joint_ids_map"].as<std::vector<int>>();
        const auto defaults = cfg["default_joint_pos"].as<std::vector<float>>();
        for (int i = 0; i < 29; ++i) {
            FSMState::lowstate->msg_.motors[ids[i]].q() = defaults[i];
            FSMState::lowcmd->msg_.motors[ids[i]].q() = defaults[i];
        }
        std::ostringstream diagnostic_log;
        auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(diagnostic_log);
        spdlog::default_logger()->sinks().push_back(sink);
        State_Walk state(3, "Velocity");
        // Populate real-time history through the production control observer.
        for (int i = 0; i < 90; ++i) {
            State_Walk::record_control_frame();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (int entry = 0; entry < 3; ++entry) {
            g1::JointVector last_sent;
            for (int i = 0; i < 29; ++i) {
                const int m = ids[i];
                last_sent[m] = defaults[i] + .04f * (entry + 1);
                FSMState::lowcmd->msg_.motors[m].q() = last_sent[m];
            }
            FSMState::lowcmd->unlockAndPublish();
            state.enter();
            bool changed = false;
            for (int tick = 0; tick < 70; ++tick) {
                state.pre_run();
                state.run();
                state.post_run();
                if (tick == 0) require(FSMState::lowcmd->published == last_sent, "first control tick published a stale/candidate target");
                for (int m = 0; m < 29; ++m) {
                    const float sent = FSMState::lowcmd->published[m];
                    require(std::isfinite(sent), "non-finite motor output");
                    changed = changed || std::fabs(sent - last_sent[m]) > 1e-5f;
                }
                for (const auto& check : state.registered_checks) require(!check.first(), "unexpected FSM transition/fault during entry");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            require(changed, "inference worker never delivered a live target");
            if (entry == 2) {
                {
                    std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
                    FSMState::lowstate->msg_.imu.quat = {std::cos(.525f), std::sin(.525f), 0, 0};
                }
                state.pre_run();
                state.run();
                require(state.registered_checks.front().first(), "orientation protection did not trigger");
                require(diagnostic_log.str().find("reason=orientation mask=64") != std::string::npos,
                        "orientation protection was not distinguished from inference/sensor faults");
            }
            const auto start = std::chrono::steady_clock::now();
            state.exit();
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            require(elapsed < 200, "worker exit exceeded inference watchdog period");
            std::cout << "PASS production State_Walk entry " << entry + 1 << ": first sent delta 0, worker exit " << elapsed << " ms\n";
        }
        FSMState::control_observer = {};
        std::cout << "PASS real state class, observer, worker and ONNX with fake DDS types; no robot I/O\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
