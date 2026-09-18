// Production State_Walk configured as the perceptive Velocity_Rough state, with the
// fake DDS types and the fake height-map source. No robot I/O.
//   usage: test_state_walk_rough <g1_project_directory> <synthetic_rough_policy_dir>
#include "State_Walk.h"
#include "dds_height_map_source.h"  // resolves to fakes/ in this build
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

static g1::HeightMapGrid make_grid(double stamp, float value)
{
    g1::HeightMapGrid grid;
    grid.stamp = stamp;
    grid.width = 17;
    grid.height = 11;
    grid.resolution = 0.1f;
    grid.origin = {-0.8f, -0.5f};
    grid.data.assign(187, value);
    return grid;
}

int main(int argc, char** argv)
{
    try {
        if (argc != 3) throw std::runtime_error("usage: test_state_walk_rough <g1_project_directory> <rough_policy_dir>");
        param::proj_dir = std::filesystem::absolute(argv[1]);
        const auto policy_dir = std::filesystem::absolute(argv[2]);
        param::config = YAML::LoadFile(param::proj_dir / "config/config.yaml");
        // The shipped config keeps Velocity_Rough disabled until a policy is exported;
        // enable it here against the synthetic fixture.
        param::config["FSM"]["_"]["Velocity_Rough"]["id"] = 6;
        param::config["FSM"]["_"]["Velocity_Rough"]["type"] = "Walk";
        auto rough = param::config["FSM"]["Velocity_Rough"];
        require(rough && rough["height_map"] && rough["height_map"]["grid"], "config.yaml must ship the Velocity_Rough height_map section");
        rough["policy_dir"] = policy_dir.string();
        rough["walk_trace"]["enabled"] = false;
        for (const auto& item : param::config["FSM"]["_"]) {
            FSMStringMap.insert({item.second["id"].as<int>(), item.first.as<std::string>()});
        }
        FSMState::lowcmd = std::make_unique<LowCmd_t>();
        FSMState::lowstate = std::make_shared<LowState_t>();
        FSMState::control_observer = State_Walk::record_control_frame;
        const auto cfg = YAML::LoadFile(policy_dir / "params/deploy.yaml");
        const auto ids = cfg["joint_ids_map"].as<std::vector<int>>();
        const auto defaults = cfg["default_joint_pos"].as<std::vector<float>>();
        for (int i = 0; i < 29; ++i) {
            FSMState::lowstate->msg_.motors[ids[i]].q() = defaults[i];
            FSMState::lowcmd->msg_.motors[ids[i]].q() = defaults[i];
        }
        std::ostringstream diagnostic_log;
        spdlog::default_logger()->sinks().push_back(std::make_shared<spdlog::sinks::ostream_sink_mt>(diagnostic_log));

        // 1. A perceptive policy without a mapping source must not start at all.
        {
            const auto saved = YAML::Clone(rough["height_map"]);
            rough.remove("height_map");
            bool rejected = false;
            try { State_Walk state(6, "Velocity_Rough"); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "perceptive policy without height_map config was accepted");
            rough["height_map"] = saved;
        }
        // 2. A grid that does not match the policy's scan must not start either.
        {
            rough["height_map"]["grid"]["width"] = 16;
            bool rejected = false;
            try { State_Walk state(6, "Velocity_Rough"); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "mismatched height_map grid was accepted");
            rough["height_map"]["grid"]["width"] = 17;
        }

        State_Walk state(6, "Velocity_Rough");
        auto* source = g1::DdsHeightMapSource::instance;
        require(source && source->topic() == "rt/perceptive/height_map", "DDS height map source not created from config");
        require(State_Walk::height_map_status() == "unconfigured" || State_Walk::height_map_status() == "none", "status before frames");

        // 3. Frames before any map: flat fallback, logged once, no fault.
        State_Walk::record_control_frame();
        require(State_Walk::height_map_status() == "none", "no message yet must read as none");
        require(diagnostic_log.str().find("Height map none") != std::string::npos, "missing-map fallback was not logged");

        const auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        source->publish(make_grid(now(), -0.78f));
        State_Walk::record_control_frame();
        require(State_Walk::height_map_status() == "live", "fresh map must read as live");
        require(diagnostic_log.str().find("Height map live") != std::string::npos, "live transition was not logged");

        source->publish(make_grid(now() - 5.0, -0.78f));
        State_Walk::record_control_frame();
        require(State_Walk::height_map_status() == "stale", "old map must read as stale");
        require(diagnostic_log.str().find("Height map stale") != std::string::npos, "stale transition was not logged");

        auto wrong = make_grid(now(), -0.78f);
        wrong.width = 11;
        wrong.height = 17;
        source->publish(wrong);
        State_Walk::record_control_frame();
        require(State_Walk::height_map_status() == "invalid", "mismatched map must read as invalid");

        // 4. Full entry with a live map that keeps arriving: the worker must deliver targets and no fault fires.
        for (int i = 0; i < 90; ++i) {
            source->publish(make_grid(now(), -0.78f));
            State_Walk::record_control_frame();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (int scenario = 0; scenario < 2; ++scenario) {
            const bool with_map = scenario == 0;
            // Scenario 2: the mapper stopped 5 s ago; the gate must hand the policy flat ground and keep walking.
            if (!with_map) source->publish(make_grid(now() - 5.0, -0.78f));
            g1::JointVector last_sent;
            for (int i = 0; i < 29; ++i) {
                last_sent[ids[i]] = defaults[i] + .03f;
                FSMState::lowcmd->msg_.motors[ids[i]].q() = last_sent[ids[i]];
            }
            FSMState::lowcmd->unlockAndPublish();
            state.enter();
            bool changed = false;
            for (int tick = 0; tick < 70; ++tick) {
                if (with_map) source->publish(make_grid(now(), -0.78f));  // 5: no fresh map -> flat fallback must still walk
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
            require(State_Walk::height_map_status() == (with_map ? "live" : "stale"), "height map status during entry");
            state.exit();
            std::cout << "PASS Velocity_Rough entry " << (with_map ? "with live map" : "with stale map (flat fallback)") << '\n';
        }
        require(diagnostic_log.str().find("Velocity entry height map: live") != std::string::npos, "entry log must report the map status");
        FSMState::control_observer = {};
        std::cout << "PASS perceptive State_Walk: config validation, fallback/live/stale/invalid transitions, worker with fake DDS map\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
