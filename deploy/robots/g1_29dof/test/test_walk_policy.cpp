#include "walk_policy.h"
#include <atomic>
#include <iostream>
#include <thread>

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
static void near(float a, float b, const char* message)
{
    if (std::fabs(a - b) > 2e-5f) throw std::runtime_error(message);
}

int main(int argc, char** argv)
{
    try {
        if (argc != 3) throw std::runtime_error("usage: test_walk_policy <velocity_policy_dir> <jump3_csv>");
        const std::filesystem::path dir = argv[1];
        auto cfg = YAML::LoadFile(dir / "params/deploy.yaml");
        g1::WalkPolicy policy(cfg);
        policy.load((dir / "exported/policy.onnx").string());
        auto& env = policy.env();
        const auto ids = cfg["joint_ids_map"].as<std::vector<int>>();
        const auto offset = cfg["default_joint_pos"].as<std::vector<float>>();
        const auto raw_clip = cfg["actions"]["JointPositionAction"]["raw_clip"].as<std::vector<float>>();
        require(raw_clip == std::vector<float>({-10, 10}), "deployed yawfix must use its training action clip");

        // Regression: reset must clear the physical targets, and invalid action
        // data must leave the previous complete target unchanged.
        env.action_manager->process_action(std::vector<float>(29, 30));
        for (float a : env.action_manager->action()) near(a, 10, "raw action clipping");
        env.action_manager->reset();
        const auto reset_target = env.action_manager->processed_actions();
        for (int i = 0; i < 29; ++i) near(reset_target[i], offset[i], "reset retained stale physical target");
        bool rejected = false;
        try { env.action_manager->process_action(std::vector<float>(28, 0)); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "wrong action dimension was not rejected");
        auto bad = std::vector<float>(29, 0);
        bad[3] = NAN;
        rejected = false;
        try { env.action_manager->process_action(bad); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && env.action_manager->processed_actions() == reset_target, "invalid output changed target cache");

        std::vector<g1::ControlFrame> history(5);
        for (int k = 0; k < 5; ++k) {
            auto& f = history[k];
            f.time = .02 * k;
            f.angular_velocity = Eigen::Vector3f(.1 * k, .2 * k, -.1 * k);
            f.quaternion = Eigen::AngleAxisf(.03f * k, Eigen::Vector3f::UnitY());
            f.command = {.7f, -.9f, .9f};
            for (int i = 0; i < 29; ++i) {
                f.q[ids[i]] = offset[i] + .01f * i + .02f * k;
                f.dq[ids[i]] = .1f * i - .2f * k;
                // Deliberately unlike measured q; catches seeding from measured
                // joint pose or raw Jump action instead of the last sent target.
                f.previous_target[ids[i]] = offset[i] + .25f * (.1f * i + k);
            }
        }
        history[2].previous_target[ids[8]] = offset[8] + 5;  // equivalent raw 20 -> clipped to 10
        policy.reset({history.begin(), history.end() - 1});
        policy.prepare(history.back(), 0);
        const auto observed = env.observation_manager->compute().at("obs");
        // Independent expected 6-term × 5-frame layout, with fixed exported
        // scales. This catches ordering, mapping and startup-history mistakes.
        std::vector<float> expected;
        for (const auto& f : history) for (int j = 0; j < 3; ++j) expected.push_back(.2f * f.angular_velocity[j]);
        for (const auto& f : history) {
            Eigen::Vector3f g = f.quaternion.conjugate() * Eigen::Vector3f(0, 0, -1);
            expected.insert(expected.end(), g.data(), g.data() + 3);
        }
        expected.insert(expected.end(), 15, 0);  // entry has zero velocity commands
        for (const auto& f : history) for (int i = 0; i < 29; ++i) expected.push_back(f.q[ids[i]] - offset[i]);
        for (const auto& f : history) for (int i = 0; i < 29; ++i) expected.push_back(.05f * f.dq[ids[i]]);
        for (const auto& f : history) for (int i = 0; i < 29; ++i) {
            expected.push_back(std::clamp((f.previous_target[ids[i]] - offset[i]) / .25f, -10.0f, 10.0f));
        }
        require(observed.size() == 480 && expected.size() == 480, "observation dimension");
        for (int i = 0; i < 480; ++i) near(observed[i], expected[i], "warm history differs from expected sensor/command sequence");

        const auto raw_expected = env.alg->act({{"obs", expected}});
        rejected = false;
        try { env.alg->act({{"obs", std::vector<float>(479, 0)}}); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "wrong ONNX input dimension must be rejected before reading the tensor buffer");
        policy.reset({history.begin(), history.end() - 1});
        const auto target = policy.step(history.back(), 0);
        for (int i = 0; i < 29; ++i) {
            near(target[ids[i]], offset[i] + .25f * std::clamp(raw_expected[i], -10.0f, 10.0f), "ONNX/physical target mapping mismatch");
        }
        auto applied_frame = history.back();
        for (int i = 0; i < 29; ++i) applied_frame.previous_target[ids[i]] = offset[i] + .025f;
        policy.prepare(applied_frame, .5f);
        for (float a : env.action_manager->action()) near(a, .1f, "last_action must track applied blended target, not unexecuted candidate");
        const auto command = isaaclab::mdp::velocity_commands(&env, YAML::Node());
        near(command[0], .35f, "command ramp x");
        near(command[1], -.4f, "command must be clamped before ramp y");
        near(command[2], .4f, "command must be clamped before ramp yaw");
        std::cout << "PASS raw_clip/reset, 480-value history parity, applied-action feedback, actual ONNX and motor mapping\n";

        // Real CSV endpoint regression (surrogate sensor data, no physics).
        isaaclab::MotionLoader motion(argv[2], 120);
        std::vector<g1::ControlFrame> terminal;
        for (int k = 4; k >= 0; --k) {
            const int index = motion.num_frames - 1 - static_cast<int>(std::round(k * .02 * 120));
            g1::ControlFrame f;
            f.time = 50 - k * .02;
            for (int m = 0; m < 29; ++m) {
                f.q[m] = f.previous_target[m] = motion.dof_positions[index][m];
                f.dq[m] = motion.dof_velocities[index][m];
            }
            f.quaternion = motion.root_quaternions[index];
            const int a = std::min(index, motion.num_frames - 2);
            const Eigen::AngleAxisf rotation(motion.root_quaternions[a].conjugate() * motion.root_quaternions[a + 1]);
            f.angular_velocity = rotation.axis() * rotation.angle() * 120;
            terminal.push_back(f);
        }
        policy.reset({terminal.begin(), terminal.end() - 1});
        const auto candidate = policy.step(terminal.back(), 0);
        g1::WalkHandoff handoff;
        handoff.begin(terminal.back().previous_target, 50);
        require(handoff.sample(50.05) == terminal.back().previous_target, "CSV endpoint delay lost outgoing command");
        handoff.publish(candidate, 50.05);
        require(handoff.sample(50.05) == terminal.back().previous_target, "CSV endpoint first candidate produced a target discontinuity");
        float unblended_delta = 0, max_step = 0;
        auto last = terminal.back().previous_target;
        for (int m = 0; m < 29; ++m) unblended_delta = std::max(unblended_delta, std::fabs(candidate[m] - last[m]));
        for (int ms = 1; ms <= 300; ++ms) {
            const double t = 50.05 + ms * .001;
            if (ms % 20 == 0) handoff.publish(candidate, t);
            const auto sent = handoff.sample(t);
            for (int m = 0; m < 29; ++m) max_step = std::max(max_step, std::fabs(sent[m] - last[m]));
            last = sent;
        }
        for (int m = 0; m < 29; ++m) near(last[m], candidate[m], "CSV blend did not finish");
        std::cout << "PASS actual Jump3 CSV endpoint -> ONNX -> handoff: first sent delta 0 rad; candidate delta "
                  << unblended_delta << " rad; fixed-candidate 1 ms max delta " << max_step << " rad\n";
        std::cout << "Scope: offline reference input and command-path checks, not closed-loop robot stability.\n";

        policy.cancel_inference();
        rejected = false;
        try { policy.step(terminal.back(), 0); } catch (const Ort::Exception&) { rejected = true; }
        require(rejected, "canceled inference must stop instead of delaying FSM exit");
        policy.resume_inference();
        policy.step(terminal.back(), 0);
        std::cout << "PASS inference cancellation and reuse on next entry\n";

        // ActionManager is also used by the outgoing Mimic worker and 1 kHz
        // writer. Check that all 29 processed values come from one update.
        env.action_manager->process_action(std::vector<float>(29, 0));
        std::atomic<bool> failed{false};
        std::thread writer([&] {
            for (int n = 0; n < 10000; ++n) env.action_manager->process_action(std::vector<float>(29, n % 2 ? 1 : -1));
        });
        for (int n = 0; n < 10000; ++n) {
            const auto a = env.action_manager->processed_actions();
            const float base = a[0] - offset[0];
            for (int i = 1; i < 29; ++i) if (std::fabs(a[i] - offset[i] - base) > 1e-6f) failed = true;
        }
        writer.join();
        require(!failed, "ActionManager published a partially updated vector");
        std::cout << "PASS concurrent ActionManager snapshot stress test\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
