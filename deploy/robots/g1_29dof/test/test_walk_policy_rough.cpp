// Perceptive walk policy (six proprioceptive terms x 5 frames + height_scan x 1 frame)
// through the production WalkPolicy / ObservationManager / ONNX path.
//   usage: test_walk_policy_rough <synthetic_rough_policy_dir> <deployed_velocity_policy_dir>
#include "walk_policy.h"
#include <iostream>

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
        if (argc != 3) throw std::runtime_error("usage: test_walk_policy_rough <rough_policy_dir> <velocity_policy_dir>");
        const std::filesystem::path rough_dir = argv[1], blind_dir = argv[2];
        auto rough_cfg = YAML::LoadFile(rough_dir / "params/deploy.yaml");
        g1::WalkPolicy rough(rough_cfg);
        rough.load((rough_dir / "exported/policy.onnx").string());
        require(rough.uses_height_scan() && rough.height_scan_size() == 187, "rough policy must declare a 187-cell height scan");

        auto blind_cfg = YAML::LoadFile(blind_dir / "params/deploy.yaml");
        g1::WalkPolicy blind(blind_cfg);
        blind.load((blind_dir / "exported/policy.onnx").string());
        require(!blind.uses_height_scan(), "deployed velocity policy is proprioceptive");

        const auto ids = rough_cfg["joint_ids_map"].as<std::vector<int>>();
        const auto offset = rough_cfg["default_joint_pos"].as<std::vector<float>>();
        const float scan_offset = rough_cfg["observations"]["height_scan"]["params"]["offset"].as<float>();
        near(scan_offset, 0.5f, "fixture must use the training height_scan offset");

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
                f.previous_target[ids[i]] = offset[i] + .25f * (.1f * i + k);
            }
            // z_ground - z_torso per cell; a different map in every frame so a
            // multi-frame or stale-frame height history would be caught.
            f.height_map.assign(187, -0.78f - 0.01f * k);
            f.height_map[3 * 17 + 5] = -0.63f + 0.01f * k;  // step in front-left
            f.height_map[100] = -5.0f;                        // deep hole -> clipped to +1
            f.height_map[101] = 5.0f;                         // wall -> clipped to -1
        }
        rough.reset({history.begin(), history.end() - 1});
        rough.prepare(history.back(), 0);
        const auto observed = rough.env().observation_manager->compute().at("obs");
        require(observed.size() == 667, "rough observation must be 480 + 187");

        // The proprioceptive prefix must be exactly what the deployed blind policy
        // computes from the same frames: per-term history blocks, unchanged order.
        blind.reset({history.begin(), history.end() - 1});
        blind.prepare(history.back(), 0);
        const auto blind_obs = blind.env().observation_manager->compute().at("obs");
        require(blind_obs.size() == 480, "blind observation dimension");
        for (int i = 0; i < 480; ++i) near(observed[i], blind_obs[i], "proprioceptive prefix differs from the blind policy");

        // Tail: -(z_ground - z_torso) - offset from the newest frame only, clipped to [-1, 1].
        const auto& newest = history.back();
        for (int i = 0; i < 187; ++i) {
            const float expected = std::clamp(-newest.height_map[i] - scan_offset, -1.0f, 1.0f);
            near(observed[480 + i], expected, "height_scan cell value");
        }
        near(observed[480 + 100], 1.0f, "deep hole clips to +1");
        near(observed[480 + 101], -1.0f, "wall clips to -1");
        near(observed[480 + 3 * 17 + 5], -(-0.63f + 0.04f) - 0.5f, "step cell from the newest frame");

        // Frames without (or with a wrong-sized) map are rejected before touching the network.
        auto bare = history.back();
        bare.height_map.clear();
        bool rejected = false;
        try { rough.prepare(bare, 0); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "frame without height map was not rejected");
        bare.height_map.assign(186, -0.78f);
        rejected = false;
        try { rough.prepare(bare, 0); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "wrong-sized height map was not rejected");
        require(rough.env().observation_manager->compute().at("obs").size() == 667, "rejected frames must not corrupt the manager");

        // The blind policy ignores a map carried by the frame (shared control history).
        blind.prepare(history.back(), 0);
        require(blind.env().observation_manager->compute().at("obs").size() == 480, "blind policy must ignore the height map");

        // End to end: the synthetic network reads the scan, so a different map changes the action.
        auto flat = history.back();
        flat.height_map.assign(187, -0.78f);
        auto stairs = flat;
        for (int iy = 0; iy < 11; ++iy) for (int ix = 9; ix < 17; ++ix) stairs.height_map[iy * 17 + ix] = -0.78f + 0.15f * (ix - 8);
        const auto a_flat = rough.step(flat, 1.0f);
        const auto a_stairs = rough.step(stairs, 1.0f);
        float delta = 0;
        for (int m = 0; m < 29; ++m) {
            require(std::isfinite(a_flat[m]) && std::isfinite(a_stairs[m]), "non-finite action");
            delta = std::max(delta, std::fabs(a_flat[m] - a_stairs[m]));
        }
        require(delta > 1e-4f, "height map did not reach the network");

        std::cout << "PASS perceptive walk policy: 667-dim layout, blind prefix parity, scan clipping, map validation, ONNX\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
