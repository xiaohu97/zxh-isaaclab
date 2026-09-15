#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/observations/motion_observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

static Eigen::Quaternionf init_quat = Eigen::Quaternionf::Identity(); // only the active Mimic worker uses this

Eigen::Quaternionf torso_quat_w(isaaclab::ManagerBasedRLEnv* env) {
    auto root_quat = env->robot->data.root_quat_w;
    // Use the same locked sensor snapshot as root_quat, not a concurrently
    // updated DDS message. joint_pos is policy order, so map waist motor IDs.
    auto waist_q = [&](int motor) {
        const auto& ids = env->robot->data.joint_ids_map;
        return env->robot->data.joint_pos[std::distance(ids.begin(), std::find(ids.begin(), ids.end(), motor))];
    };

    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(waist_q(12), Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(waist_q(13), Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(waist_q(14), Eigen::Vector3f::UnitY()) \
    ;
    return torso_quat;
};

Eigen::Quaternionf anchor_quat_w(isaaclab::MotionLoader* loader)
{
    const auto root_quat = loader->root_quaternion();
    const auto joint_pos = loader->joint_pos();
    Eigen::Quaternionf torso_quat = root_quat \
        * Eigen::AngleAxisf(joint_pos[12], Eigen::Vector3f::UnitZ()) \
        * Eigen::AngleAxisf(joint_pos[13], Eigen::Vector3f::UnitX()) \
        * Eigen::AngleAxisf(joint_pos[14], Eigen::Vector3f::UnitY()) \
    ;
    return torso_quat;
}


namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    // auto & robot = env->robot;
    auto real_quat_w = torso_quat_w(env);
    auto ref_quat_w = anchor_quat_w(env->robot->data.motion_loader);

    auto rot_ = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    const Eigen::Matrix3f rot = rot_.toRotationMatrix().transpose();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

}
}


State_Mimic::State_Mimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if(!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    articulation->data.motion_loader = new isaaclab::MotionLoader(motion_file.string(), cfg["fps"].as<float>());
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    registered_checks.insert(registered_checks.begin(), {
        [this] { return policy_fault_.load() || bad_orientation_.load(); },
        FSMStringMap.right.at("Passive")});
    this->registered_checks.emplace_back(
        std::make_pair(
            [this]()->bool{ return elapsed_.load() > env->robot->data.motion_loader->duration; }, // completed policy steps
            FSMStringMap.right.at("Velocity")
        )
    );
}

void State_Mimic::enter()
{
    action_ready_ = false;
    policy_fault_ = false;
    bad_orientation_ = false;
    elapsed_ = 0;
    // set gain
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    env->reset(); // Update robot state for init_quat calculation
    env->alg->reset_inference_cancel();
    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        try {
        using clock = std::chrono::steady_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        // Initialize timing
        const auto start = clock::now();
        auto sleepTill = start + dt;

        auto ref_yaw = isaaclab::yawQuaternion(env->robot->data.motion_loader->root_quaternion()).toRotationMatrix();
        auto robot_yaw = isaaclab::yawQuaternion(torso_quat_w(env.get())).toRotationMatrix();
        init_quat = robot_yaw * ref_yaw.transpose();
        env->reset();

        while (policy_thread_running)
        {
            env->step();
            const auto target = env->action_manager->processed_actions();
            if (!std::all_of(target.begin(), target.end(), [](float v) { return std::isfinite(v); })) {
                throw std::runtime_error("Non-finite Mimic target");
            }
            const float gz = env->robot->data.projected_gravity_b.z();
            bad_orientation_ = !std::isfinite(gz) || std::acos(std::clamp(-gz, -1.0f, 1.0f)) > 1.0f;
            elapsed_ = env->episode_length * env->step_dt;
            action_ready_ = true;

            // Sleep
            std::unique_lock<std::mutex> lock(wake_mutex_);
            if (wake_.wait_until(lock, sleepTill, [this] { return !policy_thread_running.load(); })) break;
            sleepTill += dt;
        }
        } catch (const std::exception& e) {
            if (policy_thread_running) {
                policy_fault_ = true;
                spdlog::error("Mimic inference failed: {}", e.what());
            }
        }
    });
}


void State_Mimic::run()
{
    if (!action_ready_ || policy_fault_) return;  // hold the last sent target until the first complete result
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
