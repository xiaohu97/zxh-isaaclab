// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/envs/mdp/commands/motion_command.h"
#include "isaaclab/envs/mdp/commands/gait_command.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // 可控步态命令（Run 任务）：deploy.yaml 里有 commands.base_velocity.gait 才创建。
        // 必须先于 ObservationManager，它构造时会把每个观测项调用一次来填历史。
        {
            const YAML::Node c = cfg;  // const 访问不会往 map 里插空节点
            if (c["commands"] && c["commands"]["base_velocity"] && c["commands"]["base_velocity"]["gait"]) {
                gait_command = std::make_unique<GaitCommand>(c["commands"]["base_velocity"], this->step_dt);
            }
        }

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        if(robot->data.motion_loader) {
            robot->data.motion_loader->reset(robot->data);
        }
        if(gait_command) {
            gait_command->reset(robot->data.joystick);
        }
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        if(robot->data.motion_loader) {
            robot->data.motion_loader->update(episode_length * step_dt);
        }
        if(gait_command) {
            gait_command->update(robot->data.joystick);  // 观测之前更新，gait_commands/gait_clock 只读
        }
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);
        action_manager->process_action(action);
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    std::unique_ptr<GaitCommand> gait_command;  // Run 任务的可控步态指令；其它任务为空
    long episode_length = 0;
    float global_phase = 0.0f;
};

};