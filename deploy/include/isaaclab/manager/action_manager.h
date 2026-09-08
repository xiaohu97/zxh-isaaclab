// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/manager/manager_term_cfg.h"
#include <algorithm>
#include <numeric>

namespace isaaclab
{

class ActionTerm 
{
public:
    ActionTerm(YAML::Node cfg, ManagerBasedRLEnv* env): cfg(cfg), env(env) {}

    virtual int action_dim() = 0;
    virtual std::vector<float> raw_actions() = 0;
    virtual std::vector<float> processed_actions() = 0;
    virtual void process_actions(std::vector<float> actions) = 0;
    virtual void reset(){};

protected:
    YAML::Node cfg;
    ManagerBasedRLEnv* env;
};

inline std::map<std::string, std::function<std::unique_ptr<ActionTerm>(YAML::Node, ManagerBasedRLEnv*)>>& actions_map() {
    static std::map<std::string, std::function<std::unique_ptr<ActionTerm>(YAML::Node, ManagerBasedRLEnv*)>> instance;
    return instance;
}

#define REGISTER_ACTION(name) \
    inline struct name##_registrar { \
        name##_registrar() { \
            actions_map()[#name] = [](YAML::Node cfg, ManagerBasedRLEnv* env) { \
                return std::make_unique<name>(cfg, env); \
            }; \
        } \
    } name##_registrar_instance;

class ActionManager
{
public:
    ActionManager(YAML::Node cfg, ManagerBasedRLEnv* env)
    : cfg(cfg), env(env)
    {
        _prepare_terms();
        _action.resize(total_action_dim(), 0.0f);
    }

    void reset()
    {
        _action.assign(total_action_dim(), 0.0f);
        for(auto & term : _terms)
        {
            term->reset();
        }
    }

    std::vector<float> action()
    {
        return _action;
    }

    std::vector<float> processed_actions()
    {
        std::vector<float> actions;
        for(auto & term : _terms)
        {
            auto term_action = term->processed_actions();
            actions.insert(actions.end(), term_action.begin(), term_action.end());
        }
        return actions;
    }

    void process_action(std::vector<float> action)
    {
        // 与训练一致：rsl_rl 的 RslRlVecEnvWrapper 在环境之前裁剪原始动作（RunnerCfg.clip_actions），
        // 网络/ONNX 里没有这一步。在这里裁而不是在各 term 里裁，是因为 last_action 观测读的就是
        // _action，训练时它看到的也是裁剪后的值。
        {
            int idx = 0;
            for(size_t t = 0; t < _terms.size(); ++t)
            {
                const int dim = _terms[t]->action_dim();
                if(!_raw_clips[t].empty()) {
                    for(int i = idx; i < idx + dim; ++i) {
                        action[i] = std::clamp(action[i], _raw_clips[t][0], _raw_clips[t][1]);
                    }
                }
                idx += dim;
            }
        }
        _action = action;
        int idx = 0;
        for(auto & term : _terms)
        {
            auto term_action = std::vector<float>(action.begin() + idx, action.begin() + idx + term->action_dim());
            term->process_actions(term_action);
            idx += term->action_dim();
        }
    }

    int total_action_dim()
    {
        auto dims = action_dim();
        
        return std::accumulate(dims.begin(), dims.end(), 0);
    }

    std::vector<int> action_dim()
    {
        std::vector<int> dims;
        for (auto & term : _terms)
        {
            dims.push_back(term->action_dim());
        }
        return dims;
    }

    YAML::Node cfg;
    ManagerBasedRLEnv* env;

private:
    void _prepare_terms()
    {
        for(auto it = this->cfg.begin(); it != this->cfg.end(); ++it)
        {
            std::string action_name = it->first.as<std::string>();
            if(actions_map().find(action_name) == actions_map().end())
            {
                throw std::runtime_error("Action term '" + action_name + "' is not registered.");
            }

            auto term = actions_map()[action_name](it->second, env);
            _terms.push_back(std::move(term));

            // 可选 raw_clip: [lo, hi]，对该项的原始网络输出裁剪（deploy.yaml 由 export_deploy_cfg 写入）
            const YAML::Node term_cfg = it->second;
            if(term_cfg["raw_clip"] && !term_cfg["raw_clip"].IsNull()) {
                _raw_clips.push_back(term_cfg["raw_clip"].as<std::vector<float>>());
            } else {
                _raw_clips.push_back({});
            }
        }
    }

    std::vector<float> _action;
    std::vector<std::unique_ptr<ActionTerm>> _terms;
    std::vector<std::vector<float>> _raw_clips;  // 与 _terms 一一对应；空 = 不裁
};

};