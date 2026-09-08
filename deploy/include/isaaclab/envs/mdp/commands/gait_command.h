// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// GaitCommand: C++ 版的 unitree_rl_lab.tasks.locomotion.robots.g1.run.gait_command.GaitCommand。
//
// 训练侧那个类做两件事：采样目标指令（课程、随机）和"执行"目标指令（可行性约束、站立、斜率限制、
// 相位积分、观测归一化）。部署只需要后半部分，目标改由手柄给出。下面每一步都标注了对应的
// Python 位置，改训练逻辑时请同步。训练侧参数（斜率、阈值、区间）从 deploy.yaml 的
// commands.base_velocity 读，控制侧的手柄映射从 config.yaml 的 FSM.Run.gait 读（configure()）。

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

namespace isaaclab
{

class GaitCommand
{
public:
    static constexpr int DIM = 8;
    // 与 Python 的 IDX_* 一致
    enum Idx { LIN_VEL_X = 0, LIN_VEL_Y, ANG_VEL_Z, GAIT_FREQ, STANCE_RATIO, SWING_HEIGHT, BODY_HEIGHT, BODY_PITCH };
    using Vec = std::array<float, DIM>;

    /**
     * @param cmd_cfg  deploy.yaml 的 commands.base_velocity 节点（ranges = 训练的 limit_ranges，gait = GaitCommandCfg 标量）
     * @param step_dt  策略步长 [s]
     */
    GaitCommand(const YAML::Node& cmd_cfg, float step_dt)
    : dt_(step_dt)
    {
        const auto ranges = cmd_cfg["ranges"];
        const char* names[DIM] = {"lin_vel_x", "lin_vel_y", "ang_vel_z", "gait_freq", "stance_ratio",
                                  "swing_height", "body_height", "body_pitch"};
        for (int i = 0; i < DIM; ++i) {
            if (!ranges[names[i]]) {
                throw std::runtime_error(std::string("deploy.yaml: commands.base_velocity.ranges 缺少 ") + names[i]);
            }
            auto r = ranges[names[i]].as<std::vector<float>>();
            lo_[i] = r.at(0);
            hi_[i] = r.at(1);
        }

        const auto gait = cmd_cfg["gait"];
        if (!gait) {
            throw std::runtime_error(
                "deploy.yaml: commands.base_velocity.gait 缺失。用新版 export_deploy_cfg.py 重新导出，"
                "它会写入斜率限制、站立阈值等 GaitCommandCfg 参数；没有这些，部署行为和训练对不上。");
        }
        auto slew = gait["slew_rates"].as<std::vector<float>>();
        if (slew.size() != DIM) {
            throw std::runtime_error("deploy.yaml: gait.slew_rates 需要 8 个值");
        }
        for (int i = 0; i < DIM; ++i) slew_[i] = slew[i];
        standing_threshold_ = gait["standing_threshold"].as<float>();
        standing_stance_ratio_ = gait["standing_stance_ratio"].as<float>();
        standing_swing_height_max_ = gait["standing_swing_height_max"].as<float>();
        settled_stance_ratio_ = gait["settled_stance_ratio"].as<float>();
        max_stride_length_ = gait["max_stride_length"].as<float>();
        flight_speed_threshold_ = gait["flight_speed_threshold"].as<float>();
        running_stance_ratio_ = gait["running_stance_ratio"].as<float>();

        // 默认预设表：按指令速度 |vx| [m/s] 插值（与 scripts/rsl_rl/play_run.py 一致），可被 configure() 覆盖
        presets_speed_ = {0.0f, 1.0f, 2.0f, 2.5f, 3.0f};
        presets_[0] = {1.5f, 1.5f, 2.4f, 2.8f, 3.0f};       // 步频 [Hz]
        presets_[1] = {0.55f, 0.55f, 0.40f, 0.35f, 0.32f};  // 支撑相比例
        presets_[2] = {0.12f, 0.12f, 0.15f, 0.18f, 0.20f};  // 摆动足高 [m]
        presets_[3] = {0.74f, 0.74f, 0.72f, 0.70f, 0.70f};  // 躯干高 [m]
        presets_[4] = {0.0f, 0.0f, 0.10f, 0.15f, 0.20f};    // 俯仰 [rad]

        reset(nullptr);
    }

    /**
     * 控制侧配置（config.yaml FSM.<state>.gait，可为空节点）::
     *
     *   max_lin_vel_x: 1.5        # 总限速；不写 = 训练上限。首次上机先 1.5
     *   max_lin_vel_y: 0.3
     *   max_ang_vel_z: 0.6
     *   run_enable: RT            # 跑步使能：RT / LT / always / never
     *   walk_max_lin_vel_x: 1.5   # 未使能时的限速（走路），且支撑相不低于 walk_min_stance_ratio
     *   walk_min_stance_ratio: 0.5
     *   presets:                  # 按指令速度 |vx| 插值的步态预设，speed 升序，其余各列等长
     *     speed:  [0.0, 1.0, 2.0, 2.5, 3.0]
     *     freq:   [1.5, 1.5, 2.4, 2.8, 3.0]
     *     stance: [0.55, 0.55, 0.40, 0.35, 0.32]
     *     swing:  [0.12, 0.12, 0.15, 0.18, 0.20]
     *     height: [0.74, 0.74, 0.72, 0.70, 0.70]
     *     pitch:  [0.0, 0.0, 0.10, 0.15, 0.20]
     *
     * 为什么不是"RT 当模拟档位"：Unitree 手柄的 LT/RT 在 UnitreeJoystick::extract 里喂的是
     * btn.L2/R2（0/1 按键），Axis 只是给它加了平滑，读出来只有 0 和 1。所以 RT 只做跑步使能，
     * 步态参数按速度插值，保证任何速度下步频/支撑相都和速度自洽。
     */
    void configure(const YAML::Node& ctrl)
    {
        if (!ctrl) return;
        if (ctrl["max_lin_vel_x"]) cap_[LIN_VEL_X] = ctrl["max_lin_vel_x"].as<float>();
        if (ctrl["max_lin_vel_y"]) cap_[LIN_VEL_Y] = ctrl["max_lin_vel_y"].as<float>();
        if (ctrl["max_ang_vel_z"]) cap_[ANG_VEL_Z] = ctrl["max_ang_vel_z"].as<float>();
        if (ctrl["run_enable"]) run_enable_ = ctrl["run_enable"].as<std::string>();
        if (ctrl["walk_max_lin_vel_x"]) walk_max_lin_vel_x_ = ctrl["walk_max_lin_vel_x"].as<float>();
        if (ctrl["walk_min_stance_ratio"]) walk_min_stance_ratio_ = ctrl["walk_min_stance_ratio"].as<float>();
        if (ctrl["presets"]) {
            const auto p = ctrl["presets"];
            presets_speed_ = p["speed"].as<std::vector<float>>();
            const char* cols[5] = {"freq", "stance", "swing", "height", "pitch"};
            for (int k = 0; k < 5; ++k) {
                presets_[k] = p[cols[k]].as<std::vector<float>>();
                if (presets_[k].size() != presets_speed_.size()) {
                    throw std::runtime_error(std::string("gait.presets.") + cols[k] + " 长度与 speed 不一致");
                }
            }
        }
        spdlog::info("GaitCommand: vx cap {:.2f} (train max {:.2f}), run enable = {}, walk cap {:.2f}, {} preset knots",
                     effective_hi(LIN_VEL_X), hi_[LIN_VEL_X], run_enable_, walk_max_lin_vel_x_, presets_speed_.size());
    }

    /**
     * 进入状态时调用。部署侧刻意不照搬训练的"reset 时指令直接跳到采样目标"：切进来时手柄可能正被
     * 推着，指令从站立值（v=0, θ=1）起步再按斜率滑向摇杆目标，机器人从站姿平滑起跑。
     */
    void reset(const unitree::common::UnitreeJoystick* joy)
    {
        compute_target(joy);
        command_ = target_;
        command_[LIN_VEL_X] = command_[LIN_VEL_Y] = command_[ANG_VEL_Z] = 0.0f;
        command_[STANCE_RATIO] = standing_stance_ratio_;
        command_[SWING_HEIGHT] = std::min(command_[SWING_HEIGHT], standing_swing_height_max_);
        phase_ = 0.0f;
    }

    /** 每个策略步调用一次，必须在观测计算之前（gait_commands / gait_clock 只读这里的状态）。 */
    void update(const unitree::common::UnitreeJoystick* joy)
    {
        compute_target(joy);
        // GaitCommand._update_command: 斜率限制
        for (int i = 0; i < DIM; ++i) {
            float delta = target_[i] - command_[i];
            if (slew_[i] > 0.0f) {
                float step = slew_[i] * dt_;
                delta = std::clamp(delta, -step, step);
            }
            command_[i] += delta;
        }
        // 相位积分，用限幅后的步频
        phase_ = std::fmod(phase_ + command_[GAIT_FREQ] * dt_, 1.0f);
        if (phase_ < 0.0f) phase_ += 1.0f;
    }

    /** 测试 / 外部规划器接口：直接写入限幅后的指令与相位（绕过手柄和斜率）。 */
    void set_command(const Vec& cmd, float phase)
    {
        command_ = cmd;
        target_ = cmd;
        phase_ = phase;
    }

    /** GaitCommand.command_obs：速度保物理量纲，5 个步态参数按 limit_ranges 归一到 [-1, 1]。 */
    std::vector<float> obs() const
    {
        std::vector<float> o(command_.begin(), command_.end());
        for (int i = GAIT_FREQ; i < DIM; ++i) {
            float span = std::max(hi_[i] - lo_[i], 1e-6f);
            o[i] = 2.0f * (command_[i] - lo_[i]) / span - 1.0f;
        }
        return o;
    }

    /** observations.gait_clock */
    std::vector<float> clock() const
    {
        const float a = phase_ * 2.0f * static_cast<float>(M_PI);
        return {std::sin(a), std::cos(a)};
    }

    /** GaitCommand.is_standing：判据是观测里的速度指令范数 */
    bool is_standing() const
    {
        return vel_norm(command_) < standing_threshold_;
    }

    /** GaitCommand.is_settled：站立且支撑相已接近 1，即真的双脚站定 */
    bool is_settled() const
    {
        return is_standing() && command_[STANCE_RATIO] > settled_stance_ratio_;
    }

    const Vec& command() const { return command_; }
    const Vec& target() const { return target_; }
    float phase() const { return phase_; }
    bool run_active() const { return run_active_; }
    float effective_hi(int i) const { return std::isnan(cap_[i]) ? hi_[i] : std::min(hi_[i], cap_[i]); }
    float effective_lo(int i) const { return std::isnan(cap_[i]) ? lo_[i] : std::max(lo_[i], -cap_[i]); }

private:
    static float vel_norm(const Vec& c)
    {
        return std::sqrt(c[LIN_VEL_X] * c[LIN_VEL_X] + c[LIN_VEL_Y] * c[LIN_VEL_Y] + c[ANG_VEL_Z] * c[ANG_VEL_Z]);
    }

    float interp(const std::vector<float>& xs, const std::vector<float>& ys, float x) const
    {
        if (x <= xs.front()) return ys.front();
        if (x >= xs.back()) return ys.back();
        for (size_t k = 1; k < xs.size(); ++k) {
            if (x <= xs[k]) {
                float t = (x - xs[k - 1]) / std::max(xs[k] - xs[k - 1], 1e-6f);
                return ys[k - 1] + t * (ys[k] - ys[k - 1]);
            }
        }
        return ys.back();
    }

    // 摇杆 -> 目标指令（对应 Python 的 _resample_command + _apply_feasibility + 站立分支）
    void compute_target(const unitree::common::UnitreeJoystick* joy)
    {
        float ly = 0.0f, lx = 0.0f, rx = 0.0f;
        bool run = (run_enable_ == "always");
        if (joy) {
            // const_cast：UnitreeJoystick 的 Axis::operator()() 没有 const 版本，这里只读
            auto* j = const_cast<unitree::common::UnitreeJoystick*>(joy);
            ly = j->ly();
            lx = j->lx();
            rx = j->rx();
            if (run_enable_ == "RT") run = j->RT() > 0.5f;
            else if (run_enable_ == "LT") run = j->LT() > 0.5f;
        }
        run_active_ = run;

        // 速度：摇杆 ±1 按"当前模式"的上限缩放，满杆 = 该模式的最大速度（velocity_commands 那种
        // 直接 clamp 会让满杆停在 1.0）。不按 RT 时上限是走路限速，按下后上限抬到总限速，
        // 指令由斜率限制平滑过渡。
        auto axis_to_vel = [this](float a, float lo, float hi) {
            a = std::clamp(a, -1.0f, 1.0f);
            return a >= 0.0f ? a * hi : a * (-lo);
        };
        float vx_hi = effective_hi(LIN_VEL_X), vx_lo = effective_lo(LIN_VEL_X);
        if (!run) {
            vx_hi = std::min(vx_hi, walk_max_lin_vel_x_);
            vx_lo = std::max(vx_lo, -walk_max_lin_vel_x_);
        }
        target_[LIN_VEL_X] = axis_to_vel(ly, vx_lo, vx_hi);
        target_[LIN_VEL_Y] = axis_to_vel(-lx, effective_lo(LIN_VEL_Y), effective_hi(LIN_VEL_Y));   // 与 velocity_commands 同号约定
        target_[ANG_VEL_Z] = axis_to_vel(-rx, effective_lo(ANG_VEL_Z), effective_hi(ANG_VEL_Z));

        // 步态参数：按指令速度插值，再夹进训练区间；未使能跑步时支撑相不低于 walk_min_stance_ratio（无腾空）
        const float speed = std::fabs(target_[LIN_VEL_X]);
        for (int k = 0; k < 5; ++k) {
            const int i = GAIT_FREQ + k;
            target_[i] = std::clamp(interp(presets_speed_, presets_[k], speed), lo_[i], hi_[i]);
        }
        if (!run) {
            target_[STANCE_RATIO] = std::max(target_[STANCE_RATIO], walk_min_stance_ratio_);
        }

        // _apply_feasibility 1) |vx| <= f * max_stride_length
        if (max_stride_length_ > 0.0f) {
            float v_cap = target_[GAIT_FREQ] * max_stride_length_;
            target_[LIN_VEL_X] = std::clamp(target_[LIN_VEL_X], -v_cap, v_cap);
        }
        // _apply_feasibility 2) 高速必须有腾空期
        if (std::fabs(target_[LIN_VEL_X]) > flight_speed_threshold_) {
            target_[STANCE_RATIO] = std::min(target_[STANCE_RATIO], running_stance_ratio_);
        }
        // 站立：目标速度范数低于阈值 -> 支撑相目标 = 1.0（训练课程终值），摆高封顶
        if (vel_norm(target_) < standing_threshold_) {
            target_[STANCE_RATIO] = standing_stance_ratio_;
            target_[SWING_HEIGHT] = std::min(target_[SWING_HEIGHT], standing_swing_height_max_);
        }
    }

    float dt_;
    Vec lo_{}, hi_{};                 // limit_ranges
    Vec cap_{{NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN}};  // 控制侧限速（只用前 3 维）
    Vec slew_{};
    float standing_threshold_, standing_stance_ratio_, standing_swing_height_max_, settled_stance_ratio_;
    float max_stride_length_, flight_speed_threshold_, running_stance_ratio_;

    std::string run_enable_ = "RT";
    float walk_max_lin_vel_x_ = 1.5f;
    float walk_min_stance_ratio_ = 0.5f;
    bool run_active_ = false;
    std::vector<float> presets_speed_;
    std::array<std::vector<float>, 5> presets_;

    Vec target_{}, command_{};
    float phase_ = 0.0f;
};

};  // namespace isaaclab
