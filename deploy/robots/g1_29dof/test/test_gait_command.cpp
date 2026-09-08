// GaitCommand 手柄路径的行为测试（对拍测试绕过了它）：站立、走路使能、跑步使能、可行性约束、
// 斜率限制、松杆站定。数值按 Python GaitCommand 的公式手算。Axis 有 0.03 的平滑系数，
// 要重复喂 ~1000 次才收敛，和实机 1 kHz 的 lowstate 更新一致。
#include <cassert>
#include <cmath>
#include <iostream>
#include "isaaclab/envs/mdp/commands/gait_command.h"

using isaaclab::GaitCommand;
static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fails++; std::cout << "  FAIL: " << msg << "\n"; } else { std::cout << "  ok:   " << msg << "\n"; } } while (0)
static bool near(float a, float b, float tol = 1e-3f) { return std::fabs(a - b) <= tol; }

int main(int argc, char** argv)
{
    if (argc < 2) { std::cerr << "usage: test_gait_command <deploy.yaml>\n"; return 2; }
    auto cfg = YAML::LoadFile(argv[1])["commands"]["base_velocity"];
    const float dt = 0.02f;
    GaitCommand gc(cfg, dt);
    gc.configure(YAML::Load("{max_lin_vel_x: 3.0, run_enable: RT, walk_max_lin_vel_x: 1.5, walk_min_stance_ratio: 0.5}"));

    unitree::common::UnitreeJoystick joy;
    auto hold = [&](float ly, float lx, float rx, float rt) {  // 1 kHz 更新 1 s，Axis 收敛
        for (int i = 0; i < 1000; ++i) { joy.ly(ly); joy.lx(lx); joy.rx(rx); joy.RT(rt); joy.LT(0.0f); }
    };
    auto run_for = [&](float seconds) { for (int i = 0; i < int(seconds / dt + 0.5f); ++i) gc.update(&joy); };
    const auto& c = gc.command();

    std::cout << "[1] reset 时松杆：站立\n";
    hold(0, 0, 0, 0); gc.reset(&joy);
    CHECK(near(c[GaitCommand::LIN_VEL_X], 0) && near(c[GaitCommand::STANCE_RATIO], 1.0f), "指令 v=0, θ=1.0");
    CHECK(gc.is_standing() && gc.is_settled(), "is_standing && is_settled");
    CHECK(near(gc.obs()[GaitCommand::STANCE_RATIO], 2.0f * (1.0f - 0.3f) / 0.35f - 1.0f), "θ=1.0 的观测归一化 = +3.0（与训练一致）");

    std::cout << "[2] 半杆、不按 RT：走路，vx 目标 0.75，θ 不低于 0.5，斜率 2 m/s²\n";
    hold(0.5f, 0, 0, 0); run_for(0.1f);
    CHECK(near(gc.target()[GaitCommand::LIN_VEL_X], 0.75f, 0.02f), "vx 目标 = 0.5 * 1.5 (走路限速)");
    CHECK(gc.target()[GaitCommand::STANCE_RATIO] >= 0.5f, "θ 目标 >= 0.5（无腾空）");
    CHECK(near(c[GaitCommand::LIN_VEL_X], 0.2f, 0.03f), "0.1 s 后 vx 指令 ≈ 0.2（2 m/s² 斜率）");
    run_for(1.0f);
    CHECK(near(c[GaitCommand::LIN_VEL_X], 0.75f, 0.02f) && !gc.is_standing(), "1.1 s 后跟上目标，非站立");
    CHECK(gc.phase() > 0.0f && gc.phase() < 1.0f, "相位在积分");

    std::cout << "[3] 满杆 + 按住 RT：跑步预设，θ 0.32，速度经可行性约束到 3.0\n";
    hold(1.0f, 0, 0, 1.0f); run_for(0.05f);
    CHECK(gc.run_active(), "run_active");
    CHECK(near(gc.target()[GaitCommand::LIN_VEL_X], 3.0f, 0.02f), "vx 目标 3.0 (= 3.0 Hz * 1.0 m 步幅)");
    CHECK(near(gc.target()[GaitCommand::GAIT_FREQ], 3.0f, 0.02f) && near(gc.target()[GaitCommand::STANCE_RATIO], 0.32f, 0.02f), "步频 3.0, θ 0.32");
    run_for(2.0f);
    CHECK(near(c[GaitCommand::LIN_VEL_X], 3.0f, 0.02f) && near(c[GaitCommand::STANCE_RATIO], 0.32f, 0.02f), "2 s 后指令到位");
    auto o = gc.obs();
    CHECK(near(o[GaitCommand::LIN_VEL_X], 3.0f, 0.02f) && near(o[GaitCommand::STANCE_RATIO], 2.0f * (0.32f - 0.3f) / 0.35f - 1.0f, 0.02f), "观测：速度原量纲，θ 归一到 [-1,1]");

    std::cout << "[4] 满杆但不按 RT：走路限速 1.5，θ 抬回 0.5\n";
    hold(1.0f, 0, 0, 0); run_for(2.0f);
    CHECK(near(c[GaitCommand::LIN_VEL_X], 1.5f, 0.02f), "vx 指令回落到 1.5");
    CHECK(c[GaitCommand::STANCE_RATIO] >= 0.5f - 1e-3f, "θ >= 0.5");

    std::cout << "[5] 2.5 m/s 跑步时松杆：速度 2 m/s² 减到 0 要 1.25 s，θ 0.35->1.0 以 0.5/s 要 1.3 s\n";
    hold(1.0f, 0, 0, 1.0f); gc.configure(YAML::Load("{max_lin_vel_x: 2.5}")); run_for(2.0f);
    CHECK(near(c[GaitCommand::LIN_VEL_X], 2.5f, 0.02f) && near(c[GaitCommand::STANCE_RATIO], 0.35f, 0.02f), "跑步中 vx 2.5, θ 0.35");
    hold(0, 0, 0, 0);
    run_for(0.5f);
    CHECK(near(c[GaitCommand::LIN_VEL_X], 1.5f, 0.05f) && !gc.is_standing(), "0.5 s: vx ≈ 1.5，仍在减速");
    run_for(0.5f);
    CHECK(near(c[GaitCommand::LIN_VEL_X], 0.5f, 0.05f) && !gc.is_standing() && !gc.is_settled(), "1.0 s: vx ≈ 0.5, θ ≈ 0.85，未站定");
    run_for(0.5f);
    CHECK(gc.is_standing() && near(c[GaitCommand::STANCE_RATIO], 1.0f, 0.02f) && gc.is_settled(), "1.5 s: 速度 0、θ 1.0，settled，允许切换");
    CHECK(c[GaitCommand::SWING_HEIGHT] <= 0.1f + 1e-3f, "站立时摆高 <= 0.10");

    std::cout << "[6] 侧向与转向符号：lx 右推 -> vy 为负，rx 右推 -> wz 为负（与 velocity_commands 一致）\n";
    hold(0, 1.0f, 1.0f, 0); run_for(1.0f);
    CHECK(gc.target()[GaitCommand::LIN_VEL_Y] < 0 && gc.target()[GaitCommand::ANG_VEL_Z] < 0, "符号约定");

    std::cout << (fails ? "\nFAIL" : "\nPASS") << " (" << fails << " failures)\n";
    return fails ? 1 : 0;
}
