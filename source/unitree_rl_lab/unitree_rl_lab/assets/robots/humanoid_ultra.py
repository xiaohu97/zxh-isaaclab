# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import tempfile
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots import ustc_actuators

HUMANOID_ULTRA_DESCRIPTION_DIR = Path(__file__).resolve().parent / "humanoid_ultra_description"
HUMANOID_ULTRA_URDF_DIR = HUMANOID_ULTRA_DESCRIPTION_DIR / "urdf"
HUMANOID_ULTRA_MESH_DIR = HUMANOID_ULTRA_DESCRIPTION_DIR / "meshes"

HUMANOID_ULTRA_12DOF_URDF = "humanoid_ultra_12dof_description.urdf"
HUMANOID_ULTRA_27DOF_URDF = "humanoid_ultra_27dof_description.urdf"
HUMANOID_ULTRA_27DOF_IDENTIFIED_URDF = "humanoid_ultra_27dof_description_identified.urdf"
HUMANOID_ULTRA_27DOF_IDENTIFIED_LEFTARM2KG_URDF = (
    "humanoid_ultra_27dof_description_identified_leftarm2kg.urdf"
)


def _prepared_urdf(filename: str) -> str:
    """Create an Isaac Sim-friendly URDF with absolute mesh paths."""
    source_path = HUMANOID_ULTRA_URDF_DIR / filename
    if not source_path.is_file():
        raise FileNotFoundError(f"Humanoid Ultra URDF does not exist: {source_path}")
    if not HUMANOID_ULTRA_MESH_DIR.is_dir():
        raise FileNotFoundError(f"Humanoid Ultra mesh directory does not exist: {HUMANOID_ULTRA_MESH_DIR}")

    mesh_prefix = f"{HUMANOID_ULTRA_MESH_DIR.as_posix()}/"
    contents = source_path.read_text(encoding="utf-8")
    for package_prefix in (
        "package://humanoidultra01/meshes/",
        "package://humanoid_ultra_description/meshes/",
        "package://humanoidultra_urdf/meshes/",
    ):
        contents = contents.replace(package_prefix, mesh_prefix)

    output_dir = Path(tempfile.gettempdir()) / "IsaacLab" / "unitree_rl_lab" / "humanoid_ultra"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != contents:
        output_path.write_text(contents, encoding="utf-8")
    return str(output_path)

HUMANOIDULTRA12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_prepared_urdf(HUMANOID_ULTRA_12DOF_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.995),
        joint_pos={
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": 0.346431,
            "left_knee_joint": 0.755514,
            "left_ankle_pitch_joint": 0.366252,
            "left_ankle_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_pitch_joint": 0.346431,
            "right_knee_joint": 0.755514,
            "right_ankle_pitch_joint": 0.366252,
            "right_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 90.0,
                ".*_hip_roll_joint": 300.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint": 300.0,
            },
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_yaw_joint": 80.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 180.0,
                ".*_knee_joint": 180.0,
            },
            damping={
                ".*_hip_yaw_joint": 0.8,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.4,
                ".*_knee_joint": 2.4,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.8,
                ".*_ankle_roll_joint": 0.4,
                },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


#腿部6个，腰部1个，手臂7个   
HUMANOIDULTRA27DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_prepared_urdf(HUMANOID_ULTRA_27DOF_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.005),
        joint_pos={
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": 0.289936,
            "left_knee_joint": 0.742326,
            "left_ankle_pitch_joint": 0.409573,
            "left_ankle_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_pitch_joint": 0.289936,
            "right_knee_joint": 0.742326,
            "right_ankle_pitch_joint": 0.409573,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.25,
            "left_shoulder_roll_joint": 0.1,
            "left_shoulder_yaw_joint": -1.5707963,
            "left_elbow_joint": -0.6,
            "left_wrist_yaw_joint": 1.5707963,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": -0.25,
            "right_shoulder_roll_joint": -0.1,
            "right_shoulder_yaw_joint": 1.5707963,
            "right_elbow_joint": 0.6,
            "right_wrist_yaw_joint": -1.5707963,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 90.0,
                ".*_hip_roll_joint": 300.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint": 300.0,
            },
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_yaw_joint": 80.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 180.0,
                ".*_knee_joint": 180.0,
            },
            damping={
                ".*_hip_yaw_joint": 0.8,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.4,
                ".*_knee_joint": 2.4,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.8,
                ".*_ankle_roll_joint": 0.4,
                },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*waist_yaw_joint",
            ],
            stiffness={
                ".*waist_yaw_joint": 150.0,
            },
            damping={
                ".*waist_yaw_joint": 2.5,
            },
            effort_limit_sim=150.0,
            velocity_limit_sim=12.56,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ), 
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
            ],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=80.0,
            damping=1.5,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),       
        "elbow": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_joint",
            ],
            stiffness={
                ".*_elbow_joint": 60.0,
            },
            damping={
                ".*_elbow_joint": 1.2,
            },
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "wrist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
            ],
            stiffness={
                ".*_wrist_yaw_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 25.0,
            },
            damping={
                ".*_wrist_yaw_joint": 0.8,
                ".*_wrist_roll_joint": 0.8,
                ".*_wrist_pitch_joint": 0.8,
            },
            effort_limit_sim=24.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)

# Identified rigid-body parameters for 27-DoF mimic tasks. Keep the nominal
# asset unchanged so existing non-mimic tasks and checkpoints remain reproducible.
HUMANOIDULTRA27DOF_IDENTIFIED_CFG = HUMANOIDULTRA27DOF_CFG.copy()
HUMANOIDULTRA27DOF_IDENTIFIED_CFG.spawn.asset_path = _prepared_urdf(HUMANOID_ULTRA_27DOF_IDENTIFIED_URDF)
HUMANOIDULTRA27DOF_IDENTIFIED_CFG.actuators["legs"].armature = {
    ".*_hip_yaw_joint": 0.01,
    ".*_hip_roll_joint": 0.15,
    ".*_hip_pitch_joint": 0.10,
    ".*_knee_joint": 0.12,
}

# Identified 27-DoF asset with the arm inertias obtained while carrying the
# 2 kg payload on the left arm.
HUMANOIDULTRA27DOF_IDENTIFIED_LEFTARM2KG_CFG = HUMANOIDULTRA27DOF_IDENTIFIED_CFG.copy()
HUMANOIDULTRA27DOF_IDENTIFIED_LEFTARM2KG_CFG.spawn.asset_path = _prepared_urdf(
    HUMANOID_ULTRA_27DOF_IDENTIFIED_LEFTARM2KG_URDF
)

# Mimic-only motor model. Keep the identified asset available to non-mimic
# tasks while applying the USTC torque-speed curves to all 27 controlled DoFs.
HUMANOIDULTRA27DOF_MIMIC_CFG = HUMANOIDULTRA27DOF_IDENTIFIED_CFG.copy()
HUMANOIDULTRA27DOF_MIMIC_CFG.actuators = {
    "hip_yaw_E8112": ustc_actuators.USTCActuatorCfg_E8112(
        joint_names_expr=[".*_hip_yaw_joint"],
        stiffness=80.0,
        damping=0.8,
        armature=0.01,
        min_delay=0,
        max_delay=2,
    ),
    "hip_roll_E10020_P24": ustc_actuators.USTCActuatorCfg_E10020_P24(
        joint_names_expr=[".*_hip_roll_joint"],
        stiffness=150.0,
        damping=2.5,
        armature=HUMANOIDULTRA27DOF_IDENTIFIED_CFG.actuators["legs"].armature[".*_hip_roll_joint"],
        min_delay=0,
        max_delay=2,
    ),
    "hip_pitch_knee_E13715": ustc_actuators.USTCActuatorCfg_E13715(
        joint_names_expr=[".*_hip_pitch_joint", ".*_knee_joint"],
        stiffness=180.0,
        damping=2.4,
        armature={
            ".*_hip_pitch_joint": 0.10,
            ".*_knee_joint": 0.12,
        },
        min_delay=0,
        max_delay=2,
    ),
    # Each simulated ankle DoF is virtual. This first version applies the
    # derated E4315 curve in joint space; the physical ankle uses two coupled
    # E4315 motors and ultimately needs motor-space Jacobian clipping.
    "ankles_E4315_P36_approx": ustc_actuators.USTCActuatorCfg_E4315_P36_Ankle(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness={
            ".*_ankle_pitch_joint": 40.0,
            ".*_ankle_roll_joint": 20.0,
        },
        damping={
            ".*_ankle_pitch_joint": 0.8,
            ".*_ankle_roll_joint": 0.4,
        },
        armature=0.01,
        min_delay=0,
        max_delay=2,
    ),
    "waist_yaw_E10020_P12": ustc_actuators.USTCActuatorCfg_E10020_P12(
        joint_names_expr=[".*waist_yaw_joint"],
        stiffness=150.0,
        damping=2.5,
        armature=0.01,
        min_delay=0,
        max_delay=2,
    ),
    "shoulders_E4315_P36": ustc_actuators.USTCActuatorCfg_E4315_P36(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
        ],
        stiffness=80.0,
        damping=1.5,
        armature=0.01,
        min_delay=0,
        max_delay=2,
    ),
    "elbows_E4315_P36": ustc_actuators.USTCActuatorCfg_E4315_P36(
        joint_names_expr=[".*_elbow_joint"],
        stiffness=60.0,
        damping=1.2,
        armature=0.01,
        min_delay=0,
        max_delay=2,
    ),
    "wrists_E4310_P36": ustc_actuators.USTCActuatorCfg_E4310_P36(
        joint_names_expr=[".*_wrist_yaw_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint"],
        stiffness=25.0,
        damping=0.8,
        armature=0.01,
        min_delay=0,
        max_delay=2,
    ),
}

#腿部6个，腰部1个，手臂7个   
HUMANOIDULTRA27DOF_AMP_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_prepared_urdf(HUMANOID_ULTRA_27DOF_URDF),
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.005),
        joint_pos={
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_hip_pitch_joint": 0.289936,
            "left_knee_joint": 0.742326,
            "left_ankle_pitch_joint": 0.409573,
            "left_ankle_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_pitch_joint": 0.289936,
            "right_knee_joint": 0.742326,
            "right_ankle_pitch_joint": 0.409573,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.25,
            "left_shoulder_roll_joint": -0.05,
            "left_shoulder_yaw_joint": -1.5707963,
            "left_elbow_joint": -0.6,
            "left_wrist_yaw_joint": 1.5707963,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": -0.25,
            "right_shoulder_roll_joint": 0.05,
            "right_shoulder_yaw_joint": 1.5707963,
            "right_elbow_joint": 0.6,
            "right_wrist_yaw_joint": -1.5707963,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 90.0,
                ".*_hip_roll_joint": 300.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint": 300.0,
            },
            velocity_limit_sim=15.0,
            stiffness={
                ".*_hip_yaw_joint": 80.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 180.0,
                ".*_knee_joint": 180.0,
            },
            damping={
                ".*_hip_yaw_joint": 0.8,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.4,
                ".*_knee_joint": 2.4,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=12.0,
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.8,
                ".*_ankle_roll_joint": 0.4,
                },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*waist_yaw_joint",
            ],
            stiffness={
                ".*waist_yaw_joint": 150.0,
            },
            damping={
                ".*waist_yaw_joint": 3.0,
            },
            effort_limit_sim=150.0,
            velocity_limit_sim=12.56,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ), 
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
            ],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=80.0,
            damping = 2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),       
        "elbow": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_joint",
            ],
            stiffness={
                ".*_elbow_joint": 60.0,
            },
            damping={
                ".*_elbow_joint": 1.6,
            },
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "wrist": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
            ],
            stiffness={
                ".*_wrist_yaw_joint": 30.0,
                ".*_wrist_roll_joint": 30.0,
                ".*_wrist_pitch_joint": 30.0,
            },
            damping={
                ".*_wrist_yaw_joint": 1.2,
                ".*_wrist_roll_joint": 1.2,
                ".*_wrist_pitch_joint": 1.2,
            },
            effort_limit_sim=24.0,
            velocity_limit_sim=10.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)
