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

import os
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robolab.tasks.humanoidultra.amp.mdp as mdp
from robolab.tasks.humanoidultra.amp.managers import MotionDataTermCfg
from robolab.tasks.humanoidultra.amp.amp_env_cfg import AmpEnvCfg, MotionDataCfg

import isaaclab.terrains as terrain_gen

##
# Pre-defined configs
##

from robolab.assets.robots.humanoidultra import HUMANOIDULTRA27DOF_AMP_CFG
from robolab import ROBOLAB_ROOT_DIR

ISAACLAB_JOINT_ORDER= [
    'left_hip_roll_joint', 
    'right_hip_roll_joint', 
    'waist_yaw_joint', 
    'left_hip_yaw_joint', 
    'right_hip_yaw_joint', 
    'left_shoulder_pitch_joint', 
    'right_shoulder_pitch_joint', 
    'left_hip_pitch_joint', 
    'right_hip_pitch_joint', 
    'left_shoulder_roll_joint', 
    'right_shoulder_roll_joint', 
    'left_knee_joint', 
    'right_knee_joint', 
    'left_shoulder_yaw_joint', 
    'right_shoulder_yaw_joint', 
    'left_ankle_pitch_joint', 
    'right_ankle_pitch_joint', 
    'left_elbow_joint', 
    'right_elbow_joint', 
    'left_ankle_roll_joint', 
    'right_ankle_roll_joint', 
    'left_wrist_yaw_joint', 
    'right_wrist_yaw_joint',
    'left_wrist_roll_joint', 
    'right_wrist_roll_joint',
    'left_wrist_pitch_joint', 
    'right_wrist_pitch_joint'
]

DATASET_JOINT_ORDER = [
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_hip_pitch_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_hip_pitch_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_yaw_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_yaw_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
]

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link", 
    "right_ankle_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
] # if changed here and symmetry is enabled, you might need to update amp.mdp.symmetry

ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 3

@configclass
class Humanoidultra27dofAmpRewards():
    """Reward terms for the MDP."""

    # -- Task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.25,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=1.25, 
        params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # -- Alive
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    
    # -- Base Link
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight= -0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight= -0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight= -1.0)

    # -- Joint
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight= -2e-4)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight= -1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight= -0.005)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight= -1.0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight= -1e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight= -2.0e-6)

    low_speed_sway_penalty = RewTerm(
        func=mdp.low_speed_sway_penalty,
        weight= -5e-2,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )
    
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_leg = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint",".*_ankle_pitch_joint",".*_ankle_roll_joint"])},
    )
    joint_deviation_arm = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint")},
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.22, "max": 0.9},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.23, "max": 0.56},
    )    
    # -- Feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight= -0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight= -0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    
    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped_,
        weight= 0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), 
            "threshold": 0.4},
    )
    #减小足端冲击的奖励函数
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight= -5e-5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight= -1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),# exclude ankle links
        },
    )


@configclass
class Humanoidultra27dofAmpEnvCfg(AmpEnvCfg):
    rewards: Humanoidultra27dofAmpRewards = Humanoidultra27dofAmpRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        self.scene.robot = HUMANOIDULTRA27DOF_AMP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "humanoidultra27dof/lab/locked_waist"
        )
        self.motion_data.motion_dataset.motion_data_weights={
            
            # '16_34': 1,

            # # # male2 walk 8
            # "C4_-_run_to_walk_a_stageii":1,
            # "C5_-_walk_to_run_stageii":1,
             
            # walk
            '0007_Walking001_stageii': 1,
            "LeftTurn01_stageii":1,
            "LeftTurn02_1_stageii":1,
            "B4_-_Stand_to_Walk_backwards_stageii": 1,
            "B14_-__Walk_turn_right_45_t2_stageii": 1,
            "B22_-__side_step_left_stageii": 1,
            "B23_-__side_step_right_stageii": 1,
            "Slow_AFig8CCW1_stageii":1,
            "Slow_Fig8_1_stageii":1,
            "Slow_SShapeLR1_stageii":1,
            "Slow_StraightLong23_stageii":1,
            "Normal_Fig8_1_stageii":1,
            "Normal_StraightLong45_stageii":1,
            "Fast_AFig8CCW2_stageii":1,
            "Fast_SShapeLR8_stageii":1,
            "Fast_StraightLong20_stageii":1,
            "walking_run05_stageii":1,
            "walking_run10_stageii":1,
            "walking_slow01_stageii":1,
            "walking_slow02_stageii":1,
            "walking_fast02_stageii":1,
            "WSTR05_J0Yfi7a_stageii":1,
            
            
            #ring
            "111_31_stageii":1,
            "111_31_stageii_":1,
            
            #run
            # '127_04_stageii': 1,
            # '127_06_stageii': 1,
            # '127_17_stageii': 1,
            # '127_18_stageii': 1,
            # "127_08_stageii": 1,
            # "C1_-_stand_to_run_stageii": 1,
            # # "C3_-_run_stageii": 1,
            # "C12_-_run_turn_left_45_stageii":1,
            # "C17_-_run_change_direction_stageii":1,
            
            #walk back
            "WalkingStraightBackwards06_stageii": 1,
            # "WalkingStraightBackwards09_stageii": 1,
            
            
            #run back
            # "C9_-_run_backwards_turn_run_forward_stageii":1,
            
            #stand
            "A1-_Stand_stageii": 1,
            # # male2 run 8


            #横移
            "RecoveryStepping_30_120_02_stageii": 1,
            "RecoveryStepping_50_45_01_stageii": 1,        
            "Push_Left_Hard01_stageii":1,


        }
        
        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
                
        # discriminator observations
        
        # self.observations.critic.key_body_pos_b.params = {
        #     "asset_cfg": SceneEntityCfg(
        #         name="robot", 
        #         body_names=KEY_BODY_NAMES, 
        #         preserve_order=True
        #     )
        # }
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        
        
        self.observations.disc.history_length = AMP_NUM_STEPS
        
        # discriminator demonstration observations
        
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_lin_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME
     
        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------

        self.events.reset_from_ref.params = {"animation": ANIMATION_TERM_NAME, "height_offset": 0.01}

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------

        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
                
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        
        # self.terminations.base_contact.params["sensor_cfg"].body_names = [
        #     ".*_hip_.*_link", "base_link", ".*_shoulder_.*_link", ".*_wrist_.*_link",".*_elbow_link",
        # ]
        
        self.terminations.base_contact = None
        
        if self.__class__.__name__ == "Humanoidultra27dofAmpEnvCfg":
            self.disable_zero_weight_rewards()
            
            
@configclass
class Humanoidultra27dofAmpEnvCfg_PLAY(Humanoidultra27dofAmpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 40
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None
        
        self.events.reset_from_ref = None
        