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

import gymnasium as gym

gym.register(
    id="USTC-Humanoid-Ultra-12dof-Flat",
    entry_point=f"{__name__}.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoidultra12dof_env_cfg:Humanoidultra12dofFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.humanoidultra12dof_env_cfg:Humanoidultra12dofFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.humanoidultra12dof_agent_cfg:Humanoidultra12dofFlatAgentCfg"
        ),
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-12dof-Rough",
    entry_point=f"{__name__}.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoidultra12dof_env_cfg:Humanoidultra12dofRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.humanoidultra12dof_env_cfg:Humanoidultra12dofRoughEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.humanoidultra12dof_agent_cfg:Humanoidultra12dofRoughAgentCfg"
        ),
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Flat",
    entry_point=f"{__name__}.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoidultra27dof_env_cfg:Humanoidultra27dofFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.humanoidultra27dof_env_cfg:Humanoidultra27dofFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.humanoidultra27dof_agent_cfg:Humanoidultra27dofFlatAgentCfg"
        ),
    },
)

gym.register(
    id="USTC-Humanoid-Ultra-27dof-Rough",
    entry_point=f"{__name__}.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoidultra27dof_env_cfg:Humanoidultra27dofRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.humanoidultra27dof_env_cfg:Humanoidultra27dofRoughEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.humanoidultra27dof_agent_cfg:Humanoidultra27dofRoughAgentCfg"
        ),
    },
)
