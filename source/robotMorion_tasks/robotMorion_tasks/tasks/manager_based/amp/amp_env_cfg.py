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

import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##

import robotMorion_tasks.tasks.manager_based.amp.mdp as mdp
from .animation_env_cfg import AnimationEnvCfg
from robotMorion_tasks.tasks.manager_based.amp.managers import AnimationTermCfg as AnimTerm
from robotMorion_tasks.tasks.manager_based.amp.managers import MotionDataTermCfg as MotionDataTerm
ANIMATION_TERM_NAME = "animation"

# 该类是 AMP 任务的场景基础配置，定义了地面（物理 + 视觉）、机器人占位符、接触力传感器、天空灯光四大核心元素；
# 关键设计：robot = MISSING 强制子类指定具体机器人模型，保证配置的通用性；接触力传感器启用历史记录和空中时间跟踪，为运动任务提供核心数据；
# 核心参数：地面高摩擦配置（static_friction=1.0）适合腿式机器人运动，避免打滑；传感器覆盖全机器人连杆，满足接触检测需求。

@configclass
class AmpSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # robot animation (for reference)
    robot_anim: ArticulationCfg = None
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##

# 该配置类定义了机器人基座速度指令的生成规则，核心是低速、小幅、全朝向的随机指令，适配初始训练阶段；
# 关键参数：10 秒固定重采样周期、2% 站立指令比例、启用朝向控制（刚度 0.5）、全方向朝向范围；
# 设计逻辑：通过小范围随机指令先训练机器人的平稳性和基础运动能力，后续可通过扩大 ranges 中的数值（如 lin_vel_x 改为 (-1.0, 2.0)）提升运动难度。

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1), heading=(-math.pi, math.pi)
        ),
    )

# 该配置类定义了机器人的动作类型为关节位置控制，动作作用于所有关节，且通过小缩放因子（0.25）限制调整幅度；
# 核心设计：启用默认偏移让智能体学习 “相对调整”，降低学习难度，适配初始训练阶段；
# 关键参数：scale=0.25 是核心，决定关节动作的最大调整幅度，小值保证训练初期的稳定性。

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

# 该配置为 AMP 任务的四大核心模块分别定制观测规则：Policy（带噪声、单步）、Critic（无噪声、3 步）、Discriminator（无噪声、10 步、时序）、DiscriminatorDemo（参考动画、时序）；
# 关键差异：Policy 加噪声模拟真实感知，Critic 用特权观测精准评估，Discriminator 用长时序区分运动序列；
# 核心目的：通过差异化的观测配置，让每个模块发挥最优作用，最终实现机器人对参考动画的精准模仿。

@configclass
class ObservationsCfg():
        
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.35, n_max=0.35))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.75, n_max=1.75))
        actions = ObsTerm(func=mdp.last_action)
        

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)
        # key_body_pos_b = ObsTerm(
        #     func=mdp.key_body_pos_b,
        #     params=MISSING,
        # )

        def __post_init__(self):
            self.history_length = 3
            self.enable_corruption = False
            self.concatenate_terms = True
    
    critic: CriticCfg = CriticCfg()
    
    @configclass
    class DiscriminatorCfg(ObsGroup):
        # root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        # key_body_pos_b = ObsTerm(
        #     func=mdp.key_body_pos_b,
        #     params=MISSING,
        # )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 10
            self.flatten_history_dim = False
            
    disc: DiscriminatorCfg = DiscriminatorCfg()
            
    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        # ref_root_local_rot_tan_norm = ObsTerm(
        #     func=mdp.ref_root_local_rot_tan_norm,
        #     params={
        #         "animation": ANIMATION_TERM_NAME,
        #         "flatten_steps_dim": False,
        #     }
        # )
        # ref_root_lin_vel_b = ObsTerm(
        #     func=mdp.ref_root_lin_vel_b,
        #     params={
        #         "animation": ANIMATION_TERM_NAME,
        #         "flatten_steps_dim": False,
        #     }
        # )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
            }
        )
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
            }
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel,
            params={
                "animation": ANIMATION_TERM_NAME,
                "flatten_steps_dim": False,
            }
        )
        # ref_key_body_pos_b = ObsTerm(
        #     func=mdp.ref_key_body_pos_b,
        #     params={
        #         "animation": MISSING,
        #         "flatten_steps_dim": False,
        #     }
        # )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
    
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()
        
# 该配置类通过 startup/reset/interval 三种触发模式，为机器人引入物理属性随机化、初始状态随机化、动态外部干扰三大类事件；
# 核心目的：提升策略的鲁棒性和泛化能力，让训练出的机器人能适应不同硬件参数、初始状态和外部干扰；
# 关键设计：随机范围控制在合理区间（如 80%~120% 缩放），避免过度随机导致训练不稳定，同时保留核心干扰场景（如推机器人）。

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}, # 0.02
        },
    )
    
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset
    reset_base=EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints=EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-1.0, 1.0)}},
    )

# 该配置类构建了 “核心奖励 + 分层惩罚” 的奖励体系，核心目标是让机器人精准跟踪速度指令、平稳无跳动、仅用足部接触地面；
# 关键设计：通过指数奖励保证速度跟踪的精准性，通过高权重惩罚约束上下跳动，通过低权重惩罚兼顾能耗和平滑性；
# 灵活性：可选惩罚项当前禁用，可根据任务需求（如强制水平、关节保护）调整权重启用。

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


# 该配置类定义了机器人仿真的四类核心终止条件：超时、非法接触、基座过低、姿态失控，覆盖了训练中所有常见的无效状态；
# 关键设计：通过 “快速检测（高度 / 姿态）+ 精准检测（接触）+ 兜底（超时）” 的分层逻辑，保证终止条件的及时性和准确性；
# 核心价值：避免无效训练数据，提升训练效率，同时约束机器人的运动安全边界（如姿态、高度）。

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MISSING), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, 
        params={
            "limit_angle": math.radians(60.0),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

@configclass
class MotionDataCfg:
    """Motion data settings for the MDP."""
    motion_dataset = MotionDataTerm(
        motion_data_dir="", 
        motion_data_weights={},
    )

# 该配置类是 AMP 算法读取参考动画数据的核心入口，定义了动画数据的提取维度、使用长度和采样规则；
# 关键参数：num_steps_to_use=10 保证时序长度匹配判别器，random_initialize/random_fetch 提升样本多样性，motion_data_components 覆盖全维度运动数据；
# 核心作用：为判别器提供多样化的参考动画序列，让机器人学习模仿自然、真实的运动模式。

@configclass
class AnimationCfg:
    """Animation settings for the MDP."""
    animation = AnimTerm(
        motion_data_term="motion_dataset",
        motion_data_components=[
            "root_pos_w",
            "root_quat",
            "root_vel_w",
            "root_ang_vel_w",
            "dof_pos",
            "dof_vel",
            "key_body_pos_b",
        ], 
        num_steps_to_use=10, 
        random_initialize=True,
        random_fetch=True,
        enable_visualization=False,
    )


##
# Environment configuration
##

# AmpEnvCfg 是 AMP 运动任务的总配置入口，整合了场景、观测、动作、奖励等所有子模块，定义了训练所需的完整环境规则；
# __post_init__ 核心作用：配置仿真步长、降采样、GPU 资源、传感器周期，适配大规模并行训练的性能与精度需求；
# disable_zero_weight_rewards 是实用工具：自动禁用无效奖励项，优化训练计算效率；
# 设计逻辑：通过 “静态声明 + 动态补全” 的方式，既保证配置的可读性 / 复用性，又适配大规模并行仿真的工程需求。

@configclass
class AmpEnvCfg(AnimationEnvCfg):
    """Configuration for the AMP locomotion environment."""

    # scene
    scene: AmpSceneCfg = AmpSceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Motion data
    motion_data: MotionDataCfg = MotionDataCfg()
    # Animation
    animation: AnimationCfg = AnimationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)