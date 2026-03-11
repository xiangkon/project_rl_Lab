import gymnasium as gym

# 基于宇数 Unitree RL Lab 的 Unitree-G1-29dof-Mimic-Dance-102 构建

gym.register(
    id="Unitree-G1-29dof-Mimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": f"robotMorion_tasks.tasks.manager_based.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)


gym.register(
    id="Unitree-G1-29dof-Mimic-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"robotMorion_tasks.tasks.manager_based.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)