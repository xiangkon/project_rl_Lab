# project_rl_Lab

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/phc_isaaclab
    ```

- 启用 conda 环境
```bash
# RL 训练环境
conda activate env_isaaclab

# 运动可视化环境
conda activate roboMotion
```

```bash
python -m pip install -e source/phc_isaaclab

# 训练
python scripts/rsl_rl/train.py --task=Unitree-G1-29dof-Velocity-v0 --headless

# 无头模式训练
python scripts/rsl_rl/train.py --task=Unitree-G1-29dof-Velocity-gym-v0 --headless

# 测试
python scripts/rsl_rl/play.py --task=Unitree-G1-29dof-Velocity-gym-v0
```

- 将宇树官方github仓库unitree_rl_lab的运动数据转换为方便可视化的pkl格式:
```bash
python scripts/motion_convert/unitree_rl_lab2pkl.py --input_file csv_file_path
```

- 运行可视化程序:
```bash
# 直接可视化
python scripts/motion_verify/direct_motion_visualization.py --motion_file pkl_file_path

# 策略可视化
python scripts/motion_verify/policy_motion_visualization.py --motion_file pkl_file_path
```