# project_rl_Lab
```bash
# roboMotion
pip install -e source/pose
pip install -e source/motion_verify
```

```bash
# env_isaaclab
pip install -e source/phc_isaaclab
pip install -e source/motion_verify
```

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

- RL 训练
    ```bash
    python -m pip install -e source/phc_isaaclab

    # 训练
    python scripts/rsl_rl/train.py --task=Unitree-G1-29dof-Velocity-v0 --headless

    # 无头模式训练
    python scripts/rsl_rl/train.py --task=Unitree-G1-29dof-Velocity-gym-v0 --headless

    # 测试
    python scripts/rsl_rl/play.py --task=Unitree-G1-29dof-Velocity-gym-v0
    ```
## 基于 TWIST 进行可视化
- 将宇树官方github仓库unitree_rl_lab的.csv运动数据转换为方便可视化的pkl格式:
    ```bash
    python scripts/data_convert/unitree_rl_lab2pkl.py --input_file csv_file_path
    ```

- 运行可视化程序:
    ```bash
    # 直接可视化
    python scripts/motion_verify/direct_motion_visualization.py --motion_file pkl_file_path

    # 策略可视化
    python scripts/motion_verify/policy_motion_visualization.py --motion_file pkl_file_path
    ```

## 基于 BeyoundMimic 进行可视化
- 通过正向运动学，将重定向后的运动转换为包含最大坐标信息（躯体位姿、躯体速度、躯体加速度）的形式。
    ```bash
    python scripts/data_convert/csv_to_npz.py --input_file asserts/motions/unitree_rl_lab/G1_Take_102.bvh_60hz.csv --input_fps 30 --headless
    ```

- IsaacLab 运动可视化
    ```bash
    python scripts/motion_verify/direct_motion_visualization_lab.py --input_file /home/hdh/work/project_rl_Lab/motions/unitree_rl_lab/G1_Take_102.bvh_60hz.npz
    
    ```

# 1. 整个流程

[Video] → Cut by scene → GVHMR → [GVHMR-Pred (smplh)] → GMR → [Robot Motion (qpos)] → Artificial Selection → Height-Adjusted → [KungfuAthlete]

## 1. 使用从视频板视频中提取出动作

### 1.1. gvhmr
```bash
conda activate gvhmr
python scripts/motion_recovery/gvhmr.py -s --video=asserts/video/kun.mp4
```
## 2. 将提取出的运动重定向到机器人上

### 2.1. gmr
```bash
conda activate gmr
python scripts/motion_retarget/gmr_gvhmr.py --gvhmr_pred_file=outputs/pipeline/gvhmr/kun/hmr4d_results.pt --save_path=outputs/pipeline/gmr/kun.csv --record_video --video_path=outputs/pipeline/gmr/kun.mp4
```