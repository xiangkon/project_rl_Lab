# project_rl_Lab

## 1. Conda 环境介绍
```bash
# roboMotion 运动可视化
pip install -e source/pose
pip install -e source/motion_verify

# env_isaaclab 高版本 rsl-rl==5.0 适配较新的训练框架
pip install -e source/robotMorion_tasks
pip install -e source/motion_verify

# env_isaaclab2 低版本 rsl-rl==3.3 适配较通用的训练框架
conda activate env_isaaclab2
pip install -e source/robotMorion_tasks
pip install -e source/rsl_rl
```

```bash
# pipeline 运动提取(gvmhr) ——> 运动重定向(gmr) ——> 训练(env_isaaclab/2)
pip install -e source/gvhmr
pip install -e source/GMR
```

## 2. RL训练脚本

### 2.1 基础任务
```bash
# 无头模式训练
python scripts/rsl_rl/train.py --task=Unitree-G1-29dof-Velocity-gym-v0 --headless

# 测试
python scripts/rsl_rl/play.py --task=Unitree-G1-29dof-Velocity-gym-Play-v0
```

### 2.2. AMP任务
```bash
python scripts/rsl_rl/train_amp.py --task=Atom01-AMP --headless

python scripts/rsl_rl/play_amp.py --task=Atom01-AMP-Play
```
 

## 3. 运动可视化
### 3.1. 基于 TWIST 进行可视化
1. 将宇树官方github仓库unitree_rl_lab的.csv运动数据转换为方便可视化的pkl格式:
```bash
python scripts/data_convert/unitree_rl_lab2pkl.py --input_file csv_file_path
```

2. 运行可视化程序:
```bash
# 直接可视化
python scripts/motion_verify/direct_motion_visualization.py --motion_file pkl_file_path

# 策略可视化，运行策略模仿运动
python scripts/motion_verify/policy_motion_visualization.py --motion_file pkl_file_path
```

### 3.2. 基于 BeyoundMimic 进行可视化
1. 通过正向运动学，将重定向后的运动转换为包含最大坐标信息（躯体位姿、躯体速度、躯体加速度）的形式(.csv ——> .npz)
```bash
python scripts/data_convert/csv_to_npz.py --input_file asserts/motions/unitree_rl_lab/G1_Take_102.bvh_60hz.csv --input_fps 30 --headless
```

2. IsaacLab 运动可视化
```bash
python scripts/motion_verify/direct_motion_visualization_lab.py --input_file /home/hdh/work/project_rl_Lab/motions/unitree_rl_lab/G1_Take_102.bvh_60hz.npz
```

## 4. 全栈流程 [运动提取(gvmhr) ——> 运动重定向(gmr) ——> 训练(env_isaaclab/2)]

[Video] → Cut by scene → GVHMR → [GVHMR-Pred (smplh)] → GMR → [Robot Motion (qpos)] → Artificial Selection → Height-Adjusted → [KungfuAthlete]

### 4.1. 使用从视频板视频中提取出动作

#### 4.1.1 gvhmr
```bash
conda activate gvhmr
python scripts/motion_recovery/gvhmr.py -s --video=asserts/video/kun.mp4
```
### 4.2. 将提取出的运动重定向到机器人上

#### 4.2.1. gmr
```bash
conda activate gmr
python scripts/motion_retarget/gmr_gvhmr.py --gvhmr_pred_file=outputs/pipeline/gvhmr/kun/hmr4d_results.pt --save_path=outputs/pipeline/gmr/kun.csv --record_video --video_path=outputs/pipeline/gmr/kun.mp4
```

### 4.3. 进行 RL 训练

#### 4.3.1. isaaclab
```bash
# 无头模式训练
python scripts/rsl_rl/train.py --task=Unitree-G1-29dof-Velocity-gym-v0 --headless

# 测试
python scripts/rsl_rl/play.py --task=Unitree-G1-29dof-Velocity-gym-Play-v0
```