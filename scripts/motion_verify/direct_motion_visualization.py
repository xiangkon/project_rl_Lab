#!/usr/bin/env python
"""
Direct Motion Visualization Script (No Physics Simulation)
==================================
This script reads robot motion data from a .pkl file and visualizes the robot motion in MuJoCo
without any physics simulation - only direct pose playback.

Usage:
    python direct_motion_visualization.py \
        --motion_file assets/example_motions/0807_yanjie_walk_001.pkl \
        --xml assets/g1/g1_sim2sim_29dof.xml
"""

import argparse
import time
import numpy as np
import torch
import mujoco
from mujoco.viewer import launch_passive
import os
import sys

# Add pose module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pose.utils.motion_lib_pkl import MotionLib
from motion_verify.data_utils.rot_utils import euler_from_quaternion_torch, quat_rotate_inverse_torch


def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_motion_steps,
    robot_type: str = "g1",
    mask_indicator: bool = False
):
    """
    Build the mimic_obs at time-step t_step and extract motion pose data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build times
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_motion_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    # Suppose we only have a single motion in the .pkl
    motion_ids = torch.zeros(len(tar_motion_steps), dtype=torch.int, device=device)
    
    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion_torch(root_rot, scalar_first=False)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # Transform velocities to root frame
    root_vel_local = quat_rotate_inverse_torch(root_rot, root_vel, scalar_first=False).reshape(1, -1, 3)
    root_ang_vel_local = quat_rotate_inverse_torch(root_rot, root_ang_vel, scalar_first=False).reshape(1, -1, 3)
    root_vel = root_vel.reshape(1, -1, 3)
    root_ang_vel = root_ang_vel.reshape(1, -1, 3)

    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])
    
    if mask_indicator:
        mimic_obs_buf = torch.cat((
                    # root position: xy velocity + z position
                    root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
                    root_pos[..., 2:3], # 1 dim (z position)
                    # root rotation: roll/pitch + yaw angular velocity
                    roll, pitch, # 2 dims (roll/pitch orientation)
                    root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
                    dof_pos,
                ), dim=-1)[:, :]  # shape (1, 1, 6 + num_dof)
        # append mask indicator 1
        mask_indicator = torch.ones(1, mimic_obs_buf.shape[1], 1).to(device)
        mimic_obs_buf = torch.cat((mimic_obs_buf, mask_indicator), dim=-1)
    else:
        mimic_obs_buf = torch.cat((
                    # root position: xy velocity + z position
                    root_vel_local[..., :2], # 2 dims (xy velocity instead of xy position)
                    root_pos[..., 2:3], # 1 dim (z position)
                    # root rotation: roll/pitch + yaw angular velocity
                    roll, pitch, # 2 dims (roll/pitch orientation)
                    root_ang_vel_local[..., 2:3], # 1 dim (yaw angular velocity)
                    dof_pos,
                ), dim=-1)[:, :]  # shape (1, 1, 6 + num_dof)

    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    
    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


class MotionVisualizer:
    def __init__(self, 
                 xml_file: str,
                 motion_file: str,
                 robot_type: str = "unitree_g1_with_hands",
                 device: str = 'cuda',
                 control_dt: float = 0.02,
                 playback_speed: float = 1.0,
                 loop: bool = True):
        """
        初始化运动可视化器（无物理仿真）
        
        Args:
            xml_file: MuJoCo模型XML文件路径
            motion_file: 运动数据PKL文件路径
            robot_type: 机器人类型
            device: 计算设备(cuda/cpu)
            control_dt: 运动数据的时间步长
            playback_speed: 播放速度倍数
            loop: 是否循环播放
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.control_dt = control_dt
        self.playback_speed = playback_speed
        self.loop = loop
        self.robot_type = robot_type
        
        # 加载运动库
        print(f"Loading motion from {motion_file}...")
        self.motion_lib = MotionLib(motion_file, device=self.device)
        
        # 创建MuJoCo仿真（仅用于可视化，无物理）
        print(f"Loading MuJoCo model from {xml_file}...")
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        # 启动Viewer
        self.viewer = launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.cam.distance = 2.5  # 相机距离
        self.viewer.cam.elevation = -10  # 相机仰角
        self.viewer.cam.azimuth = 45     # 相机方位角
        
        # 机器人配置
        self.num_dofs = 29  # G1机器人自由度数量
        
        # 目标运动步长配置
        self.tar_motion_steps = torch.tensor([1], device=self.device, dtype=torch.int)
        
        # 计算运动总长度
        motion_id = torch.tensor([0], device=self.device, dtype=torch.long)
        motion_length = self.motion_lib.get_motion_length(motion_id)
        self.num_motion_steps = int(motion_length / control_dt)
        
        # 状态变量
        self.current_step = 0
        
        print(f"Motion info:")
        print(f"  Total duration: {motion_length}s")
        print(f"  Total steps: {self.num_motion_steps}")
        print(f"  Time step: {control_dt}s")
        print(f"  Playback speed: {playback_speed}x")
        print(f"  Loop playback: {loop}")

    def reset_visualization(self):
        """重置可视化状态"""
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

    def run(self):
        """运行运动可视化（无物理仿真）"""
        print("\nStarting motion visualization (no physics simulation)...")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                # 检查是否到达运动末尾
                if self.current_step >= self.num_motion_steps:
                    if self.loop:
                        self.current_step = 0
                        print("Motion loop restarted")
                    else:
                        print("Motion playback completed")
                        break
                
                # 记录步长开始时间（控制播放速度）
                step_start_time = time.time()
                
                # 从运动库获取当前步的姿态数据
                _, root_pos, root_rot, dof_pos, _, _ = build_mimic_obs(
                    motion_lib=self.motion_lib,
                    t_step=self.current_step,
                    control_dt=self.control_dt,
                    tar_motion_steps=self.tar_motion_steps,
                    robot_type=self.robot_type,
                    mask_indicator=False
                )
                
                # ====================== 核心：直接设置机器人姿态 ======================
                # 设置根位置 (x, y, z)
                self.data.qpos[:3] = root_pos
                
                # 转换根旋转四元数到MuJoCo格式（MuJoCo: [w, x, y, z]）
                # 注意根据你的数据格式调整四元数顺序，这里假设输入是[x, y, z, w]
                root_rot_mujoco = root_rot[[3, 0, 1, 2]]  # 调整四元数顺序
                self.data.qpos[3:7] = root_rot_mujoco
                
                # 设置关节角度（跳过根的7个自由度：3位置+4旋转）
                self.data.qpos[7:7+self.num_dofs] = dof_pos
                # ====================================================================
                
                # 更新MuJoCo可视化（无物理步进，仅更新几何）
                mujoco.mj_forward(self.model, self.data)
                
                # 让相机跟随骨盆
                pelvis_id = self.model.body("pelvis").id
                if pelvis_id >= 0:
                    self.viewer.cam.lookat = self.data.xpos[pelvis_id]
                
                # 同步Viewer显示
                self.viewer.sync()
                
                # 打印进度（每100步）
                if self.current_step % 100 == 0:
                    progress = (self.current_step / self.num_motion_steps) * 100
                    print(f"Playback progress: {self.current_step}/{self.num_motion_steps} ({progress:.1f}%)")
                
                # 控制播放速度
                elapsed = time.time() - step_start_time
                target_delay = (self.control_dt / self.playback_speed) - elapsed
                if target_delay > 0:
                    time.sleep(target_delay)
                
                # 步进
                self.current_step += 1

        except KeyboardInterrupt:
            print("\nVisualization interrupted by user")
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.viewer:
                self.viewer.close()
            print("Visualization finished")


def main():
    parser = argparse.ArgumentParser(description='MuJoCo Motion Visualization (No Physics)')
    parser.add_argument('--motion_file', type=str, default="/home/hdh/work/project_rl_Lab/motions/example_motions/0807_yanjie_walk_003.pkl",
                        help='Path to motion file (.pkl)')
    parser.add_argument('--xml', type=str, default="/home/hdh/work/project_rl_Lab/unitree_model/G1/29dof/xml/g1_sim2sim_29dof.xml",
                        help='Path to MuJoCo XML model file')
    parser.add_argument('--robot_type', type=str, default='unitree_g1_with_hands',
                        choices=['unitree_g1', 'unitree_g1_with_hands'],
                        help='Robot type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--control_dt', type=float, default=0.02,
                        help='Time step of motion data (seconds)')
    parser.add_argument('--playback_speed', type=float, default=1.0,
                        help='Playback speed multiplier (1.0 = real-time)')
    parser.add_argument('--no_loop', action='store_true',
                        help='Disable loop playback (play once)')
    
    args = parser.parse_args()
    
    # 验证文件存在
    for file_path in [args.motion_file, args.xml]:
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return
    
    # 打印配置信息
    print("=" * 60)
    print("MuJoCo Motion Visualization (No Physics Simulation)")
    print("=" * 60)
    print(f"Motion file: {args.motion_file}")
    print(f"XML model: {args.xml}")
    print(f"Robot type: {args.robot_type}")
    print(f"Device: {args.device}")
    print(f"Motion time step: {args.control_dt}s")
    print(f"Playback speed: {args.playback_speed}x")
    print(f"Loop playback: {not args.no_loop}")
    print("=" * 60)
    
    # 创建并运行可视化器
    visualizer = MotionVisualizer(
        xml_file=args.xml,
        motion_file=args.motion_file,
        robot_type=args.robot_type,
        device=args.device,
        control_dt=args.control_dt,
        playback_speed=args.playback_speed,
        loop=not args.no_loop
    )
    visualizer.run()


if __name__ == "__main__":
    main()
