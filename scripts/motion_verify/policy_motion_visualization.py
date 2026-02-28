#!/usr/bin/env python
"""
Direct Motion Visualization Script
==================================
This script reads robot motion data from a .pkl file, feeds it to the TWIST2 model,
and visualizes the robot motion in MuJoCo without requiring Redis communication.

Usage:
    python direct_motion_visualization.py \
        --motion_file assets/example_motions/0807_yanjie_walk_001.pkl \
        --policy assets/ckpts/twist2_1017_20k.onnx \
        --xml assets/g1/g1_sim2sim_29dof.xml
"""

import argparse
import time
import json
import numpy as np
import torch
import mujoco
from mujoco.viewer import launch_passive
from collections import deque
import os
import sys

# Add pose module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from pose.utils.motion_lib_pkl import MotionLib
from motion_verify.data_utils.rot_utils import euler_from_quaternion_torch, quat_rotate_inverse_torch
from motion_verify.data_utils.params import DEFAULT_MIMIC_OBS


def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_motion_steps,
    robot_type: str = "g1",
    mask_indicator: bool = False
):
    """
    Build the mimic_obs at time-step t_step, referencing the code in MimicRunner.
    """
    device = torch.device("cuda")
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


class OnnxPolicyWrapper:
    """Minimal wrapper so ONNXRuntime policies mimic TorchScript call signature."""

    def __init__(self, session, input_name, output_index=0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        if not isinstance(result, np.ndarray):
            result = np.asarray(result, dtype=np.float32)
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path: str, device: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX policy inference but is not installed.")
    providers = []
    available = ort.get_available_providers()
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        else:
            print("CUDAExecutionProvider not available in onnxruntime; falling back to CPUExecutionProvider.")
    providers.append('CPUExecutionProvider')
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX policy loaded from {policy_path} using providers: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


class DirectMotionController:
    def __init__(self, 
                 xml_file: str,
                 policy_path: str,
                 motion_file: str,
                 robot_type: str = "unitree_g1_with_hands",
                 device: str = 'cuda',
                 control_dt: float = 0.02,
                 policy_frequency: int = 100,
                 measure_fps: bool = False,
                 limit_fps: bool = True):
        
        self.device = device
        self.control_dt = control_dt
        self.policy_frequency = policy_frequency
        self.measure_fps = measure_fps
        self.limit_fps = limit_fps
        self.robot_type = robot_type
        
        # Load motion library
        print(f"Loading motion from {motion_file}...")
        self.motion_lib = MotionLib(motion_file, device=device)
        
        # Load policy
        print(f"Loading policy from {policy_path}...")
        self.policy = load_onnx_policy(policy_path, device)
        
        # Create MuJoCo simulation
        print(f"Loading MuJoCo model from {xml_file}...")
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        # Launch viewer
        self.viewer = launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        self.viewer.cam.distance = 2.0
        
        # Robot configuration
        self.num_actions = 29
        self.sim_dt = 0.001
        self.sim_decimation = 1 / (policy_frequency * self.sim_dt)
        
        # Default poses and PD parameters (from server_low_level_g1_sim.py)
        self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (3)
                0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # left arm (7)
                0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
            ])
        
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([1, 0, 0, 0]),
            np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (3)
                0.0, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0, # left arm (7)
                0.0, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0, # right arm (7)
                ])
        ])
        
        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5, 0.2, 0.2, 0.2,
                5, 5, 5, 5, 0.2, 0.2, 0.2,
            ])
        
        self.torque_limits = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
                40, 40, 40, 40, 4.0, 4.0, 4.0,
            ])
        
        self.action_scale = np.array([
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            ])
        
        self.ankle_idx = [4, 5, 10, 11]
        
        # Observation dimensions (from server_low_level_g1_sim.py)
        self.n_mimic_obs = 35  # 6 + 29
        self.n_proprio = 3 + 2 + 3*29    # from config analysis
        self.n_obs_single = 35 + 3 + 2 + 3*29  # n_mimic_obs + n_proprio = 35 + 92 = 127
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs   # 127*11 + 35 = 1402
        
        # Initialize history buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))
        
        # State variables
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.current_step = 0
        
        # Target motion steps (for mimic obs)
        self.tar_motion_steps = torch.tensor([1], device=torch.device(device), dtype=torch.int)
        
        # Calculate motion length
        motion_id = torch.tensor([0], device=torch.device(device), dtype=torch.long)
        motion_length = self.motion_lib.get_motion_length(motion_id)
        self.num_motion_steps = int(motion_length / control_dt)
        
        print(f"Motion length: {motion_length}s, Steps: {self.num_motion_steps}")
        print(f"Controller initialized with:")
        print(f"  Device: {device}")
        print(f"  Control dt: {control_dt}")
        print(f"  Policy frequency: {policy_frequency}Hz")
        print(f"  Total observation size: {self.total_obs_size}")
    
    def reset_sim(self):
        """Reset simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
    
    def reset_robot(self, init_pos=None):
        """Reset robot to initial position"""
        if init_pos is None:
            init_pos = self.mujoco_default_dof_pos
        self.data.qpos[:] = init_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
    
    def extract_robot_state(self):
        """Extract robot state data from simulation"""
        n_dof = self.num_actions
        dof_pos = self.data.qpos[7:7+n_dof]
        dof_vel = self.data.qvel[6:6+n_dof]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        sim_torque = self.data.ctrl
        return dof_pos, dof_vel, quat, ang_vel, sim_torque
    
    def build_proprio_obs(self, dof_pos, dof_vel, quat, ang_vel):
        """Build proprioceptive observation from robot state"""
        from motion_verify.data_utils.rot_utils import quatToEuler
        
        rpy = quatToEuler(quat)
        obs_body_dof_vel = dof_vel.copy()
        obs_body_dof_vel[self.ankle_idx] = 0.
        
        obs_proprio = np.concatenate([
            ang_vel * 0.25,
            rpy[:2],  # only use roll and pitch
            (dof_pos - self.default_dof_pos),
            obs_body_dof_vel * 0.05,
            self.last_action
        ])
        
        return obs_proprio
    
    def run(self):
        """Main simulation loop"""
        print("Starting direct motion visualization...")
        
        # Reset simulation
        self.reset_sim()
        self.reset_robot()
        
        # Performance tracking
        fps_measurements = []
        fps_iteration_count = 0
        fps_measurement_target = 1000
        last_policy_time = None
        policy_execution_times = []
        policy_step_count = 0
        policy_fps_print_interval = 100
        
        try:
            # Main simulation loop
            sim_steps = int(100000)  # Large number, will break when motion ends
            for sim_step in range(sim_steps):
                t_start = time.time()
                
                # Extract robot state
                dof_pos, dof_vel, quat, ang_vel, sim_torque = self.extract_robot_state()
                
                # Check if we've reached the end of the motion
                if self.current_step >= self.num_motion_steps:
                    print(f"Motion completed after {self.current_step} steps.")
                    break
                
                # Run policy at the specified frequency
                if sim_step % self.sim_decimation == 0:
                    # Build proprioceptive observation
                    obs_proprio = self.build_proprio_obs(dof_pos, dof_vel, quat, ang_vel)
                    
                    # Get mimic observation from motion library
                    mimic_obs, root_pos, root_rot, target_dof_pos, root_vel, root_ang_vel = build_mimic_obs(
                        motion_lib=self.motion_lib,
                        t_step=self.current_step,
                        control_dt=self.control_dt,
                        tar_motion_steps=self.tar_motion_steps,
                        robot_type=self.robot_type,
                        mask_indicator=False
                    )
                    
                    # Construct full observation
                    obs_full = np.concatenate([mimic_obs, obs_proprio])
                    
                    # Update history
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    self.proprio_history_buf.append(obs_full)
                    
                    # Combine all observations: current + history + future (mimic_obs as future)
                    obs_buf = np.concatenate([obs_full, obs_hist, mimic_obs])
                    
                    # Ensure correct observation size
                    if obs_buf.shape[0] != self.total_obs_size:
                        print(f"Warning: Observation size mismatch. Expected {self.total_obs_size}, got {obs_buf.shape[0]}")
                        # Pad or truncate if necessary
                        if obs_buf.shape[0] < self.total_obs_size:
                            obs_buf = np.pad(obs_buf, (0, self.total_obs_size - obs_buf.shape[0]))
                        else:
                            obs_buf = obs_buf[:self.total_obs_size]
                    
                    # Run policy
                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                    
                    # Measure policy execution time
                    current_time = time.time()
                    if last_policy_time is not None:
                        policy_interval = current_time - last_policy_time
                        current_policy_fps = 1.0 / policy_interval
                        
                        # Track for frequent printing
                        policy_execution_times.append(policy_interval)
                        policy_step_count += 1
                        
                        # Print policy execution FPS every 100 steps
                        if policy_step_count % policy_fps_print_interval == 0:
                            recent_intervals = policy_execution_times[-policy_fps_print_interval:]
                            avg_interval = np.mean(recent_intervals)
                            avg_execution_fps = 1.0 / avg_interval
                            print(f"Policy Execution FPS (last {policy_fps_print_interval} steps): {avg_execution_fps:.2f} Hz")
                        
                        # For detailed measurement
                        if self.measure_fps:
                            fps_measurements.append(current_policy_fps)
                            fps_iteration_count += 1
                            
                            if fps_iteration_count == fps_measurement_target:
                                avg_fps = np.mean(fps_measurements)
                                print(f"Average Policy FPS: {avg_fps:.2f}")
                                fps_measurements = []
                                fps_iteration_count = 0
                    
                    last_policy_time = current_time
                    
                    # Update last action
                    self.last_action = raw_action
                    
                    # Scale action and compute PD target
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos
                    
                    # Update camera to follow pelvis
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    
                    # Update robot pose in viewer (for visualization of reference motion)
                    # self.data.qpos[:3] = root_pos
                    # root_rot_mujoco = root_rot[[3, 0, 1, 2]]  # Convert to MuJoCo quaternion format
                    # self.data.qpos[3:7] = root_rot_mujoco
                    # self.data.qpos[7:7+self.num_actions] = target_dof_pos
                    # mujoco.mj_forward(self.model, self.data)
                    self.viewer.sync()
                    
                    # Increment motion step
                    self.current_step += 1
                
                # PD control
                torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                
                self.data.ctrl[:] = torque
                mujoco.mj_step(self.model, self.data)
                
                # Sleep to maintain real-time pace
                if self.limit_fps:
                    elapsed = time.time() - t_start
                    if elapsed < self.sim_dt:
                        time.sleep(self.sim_dt - elapsed)
                
                # Print progress
                if sim_step % 100 == 0:
                    print(f"Simulation step: {sim_step}, Motion step: {self.current_step}/{self.num_motion_steps}")
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        except Exception as e:
            print(f"Error in simulation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.viewer:
                self.viewer.close()
            print("Simulation finished.")


def main():
    parser = argparse.ArgumentParser(description='Direct Motion Visualization for TWIST2')
    parser.add_argument('--motion_file', type=str, default="motions/example_motions/0807_yanjie_walk_003.pkl",
                        help='Path to motion file (.pkl)')
    parser.add_argument('--policy', type=str, default="logs/twist/ckpts/twist2_1017_25k.onnx",
                        help='Path to ONNX policy file')
    parser.add_argument('--xml', type=str, default='unitree_model/G1/29dof/xml/g1_sim2sim_29dof.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--robot_type', type=str, default='unitree_g1_with_hands',
                        choices=['unitree_g1', 'unitree_g1_with_hands'],
                        help='Robot type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run policy on (cuda/cpu)')
    parser.add_argument('--control_dt', type=float, default=0.02,
                        help='Control time step (seconds)')
    parser.add_argument('--policy_frequency', type=int, default=100,
                        help='Policy execution frequency (Hz)')
    parser.add_argument('--measure_fps', action='store_true', default=1,
                        help='Measure and print policy execution FPS')
    parser.add_argument('--limit_fps', action='store_true', default=True,
                        help='Limit simulation to real-time (default: True)')
    
    args = parser.parse_args()
    
    # Verify files exist
    for file_path in [args.motion_file, args.policy, args.xml]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return
    
    print("=" * 60)
    print("Direct Motion Visualization for TWIST2")
    print("=" * 60)
    print(f"Motion file: {args.motion_file}")
    print(f"Policy file: {args.policy}")
    print(f"XML file: {args.xml}")
    print(f"Robot type: {args.robot_type}")
    print(f"Device: {args.device}")
    print(f"Control dt: {args.control_dt}")
    print(f"Policy frequency: {args.policy_frequency}Hz")
    print("=" * 60)
    
    # Create and run controller
    controller = DirectMotionController(
        xml_file=args.xml,
        policy_path=args.policy,
        motion_file=args.motion_file,
        robot_type=args.robot_type,
        device=args.device,
        control_dt=args.control_dt,
        policy_frequency=args.policy_frequency,
        measure_fps=args.measure_fps,
        limit_fps=args.limit_fps
    )
    
    controller.run()


if __name__ == "__main__":
    main()
