import pandas as pd
import numpy as np
import pickle
import argparse
import os

def load_and_convert_motion_data(input_file, output_file, fps=60.0):
    """
    加载CSV格式的运动数据，转换为指定格式的字典并保存为pkl文件
    
    参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出PKL文件路径
        fps (float): 帧率，默认60.0
    """
    # 1. 校验输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 2. 定义固定的link_body_list
    link_body_list = [
        'pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
        'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'left_toe_link',
        'pelvis_contour_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
        'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'right_toe_link',
        'waist_yaw_link', 'waist_roll_link', 'torso_link', 'head_link', 'head_mocap', 'imu_in_torso',
        'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
        'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_rubber_hand',
        'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link',
        'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand'
    ]
    
    # 3. 读取CSV数据并转换为numpy数组
    try:
        motion_datas = pd.read_csv(input_file, header=None).to_numpy()
        print(f"成功读取数据，数据形状: {motion_datas.shape}")
    except Exception as e:
        raise RuntimeError(f"读取CSV文件失败: {e}")
    
    # 4. 构建运动数据字典
    motion_dicts = {
        'fps': np.float64(fps),
        'root_pos': motion_datas[:, 0:3],
        'root_rot': motion_datas[:, 3:7],
        'dof_pos': motion_datas[:, 7:],
        'local_body_pos': np.random.randn(len(motion_datas), len(link_body_list), 3).astype(np.float32),
        'link_body_list': link_body_list
    }
    
    output_file = input_file.replace(".csv", ".pkl")
    
    # 6. 保存为pkl文件
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(motion_dicts, f)
        print(f"成功保存PKL文件: {output_file}")
    except Exception as e:
        raise RuntimeError(f"保存PKL文件失败: {e}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将CSV格式的运动数据转换为PKL格式')
    parser.add_argument('--input_file', type=str, default="motions/unitree_rl_lab/G1_Take_102.bvh_60hz.csv", help='输入CSV文件的路径')
    parser.add_argument('--fps', type=float, default=60.0, help='帧率，默认值: 60.0')
    
    args = parser.parse_args()
    
    # 执行数据转换
    try:
        load_and_convert_motion_data(args.input_file, args.fps)
    except Exception as e:
        print(f"程序执行失败: {e}")
        exit(1)

if __name__ == "__main__":
    main()
