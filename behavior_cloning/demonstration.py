from behavior_cloning.actions import ActionSpace, normalize_torques
from behavior_cloning.observation import format_trajectory_observation, get_observation_space, preprocess_observation

import h5py
import pickle

from imitation.data.types import TrajectoryWithRew, DictObs
import numpy as np

from omni.isaac.lab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat
from omni.isaac.lab.utils.math import subtract_frame_transforms, compute_pose_error

import torch
def load_trajectories(data_path):
    return h5py.File(data_path, 'r')

def preprocess_trajectories(data_path, 
                            observation_type, 
                            action_space, 
                            control_rate_divider=1, 
                            robot_base_pos_w=None,
                            robot_base_quat_w=None):
    
    f = load_trajectories(data_path)
    rollouts = []


    for demo_key in f["data"].keys():
        demo = f["data"][demo_key]

        

        dones = demo["dones"][:]

        next_obs_dict = demo["next_obs"]
        temp_next_obs = {}
        if demo["actions"][:].shape[0] < dones.shape[0]:
            dones = dones[1:]
            for key, value in next_obs_dict.items():
                temp_next_obs[key] = next_obs_dict[key][:][1:]
        else:
            for key in demo["next_obs"].keys():
                temp_next_obs[key] = next_obs_dict[key][:]

        temp_obs = {}
        for key in demo["obs"].keys():
            temp_obs[key] = demo["obs"][key][:]

        demo_len = demo["actions"][:].shape[0]

        selected_idx = np.arange(0, demo_len)
        if control_rate_divider != 1:
            selected_idx = selected_idx[::control_rate_divider]
        if selected_idx[-1] != demo_len -1:
            selected_idx = np.concatenate([selected_idx, [demo_len-1,]])

        obs = preprocess_observation(temp_obs, observation_type)
        obs = {k:v[selected_idx] for k,v in obs.items()}
        next_obs = preprocess_observation(temp_next_obs, observation_type)
        next_obs = {k:v[selected_idx] for k,v in next_obs.items()}
        
        traj_obs = format_trajectory_observation(obs, next_obs, observation_type)

        extra_info = [pickle.loads(i) for i in demo["obs"]['extra_info'][:][selected_idx]]

        actions = demo["actions"][:][selected_idx]
        if action_space == ActionSpace.DeltaJointPos:
            actions = actions - obs['joint_pos']
        if action_space == ActionSpace.DeltaEndEffectorPose:
            # obs_ee_pos_b, obs_ee_quat_b = subtract_frame_transforms(
            #     robot_base_pos_w.tile((demo_len, 1)), 
            #     robot_base_quat_w.tile((demo_len, 1)), 
            #     torch.Tensor(temp_obs["end_effector_pos"][:][selected_idx]), 
            #     torch.Tensor(temp_obs["end_effector_quat"][:][selected_idx])
            # )

            # next_obs_ee_pos_b, next_obs_ee_quat_b = subtract_frame_transforms(
            #     robot_base_pos_w.tile((demo_len, 1)), 
            #     robot_base_quat_w.tile((demo_len, 1)), 
            #     torch.Tensor(temp_next_obs["end_effector_pos"][:][selected_idx]), 
            #     torch.Tensor(temp_next_obs["end_effector_quat"][:][selected_idx])
            # )

            # pos, axis_angle = compute_pose_error(
            #     obs_ee_pos_b,
            #     obs_ee_quat_b,
            #     next_obs_ee_pos_b,
            #     next_obs_ee_quat_b
            # )

            pos, axis_angle = compute_pose_error(
                torch.Tensor(temp_obs["end_effector_pos"][:]),
                torch.Tensor(temp_obs["end_effector_quat"][:]),
                torch.Tensor(temp_next_obs["end_effector_pos"][:]),
                torch.Tensor(temp_next_obs["end_effector_quat"][:])
            )
            
            actions = torch.concatenate((pos, axis_angle), dim=1)
            actions = actions[selected_idx]
        if action_space == ActionSpace.AbsoluteEndEffectorPose:
            actions = next_obs["end_effector_pose"] #- obs["end_effector_pose"]
        if action_space == ActionSpace.JointTorque:
            # actions = normalize_torques(torch.Tensor(demo["obs"]["joint_torque"][:])).numpy()
            actions = torch.Tensor(demo["obs"]["joint_torque"][:]).numpy()
        
        
        rollout = TrajectoryWithRew(
            traj_obs[1:],
            actions[1:],
            extra_info[1:],
            True,
            dones[selected_idx][1:].astype(np.float32)
        )
        rollouts.append(rollout)

    observation_space = get_observation_space(obs, observation_type)

    return rollouts, observation_space
