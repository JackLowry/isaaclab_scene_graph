from enum import Enum
from gymnasium.spaces import Dict, Box
from imitation.data.types import DictObs
from behavior_cloning.actions import normalize_vels
import torch
import numpy as np
from omni.isaac.lab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat, axis_angle_from_quat

class ObservationType(Enum):
    IMAGES = 1
    ZERO = 2
    RANDOM = 3
    BEHAVIOR_CLONING = 4

def get_observation_space(observation, observation_type):
    if observation_type == "image":
        # img_width = observation["depth"].shape[1]
        # img_height = observation["depth"].shape[2]
        observation_space = {
            # "rgb": Box(low=0, high=1, shape=(img_width, img_height, 3)),
            # "depth": Box(low=0, high=1, shape=(img_width, img_height, 1)),
            # "mask": Box(low=1, high=5, shape=(img_width, img_height, 1), dtype=np.int32)
        }
    else:
        raise ValueError(f"{observation_type} is not an expected observation type.")
    
    observation_space["end_effector_pose"] = Box(low=-4, high=4, shape=(6,))
    observation_space["joint_pos"] = Box(low=-4, high=4, shape=(observation["joint_pos"].shape[1],))
    observation_space["joint_vel"] = Box(low=-1, high=1, shape=(observation["joint_vel"].shape[1],))
    # observation_space["step"] = Box(low=0, high=1, shape=(1,))
    observation_space = Dict(observation_space)
    return observation_space

# modifies observation in place
def _preprocess_images(observation):
    depth_processed = observation["depth"]/1.5
    depth_processed[depth_processed == float("Inf")] = 1
    observation["depth"] = depth_processed

    observation["rgb"] = observation["rgb"]/255

    observation["mask"] = observation["mask"]


def preprocess_observation(observation: DictObs, observation_type: str):
    if observation_type == "image":
        observation_processed = {
            # 'rgb': observation["images-rgb"],
            # 'depth': observation["images-depth"],
            # 'mask': observation["images-mask"]
        }
        # _preprocess_images(observation_processed)
    else:
        raise ValueError(f"{observation_type} is not an expected observation type.")
    
    
    axis_angle = axis_angle_from_quat(torch.Tensor(observation["end_effector_quat"][:]))

    observation_processed["end_effector_pose"] = torch.cat((torch.Tensor(observation["end_effector_pos"][:]), 
                                                            torch.Tensor(axis_angle)), dim=1)
    observation_processed["joint_pos"] = observation["joint_pos"]
    observation_processed["joint_vel"] = normalize_vels(torch.Tensor(observation["joint_vel"]))
    return observation_processed


def format_trajectory_observation(observation, next_observation, observation_type):
    if observation_type == "image":
        obs_formatted = {
            # "rgb": np.concatenate([observation["rgb"], next_observation["rgb"][-1][np.newaxis, :]]),
            # "depth":  np.concatenate([observation["depth"], next_observation["depth"][-1][np.newaxis, :]]),
            # "mask": np.concatenate([observation["mask"], next_observation["mask"][-1][np.newaxis, :]]),
        }
    else:
        raise ValueError(f"{observation_type} is not an expected observation type.")
    
    obs_formatted["end_effector_pose"] = np.concatenate([observation["end_effector_pose"], next_observation["end_effector_pose"][-1][np.newaxis, :]])
    obs_formatted["joint_pos"] = np.concatenate([observation["joint_pos"], next_observation["joint_pos"][-1][np.newaxis, :]])
    obs_formatted["joint_vel"] = np.concatenate([observation["joint_vel"], next_observation["joint_vel"][-1][np.newaxis, :]])
    # obs_formatted["step"] = np.arange(obs_formatted["joint_pos"].shape[0])[:, np.newaxis]/obs_formatted["joint_pos"].shape[0]
    obs_formatted = DictObs(obs_formatted)
    return obs_formatted