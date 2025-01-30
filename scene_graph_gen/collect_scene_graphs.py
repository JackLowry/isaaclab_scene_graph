# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import datetime
import glob
import pickle

from omni.isaac.lab.app import AppLauncher
import math
from imitation.data.types import DictObs

from behavior_cloning.actions import ActionSpace, unnormalize_torques
from behavior_cloning.demonstration import load_trajectories, preprocess_trajectories
from behavior_cloning.observation import ObservationType, get_observation_space, preprocess_observation
from behavior_cloning.policy import PolicyType, ReplayTrajectoryPolicy, init_policy
from curobo_utils import MotionPlanner
# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--policy_path", type=str, default="policy", help="Path of policy to evaluate")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.headless = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch

from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from swipe_grab_env import FrankaSwipeGrabEnv, FrankaSwipeGrabEnvCfg
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from omni.isaac.lab.utils.math import transform_points

from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from omni.isaac.core import World

from omni.isaac.core.objects import cuboid, sphere

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.lab.sensors.frame_transformer import FrameTransformer, FrameTransformerCfg

from omni.isaac.lab.utils.math import sample_uniform, quat_from_euler_xyz, euler_xyz_from_quat, matrix_from_quat, quat_from_matrix
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms, quat_from_angle_axis
# from helper import add_extensions, add_robot_to_scene

def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    # if "Reach" in args_cli.task:
    #     # note: reach is the only one that uses a different action space
    #     # compute actions
    #     return delta_pose
    # else:
    # resolve gripper command
    gripper_vel = torch.zeros((delta_pose.shape[0], 2), dtype=torch.float, device=delta_pose.device)
    gripper_vel[:] = -1 if gripper_command else 1
    # compute actions

    
    return torch.concat([delta_pose, gripper_vel], dim=1)


def main():

    action_space_type = ActionSpace.DeltaEndEffectorPose
    policy_type = PolicyType.REPLAY_TRAJECTORY
    observation_type = "image"

    # parse configuration
    env_cfg = FrankaSwipeGrabEnvCfg()
    env_cfg.object_asset_paths = glob.glob("assets/GoogleScanOriginal/*")
    env_cfg.num_objects = 8
    env_cfg.randomize_object_positions = True
    env_cfg.headless = True
    # create environment
    env = FrankaSwipeGrabEnv(cfg=env_cfg)

    # reset environment
    obs_dict, _ = env.reset()

    num_collections = 100
    eval_idx = 0
    # simulate environment -- run everything in inference mode
    observation_type = "image"

    data_root_path = "/home/jack/research/data/isaaclab_sg"

    def format_observation_for_policy():
        obs_temp = {}
        for key, value in obs_dict["policy"].items():
            if key == "images":
                for key, value in obs_dict["policy"]["images"].items():
                    obs_temp[f"images-{key}"] = value
            elif key == "graph" or key == "extra_info":
               obs_temp[f"{key}"] = [pickle.dumps(value)]
            else:
               obs_temp[f"{key}"] = value

        return obs_temp
    obs = format_observation_for_policy()

    run_graphs = []

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        for i in range(num_collections):

            run_graphs.append(env.reset_obs)
    
            obs_dict, _ = env.reset()
            obs = format_observation_for_policy()
            

            if env.unwrapped.sim.is_stopped():
                break

    data_path = os.path.join(data_root_path, datetime.datetime.now().strftime('%m-%d-%Y:%H-%M-%S'))
    os.mkdir(data_path)

    for i, run in enumerate(run_graphs):
        os.mkdir(os.path.join(data_path, str(i)))
        for j, graph in enumerate(run):
            with open(os.path.join(data_path, str(i), f"t_{j}.pkl"), 'wb')as f:
                pickle.dump(graph, f)


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
