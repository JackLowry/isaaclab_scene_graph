# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
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
    env_cfg.demonstration_mode = True
    env_cfg.action_space_type = action_space_type

    if action_space_type == ActionSpace.AbsoluteEndEffectorPose:
        env_cfg.ik_relative_mode = False
    env_cfg.robot.actuators["panda_shoulder"].stiffness = 40000.0
    env_cfg.robot.actuators["panda_shoulder"].damping = 8000.0
    env_cfg.robot.actuators["panda_forearm"].stiffness = 40000.0
    env_cfg.robot.actuators["panda_forearm"].damping = 8000.0


    # create environment
    env = FrankaSwipeGrabEnv(cfg=env_cfg)
    past_pose = None
    target_pose = None

    trajopt_dt = 0.04
    optimize_dt = False
    max_attempts = 1
    trim_steps = [1, None]
    interpolation_dt = trajopt_dt
    enable_finetune_trajopt = False

    cmd_plan = None

    curobo_motion_planner = MotionPlanner(env)


    # reset environment
    obs_dict, _ = env.reset()
    step = 0


    target_num_evals = 30
    eval_idx = 0
    # simulate environment -- run everything in inference mode
    observation_type = "image"

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

        obs_temp = preprocess_observation(obs_temp, observation_type)
        # obs_temp["step"] = torch.Tensor([step/220]).unsqueeze(0).cuda()
        return obs_temp
    obs = format_observation_for_policy()

    observation_space = get_observation_space(obs, observation_type)

    policy_control_rate = 20
    control_rate_divider = 120//policy_control_rate


    if policy_type == PolicyType.REPLAY_TRAJECTORY:

        robot_base_pos_w = env._robot.data.body_pos_w[:, 0].clone().cpu()
        robot_base_quat_w = env._robot.data.body_quat_w[:, 0].clone().cpu()


        traj_path = "/home/jack/research/scene_graph/isaaclab_scene_graph/logs/robomimic/FrankaSwipeGrab/bc_dataset_1-16-25-always-close-left-50.hdf5"
        trajs, _ = preprocess_trajectories(traj_path, observation_type, 
                                           action_space_type, 
                                           control_rate_divider=control_rate_divider, 
                                           robot_base_pos_w=robot_base_pos_w,
                                           robot_base_quat_w=robot_base_quat_w)
        traj = trajs[0].acts
        traj = torch.Tensor(traj).cuda()
        policy = ReplayTrajectoryPolicy(traj)
    elif policy_type == PolicyType.BEHAVIOR_CLONING:
        policy_path = "/home/jack/research/scene_graph/isaaclab_scene_graph/behavior_cloning/ckpt/image_DeltaJointPos_epochs-1000_01-17-2025:10-06-42.pt"
        policy = init_policy(observation_space, action_space=action_space_type.value)
        policy.load(policy_path)
        policy.cuda()

    policy_step = 0
    sim_step = 0

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while eval_idx < target_num_evals:

            # compute actions based on environment


            # TODO: Deal with the case when reset is triggered by teleoperation device.
            #   The observations need to be recollected.
            # store signals before stepping
            # -- obs

            # obs = DictObs(obs)
            if policy_type == PolicyType.REPLAY_TRAJECTORY:
                actions = policy.forward(policy_step).clone()
            elif policy_type == PolicyType.BEHAVIOR_CLONING:
                actions, _, _ = policy.forward(obs)


            # pose_cost_metric = PoseCostMetric(
            #     hold_partial_pose=True,
            #     hold_vec_weight=torch.tensor([1, 1, 1, 0, 0, 0], device=env.device),
            # )
            # pose_cost_metric = None
                

            if action_space_type == ActionSpace.DeltaEndEffectorPose or action_space_type == ActionSpace.AbsoluteEndEffectorPose:
                if sim_step % control_rate_divider != 0:
                    if traj is None or sim_step % control_rate_divider >= traj.shape[0]:
                        actions = torch.zeros((1,7))
                    else:
                        actions = traj[sim_step % control_rate_divider].position.unsqueeze(0)# use the existing pose command given to the IK controller
                else:
                    if action_space_type == ActionSpace.DeltaEndEffectorPose:
                        ee_pos_w = env._robot.data.body_pos_w[:, env.hand_link_idx]
                        ee_quat_w = env._robot.data.body_quat_w[:, env.hand_link_idx]

                        delta_pos = actions[:, :3]
                        delta_quat = quat_from_angle_axis(torch.norm(actions[:, 3:], dim=1), actions[:, 3:])

                        target_pos_w, target_quat_w = combine_frame_transforms(ee_pos_w, ee_quat_w, delta_pos, delta_quat)
                    else:
                        target_pos_w = actions[:, :3]
                        target_quat_w = quat_from_angle_axis(torch.norm(actions[:, 3:], dim=1), actions[:, 3:])

                    robot_base_pos_w = env._robot.data.body_pos_w[:, 0].clone()
                    robot_base_quat_w = env._robot.data.body_quat_w[:, 0].clone()
                    target_pos_b, target_quat_b = subtract_frame_transforms(
                        robot_base_pos_w,
                        robot_base_quat_w,
                        target_pos_w,
                        target_quat_w,
                    )
                        
                    
                    ee_new_pose, traj = curobo_motion_planner.plan_motion(
                        env._robot.data.joint_pos,
                        env._robot.data.joint_vel,
                        env._robot.data.joint_acc,
                        target_pos_b.cuda(),
                        target_quat_b.cuda(),
                        # pose_cost_metric=pose_cost_metric
                    )
                    if traj is None:
                        actions = torch.zeros((1,7))
                    else:
                         actions = traj[0].position.unsqueeze(0)
                actions = pre_process_actions(actions, False)
            elif action_space_type == ActionSpace.JointTorque:
                # actions = unnormalize_torques(actions)
                actions = actions
        

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            sim_step += 1
            if sim_step % control_rate_divider == 0:
                policy_step += 1
            if sim_step >= 400:
                truncated[:] = True
            dones = terminated | truncated
            if dones.any():
                eval_idx += 1
                sim_step = 0
                policy_step = 0
                obs_dict, _ = env.reset()
                # if rewards[0]

            obs = format_observation_for_policy()


            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
