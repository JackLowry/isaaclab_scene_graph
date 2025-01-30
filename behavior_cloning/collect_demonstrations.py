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
# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=50, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
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

from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
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

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)

from omni.isaac.lab.utils.math import sample_uniform, quat_from_euler_xyz, euler_xyz_from_quat, matrix_from_quat, quat_from_matrix

from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

# from helper import add_extensions, add_robot_to_scene

from curobo_utils import MotionPlanner

from omni.isaac.core.utils.stage import get_current_stage

def get_gripper_action(gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    # if "Reach" in args_cli.task:
    #     # note: reach is the only one that uses a different action space
    #     # compute actions
    #     return delta_pose
    # else:
    # resolve gripper command
    gripper_vel = torch.zeros((1, 2), dtype=torch.float).cuda()
    gripper_vel[:] = -1 if gripper_command else 1
    # compute actions

    
    return gripper_vel


def main():

    # parse configuration
    env_cfg = FrankaSwipeGrabEnvCfg()
    env_cfg.demonstration_mode = True
    env_cfg.robot.actuators["panda_shoulder"].stiffness = 40.0
    env_cfg.robot.actuators["panda_shoulder"].damping = 4.0
    env_cfg.robot.actuators["panda_forearm"].stiffness = 40.0
    env_cfg.robot.actuators["panda_forearm"].damping = 4.0
    ##TransformerFrame Sensor


    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    # env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    # env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    # env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    # env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # create environment

    stage = get_current_stage()
    stage.DefinePrim("/curobo", "Xform")

    target_default_position = torch.Tensor([0.0206, 0.6459, 0.5587])
    target_default_orientation = torch.Tensor(quat_from_euler_xyz(torch.Tensor([-math.pi/2]), torch.Tensor([0]), torch.Tensor([0]))).squeeze()
    target = cuboid.VisualCuboid(
        "/World/envs/env_0/Target",
        position=target_default_position,
        orientation=target_default_orientation,
        color=torch.Tensor([1.0, 0, 0]),
        size=0.05,
    )
    env = FrankaSwipeGrabEnv(cfg=env_cfg)

    # create a curobo motion gen instance:
    num_targets = 0
    # assuming obstacles are in objects_path:


    setup_curobo_logger("warn")
    past_pose = None
    # n_obstacle_cuboids = 30
    # n_obstacle_mesh = 100

    # # warmup curobo instance
    # usd_help = UsdHelper()
    target_pose = None

    # tensor_args = TensorDeviceType()
    # robot_cfg_path = get_robot_configs_path()
    # robot_cfg = load_yaml(join_path(robot_cfg_path, "franka.yml"))["robot_cfg"]

    # j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    # default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    # # robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    # articulation_controller = None

    # trajopt_dt = None
    # optimize_dt = True
    # trajopt_tsteps = 32
    # trim_steps = None
    # max_attempts = 4
    # interpolation_dt = 0.05
    # enable_finetune_trajopt = True
    # # if args.reactive:
    # #     trajopt_tsteps = 40
    trajopt_dt = 0.04
    optimize_dt = False
    max_attempts = 1
    trim_steps = [1, None]
    interpolation_dt = trajopt_dt
    enable_finetune_trajopt = False

    # world_cfg_table = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # )
    # world_cfg_table.cuboid[0].pose[2] -= 0.02
    # world_cfg1 = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # ).get_mesh_world()
    # world_cfg1.mesh[0].name += "_mesh"
    # world_cfg1.mesh[0].pose[2] = -10.5
    # world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # def get_world_cfg(env):
    #     return usd_help.get_obstacles_from_stage(
    #                     # only_paths=[obstacles_path],
    #                     reference_prim_path=env.cfg.robot.prim_path,
    #                     ignore_substring=[
    #                         env.cfg.robot.prim_path,
    #                         "/World/envs/env_.*/Target",
    #                         "/World/ground",
    #                         "/World/envs/env_.*/Object_.*",
    #                         "/World/envs/env_.*/Shelf",
    #                         "/World",
    #                         "/curobo",
    #                     ],
    #                 )
    # usd_help.stage = stage    
    # world_cfg = get_world_cfg(env)

    # motion_gen_config = MotionGenConfig.load_from_robot_config(
    #     robot_cfg,
    #     world_cfg,
    #     tensor_args,
    #     collision_checker_type=CollisionCheckerType.MESH,
    #     num_trajopt_seeds=12,
    #     num_graph_seeds=12,
    #     interpolation_dt=interpolation_dt,
    #     collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
    #     optimize_dt=optimize_dt,
    #     trajopt_dt=trajopt_dt,
    #     trajopt_tsteps=trajopt_tsteps,
    #     trim_steps=trim_steps,
        
    # )
    # motion_gen = MotionGen(motion_gen_config)
    # # if not args.reactive:
    # print("warming up...")
    # motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    # print("Curobo is Ready")

    # plan_config = MotionGenPlanConfig(
    #     enable_graph=False,
    #     enable_graph_attempt=2,
    #     max_attempts=max_attempts,
    #     enable_finetune_trajopt=enable_finetune_trajopt,
    #     time_dilation_factor=1.0,
    # )
    cmd_plan = None
    # cmd_idx = 0
    # i = 0
    # spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None


    curobo_motion_planner = MotionPlanner(env)

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(pos_sensitivity=.4, rot_sensitivity=0.08)
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    def reset_callback():
        env.reset()
        target.set_world_pose(target_default_position, target_default_orientation)
        collector_interface.reset()
    teleop_interface.add_callback("L", reset_callback)
    # print helper
    print(teleop_interface)
    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", "FrankaSwipeGrab")
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name="franka_swipe_grab",
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=env.num_envs,
        env_config={"teleop_device": args_cli.teleop_device},
    )

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces
    teleop_interface.reset()
    collector_interface.reset()

    # usd_help.stage = stage    

    step = 0

    hardcode_cube_pos = torch.Tensor([0,0,0]).cuda()



    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not collector_interface.is_stopped():
            if step >= 500:
                if step == 50 or step % 1000 == 0.0:
                    # obstacles = get_world_cfg(env).get_collision_check_world()
                    # motion_gen.update_world(obstacles)
                    
                    print("Updated World")

                # returned in numpy (for some reason)
                cube_position, cube_orientation = target.get_world_pose()
                cube_position = torch.Tensor(cube_position)
                cube_orientation = torch.Tensor(cube_orientation)
                
                if past_pose is None:
                    past_pose = torch.zeros_like(cube_position)
                if target_pose is None:
                    target_pose = torch.zeros_like(cube_position)
                if target_orientation is None:
                    target_orientation = torch.zeros_like(cube_orientation)
                if past_orientation is None:
                    past_orientation = torch.zeros_like(cube_orientation)
                
                if (
                    (
                        torch.norm(cube_position - target_pose) > 1e-3
                        or torch.norm(cube_orientation - target_orientation) > 1e-3
                    )
                    and torch.norm(past_pose - cube_position) == 0.0
                    and torch.norm(past_orientation - cube_orientation) == 0.0
                ):
                    # Set EE teleop goals, use cube for simple non-vr init:
                    ee_translation_goal = cube_position.clone()
                    ee_orientation_teleop_goal = cube_orientation.clone()

                    #transform into robot base frame
                    robot_pos = env._robot.data.body_pos_w[0, 0]
                    robot_orientation_quat = env._robot.data.body_quat_w[0, 0]
                    transformation_matrix = torch.zeros((4,4))
                    transformation_matrix[0:3, 0:3] = matrix_from_quat(robot_orientation_quat)
                    transformation_matrix[3,3] = 1
                    transformation_matrix[0:3, 3] = robot_pos
                    # transformation_matrix = tf_matrix_from_pose(robot_pos, robot_orientation_quat)

                    target_pos_augmented = torch.cat([ee_translation_goal, torch.tensor([1])])
                    target_pos_robot_frame = torch.inverse(transformation_matrix)@target_pos_augmented.T
                    target_pos_robot_frame = target_pos_robot_frame[:3]
                    target_rot_robot_frame_euler = torch.Tensor(euler_xyz_from_quat(robot_orientation_quat.unsqueeze(0))) + torch.Tensor(euler_xyz_from_quat(ee_orientation_teleop_goal.unsqueeze(0)))
                    target_rot_robot_frame = quat_from_euler_xyz(target_rot_robot_frame_euler[0], target_rot_robot_frame_euler[1]-math.pi/2, target_rot_robot_frame_euler[2])

                    pose_cost_metric = PoseCostMetric(
                        hold_partial_pose=True,
                        hold_vec_weight=torch.tensor([1, 1, 1, 0, 0, 0], device=env.device),
                    )
                    pose_cost_metric = None

                    ee_new_pose, traj = curobo_motion_planner.plan_motion(
                        env._robot.data.joint_pos,
                        target_pos_robot_frame.cuda(),
                        target_rot_robot_frame.cuda(),
                        pose_cost_metric=pose_cost_metric
                    )
                    # result = motion_gen.plan_single(cu_js, ik_goal, plan_config)
                    # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))
                    # if num_targets == 1:
                    #     if args.constrain_grasp_approach:
                    #         pose_metric = PoseCostMetric.create_grasp_approach_metric()
                    #     if args.reach_partial_pose is not None:
                    #         reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    #         pose_metric = PoseCostMetric(
                    #             reach_partial_pose=True, reach_vec_weight=reach_vec
                    #         )
                    #     if args.hold_partial_pose is not None:
                    #         hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    #         pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
                    if traj is not None:
                        num_targets += 1

                        cmd_plan = traj

                        cmd_idx = 0
                    else:
                        print("Plan did not converge to a solution: ")
                    target_pose = cube_position
                    target_orientation = cube_orientation            
                past_pose = cube_position
                past_orientation = cube_orientation
            if cmd_plan is not None:
                cmd_state = cmd_plan[cmd_idx]
                target_pos = cmd_state.position
                target_vel = cmd_state.velocity
                past_cmd = cmd_state.clone()
                if cmd_idx < cmd_plan.shape[0]-1:
                    cmd_idx += 1
            else:
                target_pos = env._robot.data.joint_pos[0]
            

            # carb.log_info("Synced CuRobo world from stage.")
            # get keyboard command
            _, gripper_command = teleop_interface.advance()
            # convert to torch
            # compute actions based on environment
            gripper_actions = get_gripper_action(gripper_command)
            actions = torch.zeros((1, 9)).cuda()
            actions[:, 0:7] = target_pos[0:7]
            actions[:, 7:] = gripper_actions
            
            if step > 10:
                # TODO: Deal with the case when reset is triggered by teleoperation device.
                #   The observations need to be recollected.
                # store signals before stepping
                # -- obs
                for key, value in obs_dict["policy"].items():
                    if key == "images":
                        for key, value in obs_dict["policy"]["images"].items():
                            collector_interface.add(f"obs/images-{key}", value)
                    elif key == "graph" or key == "extra_info":
                        collector_interface.add(f"obs/{key}", [pickle.dumps(value)])
                    else:
                        collector_interface.add(f"obs/{key}", value)

                collector_interface.add("actions", actions)

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            if dones.any():
                target.set_world_pose(target_default_position, target_default_orientation)
                teleop_interface._close_gripper = False

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break
                
            if step > 10:
                # robomimic only cares about policy observations
                # store signals from the environment
                # -- next_obs
                for key, value in obs_dict["policy"].items():
                    if key == "images":
                        for key, value in obs_dict["policy"]["images"].items():
                            collector_interface.add(f"next_obs/images-{key}", value)
                    elif key == "graph" or key == "extra_info":
                        collector_interface.add(f"next_obs/{key}", [pickle.dumps(value)])
                    else:
                        collector_interface.add(f"next_obs/{key}", value)

                collector_interface.add("rewards", rewards)
                # -- dones
                collector_interface.add("dones", dones)

                # -- is success label
                collector_interface.add("success", rewards == 1)

            # flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)

            step += 1

            # check if enough data is collected
            if collector_interface.is_stopped():
                break

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
