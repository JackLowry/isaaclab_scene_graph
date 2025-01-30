# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import glob
import math

from behavior_cloning.actions import ActionSpace
import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform, quat_from_euler_xyz, matrix_from_quat, quat_inv, quat_rotate_inverse
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.controllers import DifferentialIKControllerCfg, DifferentialIKController
from omni.isaac.lab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
# import omni.replicator.core as rep
from omni.isaac.lab.sensors.frame_transformer import FrameTransformer, FrameTransformerCfg
from omni.isaac.lab.managers import SceneEntityCfg

from math import pi

from scene_graph_utils import create_scene_graph
from omni.isaac.lab.utils.math import subtract_frame_transforms

@configclass
class FrankaSwipeGrabEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    num_actions = 9
    num_observations = 23
    num_states = 0
    num_objects = 4
    randomize_object_positions = False

    action_space = 9
    observation_space = [64, 64, 3]
    state_space = 0

    demonstration_mode = False
    delta_joint_pos = False


    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)

    # robot

    robot_root_pos_w = torch.Tensor((0, 0.7, 0.0))
    robot_root_quat_w = quat_from_euler_xyz(torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([-pi/2]))[0]
    robot_root_pose_w = torch.cat((robot_root_pos_w, robot_root_quat_w)).unsqueeze(0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.2745,
                "panda_joint2": -1.7059,
                "panda_joint3": 0.2756,
                "panda_joint4": -2.7451,
                "panda_joint5": 2.7265,
                "panda_joint6": 3.5525,
                "panda_joint7": -1.7363,
                "panda_finger_joint.*": 0.035,
            },
            pos=robot_root_pos_w,
            rot=robot_root_quat_w,
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # shelf
    shelf=RigidObjectCfg(
        prim_path="/World/envs/env_.*/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"assets/shelf_small.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=False,
                max_linear_velocity=0,
                max_angular_velocity=0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45), 
            rot=(1, 0, 0, 0)
        ),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )



    # object_asset_paths = glob.glob("assets/GoogleScanOriginal/*")
    object_asset_paths = glob.glob("assets/GoogleScanSubset/*")
    
    # The plane of the shelf face that counts objects as being removed
    shelf_plane_y = 0.5

    # camera
    camera = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/WristCam",
        update_period=0,
        height=512,
        width=512,
        data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(.1, 0, -.08), 
                                   rot=quat_from_euler_xyz(torch.Tensor([0]), torch.Tensor([-math.pi/2]), torch.Tensor([math.pi]))[0].tolist(), 
                                   convention="world"),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    open_reward_scale = 10.0
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0

    ik_relative_mode = True

    # controller for demos
    ik_controller_config = DifferentialIKControllerCfg(
        command_type = "pose",
        use_relative_mode = ik_relative_mode,
        ik_method = 'pinv',
        ik_params = {
            "k_val": 1.0
        } # use defaults
    )
    osc_controller_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        impedance_mode="fixed",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
        gravity_compensation=False,
        motion_stiffness_task=1.0,
        motion_damping_ratio_task=1.0,
        contact_wrench_stiffness_task=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
        nullspace_control="position",
    )
    

    robot_target_transform_cfg = None
    action_space_type = ActionSpace.DeltaEndEffectorPose
    


class FrankaSwipeGrabEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaSwipeGrabEnvCfg

    def __init__(self, cfg: FrankaSwipeGrabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        drawer_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.drawer_local_grasp_pos = drawer_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.drawer_local_grasp_rot = drawer_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.drawer_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.shelf_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.drawer_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.drawer_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # self.rep_writer = rep.BasicWriter(
        #     output_dir="/home/jack/research/scene_graph/isaaclab_scene_graph/pics",
        #     frame_padding=0,
        #     colorize_instance_id_segmentation=self.scene.camera.cfg.colorize_instance_id_segmentation,
        #     colorize_instance_segmentation=True,
        #     colorize_semantic_segmentation=self.scene.camera.cfg.colorize_semantic_segmentation,
        # )

        self.target_objects = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
        
        # controller for demos
        self.ik_controller = DifferentialIKController(self.cfg.ik_controller_config, self.num_envs, self.device)
        self.osc_controller = OperationalSpaceController(self.cfg.osc_controller_cfg, self.num_envs, self.device)

        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(self.scene)

        self.action_space_type = cfg.action_space_type

    def _init_objects(self):

        # objects
        default_x_positions = []
        
        object_spacing=0.25

        #get position four objects at a time, and then cull to the actual number of objects
        starting_x_offset = -object_spacing/2
        for i in range(self.cfg.num_objects//4+1):
            default_x_positions.append(  starting_x_offset + i* object_spacing)
            default_x_positions.append(  starting_x_offset + i* object_spacing)

            default_x_positions.append(-(starting_x_offset + i*-object_spacing))
            default_x_positions.append(-(starting_x_offset + i*-object_spacing))

        default_x_positions = default_x_positions[:self.cfg.num_objects]
        default_x_positions = torch.Tensor(default_x_positions)

        # make the y positions offset for occlusion
        default_y_positions = []
        for i in range(self.cfg.num_objects):
            if i % 2 == 0:
                default_y_positions.append(object_spacing/2)
            else: 
                default_y_positions.append(-object_spacing/2)
        default_y_positions = torch.Tensor(default_y_positions)
        default_y_positions += 0.15

        
        default_z_position = torch.full([self.cfg.num_objects], 0.60)

        default_rot = quat_from_euler_xyz(torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([math.pi/2]))

        default_pos = torch.stack([default_x_positions, default_y_positions, default_z_position], dim=1)
        default_pose = torch.cat([default_pos, default_rot.repeat(self.cfg.num_objects, 1)], dim=1)
        zero_vels = torch.zeros((self.cfg.num_objects, 6))
        self.default_state = torch.cat([default_pose, zero_vels], dim=1)

        # objects starting positions (used when adding objects in and resetting)
        starting_x_positions = torch.full([self.cfg.num_objects], -1)
        starting_y_positions = torch.arange(self.cfg.num_objects)
        starting_z_positions = torch.full([self.cfg.num_objects], 0.0)

        self.starting_x_positions = starting_x_positions
        self.starting_y_positions = starting_y_positions
        self.starting_z_positions = starting_z_positions
        self.default_rot = default_rot

        starting_pos = torch.stack([starting_x_positions, starting_y_positions, starting_z_positions], dim=1)
        starting_pose = torch.cat([starting_pos, default_rot.repeat(self.cfg.num_objects, 1)], dim=1)
        self.starting_state = torch.cat([starting_pose, zero_vels], dim=1)

        self.obj_metadata = {}
    
        self.objects_dict = {}
        for o_idx in range(self.cfg.num_objects):
            obj_class = self.cfg.object_asset_paths[o_idx].split("/")[-1]
            print(o_idx, obj_class)
            obj_name = f"Object_{o_idx}"
            prim_path=f"/World/envs/env_.*/{obj_name}"
            self.objects_dict[obj_name] = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    # usd_path=f"{object_asset_paths[o_idx]}/Props/instanceable_meshes.usd",
                    usd_path=f"{self.cfg.object_asset_paths[o_idx]}/object.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        max_linear_velocity=1000.0,
                        max_angular_velocity=1000.0,
                        max_depenetration_velocity=100.0,
                        enable_gyroscopic_forces=True,
                    ),
                    # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    #     articulation_enabled=False
                    # ),
                    semantic_tags= [("class", obj_class)],
                    # force_usd_conversion=True
                    
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(self.starting_x_positions[o_idx], self.starting_z_positions[o_idx], self.starting_y_positions[o_idx]), 
                    rot=self.default_rot[0].tolist()
                ),
            )
            self.obj_metadata[prim_path] = {
                "class": obj_class,
                "sim_name": obj_name,
                "sim_idx": o_idx
            }

        self.objects = RigidObjectCollectionCfg(
            rigid_objects=self.objects_dict
        )

        self.graphs = None

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._shelf = RigidObject(self.cfg.shelf)
        self.scene.articulations["robot"] = self._robot

        self._init_objects()

        self.scene.objects = RigidObjectCollection(self.objects)

        self.scene.camera = Camera(cfg=self.cfg.camera)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _delta_pose_to_joint_control(self, delta_pose):

        ee_pos_w = self._robot.data.body_pos_w[:, self.hand_link_idx]
        ee_quat_w = self._robot.data.body_quat_w[:, self.hand_link_idx]
    
        robot_base_pos_w = self._robot.data.body_pos_w[:, 0].clone()
        robot_base_quat_w = self._robot.data.body_quat_w[:, 0].clone()

        ee_vel_w = self._robot.data.body_vel_w[:, self.hand_link_idx, :]  # Extract end-effector velocity in the world frame
        root_vel_w = self._robot.data.root_vel_w  # Extract root velocity in the world frame
        relative_vel_w = ee_vel_w - root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = quat_rotate_inverse(self._robot.data.root_quat_w, relative_vel_w[:, 0:3])  # From world to root frame
        ee_ang_vel_b = quat_rotate_inverse(self._robot.data.root_quat_w, relative_vel_w[:, 3:6])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)


        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot_base_pos_w, robot_base_quat_w, ee_pos_w, ee_quat_w
        )

        ee_pose_b = torch.cat((ee_pos_b, ee_quat_b), dim=1)

        ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        jacobian_w = self._robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]

        jacobian_b = jacobian_w.clone()
        root_rot_matrix = matrix_from_quat(quat_inv(self._robot.data.root_quat_w))
        jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

        curr_joint_pos = self._robot.data.joint_pos[:, :7]

        if delta_pose != None:
            # self.ik_controller.set_command(
            #     delta_pose,
            #     ee_pos_b,
            #     ee_quat_b
            # )

            #need to add in empty force control
            # command = torch.cat([delta_pose, torch.zeros_like(delta_pose)], dim=1)

            self.osc_controller.set_command(
                delta_pose,
                ee_pose_b
            )

        # joint_commands = self.ik_controller.compute(
        #     ee_pos_b,
        #     ee_quat_b,
        #     jacobian_b,
        #     curr_joint_pos
        # )

        joint_commands = self.osc_controller.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=None,
            mass_matrix=self._robot.root_physx_view.get_mass_matrices()[:, :7, :][:, :, :7],
            gravity=self.sim._gravity_tensor,
            current_joint_pos=self._robot.data.joint_pos[:, :7],
            current_joint_vel=self._robot.data.joint_vel[:, :7],
            nullspace_joint_pos_target=None,
        )

        return joint_commands

    # pre-physics step calls
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()#.clamp(-1.0, 1.0)

        if self.action_space_type == ActionSpace.DeltaJointPos:
            joint_targets = self._robot.data.joint_pos + actions
        # elif: 
        #     joint_targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        joint_targets = actions

        # if self.action_space_type == ActionSpace.JointTorque:
        self.robot_dof_targets[:] = joint_targets
            # self.robot_dof_targets[:] = torch.clamp(joint_targets, self.robotdo, self.robot_dof_upper_limits)
        # else:
        #     self.robot_dof_targets[:] = torch.clamp(joint_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        # if self.action_space_type == ActionSpace.JointTorque:
        # self._robot.set_joint_effort_target(self.robot_dof_targets)
        # else:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.scene.objects.update(dt=self.physics_dt)
        removed_objects = self.scene.objects.data.object_pos_w[:, :, 1] > self.cfg.shelf_plane_y
        terminated = torch.any(removed_objects,dim=1, keepdim=True)

        truncated = torch.zeros((self.num_envs)) #self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:

        removed_objects = self.scene.objects.data.object_pos_w[:, :, 1] > self.cfg.shelf_plane_y

        # boolean array for matching the target object
        target_object_boolean = torch.ones(self.num_envs, self.cfg.num_objects).to(self.device)
        target_object_boolean = target_object_boolean*torch.arange(self.cfg.num_objects).to(self.device)

        target_object_boolean = target_object_boolean == self.target_objects

        # only the target object should be removed
        successful_envs = torch.all(removed_objects == target_object_boolean, dim=1, keepdim=True)

        return successful_envs
    
        # # Refresh the intermediate values after the physics steps
        # self._compute_intermediate_values()
        # robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        # robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        # return self._compute_rewards(
        #     self.actions,
        #     self._cabinet.data.joint_pos,
        #     self.robot_grasp_pos,
        #     self.drawer_grasp_pos,
        #     self.robot_grasp_rot,
        #     self.drawer_grasp_rot,
        #     robot_left_finger_pos,
        #     robot_right_finger_pos,
        #     self.gripper_forward_axis,
        #     self.drawer_inward_axis,
        #     self.gripper_up_axis,
        #     self.drawer_up_axis,
        #     self.num_envs,
        #     self.cfg.dist_reward_scale,
        #     self.cfg.rot_reward_scale,
        #     self.cfg.open_reward_scale,
        #     self.cfg.action_penalty_scale,
        #     self.cfg.finger_reward_scale,
        #     self._robot.data.joint_pos,
        # )

    def _get_reset_object_states_idx(self, env_ids):
        positions_idxs = []
        for i in range(env_ids.shape[0]):
            if self.cfg.randomize_object_positions:
                positions_idxs.append(torch.randperm(self.cfg.num_objects))
            else:
                positions_idxs.append(torch.Tensor([0,1,2,3]).to(torch.int64))
        positions_idxs = torch.stack(positions_idxs)
        positions_idxs = positions_idxs.unsqueeze(2).repeat(1,1,13)

        default_states_env = self.default_state.repeat(env_ids.shape[0], 1, 1)

        placement_states = torch.gather(default_states_env, 1, positions_idxs)
        return placement_states

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.025,
            0.025,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # take all objects out of scene
        starting_states_env = self.starting_state.repeat(env_ids.shape[0], 1, 1)
        self.scene.objects.write_object_state_to_sim(
                object_state=starting_states_env.cuda(),
                env_ids=env_ids,
                object_ids = None
            )

        #add objects and take observations

        placement_states = self._get_reset_object_states_idx(env_ids)
        object_order_idxs = []
        for i in range(env_ids.shape[0]):
            if self.cfg.randomize_object_positions:
                object_order_idxs.append(torch.randperm(self.cfg.num_objects))
            else:
                object_order_idxs.append(torch.Tensor([0,1,2,3]).to(torch.int64))

        object_order_idxs = torch.stack(object_order_idxs)

        reset_obs = []

        reset_obs_pose = self.cfg.robot_root_pose_w
        reset_obs_pose[:, 1] += 1
        self._robot.write_root_pose_to_sim(
                root_pose=reset_obs_pose.cuda(),
                env_ids=env_ids,
            )

        for i in range(self.cfg.num_objects):
            self.scene.objects.write_object_state_to_sim(
                object_state=placement_states[:, i].cuda(),
                env_ids=env_ids,
                object_ids = object_order_idxs[:, i].cuda()
            )

            #simulate for 20 steps
            for i in range(50):
                #step sim
                self.scene.write_data_to_sim()
                # simulate
                self.sim.step(render=False)
            joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            #step sim
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

            obs = self._get_observations()
            reset_obs.append(obs)
        self.graphs = create_scene_graph(reset_obs)
        for i in range(len(self.graphs)):
            reset_obs[i]['policy']['graph'] = self.graphs[i]
        self.reset_obs = reset_obs


        self.target_objects[env_ids] = 0#torch.randint(self.cfg.num_objects, (env_ids.shape[0],))
        for env_id in env_ids:
            target_id = self.target_objects[env_id]
            obj_class = self.cfg.object_asset_paths[target_id].split("/")[-1]
            print(f"Env {env_id}: target object {target_id}, {obj_class}")

        self._robot.write_root_pose_to_sim(
            root_pose=self.cfg.robot_root_pose_w.cuda(),
            env_ids=env_ids,
        )

        # reset IK controllers
        self.ik_controller.reset(env_ids)



        return
            

    def _get_observations(self) -> dict:
        self.scene.camera.update(dt=self.sim.get_physics_dt())
        rgb = self.scene.camera.data.output["rgb"]
        mask = self.scene.camera.data.output["instance_segmentation_fast"]
        depth = self.scene.camera.data.output["distance_to_image_plane"]
        mask_info = self.scene.camera.data.info[0]["instance_segmentation_fast"]
        obj_states = self.scene.objects.data.object_pos_w

        return {"policy":{"graph": self.graphs, 
                          "images": {
                              "rgb": rgb.clone(),
                              "mask": mask.clone(),
                              "depth": depth.clone(),
                          },
                          "joint_pos":self._robot.data.joint_pos.clone(),
                          "joint_vel":self._robot.data.joint_vel.clone(),
                          "joint_torque":self._robot._root_physx_view.get_dof_projected_joint_forces().clone(),
                          "end_effector_pos":self._robot.data.body_pos_w[:, -3].clone(),
                          "end_effector_quat":self._robot.data.body_quat_w[:, -3].clone(),
                          "object_pos":self.scene.objects.data.object_pos_w.clone(),
                          "object_quat":self.scene.objects.data.object_quat_w.clone(),
                          "object_states":obj_states.clone(),
                          "extra_info": {
                            "mask_info": mask_info,
                            "target_id": self.target_objects.clone(),
                            "object_metadata": self.obj_metadata
                          }}}

    def _compute_rewards(
        self,
        actions,
        cabinet_dof_pos,
        franka_grasp_pos,
        drawer_grasp_pos,
        franka_grasp_rot,
        drawer_grasp_rot,
        franka_lfinger_pos,
        franka_rfinger_pos,
        gripper_forward_axis,
        drawer_inward_axis,
        gripper_up_axis,
        drawer_up_axis,
        num_envs,
        dist_reward_scale,
        rot_reward_scale,
        open_reward_scale,
        action_penalty_scale,
        finger_reward_scale,
        joint_positions,
    ):
        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = (
            torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of forward axis for gripper
        dot2 = (
            torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        )  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions**2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3]  # drawer_top_joint

        # penalty for distance of each finger from the drawer handle
        lfinger_dist = franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2]
        rfinger_dist = drawer_grasp_pos[:, 2] - franka_rfinger_pos[:, 2]
        finger_dist_penalty = torch.zeros_like(lfinger_dist)
        finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        rewards = (
            dist_reward_scale * dist_reward
            + rot_reward_scale * rot_reward
            + open_reward_scale * open_reward
            + finger_reward_scale * finger_dist_penalty
            - action_penalty_scale * action_penalty
        )

        self.extras["log"] = {
            "dist_reward": (dist_reward_scale * dist_reward).mean(),
            "rot_reward": (rot_reward_scale * rot_reward).mean(),
            "open_reward": (open_reward_scale * open_reward).mean(),
            "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            "left_finger_distance_reward": (finger_reward_scale * lfinger_dist).mean(),
            "right_finger_distance_reward": (finger_reward_scale * rfinger_dist).mean(),
            "finger_dist_penalty": (finger_reward_scale * finger_dist_penalty).mean(),
        }

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + 0.25, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.35, rewards + 0.25, rewards)

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    ):
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_drawer_rot, global_drawer_pos = tf_combine(
            drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos
