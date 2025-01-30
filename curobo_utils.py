from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.robot import RobotConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
# from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

class MotionPlanner:
    def __init__(self,
                 env,
                 robot_file="franka.yml",
                 world_file="collision_table.yml",
                 collision_checker=False,
                 only_paths=None,
                 reference_prim_path=None,
                 ignore_substring=None):
        robot_file = join_path(get_robot_configs_path(), robot_file)
        world_file = join_path(get_world_configs_path(), world_file)
        self.only_paths = only_paths
        self.reference_prim_path = reference_prim_path
        self.ignore_substring = ignore_substring
        # self.device = device
        self.tensor_args = TensorDeviceType()
        # # mod this later
        n_obstacle_cuboids = 30
        n_obstacle_mesh = 100
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            world_file,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": n_obstacle_cuboids,
                "mesh": n_obstacle_mesh
            },
            interpolation_dt=1)
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True)
        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)
        self.retract_cfg = self.motion_gen.get_retract_config()
        self.device = env.device
        self.env = env
        from curobo.geom.types import WorldConfig
        self.world = WorldConfig()
        self.motion_gen.update_world(self.world)
        # self.curobo_ik = IKPlanner(env, device=self.device)
        # self.usd_help = UsdHelper()
        # self.usd_help.load_stage(env.scene.stage)
        # from curobo.geom.types import Cuboid, WorldConfig
        # self.usd_help.add_world_to_stage(self.world, ba/se_frame="/World")
        self.collision_checker = collision_checker
        if self.collision_checker:
            self.init_collision_mesh()
    def plan_motion(
        self,
        qpos,
        qvel,
        qacc,
        target_position,
        target_quat,
        return_ee_pose=True,
        pose_cost_metric=None
    ):
        start = JointState.from_position(qpos[..., :7])
        start.velocity = qvel
        start.acceleration = qacc
        goal = Pose(target_position, target_quat)
        self.motion_gen.optimize_dt = False
        result = self.motion_gen.plan_single(
            start, goal, MotionGenPlanConfig(max_attempts=8, pose_cost_metric=pose_cost_metric))
        if result.success.item():
            traj = result.get_interpolated_plan()
        else:
            traj = None
        if traj is not None:
            print(
                f"Trajectory Generated: success {result.success.item()} | len {len(traj)} | optimized_dt {result.optimized_dt.item()}"
            )
        # replace joint position with ee pose
        ee_pose = None
        if return_ee_pose and traj is not None:
            ee_pose = self.kin_model.get_state(traj.position)
        return ee_pose, traj
