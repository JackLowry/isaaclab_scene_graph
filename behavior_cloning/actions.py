from enum import Enum
from gymnasium.spaces import Dict, Box, Discrete
import numpy as np

JOINT_1_4_TORQUE_MAX = 88
JOINT_5_7_TORQUE_MAX = 13

JOINT_1_4_VEL_MAX = 2.2
JOINT_5_7_VEL_MAX = 2.62

class ActionSpace(Enum):
    DeltaJointPos = Box(low=-4, high=-4, shape=(9,))
    DeltaEndEffectorPose = Box(low=-1, high=1, shape=(6,))
    JointTorque = Box(low=-1, high=1, shape=(9,))
    AbsoluteEndEffectorPose = Box(low=-5, high=5, shape=(6,))

def normalize_torques(torques):
    new_torques = torques.clone()
    new_torques[:, :4] /= JOINT_1_4_TORQUE_MAX
    new_torques[:, 4:7] /= JOINT_5_7_TORQUE_MAX
    return new_torques

def unnormalize_torques(torques):
    new_torques = torques.clone()
    new_torques[:, :4] *= JOINT_1_4_TORQUE_MAX
    new_torques[:, 4:7] *= JOINT_5_7_TORQUE_MAX
    return new_torques

def normalize_vels(vels):
    new_vels = vels.clone()
    new_vels[:, :4] /= JOINT_1_4_VEL_MAX
    new_vels[:, 4:7] /= JOINT_5_7_VEL_MAX
    return new_vels

def unnormalize_vels(vels):
    new_vels = vels.clone()
    new_vels[:, :4] *= JOINT_1_4_VEL_MAX
    new_vels[:, 4:7] *= JOINT_5_7_VEL_MAX
    return new_vels