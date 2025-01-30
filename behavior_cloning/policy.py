from enum import Enum
from stable_baselines3.common import policies, torch_layers, utils, vec_env
import gymnasium as gym
import torch
from imitation.policies import base as policy_base
from gymnasium.spaces import Dict, Box, Discrete
from torch import nn

class PolicyType(Enum):
    REPLAY_TRAJECTORY = 1
    ZERO = 2
    RANDOM = 3
    BEHAVIOR_CLONING = 4

def init_policy(observation_space, action_space=Box(-4, 4, [9])):
    extractor = (
        torch_layers.CombinedExtractor
        if isinstance(observation_space, gym.spaces.Dict)
        else torch_layers.FlattenExtractor
    )
    policy = policies.ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        # Set lr_schedule to max value to force error if policy.optimizer
        # is used by mistake (should use self.optimizer instead).
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
        features_extractor_class=extractor
    )
    return policy

class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()

def init_zero_policy(observation_space, action_space=Box(-4, 4, [9])):
    policy = BasePolicy()
    policy.forward = lambda obs: torch.zeros(action_space.shape)
    return policy

class ReplayTrajectoryPolicy(BasePolicy):
    def __init__(self, trajectory):
        super().__init__()
        self.trajectory = trajectory
        self.trajectory_length = trajectory.shape[0]
    
    def forward(self, step):
        if step >= self.trajectory_length:
            step = self.trajectory_length-1
        return self.trajectory[step].unsqueeze(0)