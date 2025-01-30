import datetime
import imitation
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew, DictObs
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common import policies, torch_layers, utils, vec_env

from behavior_cloning.actions import ActionSpace
from observation import format_trajectory_observation, get_observation_space, preprocess_observation

from demonstration import preprocess_trajectories

import h5py
import pickle

import wandb
# wandb.init(
#     project="bc-scene_graph"
# )

rng = np.random.default_rng(0)

rollout_path = "/home/jack/research/scene_graph/isaaclab_scene_graph/logs/robomimic/FrankaSwipeGrab/bc_dataset_1-16-25-always-close-left-50.hdf5"
observation_type = "image"
action_space = ActionSpace.DeltaJointPos
rollouts, observation_space = preprocess_trajectories(rollout_path, observation_type, action_space)

transitions = rollout.flatten_trajectories_with_rew(rollouts)

bc_trainer = bc.BC(
    observation_space=observation_space,
    action_space=action_space.value,
    demonstrations=transitions,
    rng=rng,
    batch_size=64,
    optimizer_kwargs={
        "lr": .0001
    },
    # custom_logger=imitation.util.logger.WandbOutputFormat()
)
n_epochs = 1000
bc_trainer.train(n_epochs=n_epochs)
save_path = "/home/jack/research/scene_graph/isaaclab_scene_graph/behavior_cloning/ckpt/"
bc_trainer.policy.save(f"{save_path}{observation_type}_{action_space.name}_epochs-{n_epochs}_{datetime.datetime.now().strftime('%m-%d-%Y:%H-%M-%S')}.pt")
# reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print("Reward:", reward)