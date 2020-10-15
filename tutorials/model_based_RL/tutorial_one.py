"""
Tutorial two: create a parallel environment for the pendulum environment and then learn the dynamics model
from random rollouts, log the data in tensorboard.
"""
from blackbox_mpc.dynamics_functions.deterministic_mlp import \
    DeterministicMLP
from blackbox_mpc.policies.random_policy import RandomPolicy
from blackbox_mpc.policies.mpc_policy import \
    MPCPolicy
from blackbox_mpc.utils.dynamics_learning import learn_dynamics_from_policy
from blackbox_mpc.environment_utils import EnvironmentWrapper
from blackbox_mpc.utils.pendulum import pendulum_reward_function
import gym
import tensorflow as tf

log_path = './tutorial_2'
env = gym.make("Pendulum-v0")
dynamics_function = DeterministicMLP(layers=[env.action_space.shape[0]+env.observation_space.shape[0],
                                              32,
                                              32,
                                              32,
                                              env.observation_space.shape[0]],
                                     activation_functions=[tf.math.tanh,
                                                           tf.math.tanh,
                                                           tf.math.tanh,
                                                           None])
policy = RandomPolicy(number_of_agents=10,
                      env_action_space=env.action_space)
dynamics_handler = learn_dynamics_from_policy(env=EnvironmentWrapper.make_standard_gym_env("Pendulum-v0",
                                                                                           num_of_agents=10),
                                              policy=policy,
                                              number_of_rollouts=5,
                                              task_horizon=200,
                                              dynamics_function=dynamics_function)
mpc_policy = MPCPolicy(reward_function=pendulum_reward_function,
                       env_action_space=env.action_space,
                       env_observation_space=env.observation_space,
                       dynamics_handler=dynamics_handler,
                       optimizer_name='CEM')

current_obs = env.reset()
for t in range(200):
    action_to_execute, expected_obs, expected_reward = mpc_policy.act(
        current_obs, t)
    current_obs, reward, _, info = env.step(action_to_execute)
    env.render()
