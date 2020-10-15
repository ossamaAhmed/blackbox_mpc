"""
Tutorial two: create a parallel environment for the pendulum environment and then learn the dynamics model
from random rollouts, log the data in tensorboard.
"""
from blackbox_mpc.dynamics_functions.deterministic_mlp import \
    DeterministicMLP
from blackbox_mpc.policies.random_policy import RandomPolicy
from blackbox_mpc.utils.iterative_mpc import learn_dynamics_iteratively_w_mpc
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
initial_policy = RandomPolicy(number_of_agents=10,
                              env_action_space=env.action_space)

learn_dynamics_iteratively_w_mpc(env=EnvironmentWrapper.make_standard_gym_env(
    "Pendulum-v0", num_of_agents=10),
                                 env_action_space=env.action_space,
                                 env_observation_space=env.observation_space,
                                 number_of_initial_rollouts=5,
                                 number_of_rollouts_for_refinement=2,
                                 number_of_refinement_steps=3,
                                 task_horizon=50,
                                 planning_horizon=10,
                                 initial_policy=initial_policy,
                                 dynamics_function=dynamics_function,
                                 num_agents=10,
                                 reward_function=pendulum_reward_function)
