"""
- instantiate an env for a pendulum
- Define an MLP to learn a dynamics model
- instantiate a random policy to collect rollouts
- learn dynamics after collecting rollouts randomly
- save the model in the log dir with frequency 1.
- log everything in tensorboard.
"""
from blackbox_mpc.dynamics_functions.deterministic_mlp import \
    DeterministicMLP
from blackbox_mpc.policies.random_policy import RandomPolicy
from blackbox_mpc.utils.dynamics_learning import learn_dynamics_from_policy
from blackbox_mpc.environment_utils import EnvironmentWrapper
import gym
import tensorflow as tf

env = gym.make("Pendulum-v0")
log_dir = './'
tf_writer = tf.summary.create_file_writer(log_dir)
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
                                              dynamics_function=dynamics_function,
                                              log_dir=log_dir,
                                              tf_writer=tf_writer)


