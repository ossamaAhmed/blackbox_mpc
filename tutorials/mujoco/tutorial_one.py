"""
- instantiate an env for a modified version of halfcheetah mujoco env.
- Define an MLP to learn a dynamics model
- instantiate a random policy to collect rollouts
- learn dynamics in an iterative mpc fashion
(collect -> learn model -> collect using mpc with learned model -> repeat)
- record everything in tensorboard
"""
from blackbox_mpc.dynamics_functions.deterministic_mlp import \
    DeterministicMLP
from blackbox_mpc.policies.random_policy import RandomPolicy
from blackbox_mpc.utils.iterative_mpc import learn_dynamics_iteratively_w_mpc
from blackbox_mpc.environment_utils import EnvironmentWrapper
from blackbox_mpc.utils.pendulum import pendulum_reward_function
import tensorflow as tf
from env_modified import HalfCheetahEnvModified
from cost_func import reward_function

log_dir = './'
tf_writer = tf.summary.create_file_writer(log_dir)
env = HalfCheetahEnvModified()
dynamics_function = DeterministicMLP(layers=[env.action_space.shape[0]+
                                             env.observation_space.shape[0],
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

learn_dynamics_iteratively_w_mpc(env=EnvironmentWrapper.make_custom_gym_env(
    HalfCheetahEnvModified, num_of_agents=10),
                                 env_action_space=env.action_space,
                                 env_observation_space=env.observation_space,
                                 number_of_initial_rollouts=5,
                                 number_of_rollouts_for_refinement=2,
                                 number_of_refinement_steps=3,
                                 task_horizon=200,
                                 planning_horizon=50,
                                 initial_policy=initial_policy,
                                 dynamics_function=dynamics_function,
                                 num_agents=10,
                                 reward_function=pendulum_reward_function,
                                 log_dir=log_dir,
                                 tf_writer=tf_writer)
