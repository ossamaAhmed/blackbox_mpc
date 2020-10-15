"""
- instantiate an env for a pendulum
- instantiate an MPC controller using the true known analytical model
- define cost/reward functions as used in the openAI gym env.
- render the resulting MPC afterwards
"""
from blackbox_mpc.policies.mpc_policy import \
    MPCPolicy
from blackbox_mpc.utils.pendulum import PendulumTrueModel, \
    pendulum_reward_function
from blackbox_mpc.utils.rollouts import perform_rollouts
import gym
from blackbox_mpc.environment_utils import EnvironmentWrapper
import tensorflow as tf

env = gym.make("Pendulum-v0")
log_dir = './'
tf_writer = tf.summary.create_file_writer(log_dir)
mpc_policy = MPCPolicy(reward_function=pendulum_reward_function,
                       env_action_space=env.action_space,
                       env_observation_space=env.observation_space,
                       true_model=True,
                       dynamics_function=PendulumTrueModel(),
                       optimizer_name='CEM',
                       tf_writer=tf_writer,
                       log_dir=log_dir,
                       num_agents=1)

perform_rollouts(env=EnvironmentWrapper.make_standard_gym_env("Pendulum-v0",
                                                              num_of_agents=1),
                 number_of_rollouts=4,
                 task_horizon=200,
                 policy=mpc_policy,
                 exploration_noise=False,
                 tf_writer=tf_writer)
