"""
Tutorial one: create a parallel environment for the modified cheetah environment and then learn the dynamics model
from random rollouts initially and use MPC to collect more samples and refine the model in an iterative fashion.
It might be slow on the CPU.
"""
from tf_neuralmpc.environment_utils import EnvironmentWrapper
from env_modified import HalfCheetahEnvModified
from cost_funcs import state_reward_function, actions_reward_function
from tf_neuralmpc.dynamics_functions import DeterministicMLP
from tf_neuralmpc import Runner
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices([gpus[1]], 'GPU')

number_of_agents = 20
single_env, parallel_env = EnvironmentWrapper.make_custom_gym_env(HalfCheetahEnvModified, random_seed=0,
                                                                  num_of_agents=number_of_agents)
log_path = './tutorial_1'
my_runner = Runner(env=[single_env, parallel_env],
                   log_path=log_path,
                   num_of_agents=number_of_agents)

state_size = single_env.observation_space.shape[0]
input_size = single_env.action_space.shape[0]
dynamics_function = DeterministicMLP()
dynamics_function.add_layer(state_size + input_size, 500, activation_function=tf.nn.tanh)
dynamics_function.add_layer(500, 500, activation_function=tf.nn.tanh)
dynamics_function.add_layer(500, 500, activation_function=tf.nn.tanh)
dynamics_function.add_layer(500, state_size)

system_dynamics_handler, mpc_policy = my_runner.learn_dynamics_iteratively_w_mpc(number_of_initial_rollouts=10,
                                                                                 number_of_rollouts_for_refinement=1,
                                                                                 number_of_refinement_steps=10,
                                                                                 dynamics_function=dynamics_function,
                                                                                 task_horizon=1000,
                                                                                 planning_horizon=30,
                                                                                 state_reward_function=state_reward_function,
                                                                                 actions_reward_function=actions_reward_function,
                                                                                 optimizer_name='PI2',
                                                                                 exploration_noise=True)

my_runner.record_rollout(horizon=500, policy=mpc_policy,
                         record_file_path=log_path+'/episode_1')

