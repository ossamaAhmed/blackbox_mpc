"""
This tutorial is meant to show how to train a NN dynamics with MPC for the
cheetah environment.
"""
from blackbox_mpc.dynamics_functions.deterministic_mlp import \
    DeterministicMLP
from blackbox_mpc.policies.random_policy import RandomPolicy
from env_modified import HalfCheetahEnvModified
from cost_func import reward_function
from blackbox_mpc.utils.iterative_mpc import learn_dynamics_iteratively_w_mpc
from blackbox_mpc.environment_utils import EnvironmentWrapper
from blackbox_mpc.utils.recording import record_rollout
import tensorflow as tf

log_dir = './'
tf_writer = tf.summary.create_file_writer(log_dir)
env = HalfCheetahEnvModified()
num_of_agents = 1
parallel_env = EnvironmentWrapper.make_custom_gym_env(
                                     HalfCheetahEnvModified,
                                     num_of_agents=num_of_agents)

dynamics_function = DeterministicMLP(layers=[env.action_space.shape[0]+
                                             env.observation_space.shape[0],
                                             500,
                                             500,
                                             500,
                                             env.observation_space.shape[0]],
                                     activation_functions=[tf.math.tanh,
                                                           tf.math.tanh,
                                                           tf.math.tanh,
                                                           None])
initial_policy = RandomPolicy(number_of_agents=num_of_agents,
                              env_action_space=env.action_space)

system_dynamics_handler, mpc_policy = learn_dynamics_iteratively_w_mpc(
                                 env=parallel_env,
                                 env_action_space=env.action_space,
                                 env_observation_space=env.observation_space,
                                 number_of_initial_rollouts=5,
                                 number_of_rollouts_for_refinement=3,
                                 number_of_refinement_steps=1,
                                 task_horizon=1000,
                                 planning_horizon=15,
                                 initial_policy=initial_policy,
                                 dynamics_function=dynamics_function,
                                 num_agents=num_of_agents,
                                 reward_function=reward_function,
                                 log_dir=log_dir,
                                 tf_writer=tf_writer,
                                 optimizer_name='RandomSearch',
                                 population_size=4048,
                                 save_model_frequency=2,
                                 batch_size=512,
                                 epochs=100)

record_rollout(env, horizon=1000, policy=mpc_policy,
               record_file_path='./current_policy_0')

for i in range(9):
    system_dynamics_handler, mpc_policy = learn_dynamics_iteratively_w_mpc(
                                     env=parallel_env,
                                     number_of_initial_rollouts=0,
                                     number_of_rollouts_for_refinement=3,
                                     number_of_refinement_steps=5,
                                     refinement_policy=mpc_policy,
                                     task_horizon=1000,
                                     system_dynamics_handler=system_dynamics_handler,
                                     batch_size=512,
                                     epochs=100,
                                     tf_writer=tf_writer,
                                     start_episode=3+(i*5*3))
    record_rollout(env, horizon=1000, policy=mpc_policy,
                   record_file_path='./current_policy_'+str(i+1))
