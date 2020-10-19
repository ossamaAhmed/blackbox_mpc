"""
This tutorial is meant to show the modular structure of the package,
and the possibility of extending the functionality of each block further
if needed in your research.(such as new optimizer or
a new trajectory evaluator method..etc)

- instantiate an env for a cheetah.
- Define an MLP to learn a dynamics model
- Define the system handler that takes care of training the model
  and processing the rollouts..etc.
- Define a trajectory evaluator that evaluates the rewards of trajectories.
- Define an optimizer.
- instantiate a random policy to collect rollouts
- instantiate an mpc policy using the previous blocks.
- learn dynamics from random policy.
- use the learned dynamics with mpc and render the result
- record everything in tensorboard
"""
from blackbox_mpc.dynamics_functions.deterministic_mlp import \
    DeterministicMLP
from blackbox_mpc.policies.random_policy import RandomPolicy
from env_modified import HalfCheetahEnvModified
from cost_func import reward_function
from blackbox_mpc.utils.iterative_mpc import learn_dynamics_iteratively_w_mpc
from blackbox_mpc.environment_utils import EnvironmentWrapper
import tensorflow as tf

log_dir = './'
tf_writer = tf.summary.create_file_writer(log_dir)
env = HalfCheetahEnvModified()
num_of_agents = 10

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

learn_dynamics_iteratively_w_mpc(env=EnvironmentWrapper.make_standard_gym_env(
                                     HalfCheetahEnvModified,
                                     num_of_agents=num_of_agents),
                                 env_action_space=env.action_space,
                                 env_observation_space=env.observation_space,
                                 number_of_initial_rollouts=5,
                                 number_of_rollouts_for_refinement=3,
                                 number_of_refinement_steps=40,
                                 task_horizon=1000,
                                 planning_horizon=15,
                                 initial_policy=initial_policy,
                                 dynamics_function=dynamics_function,
                                 num_agents=num_of_agents,
                                 reward_function=reward_function,
                                 log_dir=log_dir,
                                 tf_writer=tf_writer,
                                 optimizer_name='RandomSearch')
