"""
This tutorial is meant to show the modular structure of the package,
and the possibility of extending the functionality of each block further
if needed in your research.(such as new optimizer or
a new trajectory evaluator method..etc)

- instantiate an env for a pendulum.
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
from blackbox_mpc.utils.dynamics_learning import learn_dynamics_from_policy
from blackbox_mpc.environment_utils import EnvironmentWrapper
from blackbox_mpc.policies.mpc_policy import \
    MPCPolicy
from blackbox_mpc.trajectory_evaluators.deterministic import \
    DeterministicTrajectoryEvaluator
from blackbox_mpc.optimizers.cem import CEMOptimizer
from blackbox_mpc.dynamics_handlers.system_dynamics_handler import \
    SystemDynamicsHandler
from blackbox_mpc.utils.pendulum import pendulum_reward_function
import gym
import tensorflow as tf

log_dir = './'
env = gym.make("Pendulum-v0")
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

system_dynamics_handler = SystemDynamicsHandler(env_action_space=env.action_space,
                                                env_observation_space=env.observation_space,
                                                true_model=False,
                                                dynamics_function=dynamics_function,
                                                tf_writer=tf_writer,
                                                is_normalized=True,
                                                log_dir=log_dir,
                                                save_model_frequency=2)
trajectory_evaluator = \
                    DeterministicTrajectoryEvaluator(reward_function=pendulum_reward_function,
                                                     system_dynamics_handler=system_dynamics_handler)

optimizer = CEMOptimizer(env_action_space=env.action_space,
                         env_observation_space=env.observation_space,
                         num_agents=10,
                         planning_horizon=30,
                         max_iterations=6)

policy = MPCPolicy(trajectory_evaluator=trajectory_evaluator,
                   optimizer=optimizer,
                   tf_writer=tf_writer)

random_policy = RandomPolicy(number_of_agents=10,
                      env_action_space=env.action_space)
dynamics_handler = learn_dynamics_from_policy(env=EnvironmentWrapper.make_standard_gym_env("Pendulum-v0",
                                                                                           num_of_agents=10),
                                              policy=random_policy,
                                              number_of_rollouts=5,
                                              task_horizon=200,
                                              system_dynamics_handler=system_dynamics_handler)

current_obs = env.reset()
for t in range(200):
    action_to_execute, expected_obs, expected_reward = policy.act(
        current_obs, t)
    current_obs, reward, _, info = env.step(action_to_execute)
    env.render()
