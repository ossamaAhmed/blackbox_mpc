"""
Tutorial seven: create a parallel environment for the pendulum environment and then create the mpc controller with the
true analytical model using the low level API blocks and customizing the optimizer params, Note: each block can be
extended and modified using the base of the block itself.
"""
from tf_neuralmpc.environment_utils import EnvironmentWrapper
from tf_neuralmpc.examples.true_models import PendulumTrueModel
from tf_neuralmpc.trajectory_evaluators.deterministic import DeterministicTrajectoryEvaluator
from tf_neuralmpc.dynamics_handlers.system_dynamics_handler import SystemDynamicsHandler
from tf_neuralmpc.examples.cost_funcs import pendulum_state_reward_function, pendulum_actions_reward_function
from tf_neuralmpc.policies.mpc_policy import MPCPolicy
from tf_neuralmpc.optimizers import PSOOptimizer
import logging
import tensorflow as tf
import numpy as np
from tf_neuralmpc import Runner
logging.getLogger().setLevel(logging.INFO)

number_of_agents = 5
single_env, parallel_env = EnvironmentWrapper.make_standard_gym_env("Pendulum-v0", random_seed=0,
                                                                    num_of_agents=number_of_agents)
my_runner = Runner(env=[single_env, parallel_env],
                   log_path=None,
                   num_of_agents=number_of_agents)

max_iterations = 5
population_size = 500
c1 = 0.3
c2 = 0.5
w = 0.2
initial_velocity_fraction = 0.01
state_size = tf.constant(single_env.observation_space.shape[0], dtype=tf.int32)
input_size = tf.constant(single_env.action_space.shape[0], dtype=tf.int32)
action_upper_bound = tf.constant(single_env.action_space.high, dtype=tf.float32)
action_lower_bound = tf.constant(single_env.action_space.low, dtype=tf.float32)

system_dynamics_handler = SystemDynamicsHandler(dynamics_function=PendulumTrueModel(),
                                                dim_O=state_size,
                                                dim_U=input_size,
                                                num_of_agents=number_of_agents,
                                                true_model=True)

deterministic_trajectory_evaluator = DeterministicTrajectoryEvaluator(state_reward_function=pendulum_state_reward_function,
                                                                      actions_reward_function=pendulum_actions_reward_function,
                                                                      planning_horizon=50,
                                                                      dim_U=input_size,
                                                                      dim_O=state_size,
                                                                      system_dynamics_handler=system_dynamics_handler)
my_optimizer = PSOOptimizer(max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                            population_size=tf.constant(population_size, dtype=tf.int32),
                            c1=tf.constant(c1, dtype=tf.float32),
                            c2=tf.constant(c2, dtype=tf.float32),
                            w=tf.constant(w, dtype=tf.float32),
                            initial_velocity_fraction=tf.constant(
                                 initial_velocity_fraction, dtype=tf.float32),
                            dim_U=input_size,
                            dim_O=state_size,
                            action_upper_bound=action_upper_bound,
                            action_lower_bound=action_lower_bound,
                            num_agents=number_of_agents,
                            trajectory_evaluator=deterministic_trajectory_evaluator,
                            planning_horizon=50)

mpc_controller = MPCPolicy(system_dynamics_handler=system_dynamics_handler,
                           optimizer=my_optimizer)

current_obs = single_env.reset()
current_obs = np.tile(np.expand_dims(current_obs, 0),
                      (number_of_agents, 1))
for t in range(200):
    action_to_execute, expected_obs, expected_reward = mpc_controller.act(current_obs, t)
    current_obs, reward, _, info = single_env.step(action_to_execute[0])
    current_obs = np.tile(np.expand_dims(current_obs, 0),
                          (number_of_agents, 1))
    single_env.render()
