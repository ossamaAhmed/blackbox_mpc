"""
Tutorial one: create a parallel environment for the pendulum environment and then create an MPC controller
using the true known analytical model and the same cost/reward functions used in the openAI gym env.
Render the resulting MPC afterwards
"""
from tf_neuralmpc.environment_utils import EnvironmentWrapper
from tf_neuralmpc.examples.true_models import PendulumTrueModel
from tf_neuralmpc.examples.cost_funcs import pendulum_state_reward_function, pendulum_actions_reward_function
import numpy as np
from tf_neuralmpc import Runner

number_of_agents = 5
single_env, parallel_env = EnvironmentWrapper.make_standard_gym_env("Pendulum-v0", random_seed=0,
                                                                    num_of_agents=number_of_agents)
my_runner = Runner(env=[single_env, parallel_env],
                   log_path=None,
                   num_of_agents=number_of_agents)
mpc_controller = my_runner.make_mpc_policy(dynamics_function=PendulumTrueModel(),
                                           state_reward_function=pendulum_state_reward_function,
                                           actions_reward_function=pendulum_actions_reward_function,
                                           planning_horizon=30,
                                           optimizer_name='PI2',
                                           true_model=True)

current_obs = single_env.reset()
current_obs = np.tile(np.expand_dims(current_obs, 0),
                      (number_of_agents, 1))
for t in range(200):
    action_to_execute, expected_obs, expected_reward = mpc_controller.act(current_obs, t)
    current_obs, reward, _, info = single_env.step(action_to_execute[0])
    current_obs = np.tile(np.expand_dims(current_obs, 0),
                          (number_of_agents, 1))
    single_env.render()
