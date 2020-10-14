"""
Tutorial three: create a parallel environment for the pendulum environment and then create an MPC controller
using the true known analytical model and the same cost/reward functions used in the openAI gym env, then record
some rollouts and switch the optimizer to have a feeling of which ones are better for this problem.
"""
from tf_neuralmpc.environment_utils import EnvironmentWrapper
from tf_neuralmpc.examples.true_models import PendulumTrueModel
from tf_neuralmpc.examples.cost_funcs import pendulum_state_reward_function, pendulum_actions_reward_function
from tf_neuralmpc import Runner

number_of_agents = 1
single_env, parallel_env = EnvironmentWrapper.make_standard_gym_env("Pendulum-v0", random_seed=0,
                                                                    num_of_agents=number_of_agents)
my_runner = Runner(env=[single_env, parallel_env],
                   log_path=None,
                   num_of_agents=number_of_agents)
mpc_controller = my_runner.make_mpc_policy(dynamics_function=PendulumTrueModel(),
                                           state_reward_function=pendulum_state_reward_function,
                                           actions_reward_function=pendulum_actions_reward_function,
                                           planning_horizon=50,
                                           optimizer_name='PI2',
                                           true_model=True)

my_runner.record_rollout(horizon=300, policy=mpc_controller, record_file_path='./pi2')

mpc_controller.switch_optimizer(optimizer_name='RandomSearch')

my_runner.record_rollout(horizon=300, policy=mpc_controller, record_file_path='./random_search')