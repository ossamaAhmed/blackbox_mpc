from blackbox_mpc.policies.model_free_base_policy import ModelFreeBasePolicy
import tensorflow as tf


class RandomPolicy(ModelFreeBasePolicy):
    """This is the random policy for controlling the agent"""
    def __init__(self, number_of_agents, env_action_space):
        """
        This is the initializer function for the random policy.


        Parameters
        ---------
        number_of_agents: tf.int32
            Defines the number of runner running in parallel
        action_upper_bound: tf.float32
            Defines the actions upper bound that could be applied, shape should be 1xdim_U.
        action_lower_bound: tf.float32
            Defines the actions lower bound that could be applied, shape should be 1xdim_U.
        """
        super(RandomPolicy, self).__init__()
        self._num_of_agents = number_of_agents
        self._action_lower_bound = tf.constant(env_action_space.high,
                                               dtype=tf.float32)
        self._action_upper_bound = tf.constant(env_action_space.low,
                                               dtype=tf.float32)
        return

    def act(self, observations, t, exploration_noise=False):
        """
        This is the act function for the random policy, which should be called to provide the action
        to be executed at the current time step.


        Parameters
        ---------
        observations: tf.float32
            Defines the current observations received from the environment.
        t: tf.float32
            Defines the current timestep.


        Returns
        -------
        action: tf.float32
            The action to be executed for each of the runner (dims = runner X dim_U)
        """
        return tf.random.uniform([self._num_of_agents, *self._action_lower_bound.shape],
                                 self._action_lower_bound,
                                 self._action_upper_bound, dtype=tf.float32)

    def reset(self):
        """
        This is the reset function for the random policy, which should be called at the beginning of
        the episode.
        """
        return
