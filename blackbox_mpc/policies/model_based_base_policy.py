class ModelBasedBasePolicy(object):
    def __init__(self, trajectory_evaluator):
        """
            This is the model based policy base class for controlling the agent

        Parameters
        ---------
        trajectory_evaluator: EvaluatorBase
            Defines the trajectory evaluator to be used in the optimizer to
            evaluate trajectories.
        """
        self._trajectory_evaluator = trajectory_evaluator
        return

    def act(self,  observations, t, exploration_noise=False):
        """
        This is the act function for the model based policy base class,
        which should be called to provide the action
        to be executed at the current time step.


        Parameters
        ---------
        observations: tf.float32
            Defines the current observations received from the environment.
        t: tf.float32
            Defines the current timestep.
        exploration_noise: bool
            Defines if exploration noise should be added to the action to be executed.


        Returns
        -------
        action: tf.float32
            The action to be executed for each of the runner (dims = runner X dim_U)
        next_observations: tf.float32
            The next observations predicted using the dynamics function learned so far.
        rewards_of_next_state: tf.float32
            The predicted reward if the action was executed using the predicted observations.
        """
        raise Exception("act function is not implemented yet")

    def reset(self):
        """
        This is the reset function for the model based policy base class,
        which should be called at the beginning of the episode.
        """
        raise Exception("reset function is not implemented yet")
