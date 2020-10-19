class ModelFreeBasePolicy(object):
    def __init__(self):
        """
            This is the model free policy base class for controlling the agent
        """
        pass

    def act(self, observations, t, exploration_noise=False):
        """
        This is the act function for the model free policy base class,
        which should be called to provide the action
        to be executed at the current time step.


        Parameters
        ---------
        observations: tf.float32
            Defines the current observations received from the environment.
        t: tf.float32
            Defines the current timestep.
        exploration_noise: bool
            Defines if exploration noise should be added to the action that will be executed.
        Returns
        -------
        action: tf.float32
            The action to be executed for each of the runner (dims = runner X dim_U)
        """
        raise Exception("act function is not implemented yet")

    def reset(self):
        """
        This is the reset function for the model free policy base class,
        which should be called at the beginning of
        the episode.
        """
        raise Exception("reset function is not implemented yet")
