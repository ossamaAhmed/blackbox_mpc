"""
tf_neuralmpc/optimizers/optimizer_base.py
==========================================
"""
import tensorflow as tf


class OptimizerBase(tf.Module):
    """This is the base class of the optimizers"""
    def __init__(self, name):
        """
        This is the initializer function for the Base Optimizer.


        Parameters
        ---------
        name: String
            Defines the name of the block of the optimizer.
        """
        super(OptimizerBase, self).__init__(name=name)

    def __call__(self, current_state, time_step, exploration_noise):
        """
       This is the call function for the Base Optimizer Class.
       It is used to calculate the optimal solution for action at the current timestep given the current state.

       Parameters
       ---------
       current_state: tf.float32
           Defines the current state of the system, (dims=num_of_agents X dim_S)
       time_step: tf.float32
           Defines the current timestep of the episode.
       exploration_noise: tf.bool
           Define if the optimal action should have some noise added to it before returning it.


       Returns
        -------
        resulting_action: tf.float32
            The optimal solution for the first action to be applied in the current time step.
        next_state: tf.float32
            The next state predicted using the dynamics model in the trajectory evaluator.
        rewards_of_next_state: tf.float32
            The predicted reward achieved after applying the action given by the optimizer.
       """
        raise Exception("__call__ function is not implemented yet")

    def reset(self):
        """
          This method resets the optimizer to its default state at the beginning of the trajectory/episode.
          """
        raise Exception("reset function is not implemented yet")
