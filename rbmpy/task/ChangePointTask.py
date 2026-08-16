"""Change-point task: Implementation of the basic functions."""

import sys

import numpy as np

from rbmpy.task.TaskVars import TaskVars


class ChangePointTask:
    """Specifies attributes and methods of the Task object that models the change-point task."""

    def __init__(self, task_vars: TaskVars):
        """Creates the Task object based on the initialization object.

        Parameters
        ----------
        task_vars : TaskVars
            Object instance with task parameters.
        """

        self.sigma = task_vars.sigma
        self.kappa = task_vars.kappa
        self.h = task_vars.h
        self.min_x = task_vars.min_x
        self.max_x = task_vars.max_x
        self.min_mu = task_vars.min_mu
        self.max_mu = task_vars.max_mu
        self.new_block = task_vars.new_block
        self.variable_shield = task_vars.variable_shield
        self.shield_min = task_vars.shield_min
        self.shield_max = task_vars.shield_max
        self.shield_mu = task_vars.shield_mu
        self.safe = task_vars.safe
        self.s = self.safe
        self.circular = task_vars.circular
        self.catch_trial_prob = task_vars.catch_trial_prob

        # Initialize other variables
        self.x_t = np.nan
        self.mu = np.nan
        self.cp = np.nan
        self.shield_size = np.nan
        self.catch_trial = np.nan

    def sample_cp(self) -> None:
        """Samples change points.

        The function takes into account the hazard rate h and the safe criterion s.

        Returns
        -------
        None
            This function does not return any value.
        """

        if self.new_block == 1:
            self.cp = 1
        elif self.s == 0:
            self.cp = np.random.binomial(1, self.h)
        else:
            self.cp = 0

        # Update safe criterion
        if self.cp:
            self.s = self.safe
        else:
            self.s = max([self.s - 1, 0])

    def sample_mu(self) -> None:
        """Samples the mean of the outcome-generating distribution conditional on a change point.

        Returns
        -------
        None
            This function does not return any value.
        """

        if self.cp == 1:
            self.mu = np.random.uniform(self.min_mu, self.max_mu)

    def sample_outcome(self) -> None:
        """Samples the outcome conditional on the outcome-generating mean.

        The function works for normal and circular outcome spaces.

        Returns
        -------
        None
            This function does not return any value.
        """

        if not self.circular:

            self.x_t = round(np.random.normal(self.mu, self.sigma))
            if self.x_t <= self.min_x:
                self.x_t = self.min_x
            elif self.x_t >= self.max_x:
                self.x_t = self.max_x

        elif self.circular:

            # Sample outcome from von Mises distribution
            self.x_t = np.random.vonmises(self.mu, self.kappa) % (2 * np.pi)

        else:

            sys.exit("Invalid option for outcome space")

    def sample_shield(self) -> None:
        """Samples the size of the shield.

        Returns
        -------
        None
            This function does not return any value.
        """

        if self.variable_shield:

            # Sample shield from exponential distribution
            self.shield_size = np.nan
            while (
                np.isnan(self.shield_size)
                or self.shield_size < self.shield_min
                or self.shield_size > self.shield_max
            ):
                self.shield_size = np.random.exponential(self.shield_mu)

        else:

            self.shield_size = self.shield_mu

    def sample_catch_trial(self) -> None:
        """Samples catch trials.

        Returns
        -------
        None
            This function does not return any value.
        """

        if self.cp == 0:
            self.catch_trial = np.random.binomial(1, self.catch_trial_prob)
        else:
            self.catch_trial = 0
