"""TaskVars: Initialization of the change-point task."""

import numpy as np


class TaskVars:
    """Specifies attributes of the TaskVars object that are used for the change-point task."""

    def __init__(self):
        """Determines the default task variables."""

        self.kappa = 16
        self.sigma = np.sqrt(1 / self.kappa)
        self.h = 0.125
        self.min_x = 0
        self.max_x = 2 * np.pi
        self.min_mu = 0
        self.max_mu = 2 * np.pi
        self.new_block = 1
        self.variable_shield = True
        self.shield_min = np.deg2rad(10)
        self.shield_max = np.pi
        self.shield_mu = np.deg2rad(10)
        self.safe = 3
        self.circular = True
        self.catch_trial_prob = 0.1
