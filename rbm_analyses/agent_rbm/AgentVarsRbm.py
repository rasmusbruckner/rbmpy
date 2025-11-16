"""AgentVars: Initialization of the reduced Bayesian model"""


class AgentVars:
    """Specifies the AgentVars object for the reduced optimal model.

    futuretodo: Consider using a data class here
    """

    def __init__(self):
        # Determines the default agent variables.

        self.s = 1  # surprise sensitivity
        self.h = 0.1  # hazard rate
        self.u = 0.0  # uncertainty underestimation
        self.q = 0  # reward bias
        self.sigma = 10  # noise in the environment (standard deviation)
        self.sigma_0 = 100  # initial variance of predictive distribution
        self.sigma_H = 1  # catch-trial standard deviation of hidden-mean cue
        self.tau_0 = 0.5  # initial relative uncertainty
        self.omega_0 = 1  # initial change-point probability
        self.mu_0 = 150  # initial belief about mean
        self.max_x = 300  # maximum outcome
        self.circular = False  # circular vs. linear outcome space
