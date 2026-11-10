"""Test RegressionParent class."""

import numpy as np
import pandas as pd
from scipy.stats import norm, vonmises

from examples.RegressionChildExample import RegressionChildExample
from examples.RegVarsExample import RegVars
from rbmpy import RegressionParent


class MockMinimizeRes:
    """Mock-out the scipy minimize function."""

    def __init__(self):

        self.fun = 100
        self.x = np.array(np.arange(9))


def minimize(*args, **kwargs):
    """Mock-out the scipy minimize function."""

    return MockMinimizeRes()


def make_reg_df(n=4):
    """Creates a mock reg_df dataframe for unit testing."""

    # IDs
    ids = np.arange(1, n + 1)

    df = pd.DataFrame(
        {
            "int": np.ones(n),
            "delta_t_rad": np.ones(n),
            "a_t_rad": np.ones(n),
            "group": np.ones(n),
            "subj_num": ids,
            "ID": ids,
        }
    )
    return df


def model_estimation(*args):
    """Model-estimation mock function."""

    return [0, 1, 2, 3, 4, 5, 6, 7]


def test_regression_init():
    """Tests the estimation initialization based on EstVars."""

    # Initialize estimation based on est_vars
    reg_vars = RegVars()
    regression = RegressionParent(reg_vars)

    # Expected free parameters
    which_vars = {
        "beta_0": True,
        "beta_1": True,
        "omikron_0": True,
        "omikron_1": True,
        "lambda_0": False,
        "lambda_1": True,
    }

    # Expected prior mean and width
    prior_mean = [0, 0, 10, 0.1, 0, 0]
    prior_width = [5, 5, 20, 5, 5, 5]

    # Expected fixed coefficients
    fixed_coeffs_reg = {
        "beta_0": 0.0,
        "beta_1": 0.0,
        "lambda_0": 0.0,
        "lambda_1": 0.0,
        "omikron_0": 10.0,
        "omikron_1": 0.0,
    }

    # Expected boundaries for estimation
    beta_0_bnds = (-2, 2)
    beta_1_bnds = (-2, 2)
    omikron_0_bnds = (0.0001, 50)
    omikron_1_bnds = (0.001, 1)
    lambda_0_bnds = (0, 1)
    lambda_1_bnds = (0, 1)

    bnds = [
        beta_0_bnds,
        beta_1_bnds,
        omikron_0_bnds,
        omikron_1_bnds,
        lambda_0_bnds,
        lambda_1_bnds,
    ]

    # Expected update regressors
    which_update_regressors = ["int", "delta_t"]

    assert regression.which_vars == which_vars
    assert regression.fixed_coeffs_reg == fixed_coeffs_reg
    assert regression.prior_mean == prior_mean
    assert regression.prior_width == prior_width
    assert np.isnan(regression.n_subj)
    assert regression.n_ker == 4
    assert regression.seed == 123
    assert regression.show_ind_prog
    assert regression.rand_sp
    assert not regression.use_prior
    assert regression.n_sp == 5
    assert regression.bnds == bnds
    assert regression.which_update_regressors == which_update_regressors


def test_parallel_estimation(monkeypatch):
    """This function tests the parallelization routines.

    We mock out "estimation" and use a patched function defined above that returns
    pre-specified parameter estimates.
    """

    # Create mock data frame
    df_data = make_reg_df()

    # Create regression variables
    reg_vars = RegVars()
    reg_vars.n_subj = df_data.subj_num.nunique()
    reg_vars.n_ker = (
        1  # use single process to avoid multiprocessing issues with mocking
    )

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # prediction error
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: False,  # learning-rate noise
        reg_vars.lambda_0: False,  # perseveration intercept
        reg_vars.lambda_1: False,  # perseveration slope
    }

    # Select parameters according to selected variables and create data frame
    prior_columns = [
        reg_vars.beta_0,
        reg_vars.beta_1,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    # Initialize regression object
    regression = RegressionChildExample(reg_vars)

    # Use monkeypatch to replace the estimation method
    monkeypatch.setattr(regression, "estimation", model_estimation)

    results_df = regression.parallel_estimation(df_data, prior_columns)

    # Create expected results data frame
    output = [
        model_estimation(),
        model_estimation(),
        model_estimation(),
        model_estimation(),
    ]
    columns = [
        reg_vars.beta_0,
        reg_vars.beta_1,
        reg_vars.omikron_0,
        "llh",
        "BIC",
        "group",
        "subj_num",
        "ID",
    ]
    expected_df = pd.DataFrame(output, columns=columns)

    # Test function output
    assert expected_df.equals(results_df)


def test_estimation(monkeypatch):
    """Test the estimation method by mocking scipy's minimize function."""

    # Mock the minimize function where it's imported in RegressionParent
    monkeypatch.setattr("rbmpy.circular_regression.RegressionParent.minimize", minimize)

    # Create mock data frame
    df_data = make_reg_df()

    # Call regression variables
    reg_vars = RegVars()
    reg_vars.n_sp = 1

    # Call regression object
    regression = RegressionChildExample(reg_vars)

    # Get model estimation results
    results_list = regression.estimation(df_data)

    # Expected results list
    min_res = MockMinimizeRes()
    expected_list = min_res.x.tolist()
    expected_list.append(100)
    expected_list.append(206.93147180559944)
    expected_list.append(1)
    expected_list.append(1)
    expected_list.append(1)

    assert results_list == expected_list


def test_llh():
    """Test the llh function."""

    # Regression coefficients
    coeffs = np.array([0, 1, 10])

    # Call regression variables
    reg_vars = RegVars()
    reg_vars.n_sp = 1

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # prediction error
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: False,  # learning-rate noise
        reg_vars.lambda_0: False,  # perseveration intercept
        reg_vars.lambda_1: False,  # perseveration slope
    }

    # Create mock data frame
    df_data = make_reg_df()

    # Get data matrix from child class that is required for the model
    df_subj = RegressionChildExample.get_datamat(df_data)

    # Create regression object
    regression = RegressionChildExample(reg_vars)

    # Compute likelihood
    llh = regression.llh(coeffs, df_subj)

    assert llh == -3.2913427703385136


def test_llh_w_omikron_1():
    """Test the llh function with learning-rate noise."""

    # Regression coefficients
    coeffs = np.array([0, 1, 10, 0.2])

    # Call regression variables
    reg_vars = RegVars()
    reg_vars.n_sp = 1

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # prediction error
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: True,  # learning-rate noise
        reg_vars.lambda_0: False,  # perseveration intercept
        reg_vars.lambda_1: False,  # perseveration slope
    }

    # Create mock data frame
    df_data = make_reg_df()

    # Get data matrix from child class that is required for the model
    df_subj = RegressionChildExample.get_datamat(df_data)

    # Create regression object
    regression = RegressionChildExample(reg_vars)

    # Compute likelihood
    llh = regression.llh(coeffs, df_subj)

    assert llh == -0.17653707973173424


def test_llh_w_extremely_high_conc():
    """Test the llh function with extremely high concentration."""

    # Regression coefficients
    coeffs = np.array([1, 1, 1e-3])

    # Call regression variables
    reg_vars = RegVars()
    reg_vars.n_sp = 1

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # prediction error
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: False,  # learning-rate noise
        reg_vars.lambda_0: False,  # perseveration intercept
        reg_vars.lambda_1: False,  # perseveration slope
    }

    # Create mock data frame
    df_data = make_reg_df()

    # Get data matrix that is required for the model from child class
    df_subj = RegressionChildExample.get_datamat(df_data)

    # Create regression object
    regression = RegressionChildExample(reg_vars)

    # Compute likelihood
    llh = regression.llh(coeffs, df_subj)

    assert llh == 92.10340371976183


def test_llh_w_extremely_low_conc():
    """Test the llh function with extremely low concentration."""

    # Regression coefficients
    coeffs = np.array([1, 1, 1.0e10])

    # Call regression variables
    reg_vars = RegVars()
    reg_vars.n_sp = 1

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # prediction error
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: False,  # learning-rate noise
        reg_vars.lambda_0: False,  # perseveration intercept
        reg_vars.lambda_1: False,  # perseveration slope
    }

    # Create mock data frame
    df_data = make_reg_df()

    # Get data matrix that is required for the model from child class
    df_subj = RegressionChildExample.get_datamat(df_data)

    # Create regression object
    regression = RegressionChildExample(reg_vars)

    # Compute likelihood
    llh = regression.llh(coeffs, df_subj)

    assert llh == 7.351508265637381


def test_llh_w_prior():
    """Test the llh function with a prior over the regression coefficients."""

    # Regression coefficients
    coeffs = np.array([0, 1, 10])

    # Call regression variables
    reg_vars = RegVars()
    reg_vars.n_sp = 1
    reg_vars.use_prior = True

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # prediction error
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: False,  # learning-rate noise
        reg_vars.lambda_0: False,  # perseveration intercept
        reg_vars.lambda_1: False,  # perseveration slope
    }

    # Create mock data frame
    df_data = make_reg_df()

    # Get data matrix that is required for the model from child class
    df_subj = RegressionChildExample.get_datamat(df_data)

    # Create regression object
    regression = RegressionChildExample(reg_vars)

    # Compute likelihood
    llh = regression.llh(coeffs, df_subj)

    assert llh == 5.700080927697698


def test_hard_mixture_llh():
    """Test the llh function with the hard-mixture perseveration model."""

    # Regression coefficients
    coeffs = np.array([0, 1, 10, 5, -0.2])

    # Call regression variables
    reg_vars = RegVars()
    reg_vars.n_sp = 1

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # prediction error
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: False,  # learning-rate noise
        reg_vars.lambda_0: True,  # perseveration intercept
        reg_vars.lambda_1: True,  # perseveration slope
    }

    # Create mock data frame
    df_data = make_reg_df()
    df_data.loc[2:, "delta_t_rad"] = 0.0
    df_data.loc[2:, "a_t_rad"] = 0.0

    # Get data matrix from child class that is required for the model
    df_subj = RegressionChildExample.get_datamat(df_data)

    # Create regression object
    regression = RegressionChildExample(reg_vars)

    # Compute likelihood
    llh = regression.llh(coeffs, df_subj)

    assert llh == -1.019090642084778


def test_vonmises_binned_sum(n_ticks=180, test_kappas=[0.5, 2.0, 10.0, 50.0]):
    """Test the binned sum of the von Mises distribution that we use for normalization."""
    
    bin_width = 2 * np.pi / n_ticks

    # 1. Create evaluation grid: center of each bin across the circle
    # [-pi + bin_width/2, ..., pi - bin_width/2]
    grid_centers = np.linspace(-np.pi + bin_width / 2, np.pi - bin_width / 2, n_ticks)

    for kappa in test_kappas:
        # 2. Evaluate continuous density at each bin center
        density = vonmises.pdf(grid_centers, loc=0.0, kappa=kappa)

        # 3. Scale by bin_width to get discrete probability mass per bin
        prob_mass = density * bin_width

        total_prob = np.sum(prob_mass)
        max_single_prob = np.max(prob_mass)

        print(
            f"kappa={kappa:4.1f} | Peak density={np.max(density):6.2f} | "
            f"Max bin prob={max_single_prob:.4f} | Total sum={total_prob:.6f}"
        )

        # Checks
        assert np.isclose(
            total_prob, 1.0, atol=1e-3
        ), f"Total sum failed for kappa={kappa}!"
        assert (
            max_single_prob <= 1.0
        ), f"Bin width too coarse! Prob > 1 for kappa={kappa}."


def test_linear_llh():
    """Test the llh function for a linear (Gaussian) regression model."""

    # Parameters: beta_0 = 0.5, beta_1 = 0.8, omikron_0 = 2.0 (std dev)
    coeffs = np.array([0.5, 0.8, 2.0])

    reg_vars = RegVars()
    reg_vars.circular = False
    reg_vars.bin_width = np.nan
    reg_vars.n_sp = 1

    reg_vars.which_vars = {
        reg_vars.beta_0: True,
        reg_vars.beta_1: True,
        reg_vars.omikron_0: True,
        reg_vars.omikron_1: False,
        reg_vars.lambda_0: False,
        reg_vars.lambda_1: False,
    }

    df_data = make_reg_df()
    df_subj = RegressionChildExample.get_datamat(df_data)

    regression = RegressionChildExample(reg_vars)
    llh = regression.llh(coeffs, df_subj)

    # Hand-calculate expected negative log-likelihood:
    # int = 1.0, delta_t = 1.0, a_t = 1.0
    # a_hat = 0.5 * 1.0 + 0.8 * 1.0 = 1.3
    # residual_fun returns kappa_up = 1 / (2.0)^2 = 0.25
    # scale = sqrt(1 / 0.25) = 2.0
    # density = norm.pdf(1.0, loc=1.3, scale=2.0)
    expected_density = norm.pdf(1.0, loc=1.3, scale=2.0)
    expected_nllh = -len(df_data) * np.log(expected_density)

    assert np.isclose(llh, expected_nllh, atol=1e-12)


def test_linear_soft_mixture_llh():
    """Test the soft mixture implementation under a linear (Gaussian) model."""

    # Parameters: beta_0=0.0, beta_1=1.0, omikron_0=1.0, lambda_0=0.2, lambda_1=-0.05
    coeffs = np.array([0.0, 1.0, 1.0, 0.2, -0.05])

    reg_vars = RegVars()
    reg_vars.circular = False
    reg_vars.mixture_type = "soft_mixture"
    reg_vars.bin_width = 1.0  # Discrete step size for linear scale
    reg_vars.n_sp = 1

    reg_vars.which_vars = {
        reg_vars.beta_0: True,
        reg_vars.beta_1: True,
        reg_vars.omikron_0: True,
        reg_vars.omikron_1: False,
        reg_vars.lambda_0: True,
        reg_vars.lambda_1: True,
    }

    df_data = make_reg_df(n=4)
    # Set the second half of trials to exact 0 updates (perseveration trials)
    df_data.loc[2:, "delta_t_rad"] = 0.0
    df_data.loc[2:, "a_t_rad"] = 0.0

    df_subj = RegressionChildExample.get_datamat(df_data)
    regression = RegressionChildExample(reg_vars)

    llh = regression.llh(coeffs, df_subj)

    # Ensure the likelihood is finite and positive
    assert np.isfinite(llh)
    assert llh > 0


def test_llh_extreme_outliers():
    """Test llh robustness against antipodal updates, extreme outliers,

    and low motor noise without triggering underflow or -inf/NaN crashes.
    """
    # High precision (low motor noise omikron_0 = 0.05 rad ~ 2.8 deg, omikron_1 = 0)
    # perseveration parameters lambda_0 = 0.1, lambda_1 = -0.01
    coeffs = np.array([0.0, 1.0, 0.05, 0.1, -0.01])

    reg_vars = RegVars()
    reg_vars.mixture_type = "soft_mixture"
    reg_vars.bin_width = 2 * np.pi / 180  # 1-degree bins
    reg_vars.n_sp = 1

    reg_vars.which_vars = {
        reg_vars.beta_0: True,
        reg_vars.beta_1: True,
        reg_vars.omikron_0: True,
        reg_vars.omikron_1: False,
        reg_vars.lambda_0: True,
        reg_vars.lambda_1: True,
    }

    # Construct stress-test trials
    df_data = pd.DataFrame(
        {
            "int": [1.0, 1.0, 1.0, 1.0],
            # Trial 0: standard matching update
            # Trial 1: complete antipodal outlier (predicted ~ +pi, observed = -pi/2 or 0)
            # Trial 2: perseveration stay trial (a_t = 0) while prediction is far away (~ pi)
            # Trial 3: non-zero update (a_t = 0.5) while delta is 0
            "delta_t_rad": [0.5, np.pi, np.pi, 0.0],
            "a_t_rad": [0.5, -np.pi / 2, 0.0, 0.5],
            "group": [1, 1, 1, 1],
            "subj_num": [1, 2, 3, 4],
            "ID": [1, 2, 3, 4],
        }
    )

    df_subj = RegressionChildExample.get_datamat(df_data)
    regression = RegressionChildExample(reg_vars)

    # Compute likelihood
    llh = regression.llh(coeffs, df_subj)

    # 1. Must be finite (no -inf, +inf, or NaN)
    assert np.isfinite(llh)
    # 2. Must not be empty or degenerate
    assert not np.isnan(llh)
    # 3. Penalizing extreme mismatches must yield a positive negative log-likelihood
    assert llh > 0
