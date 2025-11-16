"""Test RegressionParent class."""

import numpy as np
import pandas as pd

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
    monkeypatch.setattr(
        "rbmpy.circular_regression.RegressionParent.minimize", minimize
    )

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
    expected_list.append(-103.46573590279972)
    expected_list.append(1)
    expected_list.append(1)
    expected_list.append(1)

    assert results_list == expected_list


def test_llh():
    """Test the llh function."""

    # Regression coefficients
    coeffs = np.array([1, 1, 1])

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

    assert llh == 6.133956476193537


def test_llh_w_omikron_1():
    """Test the llh function with learning-rate noise."""

    # Regression coefficients
    coeffs = np.array([1, 1, 1, 1])

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

    assert llh == 6.880652881374994


def test_llh_w_extremely_high_conc():
    """Test the llh function with extremely high concentration."""

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

    assert llh == 92.10340371976183


def test_llh_w_extremely_low_conc():
    """Test the llh function with extremely low concentration."""

    # Regression coefficients
    coeffs = np.array([1, 1, 1.0e-10])

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

    assert llh == 7.35150826542126


def test_llh_w_prior():
    """Test the llh function with a prior over the regression coefficients."""

    # Regression coefficients
    coeffs = np.array([1, 1, 1])

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

    assert llh == 15.246630174229747
