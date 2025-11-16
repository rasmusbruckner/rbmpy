"""Test Agent RBM: Unit tests for the RBM."""

import numpy as np
import pandas as pd
import pytest

from rbmpy import AgentVars, AlAgent


def test_agent_init_linear():
    """Tests the agent initialization for the linear case based on AgentVars."""

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent = AlAgent(agent_vars)

    assert agent.s == 1
    assert agent.h == 0.1
    assert agent.u == 0.0
    assert agent.q == 0.0
    assert agent.sigma == 10
    assert agent.sigma_t_sq == 100
    assert agent.sigma_H == 1
    assert agent.tau_t == 0.5
    assert agent.omega_t == 1
    assert agent.mu_t == 150
    assert agent.max_x == 300
    assert agent.circular == False

    assert np.isnan(agent.a_t)
    assert np.isnan(agent.alpha_t)
    assert np.isnan(agent.tot_var)
    assert np.isnan(agent.C)

def test_agent_init_circular():
    """Tests the agent initialization for the circular case based on AgentVars."""

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent_vars.circular = True
    agent_vars.max_x = 2 * np.pi
    agent = AlAgent(agent_vars)

    assert agent.s == 1
    assert agent.h == 0.1
    assert agent.u == 0.0
    assert agent.q == 0.0
    assert agent.sigma == 10
    assert agent.sigma_t_sq == 100
    assert agent.sigma_H == 1
    assert agent.tau_t == 0.5
    assert agent.omega_t == 1
    assert agent.mu_t == 150
    assert agent.max_x == 2 * np.pi
    assert agent.circular == True

    assert np.isnan(agent.a_t)
    assert np.isnan(agent.alpha_t)
    assert np.isnan(agent.tot_var)
    assert np.isnan(agent.C)

def test_agent_learn_linear_no_ct():
    """Tests the learning function of the agent model for the linear case and without catch trials."""

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent_vars.u = np.exp(0)
    agent = AlAgent(agent_vars)

    # In task_agent_int, input is based on data frame
    df = pd.DataFrame(index=range(0, 1), dtype="float")  # create this frame here
    df["delta_t"] = 50  # add prediction error
    df["b_t"] = 150  # add participant prediction
    df["r_t"] = 0  # add high-reward index
    df["v_t"] = 0  # add catch trial
    df["mu_t"] = 0  # add true heli location (here zero, bc we don't test catch trials)

    # Extract delta and high_val as in task_agent_int
    delta = df["delta_t"]
    high_val = df["r_t"] == 1  # indicates high-value trials

    # Apply learning function
    agent.learn(delta[0], df["b_t"][0], df["v_t"][0], df["mu_t"][0], high_val[0])

    assert agent.s == 1
    assert agent.h == 0.1
    assert agent.u == 1
    assert agent.q == 0
    assert agent.sigma == 10
    assert agent.sigma_H == 1

    assert agent.sigma_t_sq == pytest.approx(163.43733523, rel=1e-6)
    assert agent.tau_t == pytest.approx(0.62040308, rel=1e-6)
    assert agent.omega_t == pytest.approx(0.8718136, rel=1e-6)
    assert agent.mu_t == pytest.approx(196.79533994, rel=1e-6)
    assert np.isnan(agent.C)
    assert agent.a_t == pytest.approx(46.79533994, rel=1e-6)
    assert agent.alpha_t == pytest.approx(0.9359068, rel=1e-6)
    assert agent.tot_var == pytest.approx(200, rel=1e-6)

def test_agent_learn_circular_no_ct():
    """Tests the learning function of the agent model for the circular case and without catch trials."""

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent_vars.circular = True
    agent_vars.max_x = 2 * np.pi
    agent_vars.u = np.exp(0)
    agent_vars.sigma_0 = 0.25**2
    agent_vars.sigma = 0.25
    agent = AlAgent(agent_vars)

    # In task_agent_int, input is based on data frame
    df = pd.DataFrame(index=range(0, 1), dtype="float")  # create this frame here
    df["delta_t"] = np.deg2rad(80)  # add prediction error
    df["b_t"] = np.deg2rad(0)  # add participant prediction
    df["r_t"] = 0  # add high-reward index
    df["v_t"] = 0  # add catch trial
    df["mu_t"] = 0  # add true heli location (here, no ct)

    # Extract delta and high_val as in task_agent_int
    delta = df["delta_t"]
    high_val = df["r_t"] == 1  # indicates high-value trials

    # Apply learning function
    agent.learn(delta[0], df["b_t"][0], df["v_t"][0], df["mu_t"][0], high_val[0])

    assert agent.s == 1
    assert agent.h == 0.1
    assert agent.u == 1
    assert agent.q == 0
    assert agent.sigma == 0.25
    assert agent.sigma_H == 1

    assert agent.sigma_t_sq == pytest.approx(0.09506273147021968, rel=1e-6)
    assert agent.tau_t == pytest.approx(0.6033325938385824, rel=1e-6)
    assert agent.omega_t == pytest.approx(0.9221335051431022, rel=1e-6)
    assert agent.mu_t == pytest.approx(1.3419023331058597, rel=1e-6)
    assert np.isnan(agent.C)
    assert agent.a_t == pytest.approx(1.3419023331058597, rel=1e-6)
    assert agent.alpha_t == pytest.approx(0.961066752571551, rel=1e-6)
    assert agent.tot_var == pytest.approx(0.125, rel=1e-6)

def test_agent_learn_linear_with_rew_bias():
    """Tests the learning function of the agent model without catch trials but including the reward bias."""

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent_vars.u = np.exp(0)
    agent_vars.q = 0.025
    agent = AlAgent(agent_vars)

    # In task_agent_int, input is based on data frame
    df = pd.DataFrame(index=range(0, 1), dtype="float")  # create this frame here
    df["delta_t"] = 50  # add prediction errors
    df["b_t"] = 150  # add participant predictions
    df["r_t"] = 1  # add high-reward index
    df["v_t"] = 0  # add catch trial
    df["mu_t"] = 0  # add true heli location

    # Extract delta and high_val as in task_agent_int
    delta = df["delta_t"]
    high_val = df["r_t"] == 1  # indicates high-value trials

    # Apply learning function
    agent.learn(delta[0], df["b_t"][0], df["v_t"][0], df["mu_t"][0], high_val[0])

    assert agent.s == 1
    assert agent.h == 0.1
    assert agent.u == 1
    assert agent.q == 0.025
    assert agent.sigma == 10
    assert agent.sigma_H == 1

    assert agent.sigma_t_sq == pytest.approx(163.43733523, rel=1e-6)
    assert agent.tau_t == pytest.approx(0.62040308, rel=1e-6)
    assert agent.omega_t == pytest.approx(0.8718136, rel=1e-6)
    assert agent.mu_t == pytest.approx(198.04533994275727, rel=1e-6)
    assert np.isnan(agent.C)
    assert agent.a_t == pytest.approx(48.04533994275727, rel=1e-6)
    assert agent.alpha_t == pytest.approx(0.9609067988551455, rel=1e-6)
    assert agent.tot_var == pytest.approx(200, rel=1e-6)

def test_agent_learn_linear_with_ct():
    """Tests the learning function of the agent model for the linear case and with catch trials."""

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent_vars.u = np.exp(0)
    agent = AlAgent(agent_vars)

    # In task_agent_int, input is based on data frame
    df = pd.DataFrame(index=range(0, 1), dtype="float")  # create this frame here
    df["delta_t"] = 50  # add prediction errors
    df["b_t"] = 150  # add participant predictions
    df["r_t"] = 0  # add high-reward index
    df["v_t"] = 1  # add catch trial
    df["mu_t"] = 190  # add true heli location

    # Extract delta and high_val as in task_agent_int
    delta = df["delta_t"]
    high_val = df["r_t"] == 1  # indicates high value trials

    # Apply learning function
    agent.learn(delta[0], df["b_t"][0], df["v_t"][0], df["mu_t"][0], high_val[0])

    assert agent.s == 1
    assert agent.h == 0.1
    assert agent.u == 1
    assert agent.q == 0
    assert agent.sigma == 10
    assert agent.sigma_H == 1

    assert agent.sigma_t_sq == pytest.approx(361.24233883, rel=1e-6)
    assert agent.tau_t == pytest.approx(0.78319423, rel=1e-6)
    assert agent.omega_t == pytest.approx(0.8718136, rel=1e-6)
    assert agent.mu_t == pytest.approx(190.06728059, rel=1e-6)
    assert agent.C == pytest.approx(0.99009901, rel=1e-6)
    assert agent.a_t == pytest.approx(40.06728059, rel=1e-6)
    assert agent.alpha_t == pytest.approx(0.9359068, rel=1e-6)
    assert agent.tot_var == pytest.approx(200, rel=1e-6)

def test_agent_learn_circular_with_ct():
    """Tests the learning function of the agent model for the circular case and with catch trials."""

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent_vars.circular = True
    agent_vars.max_x = 2 * np.pi
    agent_vars.u = np.exp(0)
    agent_vars.sigma_0 = 0.25**2  # sigma_t_sq_rad2  #0.0625 #np.deg2rad(100)
    agent_vars.sigma = 0.25  # sigma_rad #0.25 #np.deg2rad(10)
    agent_vars.sigma_H = 0.25
    agent = AlAgent(agent_vars)

    # In task_agent_int, input is based on data frame
    df = pd.DataFrame(index=range(0, 1), dtype="float")  # create this frame here
    df["delta_t"] = np.deg2rad(80)  # add prediction error
    df["b_t"] = np.deg2rad(0)  # add participant prediction
    df["r_t"] = 0  # add high-reward index
    df["v_t"] = 1  # add catch trial
    df["mu_t"] = np.deg2rad(0)  # add true heli location

    # Extract delta and high_val as in task_agent_int
    delta = df["delta_t"]
    high_val = df["r_t"] == 1  # indicates high-value trials

    # Apply learning function
    agent.learn(delta[0], df["b_t"][0], df["v_t"][0], df["mu_t"][0], high_val[0])

    assert agent.s == 1
    assert agent.h == 0.1
    assert agent.u == 1
    assert agent.q == 0
    assert agent.sigma == 0.25
    assert agent.sigma_H == 0.25

    assert agent.sigma_t_sq == pytest.approx(0.12147077948673637, rel=1e-6)
    assert agent.tau_t == pytest.approx(0.6602721357469379, rel=1e-6)
    assert agent.omega_t == pytest.approx(0.9221335051431022, rel=1e-6)
    assert agent.mu_t == pytest.approx(0.6709511665529302, rel=1e-6)
    assert agent.C == 0.03125
    assert agent.a_t == pytest.approx(0.6709511665529302, rel=1e-6)
    assert agent.alpha_t == pytest.approx(0.961066752571551, rel=1e-6)
    assert agent.tot_var == pytest.approx(0.125, rel=1e-6)

def test_agent_learn_with_nan_delta():
    """Tests the learning function of the agent model when delta_t = nan
    and the agent should crash with a warning message."""

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent_vars.u = np.exp(0)
    agent = AlAgent(agent_vars)

    # In task_agent_int, input is based on data frame
    df = pd.DataFrame(index=range(0, 1), dtype="float")  # create this frame here
    df["delta_t"] = np.nan  # add prediction error = nan
    df["b_t"] = 150  # add participant predictions
    df["r_t"] = 0  # add high-reward index
    df["v_t"] = 0  # add catch trial
    df["mu_t"] = 150  # add true heli location

    # Extract delta and high_val as in task_agent_int
    delta = df["delta_t"]
    high_val = df["r_t"] == 1  # indicates high-value trials

    with pytest.raises(SystemExit):
        agent.learn(
            delta[0],
            df["b_t"][0],
            df["v_t"][0],
            df["mu_t"][0],
            high_val[0],
        )
