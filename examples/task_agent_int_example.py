"""Task-Agent Interaction Example: Interaction between reduced Bayesian model and change-point task."""

import numpy as np
import pandas as pd

from rbmpy import AgentVars, AlAgent, residual_fun
from rbmpy.task.ChangePointTask import ChangePointTask
from rbmpy.task.TaskVars import TaskVars
from rbmpy.utilities import circ_dist


def task_agent_int(
    df_subj: pd.DataFrame,
    agent: AlAgent,
    agent_vars: AgentVars,
    sel_coeffs: dict,
) -> pd.DataFrame:
    """This function models the interaction between task and agent (RBM).

    Parameters
    ----------
    df_subj : pd.DataFrame
        Data frame with relevant data.
    agent : AlAgent
        Agent-object instance.
    agent_vars : AgentVars
        Agent-variables-object instance.
    sel_coeffs : dict
        Selected model parameters.

    Returns
    -------
    pd.DataFrame
        Simulated agent behavior.
    """

    # Extract and initialize relevant variables
    # -----------------------------------------
    n_trials = len(df_subj)  # number of trials
    mu = np.full([n_trials], np.nan)  # inferred mean
    a_hat = np.full(n_trials, np.nan)  # predicted update according
    concentration = np.full(n_trials, np.nan)  # response noise
    omega = np.full(n_trials, np.nan)  # change-point probability
    tau = np.full(n_trials, np.nan)  # relative uncertainty
    alpha = np.full(n_trials, np.nan)  # learning rate
    sigma_t_sq = np.full(n_trials, np.nan)  # estimation uncertainty
    hit = np.full(len(df_subj), np.nan)  # hit vs. miss
    delta = np.full(len(df_subj), np.nan)  # prediction error

    # Initialize variables related to simulations
    sim_b_t = np.full(n_trials, np.nan)  # simulated prediction
    sim_a_t = np.full(n_trials, np.nan)  # simulated update

    # Initialize task
    task_vars = TaskVars()
    task_vars.circular = True
    task_vars.safe = 3
    task = ChangePointTask(task_vars)

    # Initialize task variables
    df_subj["c_t"] = np.nan
    df_subj["mu_t_rad"] = np.nan
    df_subj["x_t_rad"] = np.nan
    df_subj["v_t"] = np.nan

    # Cycle over trials
    # -----------------
    for t in range(0, n_trials - 1):

        # Extract noise condition
        agent.sigma = df_subj["sigma"][t].copy()

        # For first trial of new block
        # Futuretodo: create function to re-initialize agent on new block, maybe shared across motor and sampling too
        if df_subj["new_block"][t]:

            # Reset task variables to ensure change point on the first trial of the new block
            task.new_block = 1

            # Initialize estimation uncertainty, relative uncertainty, and change-point probability
            agent.sigma_t_sq = agent_vars.sigma_0
            agent.tau_t = agent_vars.tau_0
            agent.omega_t = agent_vars.omega_0

            # Record estimation uncertainty
            sigma_t_sq[t] = agent_vars.sigma_0

            # Set initial prediction
            sim_b_t[t] = agent_vars.mu_0

        # Record relative uncertainty of current trial
        tau[t] = agent.tau_t

        # Record estimation uncertainty of current trial
        sigma_t_sq[t] = agent.sigma_t_sq

        # For all but last trials of a block:
        if not df_subj["new_block"][t + 1]:

            # No reward manipulation here
            high_val = 0

            # Generate task outcomes for the current trial
            task.kappa = df_subj["kappa_t"][t].copy()
            task.sample_cp()
            task.sample_mu()
            task.sample_outcome()
            task.sample_catch_trial()
            task.new_block = 0

            # Save task variables
            df_subj.loc[t, "c_t"] = task.cp
            df_subj.loc[t, "x_t_rad"] = task.x_t
            df_subj.loc[t, "v_t"] = task.catch_trial
            df_subj.loc[t, "mu_t_rad"] = task.mu

            # Compute prediction error
            delta[t] = circ_dist(task.x_t, sim_b_t[t])

            # Run agent
            agent.learn(
                float(delta[t]),
                float(sim_b_t[t]),
                df_subj["v_t"][t],
                task.mu,
                high_val,
            )

            # Record updated belief
            mu[t] = agent.mu_t

            # Record predicted update
            a_hat[t] = agent.a_t

            # Record change-point probability
            omega[t] = agent.omega_t

            # Record learning rate
            alpha[t] = agent.alpha_t

            # Compute absolute predicted update
            # |hat{a}_t|
            abs_pred_up = abs(a_hat[t])

            # Compute response noise
            concentration[t] = residual_fun(
                abs_pred_up, sel_coeffs["omikron_0"], sel_coeffs["omikron_1"]
            )

            # Sample update from von Mises distribution
            sim_a_t[t] = np.random.vonmises(a_hat[t], concentration[t])

            # Updated prediction
            sim_b_t[t + 1] = (sim_b_t[t] + sim_a_t[t]) % agent.max_x

            # Record hit vs. miss
            if abs(delta[t]) <= df_subj["angular_shield_size"][t] / 2:
                hit[t] = 1
            else:
                hit[t] = 0

    # Attach model variables to data frame
    df_data = pd.DataFrame(index=range(0, n_trials), dtype="float")
    df_data["a_t_rad_hat"] = a_hat
    df_data["mu_t_rad"] = mu
    df_data["omega_t"] = omega
    df_data["tau_t"] = tau
    df_data["alpha_t"] = alpha
    df_data["sigma_t_sq"] = sigma_t_sq

    # Save simulation-related variables
    df_data["sim_b_t_rad"] = sim_b_t
    df_data["sim_a_t_rad"] = sim_a_t
    df_data["delta_t_rad"] = delta
    df_data["sigma"] = df_subj["sigma"].copy()
    df_data["kappa"] = df_subj["kappa_t"].copy()
    df_data["new_block"] = df_subj["new_block"].copy()
    df_data["x_t_rad"] = df_subj["x_t_rad"].copy()
    df_data["v_t"] = df_subj["v_t"].copy()
    df_data["c_t"] = df_subj["c_t"].copy()
    df_data["task_mu"] = df_subj["mu_t_rad"].copy()
    df_data["hit"] = hit

    return df_data
