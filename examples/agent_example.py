import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from allinpy import cm2inch, latex_plt
from task_agent_int_example import task_agent_int

from rbmpy import AgentVars, AlAgent

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Turn on interactive mode
plt.ion()

# Create subject data frame for simulation
n_trials = 50
df_subj = pd.DataFrame(
    index=range(n_trials),
    columns=[
        "subj_num",
        "c_t",
        "mu_t_rad",
        "x_t_rad",
        "v_t",
        "new_block",
        "sigma",
        "angular_shield_size",
    ],
)
df_subj["subj_num"] = 1
df_subj["c_t"] = np.nan
df_subj["mu_t_rad"] = np.nan
df_subj["x_t_rad"] = np.nan
df_subj["v_t"] = np.nan
df_subj["new_block"] = 0
df_subj.loc[0, "new_block"] = 1
df_subj["kappa_t"] = 8
df_subj["sigma"] = np.sqrt(1 / df_subj["kappa_t"])
df_subj["angular_shield_size"] = 2 * df_subj["sigma"]

# Set coefficients for agent
sel_coeffs = {
    "omikron_0": 1,
    "omikron_1": 0.1,
    "h": 0.1,
    "s": 1,
    "u": 0,
    "sigma_H": 0.001,
    "subj_num": 1,
}

# Create agent variables
agent_vars = AgentVars()
agent_vars.circular = True
agent_vars.max_x = 2 * np.pi

# Set agent variables
agent_vars.h = sel_coeffs["h"]
agent_vars.s = sel_coeffs["s"]
agent_vars.u = np.exp(sel_coeffs["u"])
agent_vars.sigma_H = sel_coeffs["sigma_H"]
agent_vars.tau_0 = 1
agent_vars.sigma_0 = 10

# Initialize agent based on agent_vars
agent = AlAgent(agent_vars)

# Run simulation
df_sim = task_agent_int(df_subj, agent, agent_vars, sel_coeffs)

# Plot results
# ------------

plt.figure(figsize=cm2inch(12, 10))
ax_0 = plt.subplot(4, 1, 1)
ax_0.plot(df_sim["task_mu"], "--", color="k", label="Mean")
ax_0.plot(df_sim["x_t_rad"], ".", color="k", label="Outcome")
ax_0.plot(df_sim["mu_t_rad"], "-", color="blue", label="Model")
ax_0.set_ylabel("Screen Location")
ax_0.legend(loc="upper right")

ax_1 = plt.subplot(4, 1, 2)
ax_1.plot(df_sim["omega_t"], label="CPP")
ax_1.plot(df_sim["tau_t"], label="RU")
ax_1.plot(df_sim["alpha_t"], label="Alpha")
ax_1.set_ylabel("Value")
ax_1.legend(loc="upper right")

ax_2 = plt.subplot(4, 1, 3)
ax_2.plot(df_sim["v_t"])
ax_2.set_xlabel("Trial")
ax_2.set_ylabel("Catch Trial")

ax_2 = plt.subplot(4, 1, 4)
ax_2.plot(df_sim["hit"])
ax_2.set_xlabel("Trial")
ax_2.set_ylabel("Hit")

sns.despine()
plt.tight_layout()

plt.ioff()
plt.show()
