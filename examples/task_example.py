import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from allinpy import cm2inch, latex_plt

from rbmpy import ChangePointTask, TaskVars

# Update matplotlib to use Latex and to change some defaults
matplotlib = latex_plt(matplotlib)

# Initialize task based on task_vars
task = ChangePointTask(TaskVars())
task.circular = True
task.variable_shield = True
task.safe = 3
task.h = 0.1
task.catch_trial_prob = 0.1

# Initialize task variable
cp_list = list()
mu_list = list()
outcome_list = list()
shield_list = list()
catch_trial_list = list()

# Cycle over trials
for t in range(100):

    task.sample_cp()
    cp_list.append(task.cp)

    task.sample_mu()
    mu_list.append(task.mu)

    task.sample_outcome()
    outcome_list.append(task.x_t)

    task.sample_shield()
    shield_list.append(task.shield_size)

    task.sample_catch_trial()
    catch_trial_list.append(task.catch_trial)

    task.new_block = False

# Plot results
# ------------

plt.figure(figsize=cm2inch(12, 10))
ax_0 = plt.subplot(4, 1, 1)
ax_0.plot(cp_list, label="cp")
ax_0.set_ylabel("Change Point")

ax_1 = plt.subplot(4, 1, 2)
ax_1.plot(mu_list, color="blue", label="mu")
ax_1.plot(outcome_list, ".", color="k", label="outcome")
ax_1.set_ylabel("Location")
ax_1.legend(loc="upper right")

ax_2 = plt.subplot(4, 1, 3)
ax_2.plot(shield_list, label="shield")
ax_2.set_ylabel("Shield Size")

ax_3 = plt.subplot(4, 1, 4)
ax_3.plot(catch_trial_list, label="catch trial")
ax_3.set_ylabel("Catch Trial")
ax_3.set_xlabel("Trial")

sns.despine()
plt.tight_layout()
plt.show()
