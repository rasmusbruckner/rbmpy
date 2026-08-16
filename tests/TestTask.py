"""Test Task: Unit tests for the task object."""

import numpy as np
import pytest

from rbmpy import ChangePointTask, TaskVars


def test_task_init():
    """Tests the task initialization based on TaskVars."""

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())

    assert task.kappa == 16
    assert task.sigma == np.sqrt(1 / task.kappa)
    assert task.h == 0.125
    assert task.min_x == 0
    assert task.max_x == 2 * np.pi
    assert task.min_mu == 0
    assert task.max_mu == 2 * np.pi
    assert task.new_block == True
    assert task.variable_shield == True
    assert task.shield_min == np.deg2rad(10)
    assert task.shield_max == np.pi
    assert task.shield_mu == np.deg2rad(10)
    assert task.safe == 3
    assert task.s == 3
    assert task.circular == True
    assert task.catch_trial_prob == 0.1
    assert np.isnan(task.x_t)
    assert np.isnan(task.mu)
    assert np.isnan(task.cp)
    assert np.isnan(task.shield_size)


def test_sample_cp_1(monkeypatch):
    """Tests the sample_cp function.

    Case cp = 1 and safe = 0
    """

    # Mock np.random.binomial
    def mock_binomial(n, p):
        return 1

    monkeypatch.setattr("numpy.random.binomial", mock_binomial)

    # Initialize task
    task = ChangePointTask(TaskVars())
    task.new_block = 0
    task.s = 0

    # Sample cp
    task.sample_cp()
    assert task.cp == 1
    assert task.s == 3


def test_sample_cp_0(monkeypatch):
    """Tests the sample_cp function.

    Case: cp = 0 and safe = 0
    """

    # Mock np.random.binomial
    def mock_binomial(n, p):
        return 0

    monkeypatch.setattr("numpy.random.binomial", mock_binomial)

    # Initialize task
    task = ChangePointTask(TaskVars())
    task.new_block = 0
    task.s = 0

    # Sample cp
    task.sample_cp()
    assert task.cp == 0
    assert task.s == 0


def test_sample_cp_w_safe(monkeypatch):
    """Tests the sample_cp function.

    Case cp = 1 and safe = 3
    """

    # Mock np.random.binomial
    def mock_binomial(n, p):
        return 1

    monkeypatch.setattr("numpy.random.binomial", mock_binomial)

    # Initialize task
    task = ChangePointTask(TaskVars())
    task.new_block = 0
    task.s = 3

    # Sample cp
    task.sample_cp()
    assert task.cp == 0
    assert task.s == 2


def test_sample_cp_new_block(monkeypatch):
    """Tests the sample_cp function.

    Case cp = 1 because of a new block.
    """

    # Mock np.random.binomial
    def mock_binomial(n, p):
        return 1

    monkeypatch.setattr("numpy.random.binomial", mock_binomial)

    # Initialize task
    task = ChangePointTask(TaskVars())
    task.new_block = 1
    task.s = 3

    # Sample cp
    task.sample_cp()
    assert task.cp == 1
    assert task.s == 3


def test_sample_mu_cp():
    """Tests the sample_mu function.

    We test whether on a change point, the sampled mu differs from the prior mu.
    """

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())

    # Ensure we have a change point
    task.cp = 1

    # Store prior mu
    prior_mu = task.mu

    task.sample_mu()
    assert task.mu != prior_mu


def test_sample_mu_nocp():
    """Tests the sample_mu function.

    We test whether on a non-change point, the sampled mu equals the prior mu.
    """

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())

    # Ensure we have a change point
    task.cp = 0

    # Set test mu
    task.mu = np.pi

    # Store prior mu
    prior_mu = task.mu

    task.sample_mu()
    assert task.mu == prior_mu == np.pi


def test_sample_outcome_linear(monkeypatch):
    """Tests the sample_outcome function for the linear case.

    The sampled outcome is within the range of the task.
    """

    # Mock np.random.normal
    def mock_normal(n, p):
        return 120

    monkeypatch.setattr("numpy.random.normal", mock_normal)

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())
    task.circular = False
    task.max_x = 300

    task.sample_outcome()
    assert task.x_t == 120


def test_sample_outcome_linear_max_x(monkeypatch):
    """Tests the sample_outcome function for the linear case.

    The sampled outcome is outside the upper range of the task.
    """

    # Mock np.random.normal
    def mock_normal(n, p):
        return 310

    monkeypatch.setattr("numpy.random.normal", mock_normal)

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())
    task.circular = False
    task.max_x = 300

    task.sample_outcome()
    assert task.x_t == 300


def test_sample_outcome_linear_min_x(monkeypatch):
    """Tests the sample_outcome function for the linear case.

    The sampled outcome is outside the lower range of the task.
    """

    # Mock np.random.normal
    def mock_normal(n, p):
        return -10

    monkeypatch.setattr("numpy.random.normal", mock_normal)

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())
    task.circular = False

    task.sample_outcome()
    assert task.x_t == 0


def test_sample_outcome_circular(monkeypatch):
    """Tests the sample_outcome function for the circular case.

    We test the case where the modulo is not necessary because outcome is in range [0, 2pi).
    """

    # Mock np.random.vonmises
    def mock_vonmises(n, p):
        return np.pi / 2

    monkeypatch.setattr("numpy.random.vonmises", mock_vonmises)

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())
    task.circular = True

    task.sample_outcome()
    assert task.x_t == np.pi / 2


def test_sample_outcome_circular_mod(monkeypatch):
    """Tests the sample_outcome function for the circular case.

    We test whether the function correctly applies the modulus of the outcome to ensure outcomes are in rage [0, 2pi).
    """

    # Mock np.random.vonmises
    def mock_vonmises(n, p):
        return -np.pi / 2

    monkeypatch.setattr("numpy.random.vonmises", mock_vonmises)

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())
    task.circular = True

    task.sample_outcome()
    assert task.x_t == np.pi + np.pi / 2


def test_sample_shield_fixed():
    """Tests the sample_shield function when variable_shield is False."""

    task = ChangePointTask(TaskVars())
    task.variable_shield = False
    task.shield_mu = 10
    task.sample_shield()
    assert task.shield_size == 10


def test_sample_shield_variable_retry(monkeypatch):
    """Tests the sample_shield function when first sample is out of bounds."""

    # Create an iterator that returns values in sequence
    values = iter([5, 50])  # first value too low, second value valid
    monkeypatch.setattr("numpy.random.exponential", lambda mu: next(values))

    task = ChangePointTask(TaskVars())
    task.variable_shield = True
    task.shield_min = 10
    task.shield_max = 180
    task.shield_mu = 10

    task.sample_shield()
    assert task.shield_size == 50


def test_sample_catch_trial_no_cp(monkeypatch):
    """Tests the sample_catch_trial function when cp = 0 as catch trial is possible."""

    # Mock np.random.binomial
    def mock_binomial(n, p):
        return 1

    monkeypatch.setattr("numpy.random.binomial", mock_binomial)

    task = ChangePointTask(TaskVars())
    task.cp = 0
    task.sample_catch_trial()

    assert task.catch_trial == 1


def test_sample_catch_trial_cp(monkeypatch):
    """Tests the sample_catch_trial function when cp = 1 as catch trial is not possible."""

    # Mock np.random.binomial
    def mock_binomial(n, p):
        return 1

    monkeypatch.setattr("numpy.random.binomial", mock_binomial)

    task = ChangePointTask(TaskVars())
    task.cp = 1
    task.sample_catch_trial()

    assert task.catch_trial == 0


def test_integration():
    """Integration test for the task with basic statistical checks."""

    # Fix random seed
    np.random.seed(123)

    # Initialize task based on task_vars
    task = ChangePointTask(TaskVars())
    task.circular = True
    task.variable_shield = True
    task.safe = 0
    task.h = 0.1
    task.catch_trial_prob = 0.1

    cp_list = list()
    mu_list = list()
    outcome_list = list()
    shield_list = list()
    catch_trial_list = list()
    for t in range(100000):

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

    assert np.mean(cp_list) == pytest.approx(0.1, abs=0.05)
    assert np.mean(mu_list) == pytest.approx(np.pi, abs=0.05)
    assert np.min(mu_list) >= 0
    assert np.max(mu_list) <= 2 * np.pi

    assert np.mean(outcome_list) == pytest.approx(np.pi, abs=0.05)
    assert np.min(outcome_list) >= 0
    assert np.max(outcome_list) <= 2 * np.pi

    assert np.min(shield_list) >= np.deg2rad(10)
    assert np.max(shield_list) <= np.pi

    assert np.mean(catch_trial_list) == pytest.approx(0.09, abs=0.05)
