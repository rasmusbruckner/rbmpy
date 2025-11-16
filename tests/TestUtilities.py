import numpy as np
import pytest

from rbm_analyses.utilities import (compute_bic, compute_persprob,
                                    get_sel_coeffs, normalize_angle,
                                    residual_fun)


def test_residual_fun():
    """Tests the residual function of the RBM."""

    kappa_up = residual_fun(np.array(np.pi), 10, 0.5)
    assert kappa_up == pytest.approx(0.3282806350011744)

    with pytest.raises(ValueError):
        residual_fun(np.array(-0.5), 10, 0.5)


def test_compute_persprob():
    """Tests the perseveration probability function of the RBM."""

    # Test parameters that bring down perseveration to zero
    persprob = compute_persprob(-30, -1.5, 1)
    assert persprob < 1.0e-10

    # Test parameters that lead to 0.5
    persprob = compute_persprob(0, 0, 10)
    assert persprob == pytest.approx(0.5)

    # Test parameters that lead to low perseveration probability
    persprob = compute_persprob(0, -0.1, 10)
    assert persprob == pytest.approx(0.2689414213699951, rel=1e-6)

    persprob = compute_persprob(20, -0.1, 10)
    assert persprob == pytest.approx(0.7310585786300049, rel=1e-6)


def test_get_sel_coeffs_all_free():
    """Tests the selection of coefficients function when all coefficients are free."""

    items = {"a": True, "b": True}.items()
    fixed = {"a": 0, "b": 0}
    coeffs = np.array([1.1, 2.2])
    result = get_sel_coeffs(items, fixed, coeffs)
    assert result == {"a": 1.1, "b": 2.2}


def test_get_sel_coeffs_all_fixed():
    """Tests the selection of coefficients function when all coefficients are fixed."""

    items = {"a": False, "b": False}.items()
    fixed = {"a": 0.5, "b": -0.5}
    coeffs = np.array([])
    result = get_sel_coeffs(items, fixed, coeffs)
    assert result == {"a": 0.5, "b": -0.5}


def test_get_sel_coeffs_mixed():
    """Tests the selection of coefficients function when some coefficients are fixed."""

    items = {"a": True, "b": False, "c": True}.items()
    fixed = {"a": 0.0, "b": 99.0, "c": 0.0}
    coeffs = np.array([1.0, 2.0])
    result = get_sel_coeffs(items, fixed, coeffs)
    assert result == {"a": 1.0, "b": 99.0, "c": 2.0}


def test_get_sel_coeffs_overflow():
    """Tests the selection of coefficients function when the number of coefficients is too low."""

    items = {"a": True}.items()
    fixed = {"a": 0.0}
    coeffs = np.array([])  # missing coeff
    with pytest.raises(IndexError):
        get_sel_coeffs(items, fixed, coeffs)


def test_bic():
    """Tests the computation of the Bayesian information criterion."""

    llh = -120
    n_params = 9
    n_trials = 398
    bic = compute_bic(llh, n_params, n_trials)
    assert bic == 93.06096597622003


def test_normalize_angle_pos_outside():
    """Tests the normalization of circular angles for positive overshoot."""

    angle_deg = np.array([200])
    corr_angle = normalize_angle(np.deg2rad(angle_deg))
    assert corr_angle == pytest.approx(np.deg2rad(-160))


def test_normalize_angle_neg_outside():
    """Tests the normalization of circular angles for negative overshoot."""

    angle_deg = np.array([-200])
    corr_angle = normalize_angle(np.deg2rad(angle_deg))
    assert corr_angle == pytest.approx(np.deg2rad(160))


def test_normalize_angle_no_correction():
    """Tests the normalization of circular angles for no correction."""

    angle_deg = np.array([100])
    corr_angle = normalize_angle(np.deg2rad(angle_deg))
    assert corr_angle == pytest.approx(np.deg2rad(100))


def test_normalize_angle_boundary():
    """Tests the normalization of circular angles for boundary correction."""

    angle_deg = np.array([180])
    corr_angle = normalize_angle(np.deg2rad(angle_deg))
    assert corr_angle == pytest.approx(np.deg2rad(-180))
