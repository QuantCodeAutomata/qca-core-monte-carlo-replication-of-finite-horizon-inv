"""
Tests for src/simulator.py

Validates:
  - Single path simulation correctness
  - P&L computation: Pi_T = X_T + q_T * S_T
  - Reproducibility with fixed seeds
  - Monte Carlo run structure
  - Diagnostic counters
  - Common random numbers (CRN) for sensitivity
"""

import numpy as np
import pytest

from src.config import SimParams
from src.simulator import simulate_path, run_monte_carlo, run_monte_carlo_common_rng, PathResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_params():
    return SimParams(
        S0=100.0, T=1.0, sigma=2.0, dt=0.005,
        q0=0, X0=0.0, A=140.0, k=1.5,
        gamma=0.1, N_paths=100, spread_mode="text_formula",
        seed=42, strict_lambda=False,
    )


@pytest.fixture
def small_params():
    """Small simulation for fast tests."""
    return SimParams(
        S0=100.0, T=0.1, sigma=2.0, dt=0.01,
        q0=0, X0=0.0, A=140.0, k=1.5,
        gamma=0.1, N_paths=10, spread_mode="text_formula",
        seed=123, strict_lambda=False,
    )


# ---------------------------------------------------------------------------
# Tests: SimParams validation
# ---------------------------------------------------------------------------

class TestSimParams:
    def test_n_steps_correct(self, base_params):
        """n_steps = T / dt = 200."""
        assert base_params.n_steps == 200

    def test_invalid_T(self):
        with pytest.raises(AssertionError):
            SimParams(T=-1.0)

    def test_invalid_sigma(self):
        with pytest.raises(AssertionError):
            SimParams(sigma=-1.0)

    def test_invalid_gamma(self):
        with pytest.raises(AssertionError):
            SimParams(gamma=-0.1)

    def test_invalid_dt_not_divisible(self):
        with pytest.raises(AssertionError):
            SimParams(T=1.0, dt=0.003)  # 1.0 / 0.003 is not integer


# ---------------------------------------------------------------------------
# Tests: simulate_path
# ---------------------------------------------------------------------------

class TestSimulatePath:
    def test_pnl_equals_cash_plus_inventory_times_price(self, base_params):
        """Pi_T = X_T + q_T * S_T exactly."""
        rng = np.random.default_rng(42)
        result, _ = simulate_path(base_params, "inventory", rng)
        expected_pnl = result.terminal_cash + result.terminal_inventory * result.terminal_price
        assert abs(result.terminal_pnl - expected_pnl) < 1e-10

    def test_pnl_finite(self, base_params):
        """Terminal P&L is finite."""
        rng = np.random.default_rng(42)
        result, _ = simulate_path(base_params, "inventory", rng)
        assert np.isfinite(result.terminal_pnl)

    def test_inventory_integer_valued(self, base_params):
        """Final inventory is integer-valued (unit fills)."""
        rng = np.random.default_rng(42)
        result, _ = simulate_path(base_params, "inventory", rng)
        assert result.terminal_inventory == int(result.terminal_inventory)

    def test_symmetric_strategy_runs(self, base_params):
        """Symmetric strategy completes without error."""
        rng = np.random.default_rng(42)
        result, _ = simulate_path(base_params, "symmetric", rng)
        assert np.isfinite(result.terminal_pnl)

    def test_invalid_strategy_raises(self, base_params):
        """Invalid strategy name raises ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            simulate_path(base_params, "unknown_strategy", rng)

    def test_full_path_stored_when_requested(self, base_params):
        """Full path is stored when store_full_path=True."""
        rng = np.random.default_rng(42)
        result, full = simulate_path(base_params, "inventory", rng, store_full_path=True)
        assert full is not None
        assert len(full.times) == base_params.n_steps + 1
        assert len(full.prices) == base_params.n_steps + 1

    def test_full_path_not_stored_by_default(self, base_params):
        """Full path is None when store_full_path=False."""
        rng = np.random.default_rng(42)
        result, full = simulate_path(base_params, "inventory", rng, store_full_path=False)
        assert full is None

    def test_full_path_initial_price(self, base_params):
        """Full path starts at S0."""
        rng = np.random.default_rng(42)
        _, full = simulate_path(base_params, "inventory", rng, store_full_path=True)
        assert abs(full.prices[0] - base_params.S0) < 1e-12

    def test_full_path_initial_inventory(self, base_params):
        """Full path starts at q0."""
        rng = np.random.default_rng(42)
        _, full = simulate_path(base_params, "inventory", rng, store_full_path=True)
        assert abs(full.inventories[0] - base_params.q0) < 1e-12

    def test_price_increments_are_pm_sigma_sqrt_dt(self, base_params):
        """Price increments are exactly ±sigma*sqrt(dt)."""
        rng = np.random.default_rng(42)
        _, full = simulate_path(base_params, "inventory", rng, store_full_path=True)
        increments = np.diff(full.prices)
        expected_step = base_params.sigma * np.sqrt(base_params.dt)
        # Each increment should be exactly +expected_step or -expected_step
        assert np.all(np.abs(np.abs(increments) - expected_step) < 1e-10)

    def test_reproducibility_same_seed(self, base_params):
        """Same seed produces identical results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1, _ = simulate_path(base_params, "inventory", rng1)
        r2, _ = simulate_path(base_params, "inventory", rng2)
        assert abs(r1.terminal_pnl - r2.terminal_pnl) < 1e-12

    def test_different_seeds_different_results(self, base_params):
        """Different seeds produce different results (with high probability)."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(999)
        r1, _ = simulate_path(base_params, "inventory", rng1)
        r2, _ = simulate_path(base_params, "inventory", rng2)
        # Very unlikely to be identical
        assert abs(r1.terminal_pnl - r2.terminal_pnl) > 1e-6

    def test_table_matching_mode(self, base_params):
        """Table-matching spread mode runs without error."""
        params = SimParams(
            S0=100.0, T=1.0, sigma=2.0, dt=0.005,
            gamma=0.1, N_paths=10, spread_mode="table_matching",
            seed=42,
        )
        rng = np.random.default_rng(42)
        result, _ = simulate_path(params, "inventory", rng)
        assert np.isfinite(result.terminal_pnl)

    def test_diagnostic_counters_nonnegative(self, base_params):
        """All diagnostic counters are non-negative."""
        rng = np.random.default_rng(42)
        result, _ = simulate_path(base_params, "inventory", rng)
        assert result.n_lambda_violations >= 0
        assert result.n_negative_delta_a >= 0
        assert result.n_negative_delta_b >= 0
        assert result.n_simultaneous_fills >= 0


# ---------------------------------------------------------------------------
# Tests: run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def test_returns_correct_number_of_paths(self, base_params):
        """Returns exactly N_paths results."""
        results, _ = run_monte_carlo(base_params, "inventory", save_path_index=None)
        assert len(results) == base_params.N_paths

    def test_all_pnl_finite(self, base_params):
        """All terminal P&L values are finite."""
        results, _ = run_monte_carlo(base_params, "inventory", save_path_index=None)
        pnl = [r.terminal_pnl for r in results]
        assert all(np.isfinite(p) for p in pnl)

    def test_saved_path_returned(self, base_params):
        """Saved path is returned when save_path_index is set."""
        results, saved = run_monte_carlo(base_params, "inventory", save_path_index=0)
        assert saved is not None

    def test_no_saved_path_when_none(self, base_params):
        """No saved path when save_path_index=None."""
        results, saved = run_monte_carlo(base_params, "inventory", save_path_index=None)
        assert saved is None

    def test_symmetric_strategy(self, base_params):
        """Symmetric strategy Monte Carlo runs correctly."""
        results, _ = run_monte_carlo(base_params, "symmetric", save_path_index=None)
        assert len(results) == base_params.N_paths

    def test_pnl_formula_all_paths(self, small_params):
        """Pi_T = X_T + q_T * S_T for all paths."""
        results, _ = run_monte_carlo(small_params, "inventory", save_path_index=None)
        for r in results:
            expected = r.terminal_cash + r.terminal_inventory * r.terminal_price
            assert abs(r.terminal_pnl - expected) < 1e-10


# ---------------------------------------------------------------------------
# Tests: run_monte_carlo_common_rng
# ---------------------------------------------------------------------------

class TestCommonRandomNumbers:
    def test_crn_same_seed_different_modes(self, base_params):
        """CRN produces different results for different spread modes."""
        params_text = SimParams(
            gamma=0.1, N_paths=50, spread_mode="text_formula", seed=42
        )
        params_table = SimParams(
            gamma=0.1, N_paths=50, spread_mode="table_matching", seed=42
        )
        r1, r2 = run_monte_carlo_common_rng(params_text, params_table, "inventory")
        pnl1 = np.array([r.terminal_pnl for r in r1])
        pnl2 = np.array([r.terminal_pnl for r in r2])
        # Results should differ due to different spreads
        assert not np.allclose(pnl1, pnl2)

    def test_crn_seed_mismatch_raises(self):
        """Mismatched seeds raise AssertionError."""
        params1 = SimParams(gamma=0.1, N_paths=10, spread_mode="text_formula", seed=42)
        params2 = SimParams(gamma=0.1, N_paths=10, spread_mode="table_matching", seed=99)
        with pytest.raises(AssertionError):
            run_monte_carlo_common_rng(params1, params2, "inventory")

    def test_crn_returns_correct_lengths(self, base_params):
        """CRN returns correct number of paths for both modes."""
        params_text = SimParams(gamma=0.1, N_paths=50, spread_mode="text_formula", seed=42)
        params_table = SimParams(gamma=0.1, N_paths=50, spread_mode="table_matching", seed=42)
        r1, r2 = run_monte_carlo_common_rng(params_text, params_table, "inventory")
        assert len(r1) == 50
        assert len(r2) == 50


# ---------------------------------------------------------------------------
# Tests: qualitative properties
# ---------------------------------------------------------------------------

class TestQualitativeProperties:
    def test_inventory_lower_std_than_symmetric(self):
        """Inventory strategy has lower P&L std than symmetric (gamma=0.5)."""
        params_inv = SimParams(gamma=0.5, N_paths=500, spread_mode="text_formula", seed=42)
        params_sym = SimParams(gamma=0.5, N_paths=500, spread_mode="text_formula", seed=42)

        results_inv, _ = run_monte_carlo(params_inv, "inventory", save_path_index=None)
        results_sym, _ = run_monte_carlo(params_sym, "symmetric", save_path_index=None)

        pnl_inv = np.array([r.terminal_pnl for r in results_inv])
        pnl_sym = np.array([r.terminal_pnl for r in results_sym])

        assert np.std(pnl_inv, ddof=1) < np.std(pnl_sym, ddof=1)

    def test_inventory_lower_inv_std_than_symmetric(self):
        """Inventory strategy has lower inventory std than symmetric (gamma=0.5)."""
        params_inv = SimParams(gamma=0.5, N_paths=500, spread_mode="text_formula", seed=42)
        params_sym = SimParams(gamma=0.5, N_paths=500, spread_mode="text_formula", seed=42)

        results_inv, _ = run_monte_carlo(params_inv, "inventory", save_path_index=None)
        results_sym, _ = run_monte_carlo(params_sym, "symmetric", save_path_index=None)

        q_inv = np.array([r.terminal_inventory for r in results_inv])
        q_sym = np.array([r.terminal_inventory for r in results_sym])

        assert np.std(q_inv, ddof=1) < np.std(q_sym, ddof=1)
