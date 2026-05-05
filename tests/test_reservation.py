"""
Tests for src/reservation.py

Validates:
  - Finite-horizon reservation bid/ask/mid formulas
  - Boundary conditions
  - Utility indifference conditions
  - Infinite-horizon formulas and admissibility
  - Edge cases
"""

import numpy as np
import pytest
import warnings

from src.reservation import (
    inactive_utility,
    reservation_bid_finite,
    reservation_ask_finite,
    reservation_mid_finite,
    reservation_bid_infinite,
    reservation_ask_infinite,
    admissibility_omega,
    check_admissibility,
    verify_indifference_bid,
    verify_indifference_ask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_params():
    return {"s": 100.0, "gamma": 0.1, "sigma": 2.0, "T": 1.0}


# ---------------------------------------------------------------------------
# Tests: inactive_utility
# ---------------------------------------------------------------------------

class TestInactiveUtility:
    def test_utility_is_negative(self, base_params):
        """Exponential utility is always negative."""
        u = inactive_utility(0.0, 100.0, 0.0, 0.5, **{k: v for k, v in base_params.items() if k != "s"})
        assert u < 0

    def test_utility_variance_term_increases_with_q(self, base_params):
        """Higher |q| -> larger variance term -> lower utility for fixed total wealth."""
        p = base_params
        # Fix total wealth x + q*s = 0 to isolate the variance effect
        # u = -exp(-gamma*(x+q*s)) * exp(gamma^2*q^2*sigma^2*(T-t)/2)
        # With x + q*s = 0: u = -1 * exp(gamma^2*q^2*sigma^2*(T-t)/2)
        # Higher |q| -> larger exp term -> more negative utility
        q0, q5 = 0.0, 5.0
        x0 = 0.0                    # x + 0*s = 0
        x5 = -q5 * p["s"]          # x + 5*s = 0 -> x = -500
        u0 = inactive_utility(x0, p["s"], q0, 0.5, p["gamma"], p["sigma"], p["T"])
        u5 = inactive_utility(x5, p["s"], q5, 0.5, p["gamma"], p["sigma"], p["T"])
        # More inventory -> more variance -> lower utility (more negative)
        assert u5 < u0

    def test_utility_at_terminal_time(self, base_params):
        """At t=T, utility simplifies to -exp(-gamma*(x + q*s))."""
        p = base_params
        x, q, t = 10.0, 3.0, p["T"]
        u = inactive_utility(x, p["s"], q, t, p["gamma"], p["sigma"], p["T"])
        expected = -np.exp(-p["gamma"] * (x + q * p["s"]))
        assert abs(u - expected) < 1e-12

    def test_utility_invalid_gamma(self):
        with pytest.raises(AssertionError):
            inactive_utility(0.0, 100.0, 0.0, 0.5, gamma=-0.1, sigma=2.0, T=1.0)

    def test_utility_invalid_sigma(self):
        with pytest.raises(AssertionError):
            inactive_utility(0.0, 100.0, 0.0, 0.5, gamma=0.1, sigma=-1.0, T=1.0)


# ---------------------------------------------------------------------------
# Tests: reservation_mid_finite
# ---------------------------------------------------------------------------

class TestReservationMidFinite:
    def test_zero_inventory_equals_mid_price(self, base_params):
        """r(s, 0, t) = s for all t."""
        p = base_params
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = reservation_mid_finite(p["s"], 0.0, t, p["gamma"], p["sigma"], p["T"])
            assert abs(r - p["s"]) < 1e-12, f"Failed at t={t}"

    def test_terminal_time_equals_mid_price(self, base_params):
        """r(s, q, T) = s for all q."""
        p = base_params
        for q in [-10, -5, 0, 5, 10]:
            r = reservation_mid_finite(p["s"], float(q), p["T"], p["gamma"], p["sigma"], p["T"])
            assert abs(r - p["s"]) < 1e-12, f"Failed at q={q}"

    def test_long_inventory_below_mid(self, base_params):
        """q > 0 -> r < s (dealer wants to sell)."""
        p = base_params
        r = reservation_mid_finite(p["s"], 5.0, 0.5, p["gamma"], p["sigma"], p["T"])
        assert r < p["s"]

    def test_short_inventory_above_mid(self, base_params):
        """q < 0 -> r > s (dealer wants to buy)."""
        p = base_params
        r = reservation_mid_finite(p["s"], -5.0, 0.5, p["gamma"], p["sigma"], p["T"])
        assert r > p["s"]

    def test_affine_in_inventory(self, base_params):
        """r is affine (linear) in q."""
        p = base_params
        q_grid = np.array([-5.0, 0.0, 5.0])
        r = reservation_mid_finite(p["s"], q_grid, 0.5, p["gamma"], p["sigma"], p["T"])
        # Check linearity: r[1] - r[0] == r[2] - r[1]
        assert abs((r[1] - r[0]) - (r[2] - r[1])) < 1e-12

    def test_formula_value(self, base_params):
        """Verify exact formula: r = s - q*gamma*sigma^2*(T-t)."""
        p = base_params
        q, t = 3.0, 0.5
        expected = p["s"] - q * p["gamma"] * p["sigma"]**2 * (p["T"] - t)
        r = reservation_mid_finite(p["s"], q, t, p["gamma"], p["sigma"], p["T"])
        assert abs(r - expected) < 1e-12

    def test_larger_gamma_larger_shift(self, base_params):
        """Larger gamma -> larger |r - s| for fixed q > 0."""
        p = base_params
        q, t = 5.0, 0.5
        r_small = reservation_mid_finite(p["s"], q, t, 0.01, p["sigma"], p["T"])
        r_large = reservation_mid_finite(p["s"], q, t, 0.5, p["sigma"], p["T"])
        assert abs(r_large - p["s"]) > abs(r_small - p["s"])

    def test_vectorized_q(self, base_params):
        """Function works with array inputs."""
        p = base_params
        q_arr = np.linspace(-10, 10, 21)
        r = reservation_mid_finite(p["s"], q_arr, 0.5, p["gamma"], p["sigma"], p["T"])
        assert r.shape == q_arr.shape


# ---------------------------------------------------------------------------
# Tests: reservation_bid_finite and reservation_ask_finite
# ---------------------------------------------------------------------------

class TestReservationBidAskFinite:
    def test_ask_above_bid(self, base_params):
        """Reservation ask > reservation bid for all q, t."""
        p = base_params
        for q in [-5, 0, 5]:
            for t in [0.0, 0.5, 0.9]:
                r_b = reservation_bid_finite(p["s"], float(q), t, p["gamma"], p["sigma"], p["T"])
                r_a = reservation_ask_finite(p["s"], float(q), t, p["gamma"], p["sigma"], p["T"])
                assert r_a > r_b, f"Ask <= Bid at q={q}, t={t}"

    def test_mid_is_average_of_bid_ask(self, base_params):
        """r_mid = (r_ask + r_bid) / 2."""
        p = base_params
        for q in [-3, 0, 3]:
            for t in [0.0, 0.5]:
                r_b = reservation_bid_finite(p["s"], float(q), t, p["gamma"], p["sigma"], p["T"])
                r_a = reservation_ask_finite(p["s"], float(q), t, p["gamma"], p["sigma"], p["T"])
                r_m = reservation_mid_finite(p["s"], float(q), t, p["gamma"], p["sigma"], p["T"])
                assert abs((r_a + r_b) / 2.0 - r_m) < 1e-12

    def test_bid_formula_value(self, base_params):
        """Verify exact bid formula: r^b = s + (-1 - 2q)*gamma*sigma^2*(T-t)/2."""
        p = base_params
        q, t = 2.0, 0.5
        expected = p["s"] + (-1.0 - 2.0 * q) * p["gamma"] * p["sigma"]**2 * (p["T"] - t) / 2.0
        r_b = reservation_bid_finite(p["s"], q, t, p["gamma"], p["sigma"], p["T"])
        assert abs(r_b - expected) < 1e-12

    def test_ask_formula_value(self, base_params):
        """Verify exact ask formula: r^a = s + (1 - 2q)*gamma*sigma^2*(T-t)/2."""
        p = base_params
        q, t = 2.0, 0.5
        expected = p["s"] + (1.0 - 2.0 * q) * p["gamma"] * p["sigma"]**2 * (p["T"] - t) / 2.0
        r_a = reservation_ask_finite(p["s"], q, t, p["gamma"], p["sigma"], p["T"])
        assert abs(r_a - expected) < 1e-12

    def test_spread_equals_gamma_sigma2_tau(self, base_params):
        """Reservation spread r^a - r^b = gamma*sigma^2*(T-t)."""
        p = base_params
        for q in [-5, 0, 5]:
            for t in [0.0, 0.5]:
                r_b = reservation_bid_finite(p["s"], float(q), t, p["gamma"], p["sigma"], p["T"])
                r_a = reservation_ask_finite(p["s"], float(q), t, p["gamma"], p["sigma"], p["T"])
                expected_spread = p["gamma"] * p["sigma"]**2 * (p["T"] - t)
                assert abs(r_a - r_b - expected_spread) < 1e-12


# ---------------------------------------------------------------------------
# Tests: utility indifference conditions
# ---------------------------------------------------------------------------

class TestIndifferenceConditions:
    def test_bid_indifference_q0(self, base_params):
        """Bid indifference holds at q=0."""
        p = base_params
        assert verify_indifference_bid(0.0, p["s"], 0.0, 0.5, p["gamma"], p["sigma"], p["T"])

    def test_ask_indifference_q0(self, base_params):
        """Ask indifference holds at q=0."""
        p = base_params
        assert verify_indifference_ask(0.0, p["s"], 0.0, 0.5, p["gamma"], p["sigma"], p["T"])

    def test_bid_indifference_various_q(self, base_params):
        """Bid indifference holds for various inventory levels."""
        p = base_params
        for q in [-5, -1, 0, 1, 5]:
            assert verify_indifference_bid(
                0.0, p["s"], float(q), 0.5, p["gamma"], p["sigma"], p["T"]
            ), f"Bid indifference failed at q={q}"

    def test_ask_indifference_various_q(self, base_params):
        """Ask indifference holds for various inventory levels."""
        p = base_params
        for q in [-5, -1, 0, 1, 5]:
            assert verify_indifference_ask(
                0.0, p["s"], float(q), 0.5, p["gamma"], p["sigma"], p["T"]
            ), f"Ask indifference failed at q={q}"

    def test_indifference_various_gamma(self):
        """Indifference holds for different gamma values."""
        for gamma in [0.01, 0.1, 0.5]:
            assert verify_indifference_bid(0.0, 100.0, 3.0, 0.5, gamma, 2.0, 1.0)
            assert verify_indifference_ask(0.0, 100.0, 3.0, 0.5, gamma, 2.0, 1.0)


# ---------------------------------------------------------------------------
# Tests: infinite-horizon reservation prices
# ---------------------------------------------------------------------------

class TestInfiniteHorizonReservation:
    def test_admissibility_omega_formula(self):
        """omega = 0.5*gamma^2*sigma^2*(q_max+1)^2."""
        gamma, sigma, q_max = 0.1, 2.0, 10
        omega = admissibility_omega(gamma, sigma, q_max)
        expected = 0.5 * gamma**2 * sigma**2 * (q_max + 1)**2
        assert abs(omega - expected) < 1e-12

    def test_admissibility_holds_within_q_max(self):
        """All q in [-q_max, q_max] satisfy admissibility."""
        gamma, sigma, q_max = 0.1, 2.0, 10
        omega = admissibility_omega(gamma, sigma, q_max)
        q_range = np.arange(-q_max, q_max + 1, dtype=float)
        admissible = check_admissibility(q_range, gamma, sigma, omega)
        assert np.all(admissible)

    def test_admissibility_fails_outside_q_max(self):
        """q = q_max + 1 violates admissibility."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        omega = admissibility_omega(gamma, sigma, q_max)
        q_outside = float(q_max + 1)
        admissible = check_admissibility(np.array([q_outside]), gamma, sigma, omega)
        assert not np.all(admissible)

    def test_infinite_ask_above_bid(self):
        """Infinite-horizon ask > bid for admissible q."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        omega = admissibility_omega(gamma, sigma, q_max)
        q_range = np.arange(-q_max, q_max + 1, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_b = reservation_bid_infinite(100.0, q_range, gamma, sigma, omega)
            r_a = reservation_ask_infinite(100.0, q_range, gamma, sigma, omega)
        valid = ~np.isnan(r_b) & ~np.isnan(r_a)
        assert np.all(r_a[valid] > r_b[valid])

    def test_infinite_q0_bid_below_s(self):
        """At q=0, infinite-horizon bid < s (dealer wants to buy at discount)."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        omega = admissibility_omega(gamma, sigma, q_max)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_b = reservation_bid_infinite(100.0, 0.0, gamma, sigma, omega)
        assert r_b < 100.0

    def test_infinite_q0_ask_above_s(self):
        """At q=0, infinite-horizon ask > s (dealer wants to sell at premium)."""
        gamma, sigma, q_max = 0.1, 2.0, 5
        omega = admissibility_omega(gamma, sigma, q_max)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_a = reservation_ask_infinite(100.0, 0.0, gamma, sigma, omega)
        assert r_a > 100.0

    def test_nan_for_inadmissible_q(self):
        """Returns NaN for q values violating admissibility."""
        gamma, sigma, q_max = 0.1, 2.0, 3
        omega = admissibility_omega(gamma, sigma, q_max)
        q_bad = float(q_max + 2)  # Clearly outside admissible range
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_b = reservation_bid_infinite(100.0, q_bad, gamma, sigma, omega)
        assert np.isnan(r_b)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_time_step(self):
        """Works with t very close to T."""
        r = reservation_mid_finite(100.0, 5.0, 0.999, 0.1, 2.0, 1.0)
        assert np.isfinite(r)

    def test_zero_gamma_limit(self):
        """Very small gamma -> reservation price approaches mid-price."""
        r = reservation_mid_finite(100.0, 10.0, 0.5, 1e-6, 2.0, 1.0)
        assert abs(r - 100.0) < 0.01

    def test_large_inventory(self):
        """Works with large inventory values."""
        r = reservation_mid_finite(100.0, 1000.0, 0.5, 0.1, 2.0, 1.0)
        assert np.isfinite(r)
        assert r < 100.0

    def test_vectorized_t(self):
        """Works with array of time values."""
        t_arr = np.linspace(0, 1, 11)
        r = reservation_mid_finite(100.0, 5.0, t_arr, 0.1, 2.0, 1.0)
        assert r.shape == t_arr.shape
        # r should be monotonically increasing toward s as t -> T
        assert np.all(np.diff(r) >= 0)

    def test_reservation_mid_is_average_of_bid_ask_vectorized(self):
        """r_mid = (r_ask + r_bid) / 2 holds for arrays."""
        q_arr = np.linspace(-10, 10, 21)
        r_b = reservation_bid_finite(100.0, q_arr, 0.5, 0.1, 2.0, 1.0)
        r_a = reservation_ask_finite(100.0, q_arr, 0.5, 0.1, 2.0, 1.0)
        r_m = reservation_mid_finite(100.0, q_arr, 0.5, 0.1, 2.0, 1.0)
        assert np.allclose((r_a + r_b) / 2.0, r_m, atol=1e-12)
