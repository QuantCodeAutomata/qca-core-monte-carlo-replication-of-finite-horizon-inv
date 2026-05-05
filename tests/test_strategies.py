"""
Tests for src/strategies.py

Validates:
  - Spread computation (text_formula and table_matching modes)
  - Inventory-based quoting rule
  - Symmetric benchmark quoting rule
  - Fill intensity function
  - Published spread values
"""

import numpy as np
import pytest

from src.strategies import (
    compute_spread,
    compute_initial_spread,
    inventory_quotes,
    symmetric_quotes,
    fill_intensity,
)
from src.config import PUBLISHED_SPREADS


# ---------------------------------------------------------------------------
# Tests: compute_spread
# ---------------------------------------------------------------------------

class TestComputeSpread:
    def test_text_formula_at_t0(self):
        """Text formula at t=0: Phi = gamma*sigma^2*T + (2/gamma)*ln(1+gamma/k)."""
        gamma, sigma, k, T = 0.1, 2.0, 1.5, 1.0
        phi = compute_spread(0.0, gamma, sigma, k, T, mode="text_formula")
        expected = gamma * sigma**2 * T + (2.0 / gamma) * np.log(1.0 + gamma / k)
        assert abs(phi - expected) < 1e-12

    def test_table_matching_constant(self):
        """Table-matching spread is constant in time."""
        gamma, sigma, k, T = 0.1, 2.0, 1.5, 1.0
        phi_t0 = compute_spread(0.0, gamma, sigma, k, T, mode="table_matching")
        phi_t05 = compute_spread(0.5, gamma, sigma, k, T, mode="table_matching")
        phi_t1 = compute_spread(1.0, gamma, sigma, k, T, mode="table_matching")
        assert abs(phi_t0 - phi_t05) < 1e-12
        assert abs(phi_t0 - phi_t1) < 1e-12

    def test_table_matching_formula(self):
        """Table-matching: Phi = (2/gamma)*ln(1+gamma/k)."""
        gamma, sigma, k, T = 0.1, 2.0, 1.5, 1.0
        phi = compute_spread(0.5, gamma, sigma, k, T, mode="table_matching")
        expected = (2.0 / gamma) * np.log(1.0 + gamma / k)
        assert abs(phi - expected) < 1e-12

    def test_text_formula_decreasing_in_t(self):
        """Text-formula spread decreases as t increases (tau decreases)."""
        gamma, sigma, k, T = 0.1, 2.0, 1.5, 1.0
        phi_early = compute_spread(0.0, gamma, sigma, k, T, mode="text_formula")
        phi_late = compute_spread(0.9, gamma, sigma, k, T, mode="text_formula")
        assert phi_early > phi_late

    def test_text_formula_at_tT_equals_table(self):
        """At t=T, text-formula spread equals table-matching spread."""
        gamma, sigma, k, T = 0.1, 2.0, 1.5, 1.0
        phi_text_tT = compute_spread(T, gamma, sigma, k, T, mode="text_formula")
        phi_table = compute_spread(T, gamma, sigma, k, T, mode="table_matching")
        assert abs(phi_text_tT - phi_table) < 1e-12

    def test_invalid_mode_raises(self):
        """Invalid spread mode raises ValueError."""
        with pytest.raises(ValueError):
            compute_spread(0.5, 0.1, 2.0, 1.5, 1.0, mode="invalid_mode")

    def test_spread_positive(self):
        """Spread is always positive for valid parameters."""
        for gamma in [0.01, 0.1, 0.5]:
            for mode in ["text_formula", "table_matching"]:
                phi = compute_spread(0.5, gamma, 2.0, 1.5, 1.0, mode=mode)
                assert phi > 0, f"Spread not positive for gamma={gamma}, mode={mode}"

    def test_table_matching_matches_published_spreads(self):
        """Table-matching spread reproduces published values within tolerance."""
        sigma, k, T = 2.0, 1.5, 1.0
        for gamma, pub_spread in PUBLISHED_SPREADS.items():
            phi = compute_initial_spread(gamma, sigma, k, T, mode="table_matching")
            assert abs(phi - pub_spread) < 0.01, (
                f"Table-matching spread {phi:.4f} != published {pub_spread} for gamma={gamma}"
            )


# ---------------------------------------------------------------------------
# Tests: inventory_quotes
# ---------------------------------------------------------------------------

class TestInventoryQuotes:
    def test_q0_reservation_equals_mid(self):
        """At q=0, reservation price equals mid-price."""
        r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
            100.0, 0.0, 0.5, 0.1, 2.0, 1.5, 1.0
        )
        assert abs(r_t - 100.0) < 1e-12

    def test_ask_above_bid(self):
        """Ask quote > bid quote for all q."""
        for q in [-5, 0, 5]:
            r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
                100.0, float(q), 0.5, 0.1, 2.0, 1.5, 1.0
            )
            assert p_ask > p_bid, f"Ask <= Bid at q={q}"

    def test_ask_above_mid_for_q0(self):
        """At q=0, ask > mid and bid < mid."""
        r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
            100.0, 0.0, 0.5, 0.1, 2.0, 1.5, 1.0
        )
        assert p_ask > 100.0
        assert p_bid < 100.0

    def test_long_inventory_shifts_quotes_down(self):
        """Long inventory (q > 0) shifts both quotes below mid."""
        r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
            100.0, 10.0, 0.5, 0.5, 2.0, 1.5, 1.0
        )
        # Reservation price should be well below mid
        assert r_t < 100.0

    def test_delta_a_equals_p_ask_minus_s(self):
        """delta_a = p_ask - S."""
        S = 100.0
        r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
            S, 2.0, 0.5, 0.1, 2.0, 1.5, 1.0
        )
        assert abs(delta_a - (p_ask - S)) < 1e-12

    def test_delta_b_equals_s_minus_p_bid(self):
        """delta_b = S - p_bid."""
        S = 100.0
        r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
            S, 2.0, 0.5, 0.1, 2.0, 1.5, 1.0
        )
        assert abs(delta_b - (S - p_bid)) < 1e-12

    def test_spread_equals_phi(self):
        """p_ask - p_bid = Phi(t)."""
        gamma, sigma, k, T = 0.1, 2.0, 1.5, 1.0
        t = 0.5
        r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
            100.0, 3.0, t, gamma, sigma, k, T
        )
        phi = compute_spread(t, gamma, sigma, k, T, mode="text_formula")
        assert abs(p_ask - p_bid - phi) < 1e-12

    def test_tT_reservation_equals_mid(self):
        """At t=T, reservation price equals mid-price."""
        r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
            100.0, 5.0, 1.0, 0.1, 2.0, 1.5, 1.0
        )
        assert abs(r_t - 100.0) < 1e-12


# ---------------------------------------------------------------------------
# Tests: symmetric_quotes
# ---------------------------------------------------------------------------

class TestSymmetricQuotes:
    def test_symmetric_around_mid(self):
        """Symmetric quotes are centered at mid-price."""
        S = 100.0
        p_ask, p_bid, delta_a, delta_b = symmetric_quotes(
            S, 0.0, 0.5, 0.1, 2.0, 1.5, 1.0
        )
        assert abs(p_ask - S - (S - p_bid)) < 1e-12

    def test_equal_distances(self):
        """delta_a == delta_b for symmetric strategy."""
        p_ask, p_bid, delta_a, delta_b = symmetric_quotes(
            100.0, 5.0, 0.5, 0.1, 2.0, 1.5, 1.0
        )
        assert abs(delta_a - delta_b) < 1e-12

    def test_inventory_independent(self):
        """Symmetric quotes do not depend on inventory."""
        S, t, gamma, sigma, k, T = 100.0, 0.5, 0.1, 2.0, 1.5, 1.0
        q1_ask, q1_bid, _, _ = symmetric_quotes(S, 0.0, t, gamma, sigma, k, T)
        q2_ask, q2_bid, _, _ = symmetric_quotes(S, 10.0, t, gamma, sigma, k, T)
        assert abs(q1_ask - q2_ask) < 1e-12
        assert abs(q1_bid - q2_bid) < 1e-12

    def test_spread_equals_phi(self):
        """p_ask - p_bid = Phi(t)."""
        gamma, sigma, k, T = 0.1, 2.0, 1.5, 1.0
        t = 0.5
        p_ask, p_bid, delta_a, delta_b = symmetric_quotes(
            100.0, 0.0, t, gamma, sigma, k, T
        )
        phi = compute_spread(t, gamma, sigma, k, T, mode="text_formula")
        assert abs(p_ask - p_bid - phi) < 1e-12


# ---------------------------------------------------------------------------
# Tests: fill_intensity
# ---------------------------------------------------------------------------

class TestFillIntensity:
    def test_intensity_at_zero_delta(self):
        """lambda(0) = A."""
        assert abs(fill_intensity(0.0, 140.0, 1.5) - 140.0) < 1e-12

    def test_intensity_decreasing_in_delta(self):
        """Intensity decreases as delta increases."""
        A, k = 140.0, 1.5
        lam_small = fill_intensity(0.5, A, k)
        lam_large = fill_intensity(1.0, A, k)
        assert lam_small > lam_large

    def test_intensity_formula(self):
        """lambda = A * exp(-k * delta)."""
        A, k, delta = 140.0, 1.5, 0.8
        expected = A * np.exp(-k * delta)
        assert abs(fill_intensity(delta, A, k) - expected) < 1e-12

    def test_intensity_positive_for_positive_delta(self):
        """Intensity is always positive."""
        assert fill_intensity(5.0, 140.0, 1.5) > 0

    def test_intensity_greater_than_A_for_negative_delta(self):
        """Negative delta gives lambda > A (documented behavior)."""
        lam = fill_intensity(-1.0, 140.0, 1.5)
        assert lam > 140.0

    def test_intensity_with_paper_params(self):
        """Verify intensity with paper parameters A=140, k=1.5."""
        delta = 0.5
        lam = fill_intensity(delta, 140.0, 1.5)
        expected = 140.0 * np.exp(-1.5 * 0.5)
        assert abs(lam - expected) < 1e-10
