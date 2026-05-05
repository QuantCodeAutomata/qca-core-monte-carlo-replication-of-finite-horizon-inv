"""
Tests for src/metrics.py

Validates:
  - Summary statistics computation
  - Results table building
  - Spread comparison table
  - Variance reduction ratios
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    extract_arrays,
    compute_summary,
    build_results_table,
    build_spread_comparison_table,
    variance_reduction_ratio,
)
from src.simulator import PathResult
from src.config import PUBLISHED_SPREADS


def make_results(pnl_values, inv_values):
    """Helper to create PathResult list from arrays."""
    results = []
    for pnl, inv in zip(pnl_values, inv_values):
        r = PathResult(
            terminal_pnl=pnl,
            terminal_inventory=inv,
            terminal_cash=pnl - inv * 100.0,
            terminal_price=100.0,
        )
        results.append(r)
    return results


class TestExtractArrays:
    def test_correct_pnl_extraction(self):
        pnl = [10.0, 20.0, 30.0]
        inv = [1.0, -1.0, 0.0]
        results = make_results(pnl, inv)
        arrays = extract_arrays(results)
        assert np.allclose(arrays["pnl"], pnl)

    def test_correct_inventory_extraction(self):
        pnl = [10.0, 20.0, 30.0]
        inv = [1.0, -1.0, 0.0]
        results = make_results(pnl, inv)
        arrays = extract_arrays(results)
        assert np.allclose(arrays["inventory"], inv)


class TestComputeSummary:
    def test_mean_profit_correct(self):
        pnl = [10.0, 20.0, 30.0]
        results = make_results(pnl, [0.0] * 3)
        summary = compute_summary(results)
        assert abs(summary["mean_profit"] - 20.0) < 1e-10

    def test_std_profit_correct(self):
        pnl = [10.0, 20.0, 30.0]
        results = make_results(pnl, [0.0] * 3)
        summary = compute_summary(results)
        expected_std = np.std(pnl, ddof=1)
        assert abs(summary["std_profit"] - expected_std) < 1e-10

    def test_mean_final_q_correct(self):
        inv = [2.0, -2.0, 0.0]
        results = make_results([0.0] * 3, inv)
        summary = compute_summary(results)
        assert abs(summary["mean_final_q"] - 0.0) < 1e-10

    def test_n_paths_correct(self):
        results = make_results([1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 5)
        summary = compute_summary(results)
        assert summary["n_paths"] == 5

    def test_single_path(self):
        results = make_results([42.0], [3.0])
        summary = compute_summary(results)
        assert abs(summary["mean_profit"] - 42.0) < 1e-10
        assert abs(summary["mean_final_q"] - 3.0) < 1e-10


class TestBuildResultsTable:
    def test_table_has_correct_columns(self):
        gamma_list = [0.1]
        results_dict = {
            0.1: {
                "inventory": make_results([10.0, 20.0], [1.0, -1.0]),
                "symmetric": make_results([15.0, 25.0], [2.0, -2.0]),
            }
        }
        table = build_results_table(results_dict, gamma_list)
        assert "gamma" in table.columns
        assert "strategy" in table.columns
        assert "mean_profit" in table.columns
        assert "std_profit" in table.columns
        assert "mean_final_q" in table.columns
        assert "std_final_q" in table.columns

    def test_table_has_correct_rows(self):
        gamma_list = [0.01, 0.1, 0.5]
        results_dict = {
            g: {
                "inventory": make_results([10.0, 20.0], [1.0, -1.0]),
                "symmetric": make_results([15.0, 25.0], [2.0, -2.0]),
            }
            for g in gamma_list
        }
        table = build_results_table(results_dict, gamma_list)
        assert len(table) == 6  # 3 gammas * 2 strategies


class TestBuildSpreadComparisonTable:
    def test_table_has_correct_columns(self):
        table = build_spread_comparison_table([0.1], sigma=2.0, k=1.5, T=1.0)
        assert "gamma" in table.columns
        assert "text_formula_spread" in table.columns
        assert "table_matching_spread" in table.columns
        assert "published_spread" in table.columns

    def test_table_matching_close_to_published(self):
        """Table-matching spread should be close to published values."""
        table = build_spread_comparison_table([0.01, 0.1, 0.5], sigma=2.0, k=1.5, T=1.0)
        for _, row in table.iterrows():
            assert abs(row["discrepancy_table"]) < 0.01, (
                f"Table-matching spread not close to published for gamma={row['gamma']}"
            )

    def test_text_formula_larger_than_table_at_t0(self):
        """Text-formula spread > table-matching at t=0 (extra gamma*sigma^2*T term)."""
        table = build_spread_comparison_table([0.1], sigma=2.0, k=1.5, T=1.0)
        row = table.iloc[0]
        assert row["text_formula_spread"] > row["table_matching_spread"]


class TestVarianceReductionRatio:
    def test_vrr_less_than_one_for_lower_variance(self):
        """VRR < 1 when inventory strategy has lower variance."""
        # Inventory: tight distribution
        inv_results = make_results(
            np.random.default_rng(42).normal(50, 5, 100).tolist(),
            np.random.default_rng(42).normal(0, 2, 100).tolist(),
        )
        # Symmetric: wide distribution
        sym_results = make_results(
            np.random.default_rng(42).normal(50, 15, 100).tolist(),
            np.random.default_rng(42).normal(0, 8, 100).tolist(),
        )
        vrr = variance_reduction_ratio(inv_results, sym_results)
        assert vrr["vrr_profit"] < 1.0
        assert vrr["vrr_inventory"] < 1.0

    def test_vrr_equals_one_for_equal_variance(self):
        """VRR = 1 when both strategies have same variance."""
        pnl = np.random.default_rng(42).normal(50, 10, 100).tolist()
        inv_results = make_results(pnl, [0.0] * 100)
        sym_results = make_results(pnl, [0.0] * 100)
        vrr = variance_reduction_ratio(inv_results, sym_results)
        assert abs(vrr["vrr_profit"] - 1.0) < 1e-10

    def test_vrr_keys_present(self):
        """VRR dict has expected keys."""
        results = make_results([1.0, 2.0], [0.0, 1.0])
        vrr = variance_reduction_ratio(results, results)
        assert "vrr_profit" in vrr
        assert "vrr_inventory" in vrr
