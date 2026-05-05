"""
Integration tests for experiment modules.

Tests that the experiment functions run end-to-end and produce
results consistent with the paper's methodology.

These tests use small N_paths for speed but verify structural correctness.
"""

import os
import numpy as np
import pandas as pd
import pytest
import tempfile

from src.config import SimParams, GAMMA_SCENARIOS
from src.simulator import run_monte_carlo
from src.metrics import (
    build_results_table,
    variance_reduction_ratio,
    build_spread_comparison_table,
)
from exp.exp1_core_replication import run_exp1, verify_qualitative_findings
from exp.exp2_spread_sensitivity import run_exp2, diagnose_spread_ambiguity
from exp.exp3_reservation_prices import run_exp3


# ---------------------------------------------------------------------------
# Integration tests: Experiment 1
# ---------------------------------------------------------------------------

class TestExp1Integration:
    def test_exp1_runs_without_error(self, tmp_path):
        """Experiment 1 completes without error."""
        results = run_exp1(
            gamma_list=[0.1],
            N_paths=50,
            spread_mode="text_formula",
            seed=42,
            results_dir=str(tmp_path),
            verbose=False,
        )
        assert 0.1 in results
        assert "inventory" in results[0.1]
        assert "symmetric" in results[0.1]

    def test_exp1_correct_number_of_paths(self, tmp_path):
        """Experiment 1 produces correct number of paths."""
        N = 50
        results = run_exp1(
            gamma_list=[0.1],
            N_paths=N,
            results_dir=str(tmp_path),
            verbose=False,
        )
        assert len(results[0.1]["inventory"]) == N
        assert len(results[0.1]["symmetric"]) == N

    def test_exp1_pnl_formula_holds(self, tmp_path):
        """Pi_T = X_T + q_T * S_T for all paths in Exp1."""
        results = run_exp1(
            gamma_list=[0.1],
            N_paths=20,
            results_dir=str(tmp_path),
            verbose=False,
        )
        for strat in ["inventory", "symmetric"]:
            for r in results[0.1][strat]:
                expected = r.terminal_cash + r.terminal_inventory * r.terminal_price
                assert abs(r.terminal_pnl - expected) < 1e-10

    def test_exp1_qualitative_findings_gamma05(self, tmp_path):
        """At gamma=0.5, inventory strategy has lower variance than symmetric."""
        results = run_exp1(
            gamma_list=[0.5],
            N_paths=500,
            results_dir=str(tmp_path),
            verbose=False,
        )
        verif = verify_qualitative_findings(results, [0.5])
        assert verif[0.5]["lower_pnl_std"], "Inventory should have lower P&L std at gamma=0.5"
        assert verif[0.5]["lower_inv_std"], "Inventory should have lower inv std at gamma=0.5"

    def test_exp1_saves_csv(self, tmp_path):
        """Experiment 1 saves CSV files."""
        run_exp1(
            gamma_list=[0.1],
            N_paths=20,
            results_dir=str(tmp_path),
            verbose=False,
        )
        assert os.path.exists(os.path.join(str(tmp_path), "exp1_summary.csv"))
        assert os.path.exists(os.path.join(str(tmp_path), "exp1_comparison.csv"))

    def test_exp1_saves_plots(self, tmp_path):
        """Experiment 1 saves plot files."""
        run_exp1(
            gamma_list=[0.1],
            N_paths=20,
            results_dir=str(tmp_path),
            verbose=False,
        )
        assert os.path.exists(
            os.path.join(str(tmp_path), "exp1_pnl_hist_gamma0.1.png")
        )


# ---------------------------------------------------------------------------
# Integration tests: Experiment 2
# ---------------------------------------------------------------------------

class TestExp2Integration:
    def test_exp2_runs_without_error(self, tmp_path):
        """Experiment 2 completes without error."""
        output = run_exp2(
            gamma_list=[0.1],
            N_paths=50,
            results_dir=str(tmp_path),
            verbose=False,
        )
        assert "results" in output
        assert "spread_table" in output
        assert "better_match" in output

    def test_exp2_better_match_is_table_matching(self, tmp_path):
        """Table-matching formula better matches published spreads."""
        output = run_exp2(
            gamma_list=[0.01, 0.1, 0.5],
            N_paths=50,
            results_dir=str(tmp_path),
            verbose=False,
        )
        assert output["better_match"] == "table_matching"

    def test_exp2_spread_diagnosis(self):
        """Spread diagnosis shows table-matching matches published values."""
        diag = diagnose_spread_ambiguity([0.01, 0.1, 0.5])
        for _, row in diag.iterrows():
            assert row["table_matches_published"], (
                f"Table-matching should match published for gamma={row['gamma']}"
            )

    def test_exp2_text_formula_larger_spread(self):
        """Text-formula spread > table-matching at t=0."""
        diag = diagnose_spread_ambiguity([0.1])
        row = diag.iloc[0]
        assert row["text_formula_spread_t0"] > row["table_matching_spread"]

    def test_exp2_saves_csv(self, tmp_path):
        """Experiment 2 saves CSV files."""
        run_exp2(
            gamma_list=[0.1],
            N_paths=20,
            results_dir=str(tmp_path),
            verbose=False,
        )
        assert os.path.exists(os.path.join(str(tmp_path), "exp2_spread_comparison.csv"))
        assert os.path.exists(os.path.join(str(tmp_path), "exp2_sensitivity.csv"))


# ---------------------------------------------------------------------------
# Integration tests: Experiment 3
# ---------------------------------------------------------------------------

class TestExp3Integration:
    def test_exp3_runs_without_error(self, tmp_path):
        """Experiment 3 completes without error."""
        output = run_exp3(results_dir=str(tmp_path), verbose=False)
        assert "boundary_checks" in output
        assert "indifference_checks" in output
        assert "comparative_statics" in output

    def test_exp3_all_boundary_checks_pass(self, tmp_path):
        """All boundary conditions pass for all gamma values."""
        output = run_exp3(results_dir=str(tmp_path), verbose=False)
        for gamma in [0.01, 0.1, 0.5]:
            bc = output["boundary_checks"][gamma]
            for check_name, passed in bc.items():
                assert passed, f"Boundary check {check_name} failed for gamma={gamma}"

    def test_exp3_all_indifference_checks_pass(self, tmp_path):
        """All indifference conditions pass for all gamma values."""
        output = run_exp3(results_dir=str(tmp_path), verbose=False)
        for gamma in [0.01, 0.1, 0.5]:
            ic = output["indifference_checks"][gamma]
            assert ic["bid_indifference_all_pass"], f"Bid indifference failed for gamma={gamma}"
            assert ic["ask_indifference_all_pass"], f"Ask indifference failed for gamma={gamma}"

    def test_exp3_comparative_statics_pass(self, tmp_path):
        """Comparative statics pass for all gamma values."""
        output = run_exp3(results_dir=str(tmp_path), verbose=False)
        for gamma in [0.01, 0.1, 0.5]:
            cs = output["comparative_statics"][gamma]
            assert cs["cs1_long_inv_r_below_s"], f"CS1 failed for gamma={gamma}"
            assert cs["cs2_short_inv_r_above_s"], f"CS2 failed for gamma={gamma}"
            assert cs["cs3_r_decreasing_in_q"], f"CS3 failed for gamma={gamma}"
            assert cs["cs4_larger_tau_larger_shift"], f"CS4 failed for gamma={gamma}"

    def test_exp3_saves_csv(self, tmp_path):
        """Experiment 3 saves CSV files."""
        run_exp3(results_dir=str(tmp_path), verbose=False)
        assert os.path.exists(os.path.join(str(tmp_path), "exp3_admissibility.csv"))

    def test_exp3_admissibility_all_pass(self, tmp_path):
        """Admissibility holds for all q within q_max."""
        output = run_exp3(results_dir=str(tmp_path), verbose=False)
        adm_table = output["admissibility_table"]
        assert adm_table["all_admissible"].all()


# ---------------------------------------------------------------------------
# Cross-experiment validation
# ---------------------------------------------------------------------------

class TestCrossExperimentValidation:
    def test_inventory_strategy_converges_to_symmetric_at_small_gamma(self, tmp_path):
        """At small gamma, inventory and symmetric strategies produce similar results."""
        results = run_exp1(
            gamma_list=[0.01, 0.5],
            N_paths=200,
            results_dir=str(tmp_path),
            verbose=False,
        )
        vrr_small = variance_reduction_ratio(
            results[0.01]["inventory"],
            results[0.01]["symmetric"],
        )
        vrr_large = variance_reduction_ratio(
            results[0.5]["inventory"],
            results[0.5]["symmetric"],
        )
        # At small gamma, VRR should be closer to 1 (strategies more similar)
        assert vrr_small["vrr_profit"] > vrr_large["vrr_profit"], (
            "At small gamma, inventory strategy should be more similar to symmetric"
        )

    def test_larger_gamma_stronger_inventory_control(self, tmp_path):
        """Larger gamma -> stronger inventory control (lower inventory std ratio)."""
        results = run_exp1(
            gamma_list=[0.01, 0.5],
            N_paths=200,
            results_dir=str(tmp_path),
            verbose=False,
        )
        vrr_small = variance_reduction_ratio(
            results[0.01]["inventory"],
            results[0.01]["symmetric"],
        )
        vrr_large = variance_reduction_ratio(
            results[0.5]["inventory"],
            results[0.5]["symmetric"],
        )
        # Larger gamma -> lower VRR_inventory (stronger control)
        assert vrr_large["vrr_inventory"] < vrr_small["vrr_inventory"], (
            "Larger gamma should produce stronger inventory control"
        )
