"""
Experiment 2: Reproducibility sensitivity experiment for spread-formula ambiguity.

Tests the impact of the documented inconsistency between the paper's printed
finite-horizon spread formula and the spread values shown in published tables.

Compares:
  - text_formula: Phi(t) = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)
  - table_matching: Phi = (2/gamma)*ln(1+gamma/k)  [constant in time]

Calls functions from src/ modules.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SimParams, GAMMA_SCENARIOS, PUBLISHED_SPREADS
from src.simulator import run_monte_carlo_common_rng, run_monte_carlo
from src.metrics import (
    build_results_table,
    build_spread_comparison_table,
    build_sensitivity_table,
    compare_to_published,
)
from src.strategies import compute_initial_spread
from src.plots import (
    plot_spread_comparison,
    plot_sensitivity_heatmap,
    plot_pnl_histograms,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_exp2(
    gamma_list: list = None,
    N_paths: int = 1000,
    seed: int = 42,
    results_dir: str = RESULTS_DIR,
    verbose: bool = True,
) -> dict:
    """
    Run Experiment 2: Spread formula sensitivity analysis.

    Uses common random numbers (CRN) to isolate the effect of spread
    formula choice on simulation outcomes.

    Parameters
    ----------
    gamma_list : list or None
        List of gamma values. Defaults to [0.01, 0.1, 0.5].
    N_paths : int
        Number of Monte Carlo paths per scenario.
    seed : int
        Base random seed (same for both modes via CRN).
    results_dir : str
        Directory to save results.
    verbose : bool
        Print progress and summary tables.

    Returns
    -------
    dict
        Nested dict: results[mode][gamma][strategy] = List[PathResult]
    """
    if gamma_list is None:
        gamma_list = GAMMA_SCENARIOS

    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Compute and compare spread values
    spread_table = build_spread_comparison_table(
        gamma_list,
        sigma=2.0,
        k=1.5,
        T=1.0,
    )

    if verbose:
        print("\n" + "="*70)
        print("SPREAD FORMULA COMPARISON")
        print("="*70)
        print(spread_table.to_string(index=False, float_format="{:.6f}".format))
        print("\nNote: table_matching_spread should match published values.")
        print("The text_formula adds gamma*sigma^2*(T-t) at t=0.")

    # Identify which formula better matches published values
    spread_table["text_abs_err"] = abs(spread_table["discrepancy_text"])
    spread_table["table_abs_err"] = abs(spread_table["discrepancy_table"])
    better_match = "table_matching" if (
        spread_table["table_abs_err"].mean() < spread_table["text_abs_err"].mean()
    ) else "text_formula"

    if verbose:
        print(f"\nBetter match to published spreads: {better_match}")

    # Step 2: Run Monte Carlo under both spread modes using CRN
    all_results = {"text_formula": {}, "table_matching": {}}

    for gamma in gamma_list:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running gamma = {gamma} (CRN across spread modes)")
            print(f"{'='*60}")

        all_results["text_formula"][gamma] = {}
        all_results["table_matching"][gamma] = {}

        for strategy in ["inventory", "symmetric"]:
            params_text = SimParams(
                gamma=gamma,
                N_paths=N_paths,
                spread_mode="text_formula",
                seed=seed + int(gamma * 1000),
            )
            params_table = SimParams(
                gamma=gamma,
                N_paths=N_paths,
                spread_mode="table_matching",
                seed=seed + int(gamma * 1000),  # Same seed for CRN
            )

            if verbose:
                print(f"  Simulating {strategy} strategy with CRN...")

            results_text, results_table_mode = run_monte_carlo_common_rng(
                params_text, params_table, strategy
            )

            all_results["text_formula"][gamma][strategy] = results_text
            all_results["table_matching"][gamma][strategy] = results_table_mode

    # Step 3: Build summary tables for each mode
    table_text = build_results_table(all_results["text_formula"], gamma_list)
    table_text["spread_mode"] = "text_formula"

    table_table = build_results_table(all_results["table_matching"], gamma_list)
    table_table["spread_mode"] = "table_matching"

    combined = pd.concat([table_text, table_table], ignore_index=True)

    if verbose:
        print("\n" + "="*70)
        print("RESULTS BY SPREAD MODE")
        print("="*70)
        print(combined.to_string(index=False, float_format="{:.4f}".format))

    # Step 4: Sensitivity table
    sensitivity = build_sensitivity_table(
        all_results["text_formula"],
        all_results["table_matching"],
        gamma_list,
    )

    if verbose:
        print("\n" + "="*70)
        print("SENSITIVITY TABLE")
        print("="*70)
        print(sensitivity.to_string(index=False, float_format="{:.4f}".format))

    # Step 5: Compare table_matching results to published values
    comparison_table = compare_to_published(table_table, gamma_list)
    comparison_text = compare_to_published(table_text, gamma_list)

    if verbose:
        print("\n" + "="*70)
        print("COMPARISON TO PUBLISHED (table_matching mode)")
        print("="*70)
        print(comparison_table.to_string(index=False, float_format="{:.4f}".format))

        print("\n" + "="*70)
        print("COMPARISON TO PUBLISHED (text_formula mode)")
        print("="*70)
        print(comparison_text.to_string(index=False, float_format="{:.4f}".format))

    # Save results
    spread_table.to_csv(os.path.join(results_dir, "exp2_spread_comparison.csv"), index=False)
    combined.to_csv(os.path.join(results_dir, "exp2_results_by_mode.csv"), index=False)
    sensitivity.to_csv(os.path.join(results_dir, "exp2_sensitivity.csv"), index=False)
    comparison_table.to_csv(os.path.join(results_dir, "exp2_comparison_table_mode.csv"), index=False)
    comparison_text.to_csv(os.path.join(results_dir, "exp2_comparison_text_mode.csv"), index=False)

    # Generate plots
    _generate_plots(
        all_results, spread_table, sensitivity, gamma_list, results_dir, verbose
    )

    return {
        "results": all_results,
        "spread_table": spread_table,
        "sensitivity": sensitivity,
        "better_match": better_match,
    }


def _generate_plots(
    all_results: dict,
    spread_table: pd.DataFrame,
    sensitivity: pd.DataFrame,
    gamma_list: list,
    results_dir: str,
    verbose: bool,
) -> None:
    """Generate and save all plots for Experiment 2."""
    import matplotlib.pyplot as plt

    # 1. Spread comparison bar chart
    fig = plot_spread_comparison(
        spread_table,
        save_path=os.path.join(results_dir, "exp2_spread_comparison.png"),
    )
    plt.close(fig)

    # 2. Sensitivity heatmaps
    for metric in ["std_profit", "mean_profit", "std_final_q"]:
        fig = plot_sensitivity_heatmap(
            sensitivity,
            metric=metric,
            save_path=os.path.join(results_dir, f"exp2_sensitivity_{metric}.png"),
        )
        plt.close(fig)

    # 3. P&L histograms comparing modes for gamma=0.1
    gamma_plot = 0.1 if 0.1 in gamma_list else gamma_list[0]
    for strategy in ["inventory", "symmetric"]:
        fig = plot_pnl_histograms(
            all_results["text_formula"][gamma_plot][strategy],
            all_results["table_matching"][gamma_plot][strategy],
            gamma=gamma_plot,
            save_path=os.path.join(
                results_dir,
                f"exp2_pnl_modes_{strategy}_gamma{gamma_plot}.png",
            ),
        )
        plt.close(fig)

    if verbose:
        print(f"\nAll Exp2 plots saved to {results_dir}")


def diagnose_spread_ambiguity(
    gamma_list: list = None,
    sigma: float = 2.0,
    k: float = 1.5,
    T: float = 1.0,
) -> pd.DataFrame:
    """
    Compute and display the spread discrepancy for each gamma.

    Shows explicitly that the table-matching formula reproduces published values
    while the text-formula adds an extra gamma*sigma^2*(T-t) term.

    Parameters
    ----------
    gamma_list : list or None
        List of gamma values.
    sigma : float
        Mid-price volatility.
    k : float
        Fill-intensity decay parameter.
    T : float
        Terminal horizon.

    Returns
    -------
    pd.DataFrame
        Spread diagnosis table.
    """
    if gamma_list is None:
        gamma_list = GAMMA_SCENARIOS

    rows = []
    for gamma in gamma_list:
        text_spread = compute_initial_spread(gamma, sigma, k, T, mode="text_formula")
        table_spread = compute_initial_spread(gamma, sigma, k, T, mode="table_matching")
        pub_spread = PUBLISHED_SPREADS.get(gamma, float("nan"))
        extra_term = gamma * sigma**2 * T  # The extra term in text_formula at t=0

        rows.append({
            "gamma": gamma,
            "text_formula_spread_t0": text_spread,
            "table_matching_spread": table_spread,
            "published_spread": pub_spread,
            "extra_term_gamma_sigma2_T": extra_term,
            "table_matches_published": abs(table_spread - pub_spread) < 0.01,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running Experiment 2: Spread Formula Sensitivity Analysis")

    # First show the spread diagnosis
    diag = diagnose_spread_ambiguity()
    print("\nSpread Ambiguity Diagnosis:")
    print(diag.to_string(index=False, float_format="{:.6f}".format))

    # Run full experiment
    output = run_exp2(verbose=True)
    print(f"\nBetter spread match: {output['better_match']}")
