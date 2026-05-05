"""
Experiment 1: Core Monte Carlo replication of finite-horizon inventory-based
market making vs symmetric benchmark.

Replicates the paper's main simulation comparing:
  - Inventory-based (Avellaneda-Stoikov) quoting strategy
  - Symmetric benchmark strategy

For gamma in {0.01, 0.1, 0.5} with 1000 Monte Carlo paths each.

Calls functions from src/ modules.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SimParams, GAMMA_SCENARIOS
from src.simulator import run_monte_carlo
from src.metrics import (
    build_results_table,
    compare_to_published,
    variance_reduction_ratio,
    extract_arrays,
)
from src.plots import (
    plot_illustrative_path,
    plot_pnl_histograms,
    plot_inventory_histograms,
    plot_gamma_comparison,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_exp1(
    gamma_list: list = None,
    N_paths: int = 1000,
    spread_mode: str = "text_formula",
    seed: int = 42,
    results_dir: str = RESULTS_DIR,
    verbose: bool = True,
) -> dict:
    """
    Run Experiment 1: Core Monte Carlo replication.

    Parameters
    ----------
    gamma_list : list or None
        List of gamma values. Defaults to [0.01, 0.1, 0.5].
    N_paths : int
        Number of Monte Carlo paths per scenario.
    spread_mode : str
        'text_formula' or 'table_matching'.
    seed : int
        Base random seed.
    results_dir : str
        Directory to save results.
    verbose : bool
        Print progress and summary tables.

    Returns
    -------
    dict
        Nested dict: results[gamma][strategy] = List[PathResult]
    """
    if gamma_list is None:
        gamma_list = GAMMA_SCENARIOS

    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    full_paths = {}

    for gamma in gamma_list:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running gamma = {gamma}, spread_mode = {spread_mode}")
            print(f"{'='*60}")

        all_results[gamma] = {}
        full_paths[gamma] = {}

        for strategy in ["inventory", "symmetric"]:
            params = SimParams(
                gamma=gamma,
                N_paths=N_paths,
                spread_mode=spread_mode,
                seed=seed + int(gamma * 1000),  # Different seed per gamma
            )

            if verbose:
                print(f"  Simulating {strategy} strategy ({N_paths} paths)...")

            results, saved_path = run_monte_carlo(
                params,
                strategy,
                save_path_index=0,  # Save first path for illustration
            )
            all_results[gamma][strategy] = results
            full_paths[gamma][strategy] = saved_path

    # Build summary table
    results_table = build_results_table(all_results, gamma_list)

    if verbose:
        print("\n" + "="*70)
        print("SIMULATION RESULTS SUMMARY")
        print("="*70)
        print(results_table.to_string(index=False, float_format="{:.4f}".format))

    # Compare to published values
    comparison = compare_to_published(results_table, gamma_list)

    if verbose:
        print("\n" + "="*70)
        print("COMPARISON TO PUBLISHED VALUES")
        print("="*70)
        print(comparison.to_string(index=False, float_format="{:.4f}".format))

    # Variance reduction ratios
    if verbose:
        print("\n" + "="*70)
        print("VARIANCE REDUCTION RATIOS (inventory / symmetric)")
        print("="*70)
        for gamma in gamma_list:
            vrr = variance_reduction_ratio(
                all_results[gamma]["inventory"],
                all_results[gamma]["symmetric"],
            )
            print(f"  gamma={gamma}: VRR_profit={vrr['vrr_profit']:.4f}, "
                  f"VRR_inventory={vrr['vrr_inventory']:.4f}")

    # Save CSV results
    results_table.to_csv(
        os.path.join(results_dir, "exp1_summary.csv"), index=False
    )
    comparison.to_csv(
        os.path.join(results_dir, "exp1_comparison.csv"), index=False
    )

    # Generate plots
    _generate_plots(all_results, full_paths, gamma_list, results_dir, verbose)

    return all_results


def _generate_plots(
    all_results: dict,
    full_paths: dict,
    gamma_list: list,
    results_dir: str,
    verbose: bool,
) -> None:
    """Generate and save all plots for Experiment 1."""

    # 1. Illustrative path plot for gamma=0.1 (inventory strategy)
    gamma_plot = 0.1 if 0.1 in gamma_list else gamma_list[0]
    inv_path = full_paths[gamma_plot].get("inventory")
    if inv_path is not None:
        fig = plot_illustrative_path(
            inv_path,
            gamma=gamma_plot,
            strategy="inventory",
            save_path=os.path.join(results_dir, f"exp1_path_inventory_gamma{gamma_plot}.png"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        if verbose:
            print(f"\nSaved illustrative path plot for gamma={gamma_plot}")

    # 2. P&L histograms for each gamma
    for gamma in gamma_list:
        fig = plot_pnl_histograms(
            all_results[gamma]["inventory"],
            all_results[gamma]["symmetric"],
            gamma=gamma,
            save_path=os.path.join(results_dir, f"exp1_pnl_hist_gamma{gamma}.png"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    # 3. Inventory histograms for each gamma
    for gamma in gamma_list:
        fig = plot_inventory_histograms(
            all_results[gamma]["inventory"],
            all_results[gamma]["symmetric"],
            gamma=gamma,
            save_path=os.path.join(results_dir, f"exp1_inv_hist_gamma{gamma}.png"),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    # 4. Gamma comparison plot
    results_table = build_results_table(all_results, gamma_list)
    fig = plot_gamma_comparison(
        results_table,
        save_path=os.path.join(results_dir, "exp1_gamma_comparison.png"),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    if verbose:
        print(f"\nAll plots saved to {results_dir}")


def verify_qualitative_findings(all_results: dict, gamma_list: list) -> dict:
    """
    Verify the paper's qualitative findings:
    1. Inventory strategy has lower P&L std than symmetric for all gamma.
    2. Inventory strategy has lower inventory std than symmetric for all gamma.
    3. As gamma decreases, strategies converge.
    4. Larger gamma -> stronger inventory control.

    Parameters
    ----------
    all_results : dict
        Nested results dict.
    gamma_list : list
        List of gamma values.

    Returns
    -------
    dict
        Dictionary of verification results.
    """
    from src.metrics import compute_summary

    verifications = {}

    for gamma in gamma_list:
        inv_summary = compute_summary(all_results[gamma]["inventory"])
        sym_summary = compute_summary(all_results[gamma]["symmetric"])

        lower_pnl_std = inv_summary["std_profit"] < sym_summary["std_profit"]
        lower_inv_std = inv_summary["std_final_q"] < sym_summary["std_final_q"]

        verifications[gamma] = {
            "lower_pnl_std": lower_pnl_std,
            "lower_inv_std": lower_inv_std,
            "inv_std_profit": inv_summary["std_profit"],
            "sym_std_profit": sym_summary["std_profit"],
            "inv_std_q": inv_summary["std_final_q"],
            "sym_std_q": sym_summary["std_final_q"],
        }

    # Check convergence: at small gamma, strategies should be more similar
    if len(gamma_list) >= 2:
        sorted_gammas = sorted(gamma_list)
        vrr_small = variance_reduction_ratio(
            all_results[sorted_gammas[0]]["inventory"],
            all_results[sorted_gammas[0]]["symmetric"],
        )
        vrr_large = variance_reduction_ratio(
            all_results[sorted_gammas[-1]]["inventory"],
            all_results[sorted_gammas[-1]]["symmetric"],
        )
        # At small gamma, VRR should be closer to 1 (strategies more similar)
        verifications["convergence_check"] = {
            "vrr_small_gamma": vrr_small,
            "vrr_large_gamma": vrr_large,
            "converges_at_small_gamma": vrr_small["vrr_profit"] > vrr_large["vrr_profit"],
        }

    return verifications


if __name__ == "__main__":
    print("Running Experiment 1: Core Monte Carlo Replication")
    results = run_exp1(verbose=True)

    print("\n" + "="*60)
    print("QUALITATIVE VERIFICATION")
    print("="*60)
    verif = verify_qualitative_findings(results, GAMMA_SCENARIOS)
    for gamma in GAMMA_SCENARIOS:
        v = verif[gamma]
        print(f"  gamma={gamma}: lower_pnl_std={v['lower_pnl_std']}, "
              f"lower_inv_std={v['lower_inv_std']}")
    if "convergence_check" in verif:
        cc = verif["convergence_check"]
        print(f"  Convergence at small gamma: {cc['converges_at_small_gamma']}")
