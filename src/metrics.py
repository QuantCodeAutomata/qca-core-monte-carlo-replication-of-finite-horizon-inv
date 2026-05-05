"""
Metrics module for summarizing Monte Carlo simulation results.

Computes mean, standard deviation, and other descriptive statistics
for terminal P&L and final inventory across simulation paths.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from src.simulator import PathResult
from src.config import PUBLISHED_TARGETS, PUBLISHED_SPREADS
from src.strategies import compute_initial_spread


def extract_arrays(results: List[PathResult]) -> Dict[str, np.ndarray]:
    """
    Extract terminal P&L and inventory arrays from a list of PathResult objects.

    Parameters
    ----------
    results : List[PathResult]
        List of simulation path results.

    Returns
    -------
    Dict[str, ndarray]
        Dictionary with keys 'pnl' and 'inventory'.
    """
    pnl = np.array([r.terminal_pnl for r in results], dtype=np.float64)
    inventory = np.array([r.terminal_inventory for r in results], dtype=np.float64)
    return {"pnl": pnl, "inventory": inventory}


def compute_summary(results: List[PathResult]) -> Dict[str, float]:
    """
    Compute summary statistics for a set of simulation paths.

    Parameters
    ----------
    results : List[PathResult]
        List of simulation path results.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - mean_profit: sample mean of terminal P&L
        - std_profit: sample standard deviation of terminal P&L
        - mean_final_q: sample mean of final inventory
        - std_final_q: sample standard deviation of final inventory
        - n_paths: number of paths
    """
    arrays = extract_arrays(results)
    pnl = arrays["pnl"]
    inv = arrays["inventory"]

    return {
        "mean_profit": float(np.mean(pnl)),
        "std_profit": float(np.std(pnl, ddof=1)),
        "mean_final_q": float(np.mean(inv)),
        "std_final_q": float(np.std(inv, ddof=1)),
        "n_paths": len(results),
    }


def build_results_table(
    results_dict: Dict,
    gamma_list: List[float],
    strategies: List[str] = ("inventory", "symmetric"),
) -> pd.DataFrame:
    """
    Build a summary DataFrame from nested results dictionary.

    Parameters
    ----------
    results_dict : Dict
        Nested dict: results_dict[gamma][strategy] = List[PathResult]
    gamma_list : List[float]
        List of gamma values.
    strategies : List[str]
        Strategy names.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: gamma, strategy, mean_profit, std_profit,
        mean_final_q, std_final_q.
    """
    rows = []
    for gamma in gamma_list:
        for strat in strategies:
            summary = compute_summary(results_dict[gamma][strat])
            rows.append({
                "gamma": gamma,
                "strategy": strat,
                "mean_profit": summary["mean_profit"],
                "std_profit": summary["std_profit"],
                "mean_final_q": summary["mean_final_q"],
                "std_final_q": summary["std_final_q"],
                "n_paths": summary["n_paths"],
            })
    return pd.DataFrame(rows)


def build_spread_comparison_table(
    gamma_list: List[float],
    sigma: float,
    k: float,
    T: float,
) -> pd.DataFrame:
    """
    Build a table comparing text-formula and table-matching spreads
    against published values.

    Parameters
    ----------
    gamma_list : List[float]
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
        Comparison table with columns: gamma, text_formula_spread,
        table_matching_spread, published_spread, discrepancy_text,
        discrepancy_table.
    """
    rows = []
    for gamma in gamma_list:
        text_spread = compute_initial_spread(gamma, sigma, k, T, mode="text_formula")
        table_spread = compute_initial_spread(gamma, sigma, k, T, mode="table_matching")
        pub_spread = PUBLISHED_SPREADS.get(gamma, np.nan)

        rows.append({
            "gamma": gamma,
            "text_formula_spread": text_spread,
            "table_matching_spread": table_spread,
            "published_spread": pub_spread,
            "discrepancy_text": text_spread - pub_spread,
            "discrepancy_table": table_spread - pub_spread,
        })
    return pd.DataFrame(rows)


def build_sensitivity_table(
    results_mode1: Dict,
    results_mode2: Dict,
    gamma_list: List[float],
    strategies: List[str] = ("inventory", "symmetric"),
) -> pd.DataFrame:
    """
    Build a sensitivity table comparing two spread modes.

    Parameters
    ----------
    results_mode1 : Dict
        Results under text_formula mode: results[gamma][strategy]
    results_mode2 : Dict
        Results under table_matching mode: results[gamma][strategy]
    gamma_list : List[float]
        List of gamma values.
    strategies : List[str]
        Strategy names.

    Returns
    -------
    pd.DataFrame
        Sensitivity table with spread_mode column added.
    """
    rows = []
    for gamma in gamma_list:
        for strat in strategies:
            for mode_label, res_dict in [
                ("text_formula", results_mode1),
                ("table_matching", results_mode2),
            ]:
                summary = compute_summary(res_dict[gamma][strat])
                rows.append({
                    "gamma": gamma,
                    "strategy": strat,
                    "spread_mode": mode_label,
                    "mean_profit": summary["mean_profit"],
                    "std_profit": summary["std_profit"],
                    "mean_final_q": summary["mean_final_q"],
                    "std_final_q": summary["std_final_q"],
                })
    return pd.DataFrame(rows)


def compare_to_published(
    results_table: pd.DataFrame,
    gamma_list: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compare simulation results to published target values.

    Parameters
    ----------
    results_table : pd.DataFrame
        Output of build_results_table.
    gamma_list : List[float] or None
        Gamma values to compare. Defaults to keys in PUBLISHED_TARGETS.

    Returns
    -------
    pd.DataFrame
        Comparison table with simulated and published values side by side.
    """
    if gamma_list is None:
        gamma_list = list(PUBLISHED_TARGETS.keys())

    rows = []
    for gamma in gamma_list:
        if gamma not in PUBLISHED_TARGETS:
            continue
        for strat in ("inventory", "symmetric"):
            pub = PUBLISHED_TARGETS[gamma].get(strat, {})
            sim_row = results_table[
                (results_table["gamma"] == gamma) & (results_table["strategy"] == strat)
            ]
            if sim_row.empty:
                continue
            sim = sim_row.iloc[0]
            rows.append({
                "gamma": gamma,
                "strategy": strat,
                "sim_mean_profit": sim["mean_profit"],
                "pub_mean_profit": pub.get("mean_profit", np.nan),
                "sim_std_profit": sim["std_profit"],
                "pub_std_profit": pub.get("std_profit", np.nan),
                "sim_mean_final_q": sim["mean_final_q"],
                "pub_mean_final_q": pub.get("mean_final_q", np.nan),
                "sim_std_final_q": sim["std_final_q"],
                "pub_std_final_q": pub.get("std_final_q", np.nan),
            })
    return pd.DataFrame(rows)


def variance_reduction_ratio(
    results_inv: List[PathResult],
    results_sym: List[PathResult],
) -> Dict[str, float]:
    """
    Compute variance reduction ratios: inventory vs symmetric strategy.

    VRR_profit = std(Pi_T^inv) / std(Pi_T^sym)
    VRR_inventory = std(q_T^inv) / std(q_T^sym)

    Values < 1 indicate the inventory strategy has lower variance.

    Parameters
    ----------
    results_inv : List[PathResult]
        Inventory strategy results.
    results_sym : List[PathResult]
        Symmetric strategy results.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'vrr_profit' and 'vrr_inventory'.
    """
    inv_arrays = extract_arrays(results_inv)
    sym_arrays = extract_arrays(results_sym)

    std_pnl_inv = np.std(inv_arrays["pnl"], ddof=1)
    std_pnl_sym = np.std(sym_arrays["pnl"], ddof=1)
    std_q_inv = np.std(inv_arrays["inventory"], ddof=1)
    std_q_sym = np.std(sym_arrays["inventory"], ddof=1)

    return {
        "vrr_profit": std_pnl_inv / std_pnl_sym if std_pnl_sym > 0 else np.nan,
        "vrr_inventory": std_q_inv / std_q_sym if std_q_sym > 0 else np.nan,
    }
