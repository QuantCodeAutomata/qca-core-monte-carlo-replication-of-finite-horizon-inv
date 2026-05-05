"""
Visualization module for the finite-horizon market-making simulation.

Produces:
  - Illustrative path plots (mid-price, reservation price, bid/ask quotes)
  - Terminal P&L histograms
  - Reservation price grids (exp_3)
  - Spread comparison tables as figures
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from src.simulator import FullPath, PathResult
from src.metrics import extract_arrays

# Style settings
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQUARE = (8, 6)
FIGSIZE_TALL = (10, 8)


def plot_illustrative_path(
    full_path: FullPath,
    gamma: float,
    strategy: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot an illustrative simulation path showing mid-price, reservation price,
    and bid/ask quotes over time.

    Parameters
    ----------
    full_path : FullPath
        Full time-series path data.
    gamma : float
        Risk-aversion parameter (for title).
    strategy : str
        Strategy name (for title).
    save_path : str or None
        File path to save the figure. If None, figure is not saved.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    t = full_path.times

    # Top panel: prices and quotes
    ax1 = axes[0]
    ax1.plot(t, full_path.prices, color="black", linewidth=1.5, label="Mid-price $S_t$", zorder=3)
    ax1.plot(t, full_path.ask_quotes, color="red", linewidth=1.0, linestyle="--",
             label="Ask quote $p^a_t$", alpha=0.8)
    ax1.plot(t, full_path.bid_quotes, color="blue", linewidth=1.0, linestyle="--",
             label="Bid quote $p^b_t$", alpha=0.8)

    if strategy == "inventory" and not np.all(np.isnan(full_path.reservation)):
        ax1.plot(t, full_path.reservation, color="green", linewidth=1.2, linestyle="-.",
                 label="Reservation price $r_t$", alpha=0.9)

    ax1.set_ylabel("Price", fontsize=12)
    ax1.set_title(
        f"Illustrative Path — {strategy.capitalize()} Strategy, $\\gamma={gamma}$",
        fontsize=13,
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.fill_between(t, full_path.bid_quotes, full_path.ask_quotes,
                     alpha=0.1, color="gray", label="Spread")

    # Bottom panel: inventory
    ax2 = axes[1]
    ax2.plot(t, full_path.inventories, color="purple", linewidth=1.2, label="Inventory $q_t$")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Time $t$", fontsize=12)
    ax2.set_ylabel("Inventory $q_t$", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pnl_histograms(
    results_inv: List[PathResult],
    results_sym: List[PathResult],
    gamma: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot overlapping histograms of terminal P&L for inventory and symmetric strategies.

    Parameters
    ----------
    results_inv : List[PathResult]
        Inventory strategy results.
    results_sym : List[PathResult]
        Symmetric strategy results.
    gamma : float
        Risk-aversion parameter (for title).
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    inv_pnl = extract_arrays(results_inv)["pnl"]
    sym_pnl = extract_arrays(results_sym)["pnl"]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    bins = np.linspace(
        min(inv_pnl.min(), sym_pnl.min()) - 5,
        max(inv_pnl.max(), sym_pnl.max()) + 5,
        50,
    )

    ax.hist(inv_pnl, bins=bins, alpha=0.6, color="steelblue",
            label=f"Inventory (mean={np.mean(inv_pnl):.2f}, std={np.std(inv_pnl, ddof=1):.2f})",
            density=True)
    ax.hist(sym_pnl, bins=bins, alpha=0.6, color="tomato",
            label=f"Symmetric (mean={np.mean(sym_pnl):.2f}, std={np.std(sym_pnl, ddof=1):.2f})",
            density=True)

    ax.axvline(np.mean(inv_pnl), color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(sym_pnl), color="tomato", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Terminal P&L $\\Pi_T$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Terminal P&L Distribution — $\\gamma={gamma}$", fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_inventory_histograms(
    results_inv: List[PathResult],
    results_sym: List[PathResult],
    gamma: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot overlapping histograms of final inventory for both strategies.

    Parameters
    ----------
    results_inv : List[PathResult]
        Inventory strategy results.
    results_sym : List[PathResult]
        Symmetric strategy results.
    gamma : float
        Risk-aversion parameter (for title).
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    inv_q = extract_arrays(results_inv)["inventory"]
    sym_q = extract_arrays(results_sym)["inventory"]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    all_q = np.concatenate([inv_q, sym_q])
    bins = np.arange(all_q.min() - 1, all_q.max() + 2, 1)

    ax.hist(inv_q, bins=bins, alpha=0.6, color="steelblue",
            label=f"Inventory (mean={np.mean(inv_q):.2f}, std={np.std(inv_q, ddof=1):.2f})",
            density=True)
    ax.hist(sym_q, bins=bins, alpha=0.6, color="tomato",
            label=f"Symmetric (mean={np.mean(sym_q):.2f}, std={np.std(sym_q, ddof=1):.2f})",
            density=True)

    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Final Inventory $q_T$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Final Inventory Distribution — $\\gamma={gamma}$", fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_gamma_comparison(
    results_table: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot summary statistics (mean profit, std profit, std inventory) across gamma values.

    Parameters
    ----------
    results_table : pd.DataFrame
        Output of build_results_table.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("std_profit", "Std Dev of Terminal P&L"),
        ("mean_profit", "Mean Terminal P&L"),
        ("std_final_q", "Std Dev of Final Inventory"),
    ]

    colors = {"inventory": "steelblue", "symmetric": "tomato"}

    for ax, (metric, title) in zip(axes, metrics):
        for strat in ["inventory", "symmetric"]:
            subset = results_table[results_table["strategy"] == strat].sort_values("gamma")
            ax.plot(
                subset["gamma"].values,
                subset[metric].values,
                marker="o",
                label=strat.capitalize(),
                color=colors[strat],
                linewidth=2,
            )
        ax.set_xlabel("$\\gamma$", fontsize=12)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xscale("log")
        ax.legend(fontsize=10)

    plt.suptitle("Strategy Comparison Across Risk-Aversion Levels", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_reservation_price_grid(
    q_grid: np.ndarray,
    t_grid: np.ndarray,
    s: float,
    gamma: float,
    sigma: float,
    T: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot finite-horizon reservation mid-price minus mid-price (r - S)
    as a function of inventory q at several time points.

    Parameters
    ----------
    q_grid : ndarray
        Inventory grid.
    t_grid : ndarray
        Time grid.
    s : float
        Mid-price (fixed).
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    from src.reservation import reservation_mid_finite

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    colors = plt.cm.viridis(np.linspace(0, 1, len(t_grid)))

    for t, color in zip(t_grid, colors):
        r = reservation_mid_finite(s, q_grid, t, gamma, sigma, T)
        ax.plot(q_grid, r - s, label=f"$t={t:.2f}$", color=color, linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Inventory $q$", fontsize=12)
    ax.set_ylabel("$r(s,q,t) - s$", fontsize=12)
    ax.set_title(
        f"Reservation Mid-Price Shift vs Inventory\n$\\gamma={gamma}$, $\\sigma={sigma}$, $T={T}$",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_reservation_bid_ask(
    q_grid: np.ndarray,
    t: float,
    s: float,
    gamma: float,
    sigma: float,
    T: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot finite-horizon reservation bid and ask prices versus inventory.

    Parameters
    ----------
    q_grid : ndarray
        Inventory grid.
    t : float
        Current time.
    s : float
        Mid-price.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    from src.reservation import (
        reservation_bid_finite,
        reservation_ask_finite,
        reservation_mid_finite,
    )

    r_b = reservation_bid_finite(s, q_grid, t, gamma, sigma, T)
    r_a = reservation_ask_finite(s, q_grid, t, gamma, sigma, T)
    r_m = reservation_mid_finite(s, q_grid, t, gamma, sigma, T)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    ax.plot(q_grid, r_a, color="red", linewidth=1.8, label="Reservation Ask $r^a$")
    ax.plot(q_grid, r_m, color="green", linewidth=1.8, linestyle="-.", label="Reservation Mid $r$")
    ax.plot(q_grid, r_b, color="blue", linewidth=1.8, label="Reservation Bid $r^b$")
    ax.axhline(s, color="black", linewidth=0.8, linestyle=":", label="Mid-price $S$")

    ax.set_xlabel("Inventory $q$", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.set_title(
        f"Finite-Horizon Reservation Prices\n$t={t}$, $\\gamma={gamma}$, $\\sigma={sigma}$, $T={T}$",
        fontsize=12,
    )
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_infinite_horizon_reservation(
    q_grid: np.ndarray,
    s: float,
    gamma: float,
    sigma: float,
    q_max_list: List[int],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot infinite-horizon stationary reservation bid and ask prices
    for several omega values (derived from q_max).

    Parameters
    ----------
    q_grid : ndarray
        Inventory grid.
    s : float
        Mid-price.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    q_max_list : List[int]
        List of q_max values to derive omega from.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    from src.reservation import (
        reservation_bid_infinite,
        reservation_ask_infinite,
        admissibility_omega,
    )

    fig, axes = plt.subplots(1, len(q_max_list), figsize=(5 * len(q_max_list), 5), sharey=True)
    if len(q_max_list) == 1:
        axes = [axes]

    for ax, q_max in zip(axes, q_max_list):
        omega = admissibility_omega(gamma, sigma, q_max)
        r_b = reservation_bid_infinite(s, q_grid, gamma, sigma, omega)
        r_a = reservation_ask_infinite(s, q_grid, gamma, sigma, omega)

        ax.plot(q_grid, r_a, color="red", linewidth=1.8, label="$\\bar{r}^a$")
        ax.plot(q_grid, r_b, color="blue", linewidth=1.8, label="$\\bar{r}^b$")
        ax.axhline(s, color="black", linewidth=0.8, linestyle=":", label="$S$")
        ax.set_xlabel("Inventory $q$", fontsize=11)
        ax.set_title(f"$q_{{max}}={q_max}$, $\\omega={omega:.4f}$", fontsize=11)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Price", fontsize=12)
    fig.suptitle(
        f"Infinite-Horizon Reservation Prices — $\\gamma={gamma}$, $\\sigma={sigma}$",
        fontsize=13,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_spread_comparison(
    spread_table: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot spread comparison: text-formula vs table-matching vs published.

    Parameters
    ----------
    spread_table : pd.DataFrame
        Output of build_spread_comparison_table.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(spread_table))
    width = 0.25

    ax.bar(x - width, spread_table["text_formula_spread"], width, label="Text Formula", color="steelblue")
    ax.bar(x, spread_table["table_matching_spread"], width, label="Table Matching", color="orange")
    ax.bar(x + width, spread_table["published_spread"], width, label="Published", color="green", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\gamma={g}$" for g in spread_table["gamma"]], fontsize=11)
    ax.set_ylabel("Spread $\\Phi$", fontsize=12)
    ax.set_title("Spread Formula Comparison: Text vs Table-Matching vs Published", fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sensitivity_heatmap(
    sensitivity_table: pd.DataFrame,
    metric: str = "std_profit",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a heatmap of a metric across gamma, strategy, and spread_mode.

    Parameters
    ----------
    sensitivity_table : pd.DataFrame
        Output of build_sensitivity_table.
    metric : str
        Column name to visualize.
    save_path : str or None
        File path to save the figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    pivot = sensitivity_table.pivot_table(
        index=["gamma", "strategy"],
        columns="spread_mode",
        values=metric,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Sensitivity of {metric} to Spread Mode", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
