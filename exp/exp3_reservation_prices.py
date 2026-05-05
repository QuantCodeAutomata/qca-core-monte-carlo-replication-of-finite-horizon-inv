"""
Experiment 3: Independent implementation and validation of reservation prices
and stationary infinite-horizon quoting formulas.

Validates:
  - Finite-horizon utility-indifference reservation bid/ask prices
  - Finite-horizon reservation mid-price r(s,q,t) = s - q*gamma*sigma^2*(T-t)
  - Stationary infinite-horizon reservation bid/ask formulas
  - Boundary conditions and comparative statics
  - Admissibility conditions for infinite-horizon formulas

Calls functions from src/reservation.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from src.plots import (
    plot_reservation_price_grid,
    plot_reservation_bid_ask,
    plot_infinite_horizon_reservation,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_exp3(
    results_dir: str = RESULTS_DIR,
    verbose: bool = True,
) -> dict:
    """
    Run Experiment 3: Reservation price theory validation.

    Parameters
    ----------
    results_dir : str
        Directory to save results.
    verbose : bool
        Print validation results.

    Returns
    -------
    dict
        Dictionary of validation results and tables.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Parameter grids
    s = 100.0
    T = 1.0
    sigma = 2.0
    gamma_list = [0.01, 0.1, 0.5]
    q_grid = np.arange(-10, 11, dtype=float)
    t_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    output = {}

    # -----------------------------------------------------------------------
    # Part 1: Finite-horizon reservation prices
    # -----------------------------------------------------------------------
    if verbose:
        print("\n" + "="*70)
        print("PART 1: FINITE-HORIZON RESERVATION PRICES")
        print("="*70)

    finite_tables = {}
    for gamma in gamma_list:
        rows = []
        for t in t_grid:
            for q in q_grid:
                r_b = reservation_bid_finite(s, q, t, gamma, sigma, T)
                r_a = reservation_ask_finite(s, q, t, gamma, sigma, T)
                r_m = reservation_mid_finite(s, q, t, gamma, sigma, T)
                rows.append({
                    "gamma": gamma,
                    "t": t,
                    "q": q,
                    "r_bid": r_b,
                    "r_ask": r_a,
                    "r_mid": r_m,
                    "r_mid_minus_s": r_m - s,
                })
        finite_tables[gamma] = pd.DataFrame(rows)

    # Verify boundary conditions
    boundary_checks = _verify_boundary_conditions(s, q_grid, t_grid, gamma_list, sigma, T, verbose)
    output["boundary_checks"] = boundary_checks

    # Verify indifference conditions
    indiff_checks = _verify_indifference_conditions(s, q_grid, t_grid, gamma_list, sigma, T, verbose)
    output["indifference_checks"] = indiff_checks

    # Verify comparative statics
    comp_statics = _verify_comparative_statics(s, q_grid, t_grid, gamma_list, sigma, T, verbose)
    output["comparative_statics"] = comp_statics

    # Save finite-horizon tables
    for gamma in gamma_list:
        finite_tables[gamma].to_csv(
            os.path.join(results_dir, f"exp3_finite_reservation_gamma{gamma}.csv"),
            index=False,
        )

    # -----------------------------------------------------------------------
    # Part 2: Infinite-horizon stationary reservation prices
    # -----------------------------------------------------------------------
    if verbose:
        print("\n" + "="*70)
        print("PART 2: INFINITE-HORIZON STATIONARY RESERVATION PRICES")
        print("="*70)

    q_max_list = [5, 10]
    infinite_tables = {}

    for gamma in gamma_list:
        rows = []
        for q_max in q_max_list:
            omega = admissibility_omega(gamma, sigma, q_max)
            admissible_q = q_grid[check_admissibility(q_grid, gamma, sigma, omega)]

            if verbose:
                print(f"\n  gamma={gamma}, q_max={q_max}, omega={omega:.6f}")
                print(f"  Admissible q range: [{admissible_q.min():.0f}, {admissible_q.max():.0f}]")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r_b_inf = reservation_bid_infinite(s, admissible_q, gamma, sigma, omega)
                r_a_inf = reservation_ask_infinite(s, admissible_q, gamma, sigma, omega)

            for q_val, rb, ra in zip(admissible_q, r_b_inf, r_a_inf):
                rows.append({
                    "gamma": gamma,
                    "q_max": q_max,
                    "omega": omega,
                    "q": q_val,
                    "r_bid_inf": rb,
                    "r_ask_inf": ra,
                    "r_bid_minus_s": rb - s,
                    "r_ask_minus_s": ra - s,
                })

        infinite_tables[gamma] = pd.DataFrame(rows)
        infinite_tables[gamma].to_csv(
            os.path.join(results_dir, f"exp3_infinite_reservation_gamma{gamma}.csv"),
            index=False,
        )

    output["finite_tables"] = finite_tables
    output["infinite_tables"] = infinite_tables

    # -----------------------------------------------------------------------
    # Part 3: Admissibility analysis
    # -----------------------------------------------------------------------
    admissibility_table = _build_admissibility_table(gamma_list, sigma, q_max_list, verbose)
    admissibility_table.to_csv(
        os.path.join(results_dir, "exp3_admissibility.csv"), index=False
    )
    output["admissibility_table"] = admissibility_table

    # -----------------------------------------------------------------------
    # Generate plots
    # -----------------------------------------------------------------------
    _generate_plots(s, q_grid, t_grid, gamma_list, q_max_list, sigma, T, results_dir, verbose)

    return output


def _verify_boundary_conditions(
    s: float,
    q_grid: np.ndarray,
    t_grid: np.ndarray,
    gamma_list: list,
    sigma: float,
    T: float,
    verbose: bool,
) -> dict:
    """
    Verify boundary conditions:
    1. r(s, 0, t) = s for all t
    2. r(s, q, T) = s for all q
    3. r^b(s, 0, t) = s - gamma*sigma^2*(T-t)/2
    4. r^a(s, 0, t) = s + gamma*sigma^2*(T-t)/2
    """
    checks = {}

    for gamma in gamma_list:
        # BC1: q=0 -> r_mid = s
        r_mid_q0 = reservation_mid_finite(s, 0.0, t_grid, gamma, sigma, T)
        bc1_pass = np.allclose(r_mid_q0, s, atol=1e-12)

        # BC2: t=T -> r_mid = s for all q
        r_mid_tT = reservation_mid_finite(s, q_grid, T, gamma, sigma, T)
        bc2_pass = np.allclose(r_mid_tT, s, atol=1e-12)

        # BC3: q=0, r_bid = s - gamma*sigma^2*(T-t)/2
        r_bid_q0 = reservation_bid_finite(s, 0.0, t_grid, gamma, sigma, T)
        expected_bid_q0 = s - gamma * sigma**2 * (T - t_grid) / 2.0
        bc3_pass = np.allclose(r_bid_q0, expected_bid_q0, atol=1e-12)

        # BC4: q=0, r_ask = s + gamma*sigma^2*(T-t)/2
        r_ask_q0 = reservation_ask_finite(s, 0.0, t_grid, gamma, sigma, T)
        expected_ask_q0 = s + gamma * sigma**2 * (T - t_grid) / 2.0
        bc4_pass = np.allclose(r_ask_q0, expected_ask_q0, atol=1e-12)

        checks[gamma] = {
            "bc1_q0_r_mid_equals_s": bc1_pass,
            "bc2_tT_r_mid_equals_s": bc2_pass,
            "bc3_q0_r_bid_correct": bc3_pass,
            "bc4_q0_r_ask_correct": bc4_pass,
        }

        if verbose:
            print(f"\n  gamma={gamma} boundary checks:")
            for k, v in checks[gamma].items():
                status = "PASS" if v else "FAIL"
                print(f"    {k}: {status}")

    return checks


def _verify_indifference_conditions(
    s: float,
    q_grid: np.ndarray,
    t_grid: np.ndarray,
    gamma_list: list,
    sigma: float,
    T: float,
    verbose: bool,
) -> dict:
    """
    Verify utility indifference conditions numerically.
    """
    checks = {}
    x = 0.0  # Cash position for testing

    for gamma in gamma_list:
        bid_passes = []
        ask_passes = []

        for t in t_grid[:-1]:  # Exclude t=T (degenerate)
            for q in q_grid:
                bid_ok = verify_indifference_bid(x, s, float(q), t, gamma, sigma, T)
                ask_ok = verify_indifference_ask(x, s, float(q), t, gamma, sigma, T)
                bid_passes.append(bid_ok)
                ask_passes.append(ask_ok)

        checks[gamma] = {
            "bid_indifference_all_pass": all(bid_passes),
            "ask_indifference_all_pass": all(ask_passes),
            "n_bid_tests": len(bid_passes),
            "n_ask_tests": len(ask_passes),
        }

        if verbose:
            print(f"\n  gamma={gamma} indifference checks:")
            print(f"    Bid: {sum(bid_passes)}/{len(bid_passes)} passed")
            print(f"    Ask: {sum(ask_passes)}/{len(ask_passes)} passed")

    return checks


def _verify_comparative_statics(
    s: float,
    q_grid: np.ndarray,
    t_grid: np.ndarray,
    gamma_list: list,
    sigma: float,
    T: float,
    verbose: bool,
) -> dict:
    """
    Verify comparative statics:
    1. Long inventory (q > 0) -> r_mid < s
    2. Short inventory (q < 0) -> r_mid > s
    3. Larger gamma -> stronger inventory adjustment
    4. Larger T-t -> stronger inventory adjustment
    """
    checks = {}

    t_mid = 0.5  # Use t=0.5 for comparative statics

    for gamma in gamma_list:
        # CS1: q > 0 -> r_mid < s
        q_pos = q_grid[q_grid > 0]
        r_mid_pos = reservation_mid_finite(s, q_pos, t_mid, gamma, sigma, T)
        cs1_pass = np.all(r_mid_pos < s)

        # CS2: q < 0 -> r_mid > s
        q_neg = q_grid[q_grid < 0]
        r_mid_neg = reservation_mid_finite(s, q_neg, t_mid, gamma, sigma, T)
        cs2_pass = np.all(r_mid_neg > s)

        # CS3: r_mid is decreasing in q (affine with negative slope)
        r_mid_all = reservation_mid_finite(s, q_grid, t_mid, gamma, sigma, T)
        cs3_pass = np.all(np.diff(r_mid_all) < 0)

        # CS4: Larger T-t -> larger |r_mid - s| for fixed q > 0
        q_test = 5.0
        r_early = reservation_mid_finite(s, q_test, 0.0, gamma, sigma, T)   # t=0, tau=T
        r_late = reservation_mid_finite(s, q_test, 0.9, gamma, sigma, T)    # t=0.9, tau=0.1
        cs4_pass = abs(r_early - s) > abs(r_late - s)

        checks[gamma] = {
            "cs1_long_inv_r_below_s": cs1_pass,
            "cs2_short_inv_r_above_s": cs2_pass,
            "cs3_r_decreasing_in_q": cs3_pass,
            "cs4_larger_tau_larger_shift": cs4_pass,
        }

        if verbose:
            print(f"\n  gamma={gamma} comparative statics:")
            for k, v in checks[gamma].items():
                status = "PASS" if v else "FAIL"
                print(f"    {k}: {status}")

    # CS5: Larger gamma -> larger |r_mid - s| for fixed q, t
    q_test = 5.0
    t_test = 0.5
    shifts = {
        gamma: abs(reservation_mid_finite(s, q_test, t_test, gamma, sigma, T) - s)
        for gamma in gamma_list
    }
    sorted_gammas = sorted(gamma_list)
    cs5_pass = all(
        shifts[sorted_gammas[i]] < shifts[sorted_gammas[i + 1]]
        for i in range(len(sorted_gammas) - 1)
    )
    checks["cs5_larger_gamma_larger_shift"] = cs5_pass

    if verbose:
        print(f"\n  Cross-gamma: larger gamma -> larger shift: {'PASS' if cs5_pass else 'FAIL'}")
        for gamma in sorted_gammas:
            print(f"    gamma={gamma}: |r-s| = {shifts[gamma]:.4f}")

    return checks


def _build_admissibility_table(
    gamma_list: list,
    sigma: float,
    q_max_list: list,
    verbose: bool,
) -> pd.DataFrame:
    """Build admissibility analysis table for infinite-horizon formulas."""
    rows = []
    for gamma in gamma_list:
        for q_max in q_max_list:
            omega = admissibility_omega(gamma, sigma, q_max)
            # Check admissibility for q in [-q_max, q_max]
            q_range = np.arange(-q_max, q_max + 1, dtype=float)
            admissible = check_admissibility(q_range, gamma, sigma, omega)
            rows.append({
                "gamma": gamma,
                "q_max": q_max,
                "omega": omega,
                "all_admissible": bool(np.all(admissible)),
                "n_admissible": int(np.sum(admissible)),
                "n_total": len(q_range),
            })

    df = pd.DataFrame(rows)

    if verbose:
        print("\n" + "="*70)
        print("ADMISSIBILITY TABLE (Infinite-Horizon)")
        print("="*70)
        print(df.to_string(index=False, float_format="{:.6f}".format))

    return df


def _generate_plots(
    s: float,
    q_grid: np.ndarray,
    t_grid: np.ndarray,
    gamma_list: list,
    q_max_list: list,
    sigma: float,
    T: float,
    results_dir: str,
    verbose: bool,
) -> None:
    """Generate and save all plots for Experiment 3."""
    import matplotlib.pyplot as plt

    # 1. Reservation mid-price grid for each gamma
    for gamma in gamma_list:
        fig = plot_reservation_price_grid(
            q_grid, t_grid, s, gamma, sigma, T,
            save_path=os.path.join(results_dir, f"exp3_res_mid_grid_gamma{gamma}.png"),
        )
        plt.close(fig)

    # 2. Reservation bid/ask at t=0 for each gamma
    for gamma in gamma_list:
        fig = plot_reservation_bid_ask(
            q_grid, t=0.0, s=s, gamma=gamma, sigma=sigma, T=T,
            save_path=os.path.join(results_dir, f"exp3_res_bid_ask_t0_gamma{gamma}.png"),
        )
        plt.close(fig)

    # 3. Infinite-horizon reservation prices
    for gamma in gamma_list:
        fig = plot_infinite_horizon_reservation(
            q_grid, s, gamma, sigma, q_max_list,
            save_path=os.path.join(results_dir, f"exp3_inf_horizon_gamma{gamma}.png"),
        )
        plt.close(fig)

    if verbose:
        print(f"\nAll Exp3 plots saved to {results_dir}")


if __name__ == "__main__":
    print("Running Experiment 3: Reservation Price Theory Validation")
    output = run_exp3(verbose=True)

    # Summary of all checks
    print("\n" + "="*70)
    print("SUMMARY OF ALL VALIDATION CHECKS")
    print("="*70)

    all_pass = True
    for gamma in [0.01, 0.1, 0.5]:
        bc = output["boundary_checks"][gamma]
        ic = output["indifference_checks"][gamma]
        cs = output["comparative_statics"][gamma]

        gamma_pass = all(bc.values()) and all(ic.values()) and all(cs.values())
        all_pass = all_pass and gamma_pass
        status = "ALL PASS" if gamma_pass else "SOME FAIL"
        print(f"  gamma={gamma}: {status}")

    cs_cross = output["comparative_statics"].get("cs5_larger_gamma_larger_shift", False)
    all_pass = all_pass and cs_cross
    print(f"  Cross-gamma comparative statics: {'PASS' if cs_cross else 'FAIL'}")
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
