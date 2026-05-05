"""
Monte Carlo simulator for the finite-horizon market-making model.

Implements discrete-time path simulation following the paper's specification:
  - Mid-price updated as S_{t+dt} = S_t ± sigma*sqrt(dt) with equal probability
  - Bernoulli fill draws with probabilities lambda*dt
  - Both sides may fill in the same time step (documented assumption)
  - Diagnostics for lambda*dt > 1 and negative deltas

# Custom — Context7 found no library equivalent (paper simulation design)
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config import SimParams
from src.strategies import (
    inventory_quotes,
    symmetric_quotes,
    fill_intensity,
)


@dataclass
class PathResult:
    """
    Container for a single simulated path's terminal values and diagnostics.

    Attributes
    ----------
    terminal_pnl : float
        Pi_T = X_T + q_T * S_T
    terminal_inventory : float
        q_T
    terminal_cash : float
        X_T
    terminal_price : float
        S_T
    n_lambda_violations : int
        Number of time steps where lambda*dt > 1.
    n_negative_delta_a : int
        Number of steps where ask distance delta_a < 0.
    n_negative_delta_b : int
        Number of steps where bid distance delta_b < 0.
    n_simultaneous_fills : int
        Number of steps where both bid and ask filled simultaneously.
    """

    terminal_pnl: float = 0.0
    terminal_inventory: float = 0.0
    terminal_cash: float = 0.0
    terminal_price: float = 0.0
    n_lambda_violations: int = 0
    n_negative_delta_a: int = 0
    n_negative_delta_b: int = 0
    n_simultaneous_fills: int = 0


@dataclass
class FullPath:
    """
    Container for a complete time-series path (used for illustration plots).

    Attributes
    ----------
    times : ndarray
        Time grid [0, dt, 2*dt, ..., T].
    prices : ndarray
        Mid-price S_t at each time step.
    reservation : ndarray
        Reservation price r_t (NaN for symmetric strategy).
    ask_quotes : ndarray
        Ask quote p^a_t at each time step.
    bid_quotes : ndarray
        Bid quote p^b_t at each time step.
    inventories : ndarray
        Inventory q_t at each time step.
    cash : ndarray
        Cash X_t at each time step.
    """

    times: np.ndarray = field(default_factory=lambda: np.array([]))
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    reservation: np.ndarray = field(default_factory=lambda: np.array([]))
    ask_quotes: np.ndarray = field(default_factory=lambda: np.array([]))
    bid_quotes: np.ndarray = field(default_factory=lambda: np.array([]))
    inventories: np.ndarray = field(default_factory=lambda: np.array([]))
    cash: np.ndarray = field(default_factory=lambda: np.array([]))


def simulate_path(
    params: SimParams,
    strategy: str,
    rng: np.random.Generator,
    store_full_path: bool = False,
) -> Tuple[PathResult, Optional[FullPath]]:
    """
    Simulate a single market-making path.

    Mid-price dynamics (paper's discrete specification):
        S_{t+dt} = S_t ± sigma*sqrt(dt)  with equal probability

    Fill simulation:
        Bernoulli(lambda_a * dt) for ask fill
        Bernoulli(lambda_b * dt) for bid fill
        Both may fire in the same step (documented implementation assumption).

    Parameters
    ----------
    params : SimParams
        Simulation parameters.
    strategy : str
        'inventory' for Avellaneda-Stoikov or 'symmetric' for benchmark.
    rng : np.random.Generator
        Random number generator (caller controls seed).
    store_full_path : bool
        If True, record full time-series arrays.

    Returns
    -------
    Tuple[PathResult, Optional[FullPath]]
        Terminal result and optionally the full path.
    """
    n = params.n_steps
    dt = params.dt
    sqrt_dt = np.sqrt(dt)

    # Pre-generate all random numbers for this path
    # Price signs: +1 or -1 with equal probability
    price_signs = rng.choice([-1.0, 1.0], size=n)
    # Uniform draws for Bernoulli fill decisions
    u_ask = rng.random(size=n)
    u_bid = rng.random(size=n)

    # State variables
    S = float(params.S0)
    q = float(params.q0)
    X = float(params.X0)

    # Diagnostics
    n_lambda_viol = 0
    n_neg_da = 0
    n_neg_db = 0
    n_simult = 0

    # Full path storage
    if store_full_path:
        times_arr = np.zeros(n + 1)
        prices_arr = np.zeros(n + 1)
        res_arr = np.full(n + 1, np.nan)
        ask_arr = np.zeros(n + 1)
        bid_arr = np.zeros(n + 1)
        inv_arr = np.zeros(n + 1)
        cash_arr = np.zeros(n + 1)

        times_arr[0] = 0.0
        prices_arr[0] = S
        inv_arr[0] = q
        cash_arr[0] = X

    for i in range(n):
        t = i * dt

        # Compute quotes based on strategy
        if strategy == "inventory":
            r_t, p_ask, p_bid, delta_a, delta_b = inventory_quotes(
                S, q, t, params.gamma, params.sigma, params.k, params.T, params.spread_mode
            )
        elif strategy == "symmetric":
            p_ask, p_bid, delta_a, delta_b = symmetric_quotes(
                S, q, t, params.gamma, params.sigma, params.k, params.T, params.spread_mode
            )
            r_t = np.nan
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'inventory' or 'symmetric'.")

        # Diagnostics: negative deltas
        if delta_a < 0:
            n_neg_da += 1
        if delta_b < 0:
            n_neg_db += 1

        # Compute fill intensities
        lam_a = fill_intensity(delta_a, params.A, params.k)
        lam_b = fill_intensity(delta_b, params.A, params.k)

        # Check lambda*dt validity
        prob_a = lam_a * dt
        prob_b = lam_b * dt

        if prob_a > 1.0 or prob_b > 1.0:
            n_lambda_viol += 1
            if params.strict_lambda:
                raise RuntimeError(
                    f"lambda*dt > 1 at step {i}: prob_a={prob_a:.4f}, prob_b={prob_b:.4f}. "
                    f"Reduce dt or clip in non-strict mode."
                )
            # Clip to [0, 1] in non-strict mode
            prob_a = min(prob_a, 1.0)
            prob_b = min(prob_b, 1.0)

        # Bernoulli fill draws
        ask_fills = 1 if u_ask[i] < prob_a else 0
        bid_fills = 1 if u_bid[i] < prob_b else 0

        # Simultaneous fill diagnostic
        if ask_fills and bid_fills:
            n_simult += 1

        # Update cash and inventory
        # Ask fill: sell 1 share -> inventory -1, cash +p_ask
        if ask_fills:
            q -= 1.0
            X += p_ask

        # Bid fill: buy 1 share -> inventory +1, cash -p_bid
        if bid_fills:
            q += 1.0
            X -= p_bid

        # Store pre-price-update state for plotting
        if store_full_path:
            times_arr[i + 1] = (i + 1) * dt
            inv_arr[i + 1] = q
            cash_arr[i + 1] = X
            ask_arr[i] = p_ask
            bid_arr[i] = p_bid
            if strategy == "inventory":
                res_arr[i] = r_t

        # Update mid-price: S_{t+dt} = S_t ± sigma*sqrt(dt)
        S += params.sigma * sqrt_dt * price_signs[i]

        if store_full_path:
            prices_arr[i + 1] = S

    # Terminal values
    terminal_pnl = X + q * S

    result = PathResult(
        terminal_pnl=terminal_pnl,
        terminal_inventory=q,
        terminal_cash=X,
        terminal_price=S,
        n_lambda_violations=n_lambda_viol,
        n_negative_delta_a=n_neg_da,
        n_negative_delta_b=n_neg_db,
        n_simultaneous_fills=n_simult,
    )

    if store_full_path:
        # Fill last quote values
        ask_arr[n] = ask_arr[n - 1]
        bid_arr[n] = bid_arr[n - 1]
        if strategy == "inventory":
            res_arr[n] = res_arr[n - 1]

        full = FullPath(
            times=times_arr,
            prices=prices_arr,
            reservation=res_arr,
            ask_quotes=ask_arr,
            bid_quotes=bid_arr,
            inventories=inv_arr,
            cash=cash_arr,
        )
        return result, full

    return result, None


def run_monte_carlo(
    params: SimParams,
    strategy: str,
    save_path_index: Optional[int] = 0,
) -> Tuple[List[PathResult], Optional[FullPath]]:
    """
    Run N_paths Monte Carlo simulations for a given strategy.

    Parameters
    ----------
    params : SimParams
        Simulation parameters including N_paths and seed.
    strategy : str
        'inventory' or 'symmetric'.
    save_path_index : int or None
        Index of the path to save as a full time-series (for plotting).
        Set to None to skip full-path storage.

    Returns
    -------
    Tuple[List[PathResult], Optional[FullPath]]
        List of terminal results and optionally one full path.
    """
    rng = np.random.default_rng(params.seed)
    results: List[PathResult] = []
    saved_path: Optional[FullPath] = None

    total_lambda_viol = 0
    total_neg_da = 0
    total_neg_db = 0
    total_simult = 0

    for i in range(params.N_paths):
        store = (save_path_index is not None) and (i == save_path_index)
        result, full = simulate_path(params, strategy, rng, store_full_path=store)
        results.append(result)

        total_lambda_viol += result.n_lambda_violations
        total_neg_da += result.n_negative_delta_a
        total_neg_db += result.n_negative_delta_b
        total_simult += result.n_simultaneous_fills

        if store and full is not None:
            saved_path = full

    # Log aggregate diagnostics
    if total_lambda_viol > 0:
        warnings.warn(
            f"[{strategy}] lambda*dt > 1 occurred {total_lambda_viol} times across all paths. "
            f"Probabilities were clipped to 1.0 (non-strict mode).",
            RuntimeWarning,
        )
    if total_neg_da > 0 or total_neg_db > 0:
        warnings.warn(
            f"[{strategy}] Negative quote distances: delta_a<0 in {total_neg_da} steps, "
            f"delta_b<0 in {total_neg_db} steps. lambda > A in those steps.",
            RuntimeWarning,
        )
    if total_simult > 0:
        # Simultaneous fills are expected and documented
        pass

    return results, saved_path


def run_monte_carlo_common_rng(
    params_mode1: SimParams,
    params_mode2: SimParams,
    strategy: str,
) -> Tuple[List[PathResult], List[PathResult]]:
    """
    Run Monte Carlo for two spread modes using common random numbers (CRN).

    CRN ensures that only the spread formula differs between the two runs,
    making sensitivity attribution cleaner.

    Parameters
    ----------
    params_mode1 : SimParams
        Parameters for spread mode 1 (text_formula).
    params_mode2 : SimParams
        Parameters for spread mode 2 (table_matching).
    strategy : str
        'inventory' or 'symmetric'.

    Returns
    -------
    Tuple[List[PathResult], List[PathResult]]
        Results for mode1 and mode2 respectively.
    """
    assert params_mode1.seed == params_mode2.seed, "Seeds must match for CRN."
    assert params_mode1.N_paths == params_mode2.N_paths, "N_paths must match for CRN."

    rng1 = np.random.default_rng(params_mode1.seed)
    rng2 = np.random.default_rng(params_mode2.seed)

    results1: List[PathResult] = []
    results2: List[PathResult] = []

    for _ in range(params_mode1.N_paths):
        r1, _ = simulate_path(params_mode1, strategy, rng1)
        r2, _ = simulate_path(params_mode2, strategy, rng2)
        results1.append(r1)
        results2.append(r2)

    return results1, results2
