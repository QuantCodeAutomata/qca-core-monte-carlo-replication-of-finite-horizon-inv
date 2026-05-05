"""
Quoting strategies for the finite-horizon market-making simulation.

Implements:
  - Spread computation (text-formula and table-matching modes)
  - Inventory-based (Avellaneda-Stoikov) quoting rule
  - Symmetric benchmark quoting rule

# Custom — Context7 found no library equivalent (paper Eq. 3.x)
"""

import numpy as np
from typing import Tuple, Literal


SpreadMode = Literal["text_formula", "table_matching"]


# ---------------------------------------------------------------------------
# Spread computation
# ---------------------------------------------------------------------------

def compute_spread(
    t: float,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    mode: SpreadMode = "text_formula",
) -> float:
    """
    Compute the total bid-ask spread Phi used by both strategies.

    Two modes are supported to address the paper's formula ambiguity:

    text_formula (primary theoretical description):
        Phi(t) = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    table_matching (constant, matches published table values):
        Phi = (2/gamma) * ln(1 + gamma/k)

    Parameters
    ----------
    t : float
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Fill-intensity decay parameter.
    T : float
        Terminal horizon.
    mode : str
        'text_formula' or 'table_matching'.

    Returns
    -------
    float
        Total spread Phi.
    """
    constant_term = (2.0 / gamma) * np.log(1.0 + gamma / k)
    if mode == "text_formula":
        tau = T - t
        return gamma * sigma**2 * tau + constant_term
    elif mode == "table_matching":
        return constant_term
    else:
        raise ValueError(f"Unknown spread_mode: {mode!r}. Use 'text_formula' or 'table_matching'.")


def compute_initial_spread(
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    mode: SpreadMode = "text_formula",
) -> float:
    """
    Compute the spread at t=0 for comparison with published table values.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Fill-intensity decay parameter.
    T : float
        Terminal horizon.
    mode : str
        'text_formula' or 'table_matching'.

    Returns
    -------
    float
        Initial spread at t=0.
    """
    return compute_spread(0.0, gamma, sigma, k, T, mode)


# ---------------------------------------------------------------------------
# Inventory-based (Avellaneda-Stoikov) strategy
# ---------------------------------------------------------------------------

def inventory_quotes(
    S: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    mode: SpreadMode = "text_formula",
) -> Tuple[float, float, float, float, float]:
    """
    Compute inventory-based bid and ask quotes.

    Reservation price:
        r_t = S_t - q_t * gamma * sigma^2 * (T - t)

    Total spread:
        Phi(t) as per selected mode

    Quotes:
        p^a = r_t + Phi(t)/2
        p^b = r_t - Phi(t)/2

    Quote distances from mid-price:
        delta^a = p^a - S_t = ((1 - 2*q)*gamma*sigma^2*(T-t))/2 + (1/gamma)*ln(1+gamma/k)
        delta^b = S_t - p^b = ((1 + 2*q)*gamma*sigma^2*(T-t))/2 + (1/gamma)*ln(1+gamma/k)

    Parameters
    ----------
    S : float
        Current mid-price.
    q : float
        Current inventory.
    t : float
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Fill-intensity decay parameter.
    T : float
        Terminal horizon.
    mode : str
        Spread mode.

    Returns
    -------
    Tuple[float, float, float, float, float]
        (reservation_price, ask_quote, bid_quote, delta_a, delta_b)
        where delta_a = p^a - S and delta_b = S - p^b (distances from mid).
    """
    tau = T - t
    r_t = S - q * gamma * sigma**2 * tau
    phi = compute_spread(t, gamma, sigma, k, T, mode)

    p_ask = r_t + phi / 2.0
    p_bid = r_t - phi / 2.0

    delta_a = p_ask - S  # distance of ask above mid
    delta_b = S - p_bid  # distance of bid below mid

    return r_t, p_ask, p_bid, delta_a, delta_b


# ---------------------------------------------------------------------------
# Symmetric benchmark strategy
# ---------------------------------------------------------------------------

def symmetric_quotes(
    S: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    mode: SpreadMode = "text_formula",
) -> Tuple[float, float, float, float]:
    """
    Compute symmetric benchmark bid and ask quotes.

    Uses the same spread Phi(t) as the inventory strategy but centers
    around the current mid-price S_t rather than the reservation price.

    p^{a,sym} = S_t + Phi(t)/2
    p^{b,sym} = S_t - Phi(t)/2

    Quote distances are both equal to Phi(t)/2.

    Parameters
    ----------
    S : float
        Current mid-price.
    q : float
        Current inventory (unused, kept for API consistency).
    t : float
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Fill-intensity decay parameter.
    T : float
        Terminal horizon.
    mode : str
        Spread mode.

    Returns
    -------
    Tuple[float, float, float, float]
        (ask_quote, bid_quote, delta_a, delta_b)
        where delta_a = delta_b = Phi(t)/2.
    """
    phi = compute_spread(t, gamma, sigma, k, T, mode)
    half_phi = phi / 2.0

    p_ask = S + half_phi
    p_bid = S - half_phi

    delta_a = half_phi
    delta_b = half_phi

    return p_ask, p_bid, delta_a, delta_b


# ---------------------------------------------------------------------------
# Fill intensity
# ---------------------------------------------------------------------------

def fill_intensity(
    delta: float,
    A: float,
    k: float,
) -> float:
    """
    Compute the Poisson fill intensity for a given quote distance.

    lambda(delta) = A * exp(-k * delta)

    Intensities are decreasing in delta (larger spread -> fewer fills).
    Note: delta can be negative for large inventory, giving lambda > A.

    Parameters
    ----------
    delta : float
        Quote distance from mid-price.
    A : float
        Baseline intensity parameter.
    k : float
        Intensity decay parameter.

    Returns
    -------
    float
        Fill intensity (events per unit time).
    """
    return A * np.exp(-k * delta)
