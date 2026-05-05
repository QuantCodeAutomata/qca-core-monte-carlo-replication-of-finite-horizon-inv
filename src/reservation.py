"""
Reservation price module for the finite-horizon and infinite-horizon
Avellaneda-Stoikov market-making model.

Implements:
  - Finite-horizon utility-indifference reservation bid/ask prices
  - Finite-horizon reservation mid-price
  - Stationary infinite-horizon reservation bid/ask prices

All formulas are implemented from scratch following the paper's derivations.
# Custom — Context7 found no library equivalent (paper Eq. 3.x / 4.x)
"""

import numpy as np
import warnings
from typing import Union

ArrayLike = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Finite-horizon utility functions
# ---------------------------------------------------------------------------

def inactive_utility(
    x: float,
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
) -> float:
    """
    Closed-form utility value for an inactive trader (no quoting) holding
    q shares until terminal time T under exponential utility and driftless
    Brownian mid-price dynamics dS_u = sigma dW_u.

    v(x, s, q, t) = -exp(-gamma*x) * exp(-gamma*q*s)
                    * exp((gamma^2 * q^2 * sigma^2 * (T-t)) / 2)

    Parameters
    ----------
    x : float
        Current cash position.
    s : float
        Current mid-price.
    q : float
        Current inventory (shares held).
    t : float
        Current time.
    gamma : float
        Risk-aversion coefficient (> 0).
    sigma : float
        Mid-price volatility (> 0).
    T : float
        Terminal horizon.

    Returns
    -------
    float
        Utility value (always negative for exponential utility).
    """
    assert gamma > 0, "gamma must be positive"
    assert sigma > 0, "sigma must be positive"
    assert T >= t, "T must be >= t"

    tau = T - t
    return (
        -np.exp(-gamma * x)
        * np.exp(-gamma * q * s)
        * np.exp(0.5 * gamma**2 * q**2 * sigma**2 * tau)
    )


# ---------------------------------------------------------------------------
# Finite-horizon reservation prices
# ---------------------------------------------------------------------------

def reservation_bid_finite(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
) -> ArrayLike:
    """
    Finite-horizon utility-indifference reservation bid price.

    Solves v(x - r^b, s, q+1, t) = v(x, s, q, t) for r^b.

    r^b(s, q, t) = s + (-1 - 2*q) * gamma * sigma^2 * (T - t) / 2

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    t : float or ndarray
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.

    Returns
    -------
    float or ndarray
        Reservation bid price.
    """
    tau = T - t
    return s + (-1.0 - 2.0 * q) * gamma * sigma**2 * tau / 2.0


def reservation_ask_finite(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
) -> ArrayLike:
    """
    Finite-horizon utility-indifference reservation ask price.

    Solves v(x + r^a, s, q-1, t) = v(x, s, q, t) for r^a.

    r^a(s, q, t) = s + (1 - 2*q) * gamma * sigma^2 * (T - t) / 2

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    t : float or ndarray
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.

    Returns
    -------
    float or ndarray
        Reservation ask price.
    """
    tau = T - t
    return s + (1.0 - 2.0 * q) * gamma * sigma**2 * tau / 2.0


def reservation_mid_finite(
    s: ArrayLike,
    q: ArrayLike,
    t: ArrayLike,
    gamma: float,
    sigma: float,
    T: float,
) -> ArrayLike:
    """
    Finite-horizon reservation mid-price (average of bid and ask reservation prices).

    r(s, q, t) = (r^a + r^b) / 2 = s - q * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    t : float or ndarray
        Current time.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Terminal horizon.

    Returns
    -------
    float or ndarray
        Reservation mid-price.
    """
    tau = T - t
    return s - q * gamma * sigma**2 * tau


# ---------------------------------------------------------------------------
# Infinite-horizon stationary reservation prices
# ---------------------------------------------------------------------------

def reservation_bid_infinite(
    s: ArrayLike,
    q: ArrayLike,
    gamma: float,
    sigma: float,
    omega: float,
) -> ArrayLike:
    """
    Stationary infinite-horizon reservation bid price.

    rbar^b(s, q) = s + (1/gamma) * ln(1 + (-1 - 2*q) * gamma^2 * sigma^2
                                          / (2*omega - gamma^2 * q^2 * sigma^2))

    Admissibility condition: omega > 0.5 * gamma^2 * sigma^2 * q^2

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount / admissibility parameter (> 0.5 * gamma^2 * sigma^2 * q_max^2).

    Returns
    -------
    float or ndarray
        Reservation bid price, or NaN where admissibility fails.
    """
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)

    denom = 2.0 * omega - gamma**2 * q**2 * sigma**2
    numerator = (-1.0 - 2.0 * q) * gamma**2 * sigma**2
    log_arg = 1.0 + numerator / denom

    # Check admissibility
    invalid = (denom <= 0) | (log_arg <= 0)
    if np.any(invalid):
        warnings.warn(
            f"Admissibility violated for some q values in reservation_bid_infinite. "
            f"Returning NaN for those entries.",
            RuntimeWarning,
        )

    result = np.where(invalid, np.nan, s + (1.0 / gamma) * np.log(np.where(invalid, 1.0, log_arg)))
    return float(result) if result.ndim == 0 else result


def reservation_ask_infinite(
    s: ArrayLike,
    q: ArrayLike,
    gamma: float,
    sigma: float,
    omega: float,
) -> ArrayLike:
    """
    Stationary infinite-horizon reservation ask price.

    rbar^a(s, q) = s + (1/gamma) * ln(1 + (1 - 2*q) * gamma^2 * sigma^2
                                          / (2*omega - gamma^2 * q^2 * sigma^2))

    Admissibility condition: omega > 0.5 * gamma^2 * sigma^2 * q^2

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount / admissibility parameter.

    Returns
    -------
    float or ndarray
        Reservation ask price, or NaN where admissibility fails.
    """
    q = np.asarray(q, dtype=float)
    s = np.asarray(s, dtype=float)

    denom = 2.0 * omega - gamma**2 * q**2 * sigma**2
    numerator = (1.0 - 2.0 * q) * gamma**2 * sigma**2
    log_arg = 1.0 + numerator / denom

    invalid = (denom <= 0) | (log_arg <= 0)
    if np.any(invalid):
        warnings.warn(
            f"Admissibility violated for some q values in reservation_ask_infinite. "
            f"Returning NaN for those entries.",
            RuntimeWarning,
        )

    result = np.where(invalid, np.nan, s + (1.0 / gamma) * np.log(np.where(invalid, 1.0, log_arg)))
    return float(result) if result.ndim == 0 else result


def admissibility_omega(
    gamma: float,
    sigma: float,
    q_max: int,
) -> float:
    """
    Compute the suggested practical omega for the infinite-horizon model.

    omega = 0.5 * gamma^2 * sigma^2 * (q_max + 1)^2

    This ensures the admissibility condition holds for all |q| <= q_max.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    q_max : int
        Maximum absolute inventory level.

    Returns
    -------
    float
        Admissible omega value.
    """
    return 0.5 * gamma**2 * sigma**2 * (q_max + 1) ** 2


def check_admissibility(
    q: ArrayLike,
    gamma: float,
    sigma: float,
    omega: float,
) -> np.ndarray:
    """
    Check whether the admissibility condition omega > 0.5*gamma^2*sigma^2*q^2
    holds for each inventory level.

    Parameters
    ----------
    q : array-like
        Inventory levels to check.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    omega : float
        Discount parameter.

    Returns
    -------
    ndarray of bool
        True where admissibility holds.
    """
    q = np.asarray(q, dtype=float)
    return omega > 0.5 * gamma**2 * sigma**2 * q**2


def verify_indifference_bid(
    x: float,
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
    tol: float = 1e-10,
) -> bool:
    """
    Numerically verify the bid indifference condition:
    v(x - r^b, s, q+1, t) == v(x, s, q, t)

    Parameters
    ----------
    x, s, q, t, gamma, sigma, T : float
        State and model parameters.
    tol : float
        Tolerance for floating-point comparison.

    Returns
    -------
    bool
        True if indifference holds within tolerance.
    """
    r_b = reservation_bid_finite(s, q, t, gamma, sigma, T)
    lhs = inactive_utility(x - r_b, s, q + 1, t, gamma, sigma, T)
    rhs = inactive_utility(x, s, q, t, gamma, sigma, T)
    return abs(lhs - rhs) < tol * abs(rhs)


def verify_indifference_ask(
    x: float,
    s: float,
    q: float,
    t: float,
    gamma: float,
    sigma: float,
    T: float,
    tol: float = 1e-10,
) -> bool:
    """
    Numerically verify the ask indifference condition:
    v(x + r^a, s, q-1, t) == v(x, s, q, t)

    Parameters
    ----------
    x, s, q, t, gamma, sigma, T : float
        State and model parameters.
    tol : float
        Tolerance for floating-point comparison.

    Returns
    -------
    bool
        True if indifference holds within tolerance.
    """
    r_a = reservation_ask_finite(s, q, t, gamma, sigma, T)
    lhs = inactive_utility(x + r_a, s, q - 1, t, gamma, sigma, T)
    rhs = inactive_utility(x, s, q, t, gamma, sigma, T)
    return abs(lhs - rhs) < tol * abs(rhs)
