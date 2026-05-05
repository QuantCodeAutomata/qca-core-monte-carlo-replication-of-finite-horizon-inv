"""
Configuration module for the finite-horizon market-making simulation.

Defines parameter containers for the Avellaneda-Stoikov style model.
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class SimParams:
    """
    Parameter container for the finite-horizon market-making simulation.

    Attributes
    ----------
    S0 : float
        Initial mid-price.
    T : float
        Horizon in years.
    sigma : float
        Mid-price volatility.
    dt : float
        Discrete time step.
    q0 : int
        Initial inventory.
    X0 : float
        Initial cash.
    A : float
        Baseline fill-intensity parameter.
    k : float
        Fill-intensity decay parameter.
    gamma : float
        Risk-aversion coefficient.
    N_paths : int
        Number of Monte Carlo paths.
    spread_mode : str
        'text_formula' uses Phi(t) = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k).
        'table_matching' uses Phi = (2/gamma)*ln(1+gamma/k) (constant in time).
    seed : int or None
        Random seed for reproducibility.
    strict_lambda : bool
        If True, raise error when lambda*dt > 1; if False, clip and log.
    """

    S0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    q0: int = 0
    X0: float = 0.0
    A: float = 140.0
    k: float = 1.5
    gamma: float = 0.1
    N_paths: int = 1000
    spread_mode: Literal["text_formula", "table_matching"] = "text_formula"
    seed: int = 42
    strict_lambda: bool = False

    def __post_init__(self) -> None:
        assert self.T > 0, "Horizon T must be positive."
        assert self.sigma > 0, "Volatility sigma must be positive."
        assert self.dt > 0, "Time step dt must be positive."
        assert self.A > 0, "Intensity parameter A must be positive."
        assert self.k > 0, "Intensity decay k must be positive."
        assert self.gamma > 0, "Risk aversion gamma must be positive."
        assert self.N_paths > 0, "Number of paths must be positive."
        n_steps = round(self.T / self.dt)
        assert abs(n_steps * self.dt - self.T) < 1e-10, (
            f"T={self.T} must be divisible by dt={self.dt}."
        )

    @property
    def n_steps(self) -> int:
        """Number of discrete time steps."""
        return round(self.T / self.dt)


# Default scenario grid as described in the paper
GAMMA_SCENARIOS: List[float] = [0.01, 0.1, 0.5]

# Published reference values from the paper (Table 1 equivalent)
# Format: {gamma: {strategy: {metric: value}}}
PUBLISHED_TARGETS = {
    0.1: {
        "inventory": {
            "mean_profit": 62.94,
            "std_profit": 5.89,
            "mean_final_q": 0.10,
            "std_final_q": 2.80,
        },
        "symmetric": {
            "mean_profit": 67.21,
            "std_profit": 13.43,
            "mean_final_q": -0.018,
            "std_final_q": 8.66,
        },
    },
    0.01: {
        "inventory": {
            "mean_profit": 66.78,
            "std_profit": 8.76,
            "mean_final_q": -0.02,
            "std_final_q": 4.70,
        },
        "symmetric": {
            "mean_profit": 67.36,
            "std_profit": 13.40,
            "mean_final_q": -0.31,
            "std_final_q": 8.65,
        },
    },
    0.5: {
        "inventory": {
            "mean_profit": 33.92,
            "std_profit": 4.72,
            "mean_final_q": -0.02,
            "std_final_q": 1.88,
        },
        "symmetric": {
            "mean_profit": 66.20,
            "std_profit": 14.53,
            "mean_final_q": 0.25,
            "std_final_q": 9.06,
        },
    },
}

# Published spread values from the paper's table
PUBLISHED_SPREADS = {
    0.01: 1.33,
    0.1: 1.29,
    0.5: 1.15,
}
