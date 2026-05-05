# Finite-Horizon Inventory-Based Market Making: Monte Carlo Replication

A Python implementation replicating the Avellaneda-Stoikov finite-horizon market-making model, comparing an inventory-aware quoting strategy against a symmetric benchmark via Monte Carlo simulation.

## Overview

This repository implements three experiments:

1. **Exp 1 — Core Monte Carlo Replication**: Compares inventory-based vs symmetric benchmark strategies across γ ∈ {0.01, 0.1, 0.5} with 1000 paths each.
2. **Exp 2 — Spread Formula Sensitivity**: Diagnoses the paper's spread-formula ambiguity (text-formula vs table-matching) and measures its impact on results.
3. **Exp 3 — Reservation Price Validation**: Validates finite-horizon and infinite-horizon reservation price formulas analytically and numerically.

## Model

The model follows Avellaneda & Stoikov (2008):

- **Mid-price**: `S_{t+dt} = S_t ± σ√dt` (equal probability)
- **Reservation price**: `r_t = S_t - q_t · γ · σ² · (T - t)`
- **Spread**: `Φ(t) = γσ²(T-t) + (2/γ)ln(1 + γ/k)` (text formula)
- **Fill intensity**: `λ(δ) = A · exp(-k · δ)`
- **Terminal P&L**: `Π_T = X_T + q_T · S_T`

## Parameters

| Parameter | Value |
|-----------|-------|
| S₀ | 100 |
| T | 1 |
| σ | 2 |
| dt | 0.005 |
| A | 140 |
| k | 1.5 |
| γ | {0.01, 0.1, 0.5} |
| N_paths | 1000 |

## Repository Structure

```
├── data/                  # Data directory (synthetic data only)
├── src/                   # Core implementation
│   ├── config.py          # Parameters and scenario definitions
│   ├── reservation.py     # Reservation price formulas
│   ├── strategies.py      # Quoting strategies
│   ├── simulator.py       # Monte Carlo path simulation
│   ├── metrics.py         # Summary statistics
│   └── plots.py           # Visualization
├── exp/                   # Experiment scripts
│   ├── exp1_core_replication.py
│   ├── exp2_spread_sensitivity.py
│   └── exp3_reservation_prices.py
├── tests/                 # pytest tests
│   ├── test_reservation.py
│   ├── test_strategies.py
│   ├── test_simulator.py
│   ├── test_metrics.py
│   └── test_exp_integration.py
├── results/               # Outputs (plots, tables, RESULTS.md)
├── main.py                # Entry point for all experiments
└── README.md
```

## Installation

```bash
pip install numpy pandas matplotlib seaborn scipy pytest tqdm
```

## Usage

### Run all experiments

```bash
python main.py
```

### Run individual experiments

```bash
python exp/exp1_core_replication.py
python exp/exp2_spread_sensitivity.py
python exp/exp3_reservation_prices.py
```

### Run tests

```bash
pytest tests/ -v
```

## Key Findings

See `results/RESULTS.md` for detailed results.

**Qualitative findings (consistent with paper)**:
- Inventory strategy has lower P&L standard deviation than symmetric for all γ
- Inventory strategy has lower final inventory standard deviation for all γ
- As γ → 0, strategies converge
- Larger γ → stronger inventory control

**Spread formula ambiguity**:
- The table-matching formula `Φ = (2/γ)ln(1+γ/k)` reproduces published spread values
- The text-formula adds `γσ²(T-t)` and produces larger initial spreads

## Implementation Notes

- Mid-price uses discrete `±σ√dt` increments (paper's specification)
- Both bid and ask may fill in the same time step (documented assumption)
- Negative quote distances (λ > A) are allowed and logged
- λ·dt > 1 violations are clipped in non-strict mode and logged
- Common random numbers (CRN) used in Exp 2 for fair sensitivity comparison
