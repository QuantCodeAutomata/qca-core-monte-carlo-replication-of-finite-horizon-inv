# RESULTS.md — Market-Making Simulation Experiments
**Generated with N_paths=1000, seed=42**
**Spread mode for Exp1: text_formula**

---
## Experiment 3: Reservation Price Theory Validation
### Boundary Conditions
- gamma=0.01: ✓ ALL PASS
- gamma=0.1: ✓ ALL PASS
- gamma=0.5: ✓ ALL PASS
- Cross-gamma comparative statics: ✓ PASS

**Overall Exp3 validation: ✓ ALL CHECKS PASSED**

### Key Findings
- Finite-horizon reservation mid-price r(s,q,t) = s - q·γ·σ²·(T-t) validated analytically.
- Boundary conditions: r(s,0,t)=s and r(s,q,T)=s confirmed numerically.
- Utility indifference conditions verified to machine precision.
- Infinite-horizon formulas valid within admissibility region.

---
## Experiment 1: Core Monte Carlo Replication
**N_paths=1000, spread_mode=text_formula**

### Simulation Results vs Published Values

| gamma | strategy | sim_mean_profit | pub_mean_profit | sim_std_profit | pub_std_profit | sim_mean_q | pub_mean_q | sim_std_q | pub_std_q |
|-------|----------|----------------|----------------|----------------|----------------|------------|------------|-----------|----------|
| 0.01 | inventory | 68.23 | 66.78 | 9.06 | 8.76 | 0.149 | -0.020 | 5.13 | 4.70 |
| 0.01 | symmetric | 68.35 | 67.36 | 14.13 | 13.40 | 0.364 | -0.310 | 8.62 | 8.65 |
| 0.1 | inventory | 64.62 | 62.94 | 6.60 | 5.89 | 0.118 | 0.100 | 2.90 | 2.80 |
| 0.1 | symmetric | 67.75 | 67.21 | 13.31 | 13.43 | 0.056 | -0.018 | 8.45 | 8.66 |
| 0.5 | inventory | 48.53 | 33.92 | 5.98 | 4.72 | -0.086 | -0.020 | 2.02 | 1.88 |
| 0.5 | symmetric | 58.00 | 66.20 | 11.77 | 14.53 | -0.409 | 0.250 | 7.15 | 9.06 |

### Variance Reduction Ratios (inventory / symmetric)

| gamma | VRR_profit | VRR_inventory | lower_pnl_std | lower_inv_std |
|-------|-----------|---------------|---------------|---------------|
| 0.01 | 0.6411 | 0.5953 | ✓ | ✓ |
| 0.1 | 0.4960 | 0.3428 | ✓ | ✓ |
| 0.5 | 0.5076 | 0.2819 | ✓ | ✓ |

### Qualitative Findings
- Inventory strategy has lower P&L std for all gamma: ✓
- Inventory strategy has lower inventory std for all gamma: ✓
- Strategies converge as gamma→0: ✓

---
## Experiment 2: Spread Formula Sensitivity Analysis

### Spread Formula Comparison

| gamma | text_formula_spread | table_matching_spread | published_spread | table_matches_published |
|-------|--------------------|-----------------------|-----------------|------------------------|
| 0.01 | 1.368909 | 1.328909 | 1.33 | ✓ |
| 0.1 | 1.690770 | 1.290770 | 1.29 | ✓ |
| 0.5 | 3.150728 | 1.150728 | 1.15 | ✓ |

**Better match to published spreads: table_matching**

### Key Finding
The table-matching formula Φ = (2/γ)·ln(1+γ/k) reproduces the published spread values,
while the text-formula adds an extra γ·σ²·(T-t) term at t=0.
This confirms the paper's spread table was generated with the constant (table-matching) formula.

### Sensitivity of std_profit to Spread Mode (gamma=0.1)

| strategy | spread_mode | std_profit | std_final_q |
|----------|-------------|-----------|-------------|
| inventory | text_formula | 6.6003 | 2.8966 |
| inventory | table_matching | 6.0220 | 2.9128 |
| symmetric | text_formula | 13.3074 | 8.4497 |
| symmetric | table_matching | 13.9455 | 8.9592 |

---
## Plots Generated

### Experiment 1
- `exp1_path_inventory_gamma0.1.png` — Illustrative path (mid-price, reservation, quotes)
- `exp1_pnl_hist_gamma{γ}.png` — Terminal P&L histograms for each γ
- `exp1_inv_hist_gamma{γ}.png` — Final inventory histograms for each γ
- `exp1_gamma_comparison.png` — Summary statistics across γ values

### Experiment 2
- `exp2_spread_comparison.png` — Bar chart: text vs table-matching vs published spreads
- `exp2_sensitivity_{metric}.png` — Heatmaps of metric sensitivity to spread mode

### Experiment 3
- `exp3_res_mid_grid_gamma{γ}.png` — Reservation mid-price vs inventory at several times
- `exp3_res_bid_ask_t0_gamma{γ}.png` — Reservation bid/ask at t=0
- `exp3_inf_horizon_gamma{γ}.png` — Infinite-horizon reservation prices
