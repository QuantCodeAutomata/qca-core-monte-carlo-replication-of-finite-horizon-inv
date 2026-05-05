# Repository Knowledge: qca-core-monte-carlo-replication-of-finite-horizon-inv

## Project Overview
Monte Carlo replication of the Avellaneda-Stoikov finite-horizon market-making model.
Compares inventory-based quoting strategy vs symmetric benchmark.

## Key Architecture
- `src/config.py` — SimParams dataclass, GAMMA_SCENARIOS, PUBLISHED_TARGETS, PUBLISHED_SPREADS
- `src/reservation.py` — Finite/infinite-horizon reservation price formulas
- `src/strategies.py` — compute_spread(), inventory_quotes(), symmetric_quotes(), fill_intensity()
- `src/simulator.py` — simulate_path(), run_monte_carlo(), run_monte_carlo_common_rng()
- `src/metrics.py` — compute_summary(), build_results_table(), variance_reduction_ratio()
- `src/plots.py` — All visualization functions (matplotlib/seaborn, Agg backend)
- `exp/exp1_core_replication.py` — Main Monte Carlo experiment
- `exp/exp2_spread_sensitivity.py` — Spread formula ambiguity analysis
- `exp/exp3_reservation_prices.py` — Theory validation (no simulation needed)
- `main.py` — Runs all three experiments, writes RESULTS.md

## Critical Implementation Details
1. **Mid-price dynamics**: Uses discrete `S_{t+dt} = S_t ± σ√dt` (NOT Gaussian increments)
2. **Spread modes**: 
   - `text_formula`: Φ(t) = γσ²(T-t) + (2/γ)ln(1+γ/k) — primary theoretical formula
   - `table_matching`: Φ = (2/γ)ln(1+γ/k) — matches published table values
3. **Fill simulation**: Independent Bernoulli draws; both sides can fill simultaneously
4. **Negative deltas**: Allowed (λ > A), logged as diagnostic
5. **λ·dt > 1**: Clipped in non-strict mode, logged
6. **CRN**: Common random numbers used in Exp2 for fair sensitivity comparison

## Paper Ambiguities Documented
- Spread formula in text ≠ spread values in table
- Table-matching formula reproduces published spreads {0.01: 1.33, 0.1: 1.29, 0.5: 1.15}
- Simultaneous bid/ask fills are possible under independent Bernoulli simulation

## Test Structure
- `tests/test_reservation.py` — Unit tests for reservation price formulas
- `tests/test_strategies.py` — Unit tests for quoting strategies
- `tests/test_simulator.py` — Unit tests for simulator
- `tests/test_metrics.py` — Unit tests for metrics
- `tests/test_exp_integration.py` — Integration tests for all experiments

## Running
```bash
python main.py          # All experiments
pytest tests/ -v        # All tests
```

## Dependencies
numpy, pandas, matplotlib, seaborn, scipy, pytest
No specialized quant library needed (custom implementation throughout).
