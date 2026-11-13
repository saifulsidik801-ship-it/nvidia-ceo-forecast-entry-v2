# Polymarket NVIDIA CEO price forecaster — variant 2 (per-horizon realized sigma + Student-t fan)

Entry for make.fun competition `polymarket-nvidia-ceo-forecast-01`: forecast
Polymarket's "Next CEO of NVIDIA?" prices minute by minute, 4 hours ahead.

## Model

Per-market, per-horizon realized-sigma fan in logit space, with Student-t(8)
quantile multipliers.

Prediction-market prices do **not** spread like `sigma * h**beta`. Their
uncertainty grows fast in the first hour (news lands, then the market digests
it) and saturates afterwards. Instead of imposing a shape, this model
**measures** it: for each horizon h = 1..240 minutes, sigma_h is the realized
standard deviation of h-minute log-odds changes over the last 300 minutes,
winsorized at +/-0.5 (a few birth-phase jumps must not inflate the fan), then
shrunk halfway toward the power-law anchor `sigma_1 * h**0.4` so noisy
long-horizon estimates stay anchored.

- **Center:** log-odds of the market's last price (a prediction-market price
  is a martingale, so the last price is the best median).
- **sigma_1:** realized 1-minute log-odds sigma over the last 180 minutes,
  shrunk halfway toward a pooled 0.02.
- **Quantiles:** `inv_logit(center + t_tau(df=8) * sigma_h)`. A t fan fits
  better than a Gaussian one because price jumps are fat-tailed: the outer
  quantiles need to sit further out relative to the inner ones than a
  Gaussian allows. The t quantiles are computed at import from the
  incomplete beta function (stdlib only, deterministic).

## Backtest

Mean pinball loss (percentage points), scored on the house's exact grid rule:

| model | practice p00–p05 | validation (5 live cases, 2026-11-13) |
|---|---|---|
| house baseline | 0.363589 | 0.216030 |
| first entry (Gaussian power-law fan, beta 0.4) | 0.272936 | 0.061381 |
| **this entry** | **0.268807** | **0.055536** |

The win is consistent across the whole model family (window 270–480 all beat
the first entry on both sets), not a single cherry-picked point. Runtime
~0.53s per case (limit 8s); two runs byte-identical.

## Files

- `main.py` — the entrypoint (Python 3.13, stdlib only). Reads every `*.in`
  in `$MAKEFUN_INPUT_DIR` (else `/input`), writes one `*.out` per case to
  `$MAKEFUN_OUTPUT_DIR` (else `/output`).
- `LICENSE` — MIT.
- `.makefun/owner.json` — payout wallet.
