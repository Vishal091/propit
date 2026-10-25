"""Conservative discrete prop probabilities, market shrinkage, and staking helpers.

Probabilities are experimental. These utilities model integer-valued stat outcomes,
explicitly account for pushes, and never manufacture a market probability.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Any

MAX_MODEL_PROBABILITY = 0.72
MAX_UNCERTAIN_PROBABILITY = 0.65
_PROBABILITY_EPS = 1e-9
_Z_90 = 1.2815515655446004


@dataclass(frozen=True)
class PropOutcomes:
    """Unconditional probabilities for an integer-valued stat."""

    p_over: float
    p_under: float
    p_push: float
    distribution: str
    estimated_std: float


def _log_pmf(k: int, mean: float, variance: float) -> float:
    if mean <= _PROBABILITY_EPS:
        return 0.0 if k == 0 else -math.inf
    if variance <= mean + 1e-8:
        return -mean + k * math.log(mean) - math.lgamma(k + 1.0)
    size = mean * mean / max(variance - mean, 1e-8)
    prob = size / (size + mean)
    return (
        math.lgamma(k + size) - math.lgamma(size) - math.lgamma(k + 1.0)
        + size * math.log(prob) + k * math.log1p(-prob)
    )


def _pmf(mean: float, variance: float, max_k: int) -> list[float]:
    logs = [_log_pmf(k, mean, variance) for k in range(max_k + 1)]
    finite = [value for value in logs if math.isfinite(value)]
    if not finite:
        return [1.0] + [0.0] * max_k
    peak = max(finite)
    weights = [math.exp(value - peak) if math.isfinite(value) else 0.0 for value in logs]
    total = sum(weights)
    return [value / total for value in weights]


def _sample_count_distribution(
    *,
    mean: float,
    variance: float,
    size: int,
    rng: np.random.Generator,
    points_tail_mix: bool,
) -> tuple[np.ndarray, str]:
    """Vectorized Poisson/NB draws with optional fatter points tails."""
    if size < 1:
        return np.zeros(0, dtype=int), "poisson"
    mus = np.full(size, max(0.0, float(mean)), dtype=float)
    local_variance = np.full(size, max(float(variance), 0.0), dtype=float)
    family = "negative_binomial" if variance > mean + 1e-8 else "poisson"
    if points_tail_mix:
        tail = rng.random(size) < 0.15
        local_variance[tail] *= 1.8
        family += "_mixture"
    local_variance = np.maximum(local_variance, mus)
    poisson_mask = local_variance <= mus + 1e-8
    draws = np.zeros(size, dtype=int)
    if poisson_mask.any():
        draws[poisson_mask] = rng.poisson(mus[poisson_mask])
    nb_mask = ~poisson_mask & (mus > 1e-8)
    if nb_mask.any():
        mu = mus[nb_mask]
        var = local_variance[nb_mask]
        shape = np.maximum(mu * mu / np.maximum(var - mu, 1e-8), 1e-6)
        probability = shape / (shape + mu)
        draws[nb_mask] = rng.negative_binomial(shape, probability)
    return draws, family


def _estimate_with_minutes_simulation(
    *,
    mean: float,
    variance: float,
    spread: float,
    line: float,
    target: str,
    minutes_mean: float,
    minutes_std: float,
    simulations: int,
    seed: int,
) -> PropOutcomes:
    if not math.isfinite(minutes_mean) or not math.isfinite(minutes_std) or minutes_mean <= 0 or minutes_std < 0:
        raise ValueError("minutes_mean must be > 0 and minutes_std must be >= 0")
    if simulations < 10_000:
        raise ValueError("minutes Monte Carlo requires at least 10000 simulations")
    rng = np.random.default_rng(seed)
    minutes = np.clip(rng.normal(minutes_mean, minutes_std, size=simulations), 0.0, 53.0)
    # A small generic overtime tail. Game-specific spread/total is not available
    # in every feed yet, so the tail is deliberately modest and explicitly generic.
    minutes += (rng.random(simulations) < 0.035) * 5.0
    minutes = np.clip(minutes, 0.0, 53.0)
    conditional_mean = np.maximum(0.0, mean * minutes / minutes_mean)
    conditional_variance = variance * np.maximum(conditional_mean / max(mean, 0.25), 0.45)
    conditional_variance = np.maximum(conditional_variance, conditional_mean)
    points = target.strip().upper() == "PTS"

    # Draw count outcomes conditional on the minutes mixture. Each sample gets
    # its own mean/dispersion; no single point-minutes estimate is substituted.
    local_variance = conditional_variance
    if points:
        tail = rng.random(simulations) < 0.15
        local_variance = local_variance.copy()
        local_variance[tail] *= 1.8
    poisson_mask = local_variance <= conditional_mean + 1e-8
    draws = np.zeros(simulations, dtype=int)
    if poisson_mask.any():
        draws[poisson_mask] = rng.poisson(conditional_mean[poisson_mask])
    nb_mask = ~poisson_mask & (conditional_mean > 1e-8)
    if nb_mask.any():
        mus = conditional_mean[nb_mask]
        vars_ = local_variance[nb_mask]
        shape = np.maximum(mus * mus / np.maximum(vars_ - mus, 1e-8), 1e-6)
        p = shape / (shape + mus)
        draws[nb_mask] = rng.negative_binomial(shape, p)

    integer_line = math.isclose(line, round(line), abs_tol=1e-9)
    p_over = float(np.mean(draws > line))
    p_under = float(np.mean(draws < line))
    p_push = float(np.mean(draws == int(round(line)))) if integer_line else 0.0
    total = p_over + p_under + p_push
    if total <= _PROBABILITY_EPS:
        p_over, p_under, p_push = 0.0, 0.0, 1.0
    else:
        p_over, p_under, p_push = p_over / total, p_under / total, p_push / total
    # Report the total spread after adding the minute-distribution variance.
    rate = mean / minutes_mean
    combined_std = math.sqrt(max(0.0, spread * spread + (rate * minutes_std) ** 2))
    family = "negative_binomial" if variance > mean + 1e-8 else "poisson"
    if points:
        family += "_mixture"
    family += f"+minutes_mc{simulations}"
    return PropOutcomes(p_over, p_under, p_push, family, combined_std)


def estimate_prop_outcomes(
    *,
    mean: float,
    p10: float,
    p90: float,
    line: float,
    target: str,
    fallback_std: float,
    minutes_mean: float | None = None,
    minutes_std: float | None = None,
    simulations: int = 10_000,
    seed: int = 2027,
) -> PropOutcomes:
    """Estimate OVER/UNDER/PUSH using a discrete Poisson/NB distribution.

    The q10–q90 interval supplies context-specific dispersion. PTS uses a small
    broader-distribution mixture to avoid an overly thin upper/lower tail.
    Returned over/under probabilities are unconditional and sum with push to 1.
    """
    values = (mean, p10, p90, line, fallback_std)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("mean, quantiles, line, and fallback_std must be finite")
    if line < 0 or mean < 0:
        raise ValueError("line and mean must be non-negative")
    if target.strip().upper() not in {"PTS", "REB", "AST", "3PM", "STL", "BLK", "PRA"}:
        raise ValueError(f"unsupported integer-valued prop target: {target}")

    spread = (float(p90) - float(p10)) / (2.0 * _Z_90)
    if not math.isfinite(spread) or spread <= 1e-6:
        spread = max(float(fallback_std), math.sqrt(max(float(mean), 0.25)))
    # Avoid implausibly tiny dispersion for count stats.
    spread = max(spread, math.sqrt(max(float(mean), 0.25)) * 0.65)
    variance = max(spread * spread, 1e-4)
    target_key = target.strip().upper()
    is_points = target_key == "PTS"
    if minutes_mean is not None or minutes_std is not None:
        if minutes_mean is None or minutes_std is None:
            raise ValueError("minutes_mean and minutes_std must be supplied together")
        return _estimate_with_minutes_simulation(
            mean=float(mean), variance=variance, spread=spread, line=float(line),
            target=target_key, minutes_mean=float(minutes_mean), minutes_std=float(minutes_std),
            simulations=int(simulations), seed=int(seed),
        )
    max_std = math.sqrt(max(variance * (1.8 if is_points else 1.0), mean, 0.25))
    max_k = min(2000, max(80, int(math.ceil(mean + 12.0 * max_std + 30.0))))

    base = _pmf(float(mean), variance, max_k)
    if is_points:
        broad = _pmf(float(mean), max(variance * 1.8, variance + mean), max_k)
        masses = [0.85 * a + 0.15 * b for a, b in zip(base, broad)]
        family = "negative_binomial_mixture" if variance > mean else "poisson_mixture"
    else:
        masses = base
        family = "negative_binomial" if variance > mean + 1e-8 else "poisson"

    total = sum(masses)
    masses = [value / total for value in masses]
    nearest = int(round(float(line)))
    is_integer_line = math.isclose(float(line), nearest, abs_tol=1e-9)
    p_over = sum(mass for k, mass in enumerate(masses) if k > line)
    p_under = sum(mass for k, mass in enumerate(masses) if k < line)
    p_push = masses[nearest] if is_integer_line and 0 <= nearest <= max_k else 0.0

    # Normalize numerical/truncation noise while preserving a valid push state.
    probability_total = p_over + p_under + p_push
    if probability_total <= _PROBABILITY_EPS:
        return PropOutcomes(0.0, 0.0, 1.0, family, spread)
    p_over, p_under, p_push = (
        p_over / probability_total, p_under / probability_total, p_push / probability_total
    )
    return PropOutcomes(p_over, p_under, p_push, family, spread)


def calibrate_toward_market(
    raw_probability: float,
    market_probability: float,
    *,
    model_weight: float,
) -> float:
    """Shrink a raw conditional probability toward same-market no-vig probability."""
    raw = float(raw_probability)
    market = float(market_probability)
    weight = float(model_weight)
    if not all(math.isfinite(v) for v in (raw, market, weight)):
        raise ValueError("probabilities and model_weight must be finite")
    if not 0.0 <= raw <= 1.0 or not 0.0 <= market <= 1.0:
        raise ValueError("probabilities must be in [0, 1]")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("model_weight must be in [0, 1]")
    return weight * raw + (1.0 - weight) * market


def probability_cap(probability: float, *, uncertain: bool) -> float:
    """Return the allowed probability ceiling for one side of a prop."""
    _ = probability  # Kept in the signature to make call-sites explicit.
    return MAX_UNCERTAIN_PROBABILITY if uncertain else MAX_MODEL_PROBABILITY


def exceeds_probability_cap(probability: float, *, uncertain: bool) -> bool:
    """True when either side would exceed the applicable probability ceiling."""
    cap = probability_cap(probability, uncertain=uncertain)
    value = float(probability)
    return not math.isfinite(value) or max(value, 1.0 - value) > cap + 1e-9


def quarter_kelly_fraction(
    probability: float,
    american_odds: Any,
    *,
    max_fraction: float = 0.02,
    kelly_fraction: float = 0.25,
) -> float:
    """Quarter-Kelly bankroll fraction, bounded by max_fraction; returns 0 if no edge."""
    try:
        odds = float(american_odds)
        p = float(probability)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(odds) or odds == 0 or not math.isfinite(p) or not 0 <= p <= 1:
        return 0.0
    if odds > 0:
        decimal = 1.0 + odds / 100.0
    else:
        decimal = 1.0 + 100.0 / abs(odds)
    if decimal <= 1.0 or max_fraction <= 0 or not 0 <= kelly_fraction <= 1:
        return 0.0
    b = decimal - 1.0
    full_kelly = (b * p - (1.0 - p)) / b
    return max(0.0, min(float(max_fraction), float(kelly_fraction) * full_kelly))
