"""
====================================================================
 DRAW PATTERN VALIDATOR  (Consolidated Engine -- Standalone Script)
====================================================================
STANDALONE-SCRIPT NOTE: this is a single-file export of the same
engine that ships inside the `draw-pattern-validator` Claude skill,
with that skill's game_configs.py inlined below and its run_cli.py's
argument parsing folded into `main()`, so it runs with zero external
files or imports beyond numpy. The skill version keeps the engine and
its PCSO-specific config in separate files on purpose (to make that
split visible and to prove genericity against a second, non-PCSO
config) and adds a three-stage verification suite, a hypothesis
ledger, and decision templates -- reach for the skill instead of this
file if you want those. Every claim and every test result described
below was established against this same code before it was exported
here; nothing about the analysis changed in the export.

A gated, multi-method pattern-analysis pipeline for sequential draw
games -- positional digit games (order matters, e.g. EZ2/Swertres/6D)
and sorted-pool games (e.g. Lotto 6/42 through Ultra Lotto 6/58).
TYPE_CONFIG below ships with PCSO's eight formats as the default,
worked example -- swap in any other game's slot count / value range /
positional-vs-pool shape and everything else runs unchanged.

Merges techniques scattered across 34 prototype scripts in three
upload batches:

    Batch 1 (17 scripts): the Sys_V1-V6 / SWD-WPA / 3D-to-58G family --
        sliding-window frequency analysis, wave-pressure scoring, KNN
        variants, difference-based forecasting, due-element and
        zero-frequency anticipation, segment/thematic scoring.
    Batch 2 (11 scripts): the HierarchicalComboAnalyzer lineage, a
        genetic-algorithm search, a piecewise near-miss
        reverse-engineering tool, and several frequency/gap-rule
        variants that mostly overlapped what batch 1 already covered.
    Batch 3 (7 scripts, 2 of them byte-identical duplicates): a TOPSIS
        multi-criteria combiner, Shannon-entropy / within-draw run and
        gap diagnostics, two real cognitive-bias checks, an adaptive
        method-weighting scheme, and a data-quality advisory report.

    CAPTURE   -> pull raw frequency / gap / positional facts, PLUS
                 purely descriptive diagnostics (entropy, within-draw
                 runs/gaps, confirmation-bias and availability-bias
                 checks) that make no significance claim of their own
    ANALYSIS  -> run each modeling technique independently: five
                 single-number methods (wave-pressure, KNN, difference
                 model, due-element, zero-frequency), multi-number
                 combo-support analysis, a genetic-algorithm search,
                 and a TOPSIS multi-criteria combiner
    TRIAGE    -> collect each technique's candidate + rationale
    FRAMEWORK -> combine candidates into a transparent ensemble, PLUS
                 an adaptive-weighted variant that weights each base
                 method by its own recent walk-forward performance
    VALIDATE  -> test the ensemble (and each technique -- adaptive
                 weighting included, no exemption) against a random
                 baseline: raw p-value + Benjamini-Hochberg corrected
                 p-value, surfaced plainly either way, plus an
                 explicitly-separate near-miss diagnostic that is
                 never allowed to feed into the significance verdict
    FLOW      -> orchestrate all of the above per game type, T1-T8,
                 ending with a one-shot, human-facing quality advisory
                 that never triggers an automatic rerun

STATISTICAL HONESTY DISCLAIMER
-------------------------------
These are independent, certified-random draws. Nothing in this
script -- or in any of its 34 ancestors -- has been shown to predict
them. Every output here is a descriptive pattern summary
("anticipated", never "predicted"), and the VALIDATE stage exists
specifically to show, plainly, whether any technique beats chance.
A null result is a legitimate finding, not a failure of the script.

WHAT WAS DELIBERATELY LEFT OUT
-------------------------------
`RNGCracker` -- present in 15 of the 34 source scripts across all
three upload batches -- was not ported. Its LCG/LFSR checks are
structured so they always evaluate to None regardless of input, and
its "Mersenne Twister crack" is just a seeded random draw with a
hardcoded 0.2 confidence score. The name implied a capability
(recovering a certified RNG's state from a handful of draws) that the
code never actually delivered, in any of its 15 copies. Carrying it
forward -- even with a caveat -- would misrepresent what this script
can do.

A second recurring bug was caught and NOT ported: `CLPC_Model_V2`'s
`preprocess_data` calls `sorted(set(values))` on every draw before
analysis -- including its own 4-digit sample data, where `[1, 9, 7,
1]` silently becomes `[1, 7, 9]`, dropping a repeated digit and
losing a slot. `DrawDataset` below never deduplicates or re-sorts
positional-type input; it preserves exactly what it's given.

Batch 3 also contained a class literally named `NumericLLM` that has
nothing to do with language models -- it's the average of a value at
a given position across prior draws, wrapped (with `TabularModel` and
`MonteCarlo`) into a `UnifiedPredictor` whose combination weights
(0.4/0.4/0.2) were asserted, not validated. Not ported: it's a
redundant, less-capable duplicate of `DifferenceModel` and
`FrequencyEngine` already in this file, under a name that oversells
what it does.

Two of the four checks in a `CriticalThinkingFramework` class
(confirmation-bias, availability-bias) were real and are ported below
as `DescriptiveDiagnostics`. The other two (anchoring-bias,
overconfidence) always returned the same hardcoded dict regardless of
input -- the same non-functional pattern as `RNGCracker` -- and were
not ported.

A `_calculate_p_value` function mapped a chi-square statistic to a
canned significance label via hardcoded thresholds, with its own
comment admitting "in practice, use scipy.stats." Not ported -- this
engine already has a real, non-approximated significance test
(`RandomBaselineValidator`); a second, fake one would only invite
someone to quote whichever number looked better.

A `should_rethink_analysis` / `suggest_parameter_adjustments` pair
implemented real data-quality checks (overconfidence ratio,
speculative-combo ratio, confidence-score diversity) wired into a loop
that automatically retried the analysis with adjusted parameters up
to 3 times until validation passed. The checks were legitimate; the
auto-retry loop is a mechanized version of exactly what
references/playbook.md forbids doing by hand -- re-running with
different parameters until something clears is p-hacking regardless
of whether a human or a `while` loop does it. The checks are ported
below as `assess_run_quality()`, DEFANGED: it produces a one-shot
advisory (`QualityAdvisory.should_reconsider`) and nothing in this
file ever loops on that flag or calls `run_full_pipeline` again
because of it. `scripts/verify-significance-gate.sh` checks this
statically.

FOURTH PASS: efficiency, a completeness audit, and 2 more methods
-------------------------------------------------------------------
Requested explicitly, so tracked here as its own pass rather than
folded silently into the batch-3 notes above.

**Efficiency.** `ensemble` and `adaptive_weighted` used to each
recompute every base method from scratch at every walk-forward
validation point -- `adaptive_weighted` additionally recomputed all of
them again across its own internal trailing window on top of that.
On a ~30-40 draw demo dataset this made a full `run_full_pipeline`
call take 15-30 seconds. `WalkForwardCache` now computes every base
method's candidate/hit/near-miss at each walk-forward point exactly
once and shares that pass across the per-method loop and both
ensemble variants -- measured speedup on the same demo dataset was
roughly 10x (see references/runbook.md for the current number). This
is a pure performance change: `scripts/_functional_test.py` includes
a regression test asserting the cache's hit-rate/near-miss output for
a base method is numerically IDENTICAL to the slower, independent
`RandomBaselineValidator.historical_hit_rate` computation -- if a
future edit makes the cache diverge from ground truth, that test
fails.

That equivalence test caught a real, pre-existing bug while being
written: `ZeroFrequencyAnticipator.anticipate()` used to call
`random.sample()` on the global `random` module to break ties among
zero-frequency candidates. That makes the result depend on how many
OTHER random-consuming calls happened first in the same process, not
just on the input data -- the slow, one-method-at-a-time validation
path and the new all-methods-interleaved cache path consumed the
global random state in different orders and produced different
near-miss numbers for the identical dataset. Fixed by replacing the
random tie-break with a deterministic one (largest
gap-since-last-seen, then ascending value): given the same data and
parameters, this method -- and therefore the whole pipeline -- now
always returns the same output. This was an internal bug, not
something inherited from a source script; documented here because it
directly affects reproducibility, which is part of what "statistical
honesty" means in practice, not just p-value correction.

**Completeness audit.** Re-checked every technique across all 34
files against what had actually been ported. Two real gaps were found
and closed:

- `PiecewiseBinDiffModel` -- merged from `LotteryReverseEngineer`
  (Reverse-engineering_numeric_transformations_framework_06Dec25) and
  Multi_Framework_Predictor's Segmented Transformation Model. Bins
  values by quantile and learns the average next-value change
  conditional on which bin the PREVIOUS VALUE fell in. This is a
  genuinely different hypothesis from `DifferenceModel` (which
  conditions on TIME, not value) and had not been represented by any
  existing method.
- `ConvergenceSelector` -- merged from `CLPC_Model_V2`'s
  `calculate_convergence` / `_apply_convergence_check`. A third
  FRAMEWORK combination strategy alongside majority-vote
  (`EnsembleCombiner`) and multi-criteria distance ranking
  (`TOPSISCombiner`): searches a bounded top-scored pool for the
  SET_SIZE-combination whose member scores are most internally
  consistent, rather than simply the highest-scoring individual
  values.

Everything else re-checked was judged genuinely redundant with what
the engine already does, not overlooked:
`Multi_Framework_Predictor`'s other seven named frameworks (GEM, CSR,
GMF, CSI, BSL, and DAT/PTR) reduce to variations on frequency
weighting, positional trend extrapolation, or the arithmetic-formula
approach already covered by `SixStepSetTransformer`'s exclusion below;
`EmpiricalAnalyzer_Class.py`'s `adherence_avg = 0.75` and
`cascade_acc = 0.65` are hardcoded constants the class never actually
computes from data (the same non-functional pattern as `RNGCracker`),
and its `compute_pmp_medians` hand-overrides one computed median
(`medians[4] = 41`) to match a pre-written note rather than reporting
what the data produced -- neither was ever a candidate for porting.
`SixStepSetTransformer`'s and `Multi_Framework_Predictor`'s DAT
framework both reduce to the same hand-derived arithmetic formula
(lowest/mid/highest-plus-one/half-again heuristics) with no
statistical grounding and no validation in either source script --
excluded on the same basis as the frameworks above.

**Method count.** 10 base methods (wave_pressure, difference_model,
piecewise_bin_diff, knn, due_element, zero_frequency, combo_support,
genetic_algorithm, topsis, convergence_selector) + 2 combination
variants (ensemble, adaptive_weighted) = 12 tested through BH
correction every run, up from 10.
====================================================================
"""

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

random.seed(42)
np.random.seed(42)


# ============================================================
# 1. TYPE CONFIG  (T1-T3 positional digit games, T4-T8 sorted
#    pool games) -- inlined here for a single-file standalone
#    script. The companion skill (draw-pattern-validator) keeps
#    this in a separate game_configs.py on purpose, to make the
#    engine/application split visible in the file layout -- see
#    that skill's README if you want the multi-file version with
#    a CLI, verification scripts, and a non-PCSO example config
#    proving the split is real. Swap PCSO_TYPE_CONFIG below for
#    your own game's dict (same shape: name, positional, n_slots,
#    min_val, max_val, unique) to point this same engine at an
#    entirely different game with zero changes below this line.
# ============================================================
PCSO_TYPE_CONFIG: Dict[str, dict] = {
    "T1": {"name": "EZ2",              "positional": True,  "n_slots": 2, "min_val": 0, "max_val": 9,  "unique": False},
    "T2": {"name": "Swertres",         "positional": True,  "n_slots": 3, "min_val": 0, "max_val": 9,  "unique": False},
    "T3": {"name": "6D Lotto",         "positional": True,  "n_slots": 6, "min_val": 0, "max_val": 9,  "unique": False},
    "T4": {"name": "Lotto 6/42",       "positional": False, "n_slots": 6, "min_val": 1, "max_val": 42, "unique": True},
    "T5": {"name": "Megalotto 6/45",   "positional": False, "n_slots": 6, "min_val": 1, "max_val": 45, "unique": True},
    "T6": {"name": "Superlotto 6/49",  "positional": False, "n_slots": 6, "min_val": 1, "max_val": 49, "unique": True},
    "T7": {"name": "Grandlotto 6/55",  "positional": False, "n_slots": 6, "min_val": 1, "max_val": 55, "unique": True},
    "T8": {"name": "Ultra Lotto 6/58", "positional": False, "n_slots": 6, "min_val": 1, "max_val": 58, "unique": True},
}

TYPE_CONFIG: Dict[str, Dict] = dict(PCSO_TYPE_CONFIG)


# ============================================================
# 2. CAPTURE  -- dataset wrapper + raw frequency/gap facts
# ============================================================
class DrawDataset:
    """Holds historical draws for one game type. Order is preserved
    exactly as given -- positional types (T1-T3) must NOT be sorted;
    pool types (T4-T8) are expected to already be stored sorted."""

    def __init__(self, type_key: str, historical_data: Dict[str, List[int]]):
        if type_key not in TYPE_CONFIG:
            raise ValueError(f"Unknown type key: {type_key}")
        self.type_key = type_key
        self.config = TYPE_CONFIG[type_key]
        ordered_keys = sorted(historical_data.keys(), key=lambda k: int(k.split("_")[1]))
        self.set_names = ordered_keys
        self.draws = [list(historical_data[k]) for k in ordered_keys]
        self._validate()

    def _validate(self):
        n = self.config["n_slots"]
        for name, d in zip(self.set_names, self.draws):
            if len(d) != n:
                raise ValueError(f"{self.type_key} / {name}: expected {n} slots, got {len(d)} -> {d}")
            lo, hi = self.config["min_val"], self.config["max_val"]
            for v in d:
                if not (lo <= v <= hi):
                    raise ValueError(f"{self.type_key} / {name}: value {v} outside [{lo},{hi}]")

    def __len__(self):
        return len(self.draws)

    def last(self) -> List[int]:
        return self.draws[-1]

    def value_domain(self) -> List[int]:
        return list(range(self.config["min_val"], self.config["max_val"] + 1))


class FrequencyEngine:
    """CAPTURE: frequency distribution, hot/cold, positional stats,
    even/odd balance, and gap-since-last-seen for every value."""

    def __init__(self, dataset: DrawDataset):
        self.ds = dataset
        self.cfg = dataset.config

    def frequency(self, lookback: Optional[int] = None) -> Dict[int, int]:
        draws = self.ds.draws[-lookback:] if lookback else self.ds.draws
        freq = Counter()
        for d in draws:
            freq.update(d)
        return dict(freq)

    def positional_frequency(self, lookback: Optional[int] = None) -> List[Dict[int, int]]:
        draws = self.ds.draws[-lookback:] if lookback else self.ds.draws
        n = self.cfg["n_slots"]
        pos_freq = [Counter() for _ in range(n)]
        for d in draws:
            for i, v in enumerate(d):
                pos_freq[i][v] += 1
        return [dict(c) for c in pos_freq]

    def hot_cold(self, lookback: int = 20, hot_n: int = 15, cold_threshold: int = 1):
        freq = self.frequency(lookback)
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        hot = [num for num, f in sorted_freq[:hot_n] if f > 0]
        cold = [num for num in self.ds.value_domain() if freq.get(num, 0) <= cold_threshold]
        return hot, cold

    def even_odd_balance(self) -> Dict:
        even_counts, odd_counts = [], []
        for d in self.ds.draws:
            e = sum(1 for n in d if n % 2 == 0)
            even_counts.append(e)
            odd_counts.append(len(d) - e)
        last = self.ds.last()
        last_even = sum(1 for n in last if n % 2 == 0)
        return {
            "avg_even": float(np.mean(even_counts)) if even_counts else 0.0,
            "avg_odd": float(np.mean(odd_counts)) if odd_counts else 0.0,
            "last_even": last_even,
            "last_odd": len(last) - last_even,
        }

    def gap_since_last_seen(self) -> Dict[int, int]:
        """Draws elapsed since each value last appeared (value-level,
        not slot-specific -- matches how the source scripts' due
        element / cold-number logic operated)."""
        domain = self.ds.value_domain()
        gaps: Dict[int, Optional[int]] = {num: None for num in domain}
        seen = set()
        n = len(self.ds.draws)
        for idx in range(n - 1, -1, -1):
            for num in self.ds.draws[idx]:
                if num not in seen:
                    gaps[num] = (n - 1) - idx
                    seen.add(num)
            if len(seen) == len(domain):
                break
        for num in domain:
            if gaps[num] is None:
                gaps[num] = n  # never observed in the dataset
        return gaps  # type: ignore


class DescriptiveDiagnostics:
    """CAPTURE: purely descriptive statistics about the dataset itself
    -- no significance claims, no candidate generation. Merged from
    AdaptiveLotteryScanner (entropy, within-draw runs/gaps) and
    CriticalThinkingFramework (confirmation-bias, availability-bias
    checks). The other two checks in the source
    CriticalThinkingFramework -- anchoring-bias and overconfidence --
    always returned the same hardcoded dict regardless of input and
    were not ported; see the module docstring."""

    def __init__(self, dataset: DrawDataset, freq_engine: FrequencyEngine):
        self.ds = dataset
        self.fe = freq_engine

    @staticmethod
    def shannon_entropy(values: List[int]) -> float:
        if not values:
            return 0.0
        counts = Counter(values)
        total = len(values)
        return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

    def average_draw_entropy(self) -> float:
        """How spread-out (high entropy) vs. repetitive (low entropy)
        each individual draw's values tend to be, averaged across the
        dataset. Descriptive only -- entropy near the domain's
        theoretical maximum is expected for genuinely random draws;
        this is not itself a test of randomness."""
        entropies = [self.shannon_entropy(d) for d in self.ds.draws]
        return round(float(np.mean(entropies)), 4) if entropies else 0.0

    def within_draw_runs(self) -> Dict:
        """Counts runs of 2+ consecutive values within individual
        draws (e.g. a draw containing 14, 15, 16 has one run of 3)."""
        run_lengths = []
        for draw in self.ds.draws:
            sorted_vals = sorted(set(draw))
            current = 1
            for i in range(1, len(sorted_vals)):
                if sorted_vals[i] == sorted_vals[i - 1] + 1:
                    current += 1
                else:
                    if current >= 2:
                        run_lengths.append(current)
                    current = 1
            if current >= 2:
                run_lengths.append(current)
        return {
            "total_runs_of_2plus": len(run_lengths),
            "avg_run_length": round(float(np.mean(run_lengths)), 3) if run_lengths else 0.0,
        }

    def within_draw_gap_stats(self) -> Dict:
        """Mean/consistency of the gaps between sorted values within
        each draw (a different axis than gap_since_last_seen, which
        is between-draw)."""
        avg_gaps = []
        for draw in self.ds.draws:
            sorted_vals = sorted(set(draw))
            if len(sorted_vals) < 2:
                continue
            gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
            avg_gaps.append(float(np.mean(gaps)))
        if not avg_gaps:
            return {"overall_avg_gap": 0.0, "gap_consistency": 0.0}
        overall_avg = float(np.mean(avg_gaps))
        std = float(np.std(avg_gaps))
        consistency = max(0.0, 1.0 - (std / overall_avg)) if overall_avg > 0 else 0.0
        return {"overall_avg_gap": round(overall_avg, 3), "gap_consistency": round(consistency, 3)}

    def confirmation_bias_check(self, recent_lookback: int = 5) -> Dict:
        """Overlap between this dataset's own 'hot' numbers and its
        most recent draws. Low overlap flags that a hot-number-based
        method may be chasing a pattern the recent data doesn't
        actually support -- it does not itself indicate anything about
        whether hot numbers are meaningful."""
        hot, _ = self.fe.hot_cold(lookback=20)
        recent = self.ds.draws[-recent_lookback:] if len(self.ds.draws) >= recent_lookback else self.ds.draws
        recent_values = set(v for d in recent for v in d)
        hot_set = set(hot)
        if not hot_set or not recent_values:
            return {"risk_level": "unknown", "overlap_score": 0.0}
        union = hot_set | recent_values
        overlap = len(hot_set & recent_values) / len(union) if union else 0.0
        risk = "high" if overlap < 0.3 else "medium" if overlap < 0.6 else "low"
        return {"risk_level": risk, "overlap_score": round(overlap, 3)}

    def availability_bias_check(self, recent_lookback: int = 10) -> Dict:
        """How much the recent-window frequency distribution deviates
        from the whole-history distribution. A large deviation flags
        that recency-weighted methods may be overweighting a
        short-term blip -- again, a flag to consider, not a
        significance test."""
        recent_freq = self.fe.frequency(recent_lookback)
        overall_freq = self.fe.frequency(None)
        recent_total = sum(recent_freq.values())
        overall_total = sum(overall_freq.values())
        if recent_total == 0 or overall_total == 0:
            return {"risk_level": "unknown", "avg_deviation": 0.0}
        deviations = [
            abs(recent_freq[num] / recent_total - overall_freq[num] / overall_total)
            for num in recent_freq if num in overall_freq
        ]
        avg_dev = float(np.mean(deviations)) if deviations else 0.0
        risk = "high" if avg_dev > 0.1 else "medium" if avg_dev > 0.05 else "low"
        return {"risk_level": risk, "avg_deviation": round(avg_dev, 4)}

    def full_report(self) -> Dict:
        return {
            "average_draw_entropy": self.average_draw_entropy(),
            "within_draw_runs": self.within_draw_runs(),
            "within_draw_gap_stats": self.within_draw_gap_stats(),
            "confirmation_bias_check": self.confirmation_bias_check(),
            "availability_bias_check": self.availability_bias_check(),
        }


# ============================================================
# 3. ANALYSIS  -- each independent modeling technique
# ============================================================
class SlidingWindowRunner:
    """Re-runs frequency analysis across rolling windows and exposes
    a simple trend classifier (rising / falling / flat)."""

    def __init__(self, dataset: DrawDataset, window_size: int = 20, step: int = 20):
        self.ds = dataset
        self.window_size = window_size
        self.step = step

    def run(self) -> List[Dict]:
        results = []
        n = len(self.ds.draws)
        start = 0
        while start < n:
            end = min(start + self.window_size, n)
            freq = Counter()
            for d in self.ds.draws[start:end]:
                freq.update(d)
            results.append({
                "window_start": start,
                "window_end": end,
                "n_draws": end - start,
                "top5": sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5],
            })
            start += self.step
        return results

    @staticmethod
    def detect_trend(values: List[float]) -> str:
        if len(values) < 2:
            return "insufficient_data"
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        avg_diff = float(np.mean(diffs))
        if avg_diff > 0.15:
            return "rising"
        if avg_diff < -0.15:
            return "falling"
        return "flat"


class WavePressureScorer:
    """Pressure (cold-number pull) + wave (recent momentum) + recency
    + balance, combined into one transparent composite score per value."""

    def __init__(self, dataset: DrawDataset, freq_engine: FrequencyEngine):
        self.ds = dataset
        self.fe = freq_engine

    def score_all(self, lookback: int = 20) -> Dict[int, Dict[str, float]]:
        freq = self.fe.frequency(lookback)
        recent = self.ds.draws[-5:] if len(self.ds.draws) >= 5 else self.ds.draws
        recent_counts = Counter(v for d in recent for v in d)
        scores = {}
        for num in self.ds.value_domain():
            f = freq.get(num, 0)
            pressure = max(0, 5 - min(f, 5))
            wave = min(f, 5)
            recency = recent_counts.get(num, 0)
            balance = 1.0  # explicit placeholder, not hidden inside "total"
            scores[num] = {
                "pressure": pressure, "wave": wave, "recency": recency,
                "balance": balance, "total": pressure + wave + recency + balance,
            }
        return scores

    def top_candidates(self, n_needed: int, lookback: int = 20) -> List[int]:
        scores = self.score_all(lookback)
        ranked = sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True)
        return [num for num, _ in ranked[:n_needed]]


class CustomLinearRegression:
    """Minimal from-scratch OLS (no sklearn dependency), used for the
    per-slot difference forecast below."""

    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0

    def fit(self, x: List[float], y: List[float]) -> "CustomLinearRegression":
        x_arr, y_arr = np.array(x, dtype=float), np.array(y, dtype=float)
        if len(x_arr) < 2 or np.all(x_arr == x_arr[0]):
            self.slope = 0.0
            self.intercept = float(np.mean(y_arr)) if len(y_arr) else 0.0
            return self
        x_mean, y_mean = x_arr.mean(), y_arr.mean()
        denom = np.sum((x_arr - x_mean) ** 2)
        self.slope = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / denom) if denom != 0 else 0.0
        self.intercept = float(y_mean - self.slope * x_mean)
        return self

    def predict(self, x_val: float) -> float:
        return self.slope * x_val + self.intercept


class DifferenceModel:
    """Models per-slot differences between consecutive draws and
    regresses them forward. Respects positional order for T1-T3 (no
    re-sorting the output); pool types are de-duplicated and sorted.

    Caveat shared with PiecewiseBinDiffModel below: for pool types,
    each draw is sorted independently, so "slot 0," "slot 1," etc. are
    RANKS within that draw, not a stable identity across draws. If two
    underlying quantities' values cross rank order between one draw
    and the next (one overtakes another), this model's per-slot
    pairing silently attributes one quantity's change to the other's
    slot. This is a structural limitation of per-slot pairing on
    sorted pool data, not a bug to fix here -- documented so it's
    understood rather than discovered by surprise (it was, in fact,
    discovered by surprise while testing PiecewiseBinDiffModel; see
    that class's docstring)."""

    def __init__(self, dataset: DrawDataset):
        self.ds = dataset
        self.cfg = dataset.config

    def consecutive_diffs(self) -> List[List[int]]:
        diffs = []
        for i in range(1, len(self.ds.draws)):
            prev, curr = self.ds.draws[i - 1], self.ds.draws[i]
            diffs.append([curr[j] - prev[j] for j in range(self.cfg["n_slots"])])
        return diffs

    def predict_next(self, lookback: int = 20) -> List[int]:
        diffs = self.consecutive_diffs()
        if not diffs:
            return list(self.ds.last())
        recent = diffs[-lookback:] if len(diffs) > lookback else diffs
        n_slots = self.cfg["n_slots"]
        pred_diff = []
        for slot in range(n_slots):
            series = [d[slot] for d in recent]
            reg = CustomLinearRegression().fit(list(range(len(series))), series)
            pred_diff.append(reg.predict(len(series)))
        last = self.ds.last()
        lo, hi = self.cfg["min_val"], self.cfg["max_val"]
        candidate = [int(round(max(lo, min(hi, last[i] + pred_diff[i])))) for i in range(n_slots)]
        if not self.cfg["positional"]:
            candidate = _dedupe_pool(candidate, lo, hi)
        return candidate


class PiecewiseBinDiffModel:
    """ANALYSIS: bins historical values by quantile and learns the
    average change-to-next-value conditional on which bin a value fell
    in -- merged from LotteryReverseEngineer
    (Reverse-engineering_numeric_transformations_framework_06Dec25) and
    Multi_Framework_Predictor's Segmented Transformation Model (STM),
    which independently arrived at the same core idea.

    Genuinely different hypothesis from DifferenceModel: DifferenceModel
    regresses each slot's difference against TIME (does the gap trend
    up or down across draws); this model conditions the average
    difference on the VALUE itself (do small numbers tend to grow more
    than large ones, regardless of when). Both are legitimate,
    different guesses about what drives the next value -- neither gets
    to skip the significance gate for being a different kind of trend,
    and this model does not itself claim to know which hypothesis (if
    either) is right.

    Shares DifferenceModel's rank-order-stability caveat for pool
    types (see that class's docstring): if a value grows enough to
    overtake a neighboring slot's rank after re-sorting, the per-slot
    pairing this model also relies on can misattribute the change.
    Caught via `scripts/_functional_test.py`'s unit test, which needed
    its synthetic data redesigned with headroom to avoid triggering
    exactly this, once."""

    def __init__(self, dataset: DrawDataset, num_bins: int = 4):
        self.ds = dataset
        self.num_bins = max(1, num_bins)

    def _learn(self) -> Tuple[Optional[np.ndarray], Optional[List[float]]]:
        draws = self.ds.draws
        if len(draws) < 2:
            return None, None
        n_slots = self.ds.config["n_slots"]
        all_values: List[float] = []
        all_diffs: List[float] = []
        for i in range(1, len(draws)):
            prev, curr = draws[i - 1], draws[i]
            for slot in range(n_slots):
                all_values.append(prev[slot])
                all_diffs.append(curr[slot] - prev[slot])
        values = np.array(all_values, dtype=float)
        diffs = np.array(all_diffs, dtype=float)
        quantiles = np.linspace(0, 1, self.num_bins + 1)[1:-1]
        inner_edges = np.quantile(values, quantiles) if len(values) else np.array([])
        bin_edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
        bin_indices = np.digitize(values, bin_edges[1:-1])
        bin_diffs = []
        for b in range(self.num_bins):
            mask = bin_indices == b
            bin_diffs.append(float(np.mean(diffs[mask])) if np.any(mask) else 0.0)
        return bin_edges, bin_diffs

    def predict_next(self) -> List[int]:
        bin_edges, bin_diffs = self._learn()
        cfg = self.ds.config
        last = self.ds.last()
        if bin_edges is None:
            return list(last)
        lo, hi = cfg["min_val"], cfg["max_val"]
        last_arr = np.array(last, dtype=float)
        indices = np.clip(np.digitize(last_arr, bin_edges[1:-1]), 0, len(bin_diffs) - 1)
        candidate = [
            int(round(max(lo, min(hi, last[i] + bin_diffs[indices[i]]))))
            for i in range(cfg["n_slots"])
        ]
        if not cfg["positional"]:
            candidate = _dedupe_pool(candidate, lo, hi)
        return candidate


class KNNEngine:
    """Unified KNN: replaces the three separate Historical / Weighted /
    Enhanced KNN classes from the source scripts with one class and a
    `weight_mode` switch: 'uniform' | 'distance' | 'recency'."""

    def __init__(self, dataset: DrawDataset, k: int = 5, weight_mode: str = "distance"):
        self.ds = dataset
        self.k = k
        self.weight_mode = weight_mode

    @staticmethod
    def _euclidean(a: List[int], b: List[int]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict_next(self) -> List[int]:
        draws = self.ds.draws
        cfg = self.ds.config
        if len(draws) < 2:
            return list(self.ds.last())
        target = draws[-1]
        candidates = draws[:-1]
        dists = [(self._euclidean(target, c), idx) for idx, c in enumerate(candidates)]
        dists.sort(key=lambda x: x[0])
        neighbors = dists[: self.k]
        n_slots = cfg["n_slots"]
        weighted_sum = [0.0] * n_slots
        weight_total = 0.0
        for dist, idx in neighbors:
            next_set = draws[idx + 1]
            if self.weight_mode == "uniform":
                w = 1.0
            elif self.weight_mode == "recency":
                w = (idx + 1) / len(draws)
            else:
                w = 1.0 / (1.0 + dist)
            for s in range(n_slots):
                weighted_sum[s] += next_set[s] * w
            weight_total += w
        if weight_total == 0:
            return list(self.ds.last())
        avg = [weighted_sum[s] / weight_total for s in range(n_slots)]
        lo, hi = cfg["min_val"], cfg["max_val"]
        result = [int(round(max(lo, min(hi, v)))) for v in avg]
        if not cfg["positional"]:
            result = _dedupe_pool(result, lo, hi)
        return result


class SegmentThematicEngine:
    """Buckets the value domain into low/mid/high segments and reports
    which segment split (thematic signature) recent draws favor."""

    def __init__(self, dataset: DrawDataset, n_segments: int = 3):
        self.ds = dataset
        self.n_segments = n_segments

    def _segments(self) -> List[Tuple[int, int]]:
        lo, hi = self.ds.config["min_val"], self.ds.config["max_val"]
        span = hi - lo + 1
        size = math.ceil(span / self.n_segments)
        bounds = []
        for i in range(self.n_segments):
            start = lo + i * size
            end = min(hi, start + size - 1)
            bounds.append((start, end))
        return bounds

    def segment_of(self, value: int) -> int:
        for i, (lo, hi) in enumerate(self._segments()):
            if lo <= value <= hi:
                return i
        return self.n_segments - 1

    def thematic_signature(self, lookback: int = 20) -> Tuple[int, ...]:
        draws = self.ds.draws[-lookback:] if lookback else self.ds.draws
        signatures = Counter()
        for d in draws:
            sig = [0] * self.n_segments
            for v in d:
                sig[self.segment_of(v)] += 1
            signatures[tuple(sig)] += 1
        if not signatures:
            return tuple([0] * self.n_segments)
        return signatures.most_common(1)[0][0]


class ComboSupportEngine:
    """ANALYSIS: tracks co-occurrence of value-tuples (combos), not just
    single numbers -- merged from the HierarchicalComboAnalyzer lineage
    (Combos_checker_* / A_Unified_NumericLLM_* prototypes). Works on
    value co-occurrence within a draw regardless of slot position, so
    it applies the same way to positional and pool types: it answers
    "which k-value combos tend to recur," not "which slot holds which
    value."

    Hierarchy: a combo's confidence can be boosted by the strength of
    its own sub-combos (e.g. a triple boosted by how strong its three
    constituent pairs are) -- pass a one-size-smaller
    ComboSupportEngine's analysis via `sub_engine`/`sub_analysis` to
    enable this."""

    def __init__(self, dataset: DrawDataset, combo_size: int = 2):
        self.ds = dataset
        self.combo_size = combo_size

    def extract_combos(self) -> Dict[Tuple[int, ...], List[int]]:
        """combo -> list of draw indices where all its values co-occurred."""
        occurrences: Dict[Tuple[int, ...], List[int]] = {}
        for idx, draw in enumerate(self.ds.draws):
            values = sorted(set(draw))
            if len(values) < self.combo_size:
                continue
            for combo in combinations(values, self.combo_size):
                occurrences.setdefault(combo, []).append(idx)
        return occurrences

    @staticmethod
    def _trend(indices: List[int]) -> str:
        if len(indices) < 3:
            return "insufficient_data"
        gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        midpoint = len(gaps) // 2
        first_half, second_half = gaps[:midpoint] or gaps, gaps[midpoint:] or gaps
        if np.mean(second_half) < np.mean(first_half) * 0.8:
            return "tightening"
        if np.mean(second_half) > np.mean(first_half) * 1.2:
            return "loosening"
        return "stable"

    def _confidence(self, indices: List[int]) -> float:
        """0-1: blends occurrence rate with interval consistency
        (steadier gaps between occurrences -> higher confidence)."""
        n_draws = len(self.ds.draws)
        if not indices or n_draws == 0:
            return 0.0
        occurrence_rate = len(indices) / n_draws
        if len(indices) >= 2:
            gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
            mean_gap = float(np.mean(gaps))
            gap_cv = (np.std(gaps) / mean_gap) if mean_gap > 0 else 1.0
            consistency = max(0.0, 1.0 - min(gap_cv, 1.0))
        else:
            consistency = 0.0
        return min(1.0, occurrence_rate * 3.0 * 0.6 + consistency * 0.4)

    def analyze(self) -> Dict[Tuple[int, ...], Dict]:
        result = {}
        for combo, indices in self.extract_combos().items():
            result[combo] = {
                "occurrences": len(indices),
                "indices": indices,
                "trend": self._trend(indices),
                "confidence": self._confidence(indices),
                "last_seen_idx": indices[-1],
                "gap_since_last": (len(self.ds.draws) - 1) - indices[-1],
            }
        return result

    def hierarchical_boost(self, combo: Tuple[int, ...], sub_engine: "ComboSupportEngine",
                            sub_analysis: Dict[Tuple[int, ...], Dict]) -> float:
        if sub_engine.combo_size != self.combo_size - 1:
            raise ValueError("sub_engine must be exactly one combo-size smaller")
        sub_confidences = [
            sub_analysis.get(sc, {}).get("confidence", 0.0)
            for sc in combinations(combo, sub_engine.combo_size)
        ]
        return float(np.mean(sub_confidences)) if sub_confidences else 0.0

    def rank_combos(self, top_n: int = 20, sub_engine: Optional["ComboSupportEngine"] = None,
                     sub_analysis: Optional[Dict] = None) -> List[Tuple[Tuple[int, ...], Dict]]:
        analysis = self.analyze()
        for combo, data in analysis.items():
            boost = 0.0
            if sub_engine is not None and sub_analysis is not None:
                boost = self.hierarchical_boost(combo, sub_engine, sub_analysis)
            data["hierarchical_boost"] = boost
            data["combined_score"] = data["confidence"] * 0.7 + boost * 0.3
        ranked = sorted(analysis.items(), key=lambda kv: kv[1]["combined_score"], reverse=True)
        return ranked[:top_n]

    def imminent_combos(self, top_n: int = 10) -> List[Tuple[int, ...]]:
        """"Overdue" combos: gap-since-last-seen most exceeds their own
        historical average gap -- not simply the most frequent combos."""
        overdue_scores = {}
        for combo, data in self.analyze().items():
            indices = data["indices"]
            if len(indices) < 2:
                continue
            gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
            avg_gap = float(np.mean(gaps))
            if avg_gap <= 0:
                continue
            overdue_scores[combo] = data["gap_since_last"] / avg_gap
        ranked = sorted(overdue_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [combo for combo, _ in ranked[:top_n]]

    def candidate_from_top_combo(self, n_slots: int, sub_engine: Optional["ComboSupportEngine"] = None,
                                   sub_analysis: Optional[Dict] = None,
                                   fill_pool: Optional[List[int]] = None) -> List[int]:
        """Builds a full n_slots candidate: seed with the single
        highest-ranked combo's values, then fill remaining slots from
        `fill_pool` (typically another method's candidate, so this
        doesn't need its own frequency logic) or the value domain."""
        ranked = self.rank_combos(top_n=1, sub_engine=sub_engine, sub_analysis=sub_analysis)
        seed = list(dict.fromkeys(ranked[0][0])) if ranked else []
        result = seed[:n_slots]
        pool = fill_pool or list(self.ds.value_domain())
        for v in pool:
            if len(result) >= n_slots:
                break
            if v not in result:
                result.append(v)
        if len(result) < n_slots:
            for v in self.ds.value_domain():
                if len(result) >= n_slots:
                    break
                if v not in result:
                    result.append(v)
        result = result[:n_slots]
        return result if self.ds.config["positional"] else sorted(result)


class GeneticAlgorithmSearch:
    """ANALYSIS: evolutionary search over candidate sets, merged from
    the A_Unified_NumericLLM_* prototype's genetic_algorithm_prediction.
    Uses a caller-supplied fitness function built from other engines'
    signals (e.g. mean WavePressureScorer score across the candidate's
    values) -- this is a different SEARCH STRATEGY over the same
    signal space, not a new source of information about the future.
    Its output is validated exactly like every other method; nothing
    here exempts it from the significance gate."""

    def __init__(self, dataset: DrawDataset, fitness_fn: Callable[[List[int]], float],
                 population_size: int = 40, generations: int = 25, mutation_rate: float = 0.15):
        self.ds = dataset
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def _random_candidate(self) -> List[int]:
        cfg = self.ds.config
        domain = self.ds.value_domain()
        cand = random.sample(domain, cfg["n_slots"]) if cfg["unique"] else \
            [random.choice(domain) for _ in range(cfg["n_slots"])]
        return sorted(cand) if not cfg["positional"] else cand

    def _mutate(self, candidate: List[int]) -> List[int]:
        cfg = self.ds.config
        domain = self.ds.value_domain()
        mutated = list(candidate)
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.choice(domain)
        if cfg["unique"]:
            mutated = list(dict.fromkeys(mutated))
            while len(mutated) < cfg["n_slots"]:
                v = random.choice(domain)
                if v not in mutated:
                    mutated.append(v)
            mutated = sorted(mutated[: cfg["n_slots"]])
        return mutated

    def _crossover(self, a: List[int], b: List[int]) -> List[int]:
        cfg = self.ds.config
        point = random.randint(1, len(a) - 1) if len(a) > 1 else 1
        child = a[:point] + b[point:]
        if cfg["unique"]:
            child = list(dict.fromkeys(child))
            domain = self.ds.value_domain()
            while len(child) < cfg["n_slots"]:
                v = random.choice(domain)
                if v not in child:
                    child.append(v)
            child = sorted(child[: cfg["n_slots"]])
        return child

    def run(self) -> List[int]:
        population = [self._random_candidate() for _ in range(self.population_size)]
        best, best_fitness = None, float("-inf")
        for _ in range(self.generations):
            scored = sorted(population, key=self.fitness_fn, reverse=True)
            top_fitness = self.fitness_fn(scored[0])
            if top_fitness > best_fitness:
                best, best_fitness = scored[0], top_fitness
            elite = scored[: max(2, self.population_size // 5)]
            next_gen = list(elite)
            while len(next_gen) < self.population_size:
                parent_a, parent_b = random.sample(elite, 2) if len(elite) >= 2 else (elite[0], elite[0])
                next_gen.append(self._mutate(self._crossover(parent_a, parent_b)))
            population = next_gen
        return best if best is not None else population[0]


class TOPSISCombiner:
    """ANALYSIS/FRAMEWORK: TOPSIS (Technique for Order Preference by
    Similarity to Ideal Solution) -- a real multi-criteria decision
    algorithm, merged from Systematical_Manual_Arrangement_*'s
    MCDAFramework, and a genuinely different combination strategy than
    EnsembleCombiner's majority vote: it ranks values by distance to an
    ideal point across several weighted criteria simultaneously.

    Two things were fixed relative to the source:
    1. The source silently fell back to `random.random()` scores on
       any exception -- masking bugs as data. This implementation lets
       exceptions propagate; a malformed `criteria_scores` input is a
       bug to fix, not noise to paper over.
    2. `DEFAULT_WEIGHTS` below are asserted, not fitted or validated --
       exactly as unproven as the source's were, just documented as
       such instead of commented as "most predictive." Override them
       via the constructor if you have your own (equally unvalidated)
       prior. TOPSIS producing a ranked candidate does not make these
       weights correct -- only VALIDATE, applied to this method's
       output like every other, can speak to that.
    """

    DEFAULT_WEIGHTS = {
        "pressure": 0.25,
        "wave": 0.35,
        "recency": 0.25,
        "gap_due": 0.10,
        "combo_support": 0.05,
    }

    def __init__(self, criteria_weights: Optional[Dict[str, float]] = None):
        self.criteria_weights = criteria_weights or dict(self.DEFAULT_WEIGHTS)

    def rank(self, alternatives: List[int], criteria_scores: Dict[str, Dict[int, float]]) -> List[Tuple[int, float]]:
        if not alternatives:
            return []
        normalized: Dict[str, Dict[int, float]] = {}
        for criterion in self.criteria_weights:
            scores = criteria_scores.get(criterion, {})
            if not scores:
                continue
            max_score = max(scores.values())
            if max_score > 0:
                normalized[criterion] = {num: s / max_score for num, s in scores.items()}

        ranked = []
        for num in alternatives:
            pos_dist_sq = 0.0
            neg_dist_sq = 0.0
            for criterion, weight in self.criteria_weights.items():
                score = normalized.get(criterion, {}).get(num, 0.0)
                pos_dist_sq += weight * (1 - score) ** 2
                neg_dist_sq += weight * score ** 2
            pos_dist = math.sqrt(pos_dist_sq)
            neg_dist = math.sqrt(neg_dist_sq)
            topsis_score = neg_dist / (pos_dist + neg_dist) if (pos_dist + neg_dist) > 0 else 0.0
            ranked.append((num, topsis_score))
        return sorted(ranked, key=lambda kv: kv[1], reverse=True)

    def candidate(self, ds: DrawDataset, criteria_scores: Dict[str, Dict[int, float]], n_slots: int) -> List[int]:
        ranked = self.rank(ds.value_domain(), criteria_scores)
        top = [num for num, _ in ranked[:n_slots]]
        return top if ds.config["positional"] else sorted(top)


def _build_topsis_criteria(ds: DrawDataset, lookback: int) -> Dict[str, Dict[int, float]]:
    """Projects existing engines' per-value signals into the criteria
    shape TOPSISCombiner needs -- reuses already-computed scores rather
    than inventing new ones for this method alone."""
    fe = FrequencyEngine(ds)
    wp_scores = WavePressureScorer(ds, fe).score_all(lookback)
    pressure = {v: s["pressure"] for v, s in wp_scores.items()}
    wave = {v: s["wave"] for v, s in wp_scores.items()}
    recency = {v: s["recency"] for v, s in wp_scores.items()}
    gap_due = {v: float(g) for v, g in fe.gap_since_last_seen().items()}
    combo_value_scores: Dict[int, float] = defaultdict(float)
    for combo, data in ComboSupportEngine(ds, combo_size=2).analyze().items():
        for v in combo:
            combo_value_scores[v] = max(combo_value_scores[v], data["confidence"])
    return {
        "pressure": pressure,
        "wave": wave,
        "recency": recency,
        "gap_due": gap_due,
        "combo_support": dict(combo_value_scores),
    }


class ConvergenceSelector:
    """FRAMEWORK: a third combination strategy alongside EnsembleCombiner
    (majority vote) and TOPSISCombiner (multi-criteria distance
    ranking) -- merged from CLPC_Model_V2's `calculate_convergence` /
    `_apply_convergence_check`. Rather than taking the single
    highest-scoring values independently, this searches a bounded pool
    of top-scored candidates for the SET_SIZE-combination whose member
    scores are most internally consistent (lowest coefficient of
    variation), on the premise that a genuinely coherent signal should
    produce similarly-strong scores across its members rather than one
    outlier propping up an otherwise weak set.

    That premise is itself untested -- exactly like TOPSIS's criteria
    weights, picking for internal consistency is an assumption, not a
    validated finding. This method's output goes through the same
    significance gate as everything else; a low p-value here would
    mean the CONSISTENCY premise happened to pay off on this dataset,
    not that it's correct in general.

    The candidate pool is deliberately kept small (default
    `max(n_slots + 4, 2 * n_slots)`) because the search is a full
    combinations() scan over the pool -- this is what makes the search
    tractable, not an accuracy choice."""

    def __init__(self, dataset: DrawDataset):
        self.ds = dataset

    @staticmethod
    def _convergence(scores: List[float]) -> float:
        if len(scores) < 2:
            return 0.0
        avg = float(np.mean(scores))
        std = float(np.std(scores))
        cv = std / avg if avg > 0 else float("inf")
        return max(0.0, 1.0 - min(cv, 1.0))

    def select(self, value_scores: Dict[int, float], n_slots: int, pool_size: Optional[int] = None) -> List[int]:
        ranked = sorted(value_scores.items(), key=lambda kv: kv[1], reverse=True)
        pool_size = pool_size or max(n_slots + 4, 2 * n_slots)
        pool = [num for num, _ in ranked[:pool_size]]
        if len(pool) < n_slots:
            domain = self.ds.value_domain()
            for v in domain:
                if v not in pool:
                    pool.append(v)
                if len(pool) >= n_slots:
                    break

        best_set: Optional[Tuple[int, ...]] = None
        best_score = -1.0
        for candidate_set in combinations(pool, n_slots):
            member_scores = [value_scores.get(v, 0.0) for v in candidate_set]
            convergence = self._convergence(member_scores)
            set_score = float(np.mean(member_scores)) + convergence * 0.5
            if set_score > best_score:
                best_score = set_score
                best_set = candidate_set

        result = list(best_set) if best_set else pool[:n_slots]
        return result if self.ds.config["positional"] else sorted(result)


def _build_convergence_candidate(ds: DrawDataset, n_slots: int, lookback: int) -> List[int]:
    """Shared helper: reuses WavePressureScorer's composite score as
    the value_scores input, same reuse pattern as GA/TOPSIS above."""
    scores = WavePressureScorer(ds, FrequencyEngine(ds)).score_all(lookback)
    value_scores = {v: s["total"] for v, s in scores.items()}
    return ConvergenceSelector(ds).select(value_scores, n_slots)


def _dedupe_pool(values: List[int], lo: int, hi: int) -> List[int]:
    seen, out = set(), []
    for v in values:
        while v in seen:
            v = v + 1 if v < hi else lo
        seen.add(v)
        out.append(v)
    return sorted(out)


# ============================================================
# 4. TRIAGE  -- candidate-generating anticipators
# ============================================================
class DueElementAnticipator:
    """Values with the largest gap since last appearance."""

    def __init__(self, dataset: DrawDataset, freq_engine: FrequencyEngine):
        self.ds = dataset
        self.fe = freq_engine

    def anticipate(self, n_needed: int) -> List[int]:
        gaps = self.fe.gap_since_last_seen()
        ranked = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in ranked[:n_needed]]


class ZeroFrequencyAnticipator:
    """Values with zero occurrences in the lookback window. Falls
    back to the least-frequent non-zero values if there aren't enough
    true zero-frequency values to fill the slots."""

    def __init__(self, dataset: DrawDataset, freq_engine: FrequencyEngine):
        self.ds = dataset
        self.fe = freq_engine

    def anticipate(self, n_needed: int, lookback: int = 20) -> List[int]:
        freq = self.fe.frequency(lookback)
        zero_freq = [num for num in self.ds.value_domain() if freq.get(num, 0) == 0]
        if len(zero_freq) >= n_needed:
            # Deterministic tie-break: most overdue (largest
            # gap-since-last-seen) first, then ascending value. This
            # used to be `random.sample(zero_freq, n_needed)`, which
            # made the result depend on the global random module's
            # call order rather than only on the data -- caught via a
            # cache-equivalence regression test that produced two
            # different near-miss rates for the identical dataset
            # depending on which other methods ran first in the same
            # process. Fixed for reproducibility: given the same data
            # and parameters, this method now always returns the same
            # candidate.
            gaps = self.fe.gap_since_last_seen()
            ranked = sorted(zero_freq, key=lambda n: (-gaps.get(n, 0), n))
            return sorted(ranked[:n_needed])
        remaining_needed = n_needed - len(zero_freq)
        non_zero_sorted = sorted(
            (n for n in self.ds.value_domain() if n not in zero_freq),
            key=lambda n: freq.get(n, 0),
        )
        return sorted(zero_freq + non_zero_sorted[:remaining_needed])


@dataclass
class MethodResult:
    name: str
    candidate: List[int]
    rationale: str


# ============================================================
# 5. FRAMEWORK  -- transparent ensemble combiner
# ============================================================
class EnsembleCombiner:
    """Combines every method's candidate via majority vote rather than
    silently favoring one technique. Positional types vote per slot
    position; pool types vote over the pooled value set."""

    def __init__(self, dataset: DrawDataset):
        self.ds = dataset

    def combine(self, method_results: List[MethodResult]) -> Dict:
        n_slots = self.ds.config["n_slots"]
        all_values = [v for mr in method_results for v in mr.candidate]
        vote_counts = Counter(all_values)
        ranked = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)

        if self.ds.config["positional"]:
            slot_votes = [Counter() for _ in range(n_slots)]
            for mr in method_results:
                for i, v in enumerate(mr.candidate):
                    slot_votes[i][v] += 1
            ensemble_set = [sv.most_common(1)[0][0] if sv else 0 for sv in slot_votes]
        else:
            pool = [v for v, _ in ranked]
            if len(pool) >= n_slots:
                ensemble_set = sorted(pool[:n_slots])
            else:
                lo = self.ds.config["min_val"]
                filler = [v for v in range(lo, lo + (n_slots - len(pool))) if v not in pool]
                ensemble_set = sorted(pool + filler[: n_slots - len(pool)])

        return {
            "ensemble_set": ensemble_set,
            "vote_counts": dict(ranked[:15]),
            "contributing_methods": [mr.name for mr in method_results],
        }


class AdaptiveWeightedEnsemble:
    """FRAMEWORK: like EnsembleCombiner, but each base method's vote is
    weighted by that method's OWN recent walk-forward hit rate,
    measured strictly on data available at prediction time -- merged
    from AdaptiveLotteryScanner's learn_and_adapt.

    Reimplemented to close the source's leakage risk: the original
    reweighted based on a post-hoc backtest without re-validating the
    reweighted output, which would let an "adaptive" method quietly
    bypass the significance gate. Here, the adaptively-weighted
    candidate is validated through RandomBaselineValidator exactly
    like every other method in run_full_pipeline -- adaptiveness is
    not a validation exemption.

    Weights are computed over a bounded trailing window (not the full
    history) to keep this tractable when it's itself re-evaluated at
    every step of the outer walk-forward validation loop -- see
    references/runbook.md's performance note before pointing this at
    a very large dataset."""

    def __init__(self, dataset: DrawDataset, base_candidate_fns: Dict[str, Callable[[DrawDataset], List[int]]],
                 min_history: int = 10, window: int = 15):
        self.ds = dataset
        self.base_candidate_fns = base_candidate_fns
        self.min_history = min_history
        self.window = window

    def _method_weights(self) -> Dict[str, float]:
        draws = self.ds.draws
        n = len(draws)
        cfg = self.ds.config
        k_req = max(1, cfg["n_slots"] // 2)
        hits = {name: 0 for name in self.base_candidate_fns}
        trials = 0
        start = max(self.min_history, n - self.window - 1)
        for i in range(start, max(start, n - 1)):
            partial_ds = DrawDataset.__new__(DrawDataset)
            partial_ds.type_key = self.ds.type_key
            partial_ds.config = cfg
            partial_ds.set_names = self.ds.set_names[:i]
            partial_ds.draws = draws[:i]
            actual_next = draws[i]
            for name, fn in self.base_candidate_fns.items():
                cand = fn(partial_ds)
                match = (sum(1 for a, b in zip(cand, actual_next) if a == b) if cfg["positional"]
                         else len(set(cand) & set(actual_next)))
                if match >= k_req:
                    hits[name] += 1
            trials += 1

        n_methods = len(self.base_candidate_fns)
        if trials == 0 or n_methods == 0:
            return {name: 1.0 / n_methods for name in self.base_candidate_fns} if n_methods else {}
        rates = {name: hits[name] / trials for name in hits}
        total = sum(rates.values())
        if total <= 0:
            return {name: 1.0 / n_methods for name in rates}
        return {name: r / total for name, r in rates.items()}

    def candidate(self, n_slots: int) -> List[int]:
        weights = self._method_weights()
        vote_scores: Dict[int, float] = defaultdict(float)
        for name, fn in self.base_candidate_fns.items():
            cand = fn(self.ds)
            w = weights.get(name, 0.0)
            for v in cand:
                vote_scores[v] += w
        ranked = sorted(vote_scores.items(), key=lambda kv: kv[1], reverse=True)
        top = [v for v, _ in ranked[:n_slots]]
        if len(top) < n_slots:
            for v in self.ds.value_domain():
                if len(top) >= n_slots:
                    break
                if v not in top:
                    top.append(v)
        top = top[:n_slots]
        return top if self.ds.config["positional"] else sorted(top)


# ============================================================
# 6. VALIDATE  -- the statistical-honesty layer
# ============================================================
class RandomBaselineValidator:
    """Walk-forward comparison of each method's actual historical hit
    rate against the probability of hitting by pure chance, with raw
    and Benjamini-Hochberg-corrected p-values reported side by side.

    Also computes an optional near-miss diagnostic (values within a
    tolerance of an actual value, merged from the
    Reverse-engineering_numeric_transformations prototype's hit_count
    logic). This is DESCRIPTIVE ONLY -- it is never allowed to feed
    into `binomial_p_value`, `benjamini_hochberg`, or any
    "significant" flag. Loosening the match criterion after seeing a
    weak exact-match result, to manufacture a better-looking number,
    is p-hacking with extra steps. See references/playbook.md."""

    def __init__(self, dataset: DrawDataset):
        self.ds = dataset

    def _match_count(self, candidate: List[int], actual: List[int]) -> int:
        if self.ds.config["positional"]:
            return sum(1 for a, b in zip(candidate, actual) if a == b)
        return len(set(candidate) & set(actual))

    def historical_hit_rate(self, candidate_fn, min_history: int = 10) -> Tuple[float, int]:
        """candidate_fn(partial_dataset) -> candidate list. Re-evaluated
        walk-forward at each historical point -- no future leakage."""
        hits, trials = 0, 0
        k_req = max(1, self.ds.config["n_slots"] // 2)
        for i in range(min_history, len(self.ds.draws) - 1):
            partial_ds = DrawDataset.__new__(DrawDataset)
            partial_ds.type_key = self.ds.type_key
            partial_ds.config = self.ds.config
            partial_ds.set_names = self.ds.set_names[:i]
            partial_ds.draws = self.ds.draws[:i]
            candidate = candidate_fn(partial_ds)
            actual_next = self.ds.draws[i]
            match = self._match_count(candidate, actual_next)
            hits += 1 if match >= k_req else 0
            trials += 1
        return (hits / trials if trials else 0.0), trials

    def near_miss_count(self, candidate: List[int], actual: List[int], threshold: int) -> int:
        """DIAGNOSTIC ONLY (see class docstring). Values within
        `threshold` of an actual value that were not exact hits.
        Positional types compare per-slot; pool types greedily pair
        each candidate value to its closest unused, non-exact actual
        value within tolerance."""
        if self.ds.config["positional"]:
            return sum(
                1 for c, a in zip(candidate, actual)
                if c != a and abs(c - a) <= threshold
            )
        used = set()
        near = 0
        for c in candidate:
            if c in actual:
                continue  # exact hits are not near-misses
            best = None
            for a in actual:
                if a in used or a in candidate:
                    continue
                if abs(c - a) <= threshold and (best is None or abs(c - a) < abs(c - best)):
                    best = a
            if best is not None:
                used.add(best)
                near += 1
        return near

    def historical_near_miss_rate(self, candidate_fn, threshold: int, min_history: int = 10) -> Tuple[float, int]:
        """DIAGNOSTIC ONLY. Average near-misses per trial, walk-forward.
        Do not report this as evidence of a method beating chance --
        it has no null-hypothesis baseline computed for it and is not
        BH-corrected. It exists to show *how close* misses tend to be,
        nothing more."""
        total_near, trials = 0, 0
        for i in range(min_history, len(self.ds.draws) - 1):
            partial_ds = DrawDataset.__new__(DrawDataset)
            partial_ds.type_key = self.ds.type_key
            partial_ds.config = self.ds.config
            partial_ds.set_names = self.ds.set_names[:i]
            partial_ds.draws = self.ds.draws[:i]
            candidate = candidate_fn(partial_ds)
            actual_next = self.ds.draws[i]
            total_near += self.near_miss_count(candidate, actual_next, threshold)
            trials += 1
        return (total_near / trials if trials else 0.0), trials

    @staticmethod
    def null_hit_probability(cfg: Dict) -> float:
        """Analytical probability of a random guess hitting >= half the
        slots by chance, under this game's exact constraints."""
        n_slots = cfg["n_slots"]
        domain_size = cfg["max_val"] - cfg["min_val"] + 1
        k_req = max(1, n_slots // 2)
        if cfg["unique"]:
            return sum(
                comb(n_slots, k) * comb(domain_size - n_slots, n_slots - k) / comb(domain_size, n_slots)
                for k in range(k_req, n_slots + 1)
            )
        p_single = 1.0 / domain_size
        return sum(
            comb(n_slots, k) * (p_single ** k) * ((1 - p_single) ** (n_slots - k))
            for k in range(k_req, n_slots + 1)
        )

    @staticmethod
    def binomial_p_value(hits: int, trials: int, p_null: float) -> float:
        """One-sided p-value: P(X >= hits) under Binomial(trials, p_null)."""
        if trials == 0:
            return 1.0
        p_val = sum(
            comb(trials, k) * (p_null ** k) * ((1 - p_null) ** (trials - k))
            for k in range(hits, trials + 1)
        )
        return min(1.0, p_val)

    @staticmethod
    def benjamini_hochberg(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
        items = sorted(p_values.items(), key=lambda x: x[1])  # ascending raw p
        m = len(items)
        adjusted = [0.0] * m
        running_min = 1.0
        for idx in range(m - 1, -1, -1):
            rank = idx + 1
            _, p = items[idx]
            candidate = p * m / rank
            running_min = min(running_min, candidate)
            adjusted[idx] = running_min
        result = {}
        for idx, (name, p) in enumerate(items):
            adj = min(1.0, adjusted[idx])
            result[name] = {
                "raw_p": round(p, 5),
                "bh_adjusted_p": round(adj, 5),
                "significant_after_correction": adj < alpha,
            }
        return result


class WalkForwardCache:
    """VALIDATE (performance layer): precomputes every base method's
    candidate, exact-hit flag, and near-miss count at every
    walk-forward point ONCE, and shares that single pass across the
    per-method hit-rate loop, EnsembleCombiner, and
    AdaptiveWeightedEnsemble.

    Why this exists: before it did, `ensemble` and `adaptive_weighted`
    each recomputed every one of the ~10 base methods from scratch at
    every walk-forward point during their OWN validation loop --
    `adaptive_weighted` additionally recomputed all of them again
    across its own internal trailing window at every one of those
    points. On a ~30-40 draw demo dataset this made a full
    run_full_pipeline call take 15-30 seconds. This cache turns "run
    every base method at every walk-forward point" into something that
    happens exactly once, and turns ensemble/adaptive-weighted's
    validation into cheap lookups against already-computed results.

    This is a pure performance change: it does not alter what gets
    validated, how many walk-forward trials are counted, or how
    significance is computed -- only how many times each candidate
    gets computed to arrive at the same numbers. `scripts/
    _functional_test.py` includes a check that this cache's hit-rate
    output for a base method matches `RandomBaselineValidator.
    historical_hit_rate`'s slower, independent computation exactly, to
    guard against the optimization silently changing results."""

    def __init__(self, ds: DrawDataset, base_candidate_fns: Dict[str, Callable[[DrawDataset], List[int]]],
                 min_history: int, near_miss_threshold: int):
        self.ds = ds
        self.method_names = list(base_candidate_fns.keys())
        self.k_req = max(1, ds.config["n_slots"] // 2)
        self.indices: List[int] = list(range(min_history, len(ds.draws) - 1))
        validator = RandomBaselineValidator(ds)

        self.candidates: Dict[str, Dict[int, List[int]]] = {name: {} for name in self.method_names}
        self.hits: Dict[str, Dict[int, bool]] = {name: {} for name in self.method_names}
        self.near_misses: Dict[str, Dict[int, int]] = {name: {} for name in self.method_names}

        for i in self.indices:
            partial_ds = DrawDataset.__new__(DrawDataset)
            partial_ds.type_key = ds.type_key
            partial_ds.config = ds.config
            partial_ds.set_names = ds.set_names[:i]
            partial_ds.draws = ds.draws[:i]
            actual_next = ds.draws[i]
            for name, fn in base_candidate_fns.items():
                cand = fn(partial_ds)
                self.candidates[name][i] = cand
                self.hits[name][i] = validator._match_count(cand, actual_next) >= self.k_req
                self.near_misses[name][i] = validator.near_miss_count(cand, actual_next, near_miss_threshold)

    def hit_rate(self, name: str) -> Tuple[float, int]:
        trials = len(self.indices)
        if trials == 0:
            return 0.0, 0
        hits = sum(1 for i in self.indices if self.hits[name][i])
        return hits / trials, trials

    def near_miss_rate(self, name: str) -> Tuple[float, int]:
        trials = len(self.indices)
        if trials == 0:
            return 0.0, 0
        total = sum(self.near_misses[name][i] for i in self.indices)
        return total / trials, trials

    def ensemble_stats(self, near_miss_threshold: int) -> Tuple[Tuple[float, int], Tuple[float, int]]:
        """Derives ensemble's (hit_rate, trials) and (near_miss_rate,
        trials) from cached base candidates -- no base method is ever
        recomputed here."""
        combiner = EnsembleCombiner(self.ds)
        validator = RandomBaselineValidator(self.ds)
        hits, near_total, trials = 0, 0, 0
        for i in self.indices:
            method_results = [MethodResult(name, self.candidates[name][i], "") for name in self.method_names]
            ensemble_set = combiner.combine(method_results)["ensemble_set"]
            actual_next = self.ds.draws[i]
            if validator._match_count(ensemble_set, actual_next) >= self.k_req:
                hits += 1
            near_total += validator.near_miss_count(ensemble_set, actual_next, near_miss_threshold)
            trials += 1
        hit_rate = hits / trials if trials else 0.0
        near_rate = near_total / trials if trials else 0.0
        return (hit_rate, trials), (near_rate, trials)

    def adaptive_weighted_stats(self, window: int, near_miss_threshold: int) -> Tuple[Tuple[float, int], Tuple[float, int]]:
        """Derives adaptive_weighted's (hit_rate, trials) and
        (near_miss_rate, trials) from cached base hits/candidates. The
        weight computation that used to re-run a (window+1)-step nested
        walk-forward at EVERY outer step is now a handful of dict
        lookups per step -- semantically identical to
        AdaptiveWeightedEnsemble._method_weights, just reading from the
        shared cache instead of recomputing."""
        validator = RandomBaselineValidator(self.ds)
        n_methods = len(self.method_names)
        n_slots = self.ds.config["n_slots"]
        cfg = self.ds.config
        hits, near_total, trials = 0, 0, 0
        for pos, i in enumerate(self.indices):
            window_indices = [j for j in self.indices[:pos] if j >= i - window]
            if not window_indices or n_methods == 0:
                weights = {name: 1.0 / n_methods for name in self.method_names} if n_methods else {}
            else:
                rates = {
                    name: sum(1 for j in window_indices if self.hits[name][j]) / len(window_indices)
                    for name in self.method_names
                }
                total = sum(rates.values())
                weights = ({name: 1.0 / n_methods for name in self.method_names} if total <= 0
                           else {name: r / total for name, r in rates.items()})

            vote_scores: Dict[int, float] = defaultdict(float)
            for name in self.method_names:
                w = weights.get(name, 0.0)
                for v in self.candidates[name][i]:
                    vote_scores[v] += w
            ranked = sorted(vote_scores.items(), key=lambda kv: kv[1], reverse=True)
            top = [v for v, _ in ranked[:n_slots]]
            if len(top) < n_slots:
                for v in self.ds.value_domain():
                    if len(top) >= n_slots:
                        break
                    if v not in top:
                        top.append(v)
            top = top[:n_slots]
            candidate = top if cfg["positional"] else sorted(top)

            actual_next = self.ds.draws[i]
            if validator._match_count(candidate, actual_next) >= self.k_req:
                hits += 1
            near_total += validator.near_miss_count(candidate, actual_next, near_miss_threshold)
            trials += 1

        hit_rate = hits / trials if trials else 0.0
        near_rate = near_total / trials if trials else 0.0
        return (hit_rate, trials), (near_rate, trials)


# ============================================================
# 7. FLOW  -- per-type orchestration + multi-type driver
# ============================================================
@dataclass
class QualityAdvisory:
    should_reconsider: bool
    risk_level: str
    notes: List[str]
    recommended_actions: List[str]


def assess_run_quality(report: Dict) -> QualityAdvisory:
    """One-shot, human-facing quality advisory -- merged from
    Systematical_HierarchicalComboAnalyzer_18Sept25's validation-loop
    concept, DEFANGED: the source auto-retried analysis with adjusted
    parameters up to 3 times whenever these same checks failed, which
    mechanizes exactly what references/playbook.md forbids doing by
    hand ("re-run with different parameters until something clears").

    This function only REPORTS a recommendation. Nothing in this file
    calls it in a loop, and it must never be wired to automatically
    call run_full_pipeline again. If a human decides to rerun with
    different parameters after reading this advisory, that rerun --
    and every prior attempt -- belongs in the ledger
    (ledger/example-hypothesis-ledger.md), not just whichever looked
    best. scripts/verify-significance-gate.sh checks statically that
    no loop keyed on `should_reconsider` exists in this file."""
    notes: List[str] = []
    recommended_actions: List[str] = []
    risk_level = "LOW"

    n_draws = report["n_draws_analyzed"]
    if n_draws < 30:
        notes.append(f"Only {n_draws} draws analyzed -- validation trial counts are likely "
                      f"below the 20-trial floor in references/playbook.md.")
        recommended_actions.append("Treat all significance results here as provisional; "
                                    "gather more history before trusting a null or positive result.")
        risk_level = "MEDIUM"

    validation = report["validation"]["per_method"]
    total_methods = len(validation)
    high_rate_count = sum(1 for v in validation.values() if v.get("hit_rate", 0) > 0.85)
    if total_methods > 3 and high_rate_count / total_methods > 0.3:
        notes.append("Unusually high fraction of methods show a high raw hit rate -- check for a "
                      "shared data leak (e.g. a candidate_fn peeking at data beyond the partial "
                      "dataset it was given) before assuming this reflects real signal.")
        recommended_actions.append("Re-verify each method's candidate_fn only uses the partial "
                                    "dataset passed to it in RandomBaselineValidator, not the full dataset.")
        risk_level = "HIGH"

    significant_count = sum(1 for v in validation.values() if v.get("significant_after_correction"))
    if significant_count > 1:
        notes.append(f"{significant_count} methods cleared BH correction in the same run -- unusual "
                      f"for a certified-random game. Re-check BH correction was computed across ALL "
                      f"methods tested, not a subset (playbook.md step 3).")
        risk_level = "HIGH"

    if not notes:
        notes.append("No quality flags raised.")

    return QualityAdvisory(
        should_reconsider=risk_level in ("MEDIUM", "HIGH"),
        risk_level=risk_level,
        notes=notes,
        recommended_actions=recommended_actions,
    )


def _build_combo_candidate(ds: DrawDataset, n_slots: int, fill_pool: List[int]) -> List[int]:
    """Shared helper: builds a full candidate from the top hierarchical
    combo, boosted by its sub-combos where the slot count allows it."""
    max_combo_size = min(4, n_slots)
    if max_combo_size < 2:
        return list(ds.last())
    engines = {size: ComboSupportEngine(ds, combo_size=size) for size in range(2, max_combo_size + 1)}
    analyses = {size: eng.analyze() for size, eng in engines.items()}
    top_size = max_combo_size
    sub_size = top_size - 1
    return engines[top_size].candidate_from_top_combo(
        n_slots,
        sub_engine=engines.get(sub_size),
        sub_analysis=analyses.get(sub_size),
        fill_pool=fill_pool,
    )


def _build_ga_candidate(ds: DrawDataset, lookback: int) -> List[int]:
    """Shared helper: GA search using WavePressureScorer's composite
    score as the fitness function -- ties the search to an already-
    computed, already-validated signal rather than inventing a new one."""
    scores = WavePressureScorer(ds, FrequencyEngine(ds)).score_all(lookback)

    def fitness(candidate: List[int]) -> float:
        vals = [scores.get(v, {"total": 0.0})["total"] for v in candidate]
        return float(np.mean(vals)) if vals else 0.0

    return GeneticAlgorithmSearch(ds, fitness_fn=fitness).run()


def _build_topsis_candidate(ds: DrawDataset, n_slots: int, lookback: int) -> List[int]:
    """Shared helper: TOPSIS ranking over the criteria built from
    already-computed signals (see _build_topsis_criteria)."""
    criteria = _build_topsis_criteria(ds, lookback)
    return TOPSISCombiner().candidate(ds, criteria, n_slots)


def _build_piecewise_candidate(ds: DrawDataset) -> List[int]:
    """Shared helper: PiecewiseBinDiffModel with the engine's default
    4 quantile bins."""
    return PiecewiseBinDiffModel(ds, num_bins=4).predict_next()


def run_full_pipeline(type_key: str, historical_data: Dict[str, List[int]],
                       lookback: int = 20, min_history: int = 10,
                       near_miss_threshold: int = 2, verbose: bool = True) -> Dict:
    ds = DrawDataset(type_key, historical_data)
    cfg = ds.config
    n_slots = cfg["n_slots"]

    fe = FrequencyEngine(ds)
    sw = SlidingWindowRunner(ds, window_size=lookback, step=lookback)
    seg = SegmentThematicEngine(ds)
    diagnostics = DescriptiveDiagnostics(ds, fe)

    wave_pressure_candidate = WavePressureScorer(ds, fe).top_candidates(n_slots, lookback)

    # ---- ANALYSIS + TRIAGE: run every technique on the full (live)
    #      dataset once, for the report's method_candidates section.
    #      This is NOT the walk-forward validation pass below -- it's
    #      just "what would each method anticipate right now." ----
    method_results = [
        MethodResult(
            "wave_pressure",
            wave_pressure_candidate,
            "Highest combined pressure(cold)+wave(momentum)+recency score",
        ),
        MethodResult(
            "difference_model",
            DifferenceModel(ds).predict_next(lookback),
            "Per-slot linear regression on consecutive-draw differences",
        ),
        MethodResult(
            "piecewise_bin_diff",
            _build_piecewise_candidate(ds),
            "Average next-value change conditional on which quantile bin the "
            "previous value fell in (value-conditional, not time-conditional -- "
            "see PiecewiseBinDiffModel docstring for how this differs from difference_model)",
        ),
        MethodResult(
            "knn",
            KNNEngine(ds, k=5, weight_mode="distance").predict_next(),
            "Distance-weighted average of the k nearest historical draws' successors",
        ),
        MethodResult(
            "due_element",
            DueElementAnticipator(ds, fe).anticipate(n_slots),
            "Largest gap-since-last-seen per value",
        ),
        MethodResult(
            "zero_frequency",
            ZeroFrequencyAnticipator(ds, fe).anticipate(n_slots, lookback),
            f"Zero occurrences in the last {lookback} draws",
        ),
        MethodResult(
            "combo_support",
            _build_combo_candidate(ds, n_slots, fill_pool=wave_pressure_candidate),
            "Seeded from the highest-confidence hierarchical value-combo "
            "(boosted by its own sub-combos' support), filled out with wave-pressure candidates",
        ),
        MethodResult(
            "genetic_algorithm",
            _build_ga_candidate(ds, lookback),
            "Evolutionary search maximizing mean wave-pressure score across candidate members",
        ),
        MethodResult(
            "topsis",
            _build_topsis_candidate(ds, n_slots, lookback),
            "TOPSIS multi-criteria ranking over pressure/wave/recency/gap-due/combo-support "
            "(criteria weights are asserted, not validated -- see TOPSISCombiner docstring)",
        ),
        MethodResult(
            "convergence_selector",
            _build_convergence_candidate(ds, n_slots, lookback),
            "Searches a bounded top-scored pool for the combination with the most internally "
            "consistent member scores (asserted premise, not validated -- see ConvergenceSelector docstring)",
        ),
    ]

    # These are the 10 BASE methods that both adaptive_weighted and
    # ensemble draw from. Kept as one dict, reused for the live
    # adaptive_weighted candidate below AND for the WalkForwardCache --
    # see references/runbook.md's note on this being hand-duplicated
    # rather than single-sourced.
    base_candidate_fns = {
        "wave_pressure": lambda pds: WavePressureScorer(pds, FrequencyEngine(pds)).top_candidates(n_slots, lookback),
        "difference_model": lambda pds: DifferenceModel(pds).predict_next(lookback),
        "piecewise_bin_diff": lambda pds: _build_piecewise_candidate(pds),
        "knn": lambda pds: KNNEngine(pds, k=5, weight_mode="distance").predict_next(),
        "due_element": lambda pds: DueElementAnticipator(pds, FrequencyEngine(pds)).anticipate(n_slots),
        "zero_frequency": lambda pds: ZeroFrequencyAnticipator(pds, FrequencyEngine(pds)).anticipate(n_slots, lookback),
        "combo_support": lambda pds: _build_combo_candidate(
            pds, n_slots, fill_pool=WavePressureScorer(pds, FrequencyEngine(pds)).top_candidates(n_slots, lookback)
        ),
        "genetic_algorithm": lambda pds: _build_ga_candidate(pds, lookback),
        "topsis": lambda pds: _build_topsis_candidate(pds, n_slots, lookback),
        "convergence_selector": lambda pds: _build_convergence_candidate(pds, n_slots, lookback),
    }

    method_results.append(MethodResult(
        "adaptive_weighted",
        AdaptiveWeightedEnsemble(ds, base_candidate_fns, min_history=min_history).candidate(n_slots),
        "Majority vote over the 10 base methods, weighted by each method's own trailing "
        "walk-forward hit rate (not a validation exemption -- tested below like every other method)",
    ))

    # ---- FRAMEWORK: transparent ensemble ----
    ensemble_result = EnsembleCombiner(ds).combine(method_results)

    # ---- VALIDATE: walk-forward hit rate vs. random baseline, per method
    #      AND for both ensemble variants, with BH correction across all
    #      of them (multiple-testing correction is not optional here).
    #
    #      Performance note: this used to call historical_hit_rate/
    #      historical_near_miss_rate once per method (each its own O(n)
    #      walk-forward loop recomputing every base method from
    #      scratch for ensemble/adaptive_weighted). WalkForwardCache
    #      computes every base method's candidate/hit/near-miss at
    #      every walk-forward point exactly ONCE, and ensemble/
    #      adaptive_weighted derive their stats from that single pass
    #      instead of re-running the base methods themselves. See
    #      references/runbook.md for the measured speedup. ----
    cache = WalkForwardCache(ds, base_candidate_fns, min_history, near_miss_threshold)

    raw_p_values, hit_rates, near_miss_rates = {}, {}, {}
    validator = RandomBaselineValidator(ds)
    p_null = RandomBaselineValidator.null_hit_probability(cfg)

    for name in base_candidate_fns:
        rate, trials = cache.hit_rate(name)
        hits = round(rate * trials)
        raw_p_values[name] = validator.binomial_p_value(hits, trials, p_null)
        hit_rates[name] = {"hit_rate": round(rate, 4), "trials": trials, "hits": hits}
        near_rate, near_trials = cache.near_miss_rate(name)
        near_miss_rates[name] = {"avg_near_misses": round(near_rate, 3), "trials": near_trials}

    (ens_rate, ens_trials), (ens_near_rate, ens_near_trials) = cache.ensemble_stats(near_miss_threshold)
    ens_hits = round(ens_rate * ens_trials)
    raw_p_values["ensemble"] = validator.binomial_p_value(ens_hits, ens_trials, p_null)
    hit_rates["ensemble"] = {"hit_rate": round(ens_rate, 4), "trials": ens_trials, "hits": ens_hits}
    near_miss_rates["ensemble"] = {"avg_near_misses": round(ens_near_rate, 3), "trials": ens_near_trials}

    (adw_rate, adw_trials), (adw_near_rate, adw_near_trials) = cache.adaptive_weighted_stats(
        window=15, near_miss_threshold=near_miss_threshold)
    adw_hits = round(adw_rate * adw_trials)
    raw_p_values["adaptive_weighted"] = validator.binomial_p_value(adw_hits, adw_trials, p_null)
    hit_rates["adaptive_weighted"] = {"hit_rate": round(adw_rate, 4), "trials": adw_trials, "hits": adw_hits}
    near_miss_rates["adaptive_weighted"] = {"avg_near_misses": round(adw_near_rate, 3), "trials": adw_near_trials}

    bh_results = RandomBaselineValidator.benjamini_hochberg(raw_p_values)
    all_method_names = list(base_candidate_fns.keys()) + ["ensemble", "adaptive_weighted"]

    report = {
        "type": type_key,
        "game_name": cfg["name"],
        "n_draws_analyzed": len(ds),
        "last_draw": ds.last(),
        "even_odd_balance": fe.even_odd_balance(),
        "segment_signature": seg.thematic_signature(lookback),
        "sliding_windows": sw.run(),
        "descriptive_diagnostics": diagnostics.full_report(),
        "method_candidates": {mr.name: {"candidate": mr.candidate, "rationale": mr.rationale} for mr in method_results},
        "ensemble": ensemble_result,
        "validation": {
            "null_hypothesis_hit_prob": round(p_null, 5),
            "per_method": {name: {**hit_rates[name], **bh_results[name]} for name in all_method_names},
        },
        "near_miss_diagnostic": {
            "note": "DESCRIPTIVE ONLY -- not part of the significance gate. "
                    f"Within +/-{near_miss_threshold} of an actual value, not counted elsewhere as a hit.",
            "per_method": near_miss_rates,
        },
    }

    quality_advisory = assess_run_quality(report)
    report["quality_advisory"] = {
        "should_reconsider": quality_advisory.should_reconsider,
        "risk_level": quality_advisory.risk_level,
        "notes": quality_advisory.notes,
        "recommended_actions": quality_advisory.recommended_actions,
        "note": "ADVISORY ONLY -- this run was NOT automatically rerun or adjusted because of this flag. "
                "See assess_run_quality() docstring and references/playbook.md.",
    }

    if verbose:
        _print_report(report)
    return report


def _print_report(report: Dict) -> None:
    print("=" * 78)
    print(f"  {report['game_name']}  ({report['type']})  -  {report['n_draws_analyzed']} draws analyzed")
    print("=" * 78)
    print(f"Last draw: {report['last_draw']}")
    eo = report["even_odd_balance"]
    print(f"Even/Odd balance - avg {eo['avg_even']:.1f}/{eo['avg_odd']:.1f}, last {eo['last_even']}/{eo['last_odd']}")
    print(f"Segment signature (recent modal split): {report['segment_signature']}")

    dd = report["descriptive_diagnostics"]
    print("\n--- Descriptive diagnostics (no significance claim) ---")
    print(f"  Avg draw entropy: {dd['average_draw_entropy']}")
    print(f"  Within-draw runs (2+): {dd['within_draw_runs']}")
    print(f"  Within-draw gap stats: {dd['within_draw_gap_stats']}")
    print(f"  Confirmation-bias check: {dd['confirmation_bias_check']}")
    print(f"  Availability-bias check: {dd['availability_bias_check']}")

    print("\n--- Method candidates (ANTICIPATED, not predicted) ---")
    for name, info in report["method_candidates"].items():
        print(f"  [{name}] {info['candidate']}   <- {info['rationale']}")

    print("\n--- Ensemble consensus ---")
    print(f"  Ensemble set: {report['ensemble']['ensemble_set']}")
    print(f"  Top shared values (value -> vote count): {report['ensemble']['vote_counts']}")

    print(f"\n--- Statistical validation (null hit prob = {report['validation']['null_hypothesis_hit_prob']:.5f}) ---")
    print(f"  {'Method':<18}{'HitRate':>9}{'Trials':>8}{'RawP':>10}{'BH-adjP':>10}  Significant?")
    for name, v in report["validation"]["per_method"].items():
        print(f"  {name:<18}{v['hit_rate']:>9}{v['trials']:>8}{v['raw_p']:>10}{v['bh_adjusted_p']:>10}  "
              f"{'YES' if v['significant_after_correction'] else 'no'}")

    nm = report["near_miss_diagnostic"]
    print(f"\n--- Near-miss diagnostic ({nm['note']}) ---")
    print(f"  {'Method':<18}{'AvgNearMisses':>15}{'Trials':>8}")
    for name, v in nm["per_method"].items():
        print(f"  {name:<18}{v['avg_near_misses']:>15}{v['trials']:>8}")

    qa = report["quality_advisory"]
    print(f"\n--- Quality advisory ({qa['note']}) ---")
    print(f"  Risk level: {qa['risk_level']}   Should reconsider: {qa['should_reconsider']}")
    for n in qa["notes"]:
        print(f"  - {n}")
    for a in qa["recommended_actions"]:
        print(f"    -> {a}")
    print("=" * 78)


def run_all_types(all_historical_data: Dict[str, Dict[str, List[int]]],
                   lookback: int = 20, min_history: int = 10) -> Dict:
    all_reports = {}
    for type_key, hist in all_historical_data.items():
        if len(hist) < min_history + 5:
            print(f"[skip] {type_key}: needs at least {min_history + 5} draws to validate, has {len(hist)}.")
            continue
        all_reports[type_key] = run_full_pipeline(type_key, hist, lookback=lookback, min_history=min_history)
    return all_reports


# ============================================================
# 8. CLI + DEMO
# ============================================================
def _run_demo() -> None:
    """Clearly-labeled synthetic data, NOT real PCSO draws."""
    print("No --type/--data given -- running the built-in demo instead.")
    print("(All data below is randomly generated for illustration only.)\n")
    demo_data = {
        "T2": {f"SET_{i+1}": [random.randint(0, 9) for _ in range(3)] for i in range(40)},
        "T8": {f"SET_{i+1}": sorted(random.sample(range(1, 59), 6)) for i in range(40)},
    }
    run_all_types(demo_data, lookback=20, min_history=10)


def _list_types() -> None:
    for key, cfg in TYPE_CONFIG.items():
        shape = "positional" if cfg["positional"] else "pool"
        print(f"  {key:<6} {cfg['name']:<22} {shape:<10} "
              f"{cfg['n_slots']} slots, range {cfg['min_val']}-{cfg['max_val']}")


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Draw Pattern Validator -- standalone single-file engine. "
                    "Run with no arguments for a synthetic demo; add --type/--data for real data.")
    parser.add_argument("--type", dest="type_key", help="Type key from TYPE_CONFIG, e.g. T4.")
    parser.add_argument("--data", dest="data_path", help='Path to a JSON file: {"SET_1": [...], ...}')
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument("--near-miss-threshold", type=int, default=2)
    parser.add_argument("--list-types", action="store_true", help="Print available type keys and exit.")
    args = parser.parse_args()

    if args.list_types:
        _list_types()
        return

    if not args.type_key or not args.data_path:
        if args.type_key or args.data_path:
            print("--type and --data must be given together.\n")
            parser.print_help()
            return
        _run_demo()
        return

    if args.type_key not in TYPE_CONFIG:
        print(f"Type key {args.type_key!r} not found in TYPE_CONFIG. Available:")
        _list_types()
        return

    with open(args.data_path) as f:
        historical_data = json.load(f)

    print(f"NOTE: loaded {len(historical_data)} draws from {args.data_path} for type "
          f"{args.type_key}. If this isn't real historical draw data, say so before "
          f"sharing any results.\n")

    run_full_pipeline(args.type_key, historical_data, lookback=args.lookback,
                       min_history=args.min_history, near_miss_threshold=args.near_miss_threshold,
                       verbose=True)


if __name__ == "__main__":
    main()
