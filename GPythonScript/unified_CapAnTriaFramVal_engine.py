"""
====================================================================
 UNIFIED PCSO PATTERN ANALYSIS ENGINE  (Consolidated Build)
====================================================================
Merges the techniques scattered across the 17 prototype scripts
(Sys_V1 through Sys_V6, plus the flat SWD/WPA family and the
3D-to-58G multi-type family) into one coherent pipeline:

    CAPTURE   -> pull raw frequency / gap / positional facts
    ANALYSIS  -> run each modeling technique independently
    TRIAGE    -> collect each technique's candidate + rationale
    FRAMEWORK -> combine candidates into a transparent ensemble
    VALIDATE  -> test the ensemble (and each technique) against a
                 random baseline: raw p-value + Benjamini-Hochberg
                 corrected p-value, surfaced plainly either way
    FLOW      -> orchestrate all of the above per game type, T1-T8

STATISTICAL HONESTY DISCLAIMER
-------------------------------
PCSO draws are independent, certified-random events. Nothing in
this script -- or in any of its 17 ancestors -- has been shown to
predict them. Every output here is a descriptive pattern summary
("anticipated", never "predicted"), and the VALIDATE stage exists
specifically to show, plainly, whether any technique beats chance.
A null result is a legitimate finding, not a failure of the script.

WHAT WAS DELIBERATELY LEFT OUT
-------------------------------
`RNGCracker` (from the V1-V6 class lineage) was not ported. Its
LCG/LFSR checks are structured so they always evaluate to None
regardless of input, and its "Mersenne Twister crack" is just a
seeded random draw with a hardcoded 0.2 confidence score. The name
implied a capability (recovering a hardware/certified RNG's state
from a handful of draws) that the code never actually delivered.
Carrying it forward -- even with a caveat -- would misrepresent
what the script can do.
====================================================================
"""

import math
import random
from collections import Counter
from dataclasses import dataclass
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np

random.seed(42)
np.random.seed(42)


# ============================================================
# 1. TYPE CONFIG  (T1-T3 positional digit games, T4-T8 sorted
#    pool games) -- mirrors real PCSO game formats.
# ============================================================
TYPE_CONFIG = {
    "T1": {"name": "EZ2",              "positional": True,  "n_slots": 2, "min_val": 0, "max_val": 9,  "unique": False},
    "T2": {"name": "Swertres",         "positional": True,  "n_slots": 3, "min_val": 0, "max_val": 9,  "unique": False},
    "T3": {"name": "6D Lotto",         "positional": True,  "n_slots": 6, "min_val": 0, "max_val": 9,  "unique": False},
    "T4": {"name": "Lotto 6/42",       "positional": False, "n_slots": 6, "min_val": 1, "max_val": 42, "unique": True},
    "T5": {"name": "Megalotto 6/45",   "positional": False, "n_slots": 6, "min_val": 1, "max_val": 45, "unique": True},
    "T6": {"name": "Superlotto 6/49",  "positional": False, "n_slots": 6, "min_val": 1, "max_val": 49, "unique": True},
    "T7": {"name": "Grandlotto 6/55",  "positional": False, "n_slots": 6, "min_val": 1, "max_val": 55, "unique": True},
    "T8": {"name": "Ultra Lotto 6/58", "positional": False, "n_slots": 6, "min_val": 1, "max_val": 58, "unique": True},
}


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
    re-sorting the output); pool types are de-duplicated and sorted."""

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
            return sorted(random.sample(zero_freq, n_needed))
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


# ============================================================
# 6. VALIDATE  -- the statistical-honesty layer
# ============================================================
class RandomBaselineValidator:
    """Walk-forward comparison of each method's actual historical hit
    rate against the probability of hitting by pure chance, with raw
    and Benjamini-Hochberg-corrected p-values reported side by side."""

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


# ============================================================
# 7. FLOW  -- per-type orchestration + multi-type driver
# ============================================================
def run_full_pipeline(type_key: str, historical_data: Dict[str, List[int]],
                       lookback: int = 20, min_history: int = 10,
                       verbose: bool = True) -> Dict:
    ds = DrawDataset(type_key, historical_data)
    cfg = ds.config
    n_slots = cfg["n_slots"]

    fe = FrequencyEngine(ds)
    sw = SlidingWindowRunner(ds, window_size=lookback, step=lookback)
    seg = SegmentThematicEngine(ds)

    # ---- ANALYSIS + TRIAGE: run every technique on the full dataset ----
    method_results = [
        MethodResult(
            "wave_pressure",
            WavePressureScorer(ds, fe).top_candidates(n_slots, lookback),
            "Highest combined pressure(cold)+wave(momentum)+recency score",
        ),
        MethodResult(
            "difference_model",
            DifferenceModel(ds).predict_next(lookback),
            "Per-slot linear regression on consecutive-draw differences",
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
    ]

    # ---- FRAMEWORK: transparent ensemble ----
    ensemble_result = EnsembleCombiner(ds).combine(method_results)

    # ---- VALIDATE: walk-forward hit rate vs. random baseline, per method
    #      AND for the ensemble itself, with BH correction across all of
    #      them (multiple-testing correction is not optional here) ----
    def _ensemble_candidate_fn(pds: DrawDataset) -> List[int]:
        pfe = FrequencyEngine(pds)
        results = [
            MethodResult("wave_pressure", WavePressureScorer(pds, pfe).top_candidates(n_slots, lookback), ""),
            MethodResult("difference_model", DifferenceModel(pds).predict_next(lookback), ""),
            MethodResult("knn", KNNEngine(pds, k=5, weight_mode="distance").predict_next(), ""),
            MethodResult("due_element", DueElementAnticipator(pds, pfe).anticipate(n_slots), ""),
            MethodResult("zero_frequency", ZeroFrequencyAnticipator(pds, pfe).anticipate(n_slots, lookback), ""),
        ]
        return EnsembleCombiner(pds).combine(results)["ensemble_set"]

    candidate_fns = {
        "wave_pressure": lambda pds: WavePressureScorer(pds, FrequencyEngine(pds)).top_candidates(n_slots, lookback),
        "difference_model": lambda pds: DifferenceModel(pds).predict_next(lookback),
        "knn": lambda pds: KNNEngine(pds, k=5, weight_mode="distance").predict_next(),
        "due_element": lambda pds: DueElementAnticipator(pds, FrequencyEngine(pds)).anticipate(n_slots),
        "zero_frequency": lambda pds: ZeroFrequencyAnticipator(pds, FrequencyEngine(pds)).anticipate(n_slots, lookback),
        "ensemble": _ensemble_candidate_fn,
    }

    validator = RandomBaselineValidator(ds)
    p_null = RandomBaselineValidator.null_hit_probability(cfg)

    raw_p_values, hit_rates = {}, {}
    for name, fn in candidate_fns.items():
        rate, trials = validator.historical_hit_rate(fn, min_history=min_history)
        hits = round(rate * trials)
        raw_p_values[name] = validator.binomial_p_value(hits, trials, p_null)
        hit_rates[name] = {"hit_rate": round(rate, 4), "trials": trials, "hits": hits}

    bh_results = RandomBaselineValidator.benjamini_hochberg(raw_p_values)

    report = {
        "type": type_key,
        "game_name": cfg["name"],
        "n_draws_analyzed": len(ds),
        "last_draw": ds.last(),
        "even_odd_balance": fe.even_odd_balance(),
        "segment_signature": seg.thematic_signature(lookback),
        "sliding_windows": sw.run(),
        "method_candidates": {mr.name: {"candidate": mr.candidate, "rationale": mr.rationale} for mr in method_results},
        "ensemble": ensemble_result,
        "validation": {
            "null_hypothesis_hit_prob": round(p_null, 5),
            "per_method": {name: {**hit_rates[name], **bh_results[name]} for name in candidate_fns},
        },
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
# 8. DEMO  -- clearly-labeled synthetic data, NOT real PCSO draws
# ============================================================
if __name__ == "__main__":
    print("NOTE: the historical_data below is randomly generated for")
    print("illustration only -- swap in your own real draw history per")
    print("type to get a report that means anything.\n")

    demo_data = {
        "T2": {f"SET_{i+1}": [random.randint(0, 9) for _ in range(3)] for i in range(40)},
        "T8": {f"SET_{i+1}": sorted(random.sample(range(1, 59), 6)) for i in range(40)},
    }
    run_all_types(demo_data, lookback=20, min_history=10)
