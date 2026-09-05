from typing import List, Dict
import numpy as np
import random

# =========================
# SMART CONTEXT-AWARE NUMBER FILLER
# =========================
def smart_fill(
    current_set: List[int],
    needed: int,
    historical_sets: List[List[int]],
    max_number: int = 59
) -> List[int]:
    if needed <= 0:
        return []

    all_past = [num for s in historical_sets for num in s]
    recent_set = sorted(historical_sets[-1]) if historical_sets else []
    last_few_sets = historical_sets[-5:] if len(historical_sets) >= 5 else historical_sets
    recent_numbers = [num for s in last_few_sets for num in s]

    hot = [x for x in recent_numbers if recent_numbers.count(x) > 1]

    nearby = set()
    for n in recent_set:
        nearby.update(range(max(1, n-12), min(max_number+1, n+13)))

    freq = {i: all_past.count(i)+1 for i in range(1, max_number+1)}

    weights = []
    candidates = list(range(1, max_number+1))
    for num in candidates:
        w = freq[num]
        if num in hot:
            w *= 8
        if num in nearby:
            w *= 5
        if num in recent_set:
            w *= 3
        weights.append(w)

    total = sum(weights)
    if total == 0:
        weights = None
    else:
        weights = [w / total for w in weights]

    filled = []
    attempts = 0
    while len(filled) < needed and attempts < 100:
        num = np.random.choice(candidates, p=weights)
        if num not in current_set and num not in filled:
            filled.append(num)
        attempts += 1

    if len(filled) < needed:
        for i in range(1, max_number+1):
            if i not in current_set and i not in filled:
                filled.append(i)
                if len(filled) == needed:
                    break

    return filled[:needed]

# =========================
# FRAMEWORK 1 — Difference & Drift Mapping (DDM)
# =========================
def difference_drift_mapping(historical_sets: List[List[int]]) -> List[int]:
    historical_sets = np.array([sorted(s) for s in historical_sets])
    if len(historical_sets) < 2:
        return sorted(random.sample(range(1, 60), 6))
    diffs = np.diff(historical_sets, axis=0)
    avg_drift = np.round(diffs.mean(axis=0)).astype(int)
    predicted = historical_sets[-1] + avg_drift
    predicted = np.clip(predicted, 1, 59).astype(int).tolist()

    result = []
    seen = set()
    for x in predicted:
        if 1 <= x <= 59 and x not in seen:
            result.append(x)
            seen.add(x)
    missing = 6 - len(result)
    if missing > 0:
        result += smart_fill(result, missing, historical_sets)
    return sorted(result[:6])

# =========================
# FRAMEWORK 2 — Segmented Transformation Model (STM)
# =========================
def segmented_transformation_model(historical_sets: List[List[int]], num_bins: int = 4) -> List[int]:
    all_values = np.concatenate(historical_sets)
    arr = np.array([sorted(s) for s in historical_sets])
    diffs_per_position = np.diff(arr, axis=0)
    value_at_prev = arr[:-1].flatten()
    diff_at_next = diffs_per_position.flatten()

    valid = np.isfinite(diff_at_next)
    value_at_prev = value_at_prev[valid]
    diff_at_next = diff_at_next[valid].astype(float)

    quantiles = np.linspace(0, 1, num_bins + 1)[1:-1]
    bin_edges = np.quantile(all_values, quantiles)
    bin_edges = np.concatenate([[-np.inf], bin_edges, [np.inf]])

    bin_labels = np.digitize(value_at_prev, bin_edges[1:-1])
    bin_diffs = []
    for i in range(num_bins):
        mask = (bin_labels == i)
        avg = diff_at_next[mask].mean() if mask.sum() > 0 else 0
        bin_diffs.append(int(round(avg)))

    last_set = np.array(historical_sets[-1])
    indices = np.clip(np.digitize(last_set, bin_edges[1:-1]), 0, num_bins - 1)
    predicted = [int(round(val + bin_diffs[idx])) for val, idx in zip(last_set, indices)]

    result = []
    seen = set()
    for x in predicted:
        x = np.clip(x, 1, 59)
        if x not in seen:
            result.append(x)
            seen.add(x)

    missing = 6 - len(result)
    if missing > 0:
        result += smart_fill(result, missing, historical_sets)

    return sorted(result[:6])

# =========================
# FRAMEWORK 3 — Derived Arithmetic Transformation (DAT)
# =========================
def derived_arithmetic_transformation(numbers: List[int], historical_sets: List[List[int]]) -> List[int]:
    numbers = sorted(numbers)
    lowest, mid, highest = numbers[0], numbers[2], numbers[5]
    hp1 = highest + 1
    half_high = hp1 / 2
    half_again = round(half_high / 2) + lowest
    sum_small = lowest + numbers[1] + mid

    candidates = [lowest, mid, hp1, int(half_high), half_again, sum_small]
    candidates = [np.clip(x, 1, 59) for x in candidates]

    result = []
    seen = set()
    for x in candidates:
        if x not in seen:
            result.append(x)
            seen.add(x)

    missing = 6 - len(result)
    if missing > 0:
        result += smart_fill(result, missing, historical_sets)

    return sorted(result[:6])

# =========================
# FRAMEWORK 4 — Positional Trend Regression (PTR)
# =========================
def positional_trend_regression(historical_sets: List[List[int]]) -> List[int]:
    hs = np.array([sorted(s) for s in historical_sets])
    n = len(hs)
    predicted = []
    for pos in range(6):
        y = hs[:, pos]
        x = np.arange(n)
        slope, intercept = np.polyfit(x, y, 1)
        val = int(round(slope * n + intercept))
        val = np.clip(val, 1, 59)
        predicted.append(val)

    result = []
    seen = set()
    for x in predicted:
        if x not in seen:
            result.append(x)
            seen.add(x)

    missing = 6 - len(result)
    if missing > 0:
        result += smart_fill(result, missing, historical_sets)

    return sorted(result[:6])

# =========================
# FRAMEWORK 5 — Gap Evolution Model (GEM)
# =========================
def gap_evolution_model(historical_sets: List[List[int]]) -> List[int]:
    hs = [sorted(s) for s in historical_sets]
    lowest = 1
    hp1 = 59
    if len(hs) > 0:
        lowest = min(min(s) for s in hs)
        hp1 = max(max(s) for s in hs)
    gaps = [[s[i+1]-s[i] for i in range(5)] for s in hs]
    gaps = np.array(gaps)
    if len(gaps) < 2:
        return sorted(random.sample(range(lowest, hp1+1), 6))
    change = np.round(np.diff(gaps, axis=0).mean(axis=0)).astype(int)
    new_gaps = np.clip(gaps[-1] + change, 1, 30)
    pred = [hs[-1][0]]
    for g in new_gaps:
        pred.append(pred[-1] + g)
    pred = [np.clip(x, 1, 59) for x in pred]

    result = []
    seen = set()
    for x in pred:
        if x not in seen:
            result.append(int(x))
            seen.add(int(x))

    missing = 6 - len(result)
    if missing > 0:
        result += smart_fill(result, missing, historical_sets)

    return sorted(result[:6])

# =========================
# FRAMEWORK 6–9
# =========================
def constraint_system_reconstruction(historical_sets: List[List[int]]) -> List[int]:
    hs = np.array([sorted(s) for s in historical_sets])
    avg_low = int(round(hs[:,0].mean()))
    avg_mid = int(round(hs[:,2].mean()))
    avg_high = int(round(hs[:,-1].mean()))
    pred = [avg_low, avg_mid, avg_high-5, avg_high-2, avg_high-1, avg_high]
    pred = [np.clip(x, 1, 59) for x in pred]

    result = []
    seen = set()
    for x in pred:
        if x not in seen:
            result.append(x)
            seen.add(x)
    missing = 6 - len(result)
    if missing > 0:
        result += smart_fill(result, missing, historical_sets)
    return sorted(result[:6])

def generative_modeling_framework(historical_sets: List[List[int]]) -> List[int]:
    lowest, hp1 = 1, 59
    flat = [n for s in historical_sets for n in s]
    probs = [flat.count(i) + 1 for i in range(lowest, hp1+1)]
    total = sum(probs)
    probs = [p/total for p in probs]
    return sorted(np.random.choice(range(lowest, hp1+1), size=6, replace=False, p=probs).tolist())

def constraint_satisfaction_inversion(historical_sets: List[List[int]]) -> List[int]:
    flat = [n for s in historical_sets for n in s]
    freq = np.bincount(flat, minlength=60)[1:]
    hot = np.where(freq > 1)[0] + 1
    result = list(hot[:6])
    result += smart_fill(result, 6 - len(result), historical_sets)
    return sorted(result[:6])

def bayesian_structural_learning(historical_sets: List[List[int]]) -> List[int]:
    hs = np.array([sorted(s) for s in historical_sets])
    pred = []
    for pos in range(6):
        col = hs[:, pos]
        pred.append(int(np.bincount(col).argmax()))
    result = []
    seen = set()
    for x in pred:
        if x not in seen:
            result.append(x)
            seen.add(x)
    missing = 6 - len(result)
    if missing > 0:
        result += smart_fill(result, missing, historical_sets)
    return sorted(result[:6])

# =========================
# MASTER FUNCTION
# =========================
def run_all_frameworks_full(historical_sets_dict: Dict[str, List[int]]) -> Dict[str, List[int]]:
    sets = [sorted(historical_sets_dict[label]) for label in historical_sets_dict]

    return {
        "DDM": difference_drift_mapping(sets),
        "STM": segmented_transformation_model(sets),
        "DAT": derived_arithmetic_transformation(sets[-1], sets),
        "PTR": positional_trend_regression(sets),
        "GEM": gap_evolution_model(sets),
        "CSR": constraint_system_reconstruction(sets),
        "GMF": generative_modeling_framework(sets),
        "CSI": constraint_satisfaction_inversion(sets),
        "BSL": bayesian_structural_learning(sets)
    }

# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    historical_sets = {
        "SET_1": [15, 18, 41, 50, 51, 58],
        "SET_2": [2, 12, 16, 45, 46, 50],
        "SET_3": [13, 17, 23, 35, 43, 58],
        "SET_4": [1, 3, 12, 14, 19, 49],
        "SET_5": [4, 11, 16, 30, 32, 41],
    }

    results = run_all_frameworks_full(historical_sets)
    print("LOTTERY PREDICTION ENSEMBLE (Smart Fill Enabled)\n")
   
    # Initialize list to collect all numbers
    all_numbers = []

    for name, nums in results.items():
        print(f"{name}: {', '.join(f'{x:2}' for x in nums)}")
        all_numbers.append(nums)
   
    # Compute MEAN line safely
    all_numbers_arr = np.array(all_numbers)
    mean_line = np.rint(all_numbers_arr.mean(axis=0)).astype(int)
    mean_line = [np.clip(x, 1, 59) for x in mean_line]
    print("\nMEAN:", ', '.join(f"{x}" for x in mean_line))


