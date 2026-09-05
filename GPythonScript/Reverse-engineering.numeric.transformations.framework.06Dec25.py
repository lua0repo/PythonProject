from typing import List, Dict, Tuple
import numpy as np

class LotteryReverseEngineer:
    """
    Professional reverse-engineering tool for sequential lottery results.
    Learns number transformation patterns and predicts the next draw.
    """

    def __init__(self, num_bins: int = 4, near_threshold: int = 2):
        self.num_bins = max(1, num_bins)
        self.near_threshold = near_threshold
        self.bin_edges = None
        self.bin_diffs = None

    def _learn_transformation(self, sources: List[List[int]], targets: List[List[int]]) -> None:
        """Learn average difference per number range (piecewise)"""
        all_values = []
        all_diffs = []

        for src, tgt in zip(sources, targets):
            src_arr = np.array(src)
            tgt_arr = np.array(tgt)
            all_values.extend(src_arr)
            all_diffs.extend(tgt_arr - src_arr)

        values = np.array(all_values)
        diffs = np.array(all_diffs)

        # Dynamic quantile-based bins
        quantiles = np.linspace(0, 1, self.num_bins + 1)[1:-1]
        self.bin_edges = np.quantile(values, quantiles, method='midpoint')
        self.bin_edges = np.concatenate([[-np.inf], self.bin_edges, [np.inf]])

        # Mean diff per bin
        bin_indices = np.digitize(values, self.bin_edges[1:-1], right=False)
        self.bin_diffs = []
        for i in range(1, len(self.bin_edges)):
            mask = (bin_indices == i)
            avg_diff = int(round(np.mean(diffs[mask]))) if np.any(mask) else 0
            self.bin_diffs.append(avg_diff)

    def fit(self, historical_sets: List[List[int]]) -> 'LotteryReverseEngineer':
        """Fit on sequence: use each set to predict the next"""
        if len(historical_sets) < 2:
            raise ValueError("Need at least 2 sets to learn transformation.")
        self._learn_transformation(historical_sets[:-1], historical_sets[1:])
        return self

    def predict_next(self, last_set: List[int]) -> List[int]:
        """Predict next 6 numbers using learned piecewise rules"""
        if self.bin_diffs is None:
            raise ValueError("Model not fitted yet.")

        arr = np.array(last_set)
        indices = np.digitize(arr, self.bin_edges[1:-1], right=False)
        indices = np.clip(indices, 0, len(self.bin_diffs) - 1)

        predicted = [int(round(val + self.bin_diffs[idx])) for val, idx in zip(arr, indices)]
        predicted = sorted(set(predicted))  # remove duplicates

        # Filter valid range (common: 1–59)
        predicted = [x for x in predicted if 1 <= x <= 59]

        # Ensure exactly 6 numbers
        while len(predicted) < 6:
            candidate = predicted[-1] + 1 if predicted else 30
            if candidate <= 59:
                predicted.append(candidate)
            else:
                break
        predicted = predicted[:6]

        return sorted(predicted)

    def hit_count(self, pred: List[int], actual: List[int]) -> Tuple[int, List[Tuple[int, int]]]:
        """Return (exact_hits, list of near hits)"""
        exact = len(set(pred) & set(actual))
        near = []
        used = set()
        for p in pred:
            for a in actual:
                if a not in used and abs(p - a) <= self.near_threshold:
                    near.append((p, a))
                    used.add(a)
                    break
        return exact, near

    def process_sequence(self, labeled_sets: Dict[str, List[int]]):
        """Full analysis + prediction for next draw"""
        keys = list(labeled_sets.keys())
        sets = [labeled_sets[k] for k in keys]

        print("LOTTERY REVERSE ENGINEERING FRAMEWORK")
        print("=" * 72)

        if len(sets) < 2:
            print("Not enough data. Add more draws.")
            return

        # Learn from all real transitions
        self.fit(sets)

        print(f"Learned {self.num_bins}-bin piecewise transformation:")
        for i in range(len(self.bin_diffs)):
            low = self.bin_edges[i] if np.isfinite(self.bin_edges[i]) else -999
            high = self.bin_edges[i+1] if np.isfinite(self.bin_edges[i+1]) else 999
            print(f"  Bin {i+1}: [{low:>4.0f} → {high:<4.0f}) → avg change = {self.bin_diffs[i]:+3d}")
        print()

        total_exact = total_near = valid_transitions = 0

        # Historical predictions
        for i in range(len(sets) - 1):
            src = sets[i]
            actual = sets[i+1]
            pred = self.predict_next(src)
            exact, near = self.hit_count(pred, actual)
            total_exact += exact
            total_near += len(near)
            valid_transitions += 1

            print(f"{keys[i]} → {keys[i+1]}")
            print(f"   Source:     {src}")
            print(f"   Predicted:  {pred}")
            print(f"   Actual:     {actual}")
            print(f"   Hits: {exact} exact", end="")
            if near:
                near_str = ", ".join([f"{p}→{a}" for p, a in near])
                print(f" | +{len(near)} near ({near_str})")
            else:
                print()
            print("-" * 60)

        # FINAL PREDICTION FOR NEXT DRAW
        last_key = keys[-1]
        last_set = sets[-1]
        next_pred = self.predict_next(last_set)

        print(f"{last_key} → NEXT DRAW (UPCOMING)")
        print(f"   Last result:       {last_set}")
        print(f"   PREDICTED NEXT →  {next_pred}")
        print(f"   This is your model's prediction for the next draw!")
        print(f"   (Hit count will be available after the draw)")
        print("=" * 72)

        if valid_transitions > 0:
            print(f"HISTORICAL PERFORMANCE ({valid_transitions} tested draws):")
            print(f"   Average exact hits:     {total_exact / valid_transitions:.2f}")
            print(f"   Average near hits (±{self.near_threshold}): {total_near / valid_transitions:.2f}")
            print(f"   Average total close:    {(total_exact + total_near) / valid_transitions:.2f}")
        print("Analysis complete.\n")


# =============================
# YOUR DATA GOES HERE
# =============================
if __name__ == "__main__":
    data = {
        "SET_1":  [15, 18, 41, 50, 51, 58],
        "SET_2":  [2, 12, 16, 45, 46, 50],
        "SET_3":  [13, 17, 23, 35, 43, 58],
        "SET_4":  [1, 3, 12, 14, 19, 49],
        "SET_5":  [4, 11, 16, 30, 32, 41],
    }

    engine = LotteryReverseEngineer(num_bins=4, near_threshold=2)
    engine.process_sequence(data)

