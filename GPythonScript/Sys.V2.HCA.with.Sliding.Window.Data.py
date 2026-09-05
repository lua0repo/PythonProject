from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import math
import itertools
import statistics
import random
from typing import Dict, List, Tuple, Set, Any
from functools import reduce
from math import gcd
from scipy.stats import pearsonr

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ===========================
# COMPLETE HISTORICAL DATA
# ===========================
COMPLETE_HISTORICAL_DATA = {
    "SET_1": [28, 29, 32, 39, 41, 46],
    "SET_2": [15, 18, 41, 50, 51, 58],
    "SET_3": [2, 12, 16, 45, 46, 50],
    "SET_4": [13, 17, 23, 35, 43, 58],
    "SET_5": [1, 3, 12, 14, 19, 49],
    "SET_6": [4, 11, 16, 30, 32, 41],
    "SET_7": [22, 31, 34, 44, 48, 51],
    "SET_8": [2, 5, 30, 32, 52, 54],
    "SET_9": [9, 10, 15, 16, 17, 25],
    "SET_10": [4, 15, 17, 29, 32, 34],
}

# ===========================
# YOUR COMPLETE ANALYSIS SCRIPT (AS A FUNCTION)
# ===========================
def run_complete_analysis(historical_data, window_id):
    """
    This function contains ALL your analysis logic
    Returns all predictions and results
    """

    print(f"\n{'#'*28}")
    print(f"# SLIDING WINDOW RESULT {window_id}#")
    print(f"{'#'*28}\n")

    set_names = list(historical_data.keys())
    all_sets = list(historical_data.values())

    # ===========================
    # Custom Linear Regression
    # ===========================
    class CustomLinearRegression:
        def __init__(self):
            self.slope = 0
            self.intercept = 0
        def fit(self, X, y):
            X = np.array(X).reshape(-1, 1)
            y = np.array(y)
            n = len(X)
            mean_x = np.mean(X)
            mean_y = np.mean(y)
            ss_xy = np.sum((X - mean_x) * (y - mean_y))
            ss_xx = np.sum((X - mean_x) ** 2)
            self.slope = ss_xy / ss_xx if ss_xx != 0 else 0
            self.intercept = mean_y - self.slope * mean_x
        def predict(self, X_new):
            X_new = np.array(X_new).reshape(-1, 1)
            preds = self.intercept + self.slope * X_new
            if len(preds) == 1:
                return preds.item()
            return preds.flatten().tolist()

    # ===========================
    # DIFFERENCE ANALYSIS
    # ===========================
    def prepare_dataframe(data_dict):
        df = pd.DataFrame(data_dict).T
        df.columns = [f"Ball_{i+1}" for i in range(6)]
        return df
    class DifferencePredictor:
        def __init__(self, dataframe):
            self.df = dataframe
            self.diff_df = self.compute_differences()
        def compute_differences(self):
            diff_df = self.df.diff().iloc[1:]
            diff_df.index = [f"{prev} -> {curr}" for prev, curr in zip(self.df.index[:-1], self.df.index[1:])]
            return diff_df
        def display_differences(self):
            print("-" * 40)
            print("DIFFERENCES BETWEEN CONSECUTIVE SETS")
            print("-" * 40)

            predicted_diff = self.predict_diffs_regression()
            self.diff_df.loc['ML PREDICTION'] = predicted_diff

            highest_counts = []
            for col in self.diff_df.columns:
                mode_val = self.diff_df[col].mode()
                if not mode_val.empty:
                    highest_counts.append(mode_val.iloc[0])
                else:
                    highest_counts.append(np.nan)
            self.diff_df.loc['HIGHEST COUNT'] = highest_counts
            print(self.diff_df.to_string())

            last_set = [int(x) for x in self.df.iloc[-1].tolist()]
            pre_Deff_set = sorted([max(1, min(58, last_set[i] + predicted_diff[i])) for i in range(6)])
            print("\nPredicted Difference from Last SET (ML Regression) =>", ' '.join(map(str, predicted_diff)))
            print("Predicted SET_N =>", pre_Deff_set)
            print(f"✅Reusable pre_Deff_set: {pre_Deff_set}")
            return pre_Deff_set
        def predict_diffs_regression(self):
            predicted_diff = []
            time_steps = list(range(1, len(self.diff_df) + 1))
            for col in self.diff_df.columns:
                y = self.diff_df[col].tolist()
                model = CustomLinearRegression()
                model.fit(time_steps, y)
                next_time = len(time_steps) + 1
                pred = model.predict(next_time)
                predicted_diff.append(int(round(pred)))
            return predicted_diff
    df = prepare_dataframe(historical_data)
    predictor_model = DifferencePredictor(df)
    pre_Deff_set = predictor_model.display_differences()

    # ===========================
    # EVEN/ODD ANALYSIS
    # ===========================
    print("\n" + "-" * 40)
    print("EVEN And ODD Details")
    print("-" * 40)
    print("Set Name Even# Odd# ET# OT#")

    total_even = total_odd = total_et = total_ot = 0
    even_odd_profiles = []
    even_counts = []
    odd_counts = []
    et_sums = []
    ot_sums = []

    for name, numbers in historical_data.items():
        evens = [n for n in numbers if n % 2 == 0]
        odds = [n for n in numbers if n % 2 != 0]
        even_sum = sum(evens)
        odd_sum = sum(odds)
        total_even += len(evens)
        total_odd += len(odds)
        total_et += even_sum
        total_ot += odd_sum
        even_odd_profiles.append([len(evens), len(odds), even_sum, odd_sum])
        even_counts.append(len(evens))
        odd_counts.append(len(odds))
        et_sums.append(even_sum)
        ot_sums.append(odd_sum)
        print(f"{name:<10} {len(evens):<7} {len(odds):<6} {even_sum:<6} {odd_sum:<6}")

    num_sets = len(all_sets)
    time_steps = list(range(1, num_sets + 1))

    reg_even_count = CustomLinearRegression()
    reg_even_count.fit(time_steps, even_counts)
    pred_even_count = round(reg_even_count.predict(num_sets + 1))

    reg_odd_count = CustomLinearRegression()
    reg_odd_count.fit(time_steps, odd_counts)
    pred_odd_count = round(reg_odd_count.predict(num_sets + 1))

    reg_et = CustomLinearRegression()
    reg_et.fit(time_steps, et_sums)
    pred_et = round(reg_et.predict(num_sets + 1))

    reg_ot = CustomLinearRegression()
    reg_ot.fit(time_steps, ot_sums)
    pred_ot = round(reg_ot.predict(num_sets + 1))

    print(f"{'Predicted':<10} {pred_even_count:<7} {pred_odd_count:<6} {pred_et:<6} {pred_ot:<6}")
    print("-" * 40)
    print(f"TOTAL => Even Total# {total_even}, Odd Total# {total_odd} => ET# {total_et}, OT# {total_ot}")
    print(f"P Count=> Even # {pred_even_count}, Odd # {pred_odd_count} => ET# {pred_et}, OT# {pred_ot}")
    total_avg = round(pred_et + pred_ot)
    print(f"Predicted sums: {pred_ot} {pred_et} total {total_avg}")
    print("-" * 40)

    all_nums_flat = [num for nums in all_sets for num in nums]
    num_freq = Counter(all_nums_flat)
    hot_evens = [n for n, _ in num_freq.most_common() if n % 2 == 0][:10]
    hot_odds = [n for n, _ in num_freq.most_common() if n % 2 != 0][:10]
    pre_EandO_set = sorted(random.sample(hot_evens, min(pred_even_count, len(hot_evens))) +
                          random.sample(hot_odds, min(pred_odd_count, len(hot_odds))))
    print(f"✅Predicted E & O => {pre_EandO_set}")

    # ===========================
    # ENHANCED KNN PREDICTOR
    # ===========================
    class EnhancedKNNPredictor:
        def __init__(self, k=5, recency_weight=0.8):
            self.k = k
            self.recency_weight = recency_weight
            self.profiles = []
            self.sets = []

        def fit(self, profiles, sets):
            self.profiles = profiles
            self.sets = sets

        def _euclidean_distance(self, a, b):
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

        def _get_nearest_neighbors(self, input_profile):
            distances = []
            for i, profile in enumerate(self.profiles):
                dist = self._euclidean_distance(input_profile, profile)
                recency_factor = (len(self.profiles) - i) / len(self.profiles) * self.recency_weight
                weighted_dist = dist * (1 - recency_factor)
                distances.append((weighted_dist, self.sets[i]))
            distances.sort(key=lambda x: x[0])
            return [s for _, s in distances[:self.k]]

        def predict(self, input_profile):
            neighbors = self._get_nearest_neighbors(input_profile)
            predicted = []
            for pos in range(6):
                pos_nums = [s[pos] for s in neighbors if len(s) > pos]
                if pos_nums:
                    avg = round(sum(pos_nums) / len(pos_nums))
                    predicted.append(avg)
                else:
                    predicted.append(0)
            predicted = sorted(list(set(predicted)))
            while len(predicted) < 6:
                predicted.append(predicted[-1] + 1 if predicted else 1)
            return predicted[:6]

    last_profile = even_odd_profiles[-1]
    enhanced_knn = EnhancedKNNPredictor(k=5, recency_weight=0.8)
    enhanced_knn.fit(even_odd_profiles, all_sets)
    pre_EandO_set_knn = enhanced_knn.predict(last_profile)
    print(f"✅KNN-Like Predicted E & O => {pre_EandO_set_knn}")

    # ===========================
    # AVAILABLE NUMBERS
    # ===========================
    class NumberPredictor:
        def __init__(self, historical_data):
            self.historical_data = historical_data
            self.sorted_keys = sorted(historical_data.keys(), key=lambda x: int(x.split('_')[1]))
            self.max_number = max(num for nums in historical_data.values() for num in nums)
            self.frequency_count = self._compute_frequency()
            self.all_freqs = list(self.frequency_count.values())
            self.window_mean_freq = statistics.mean(self.all_freqs) if self.all_freqs else 0
        def _compute_frequency(self):
            freq = defaultdict(int)
            for key in self.sorted_keys:
                for num in self.historical_data[key]:
                    freq[num] += 1
            return freq
        def display_number_frequency(self):
            print("\n" + "-" * 40)
            print("AVAILABLE NUMBER")
            print("-" * 40)
            print("Available numbers based on Range Limit:")
            for i in range(1, self.max_number + 1):
                print(f"{i} = {self.frequency_count[i]}", end=" ")
                if i % 5 == 0:
                    print()
            print("\n" + "-" * 40)
            print("AVAILABLE NUMBER FREQUENCY (Grouped by Frequency)")
            print("-" * 40)
            grouped = defaultdict(list)
            for num in range(1, self.max_number + 1):
                f = self.frequency_count[num]
                grouped[f].append(num)
            for f in sorted(grouped, reverse=True):
                nums = sorted(grouped[f])
                print(f"Frequency {f}: {nums}")
        def display_position_frequencies(self):
            last_5_keys = self.sorted_keys[-5:]
            position_totals = [0] * 6
            print("\n" + "-" * 40)
            print("POSITIONAL FREQUENCY BREAKDOWN (Last 5 Sets):")
            print("-" * 40)
            for key in last_5_keys:
                nums = self.historical_data[key]
                line = f"{key}: "
                for i, num in enumerate(nums):
                    freq = self.frequency_count[num]
                    position_totals[i] += freq
                    line += f"{num}:{freq} "
                print(line.strip())
            print("\nTOTALS:")
            for i, total in enumerate(position_totals):
                print(f"B{i + 1}: {total}", end=" ")
            print()
            print("\nMEAN:")
            means = [round(total / 5) for total in position_totals]
            for i, mean in enumerate(means):
                print(f"B{i + 1}: {mean}", end=" ")
            print()
            return means
        def validate_actual_set(self, actual_key, actual_set, window_id, predicted_pos_means, pre_EandO_set_knn):
            sorted_actual = sorted(actual_set)
            print(f"\nACTUAL {actual_key} from Sliding window {window_id}: {sorted_actual},")
            actual_freqs = [self.frequency_count.get(num, 0) for num in sorted_actual]
            count_str = [f"[{num} = {f}]" for num, f in zip(sorted_actual, actual_freqs)]
            print(f"Count From Available number During the Sliding window {window_id}:")
            print(", ".join(count_str))

            # Summary metrics
            mean_freq = statistics.mean(actual_freqs)
            median_freq = statistics.median(actual_freqs)
            min_freq = min(actual_freqs)
            max_freq = max(actual_freqs)

            # Classification
            strong = [num for num, f in zip(sorted_actual, actual_freqs) if f > self.window_mean_freq]
            moderate = [num for num, f in zip(sorted_actual, actual_freqs) if f == self.window_mean_freq]
            weak = [num for num, f in zip(sorted_actual, actual_freqs) if 0 < f < self.window_mean_freq]
            surprise = [num for num, f in zip(sorted_actual, actual_freqs) if f == 0]

            # Hit strength
            hit_strength = (len(strong) / 6 * 100) if len(actual_set) == 6 else 0

            # Predictive alignment
            high_freq_nums = [n for n, f in self.frequency_count.items() if f > self.window_mean_freq]
            in_high_freq = sum(1 for n in sorted_actual if n in high_freq_nums)
            in_pos_regression = sum(1 for n in sorted_actual if n in predicted_pos_means)
            in_knn = sum(1 for n in sorted_actual if n in pre_EandO_set_knn)

            # Print structured summary
            print(f"SLIDING WINDOW {window_id} – ACTUAL SET VALIDATION")
            print(f"STRONG: {strong}")
            print(f"MODERATE: {moderate}")
            print(f"WEAK: {weak}")
            print(f"SURPRISE: {surprise}")
            print(f"Mean Frequency: {mean_freq:.1f}")
            print(f"Median Frequency: {median_freq}")
            print(f"Min/Max Frequency: {min_freq}/{max_freq}")
            print(f"Count of Strong: {len(strong)}, Moderate: {len(moderate)}, Weak: {len(weak)}, Surprise: {len(surprise)}")
            print(f"Hit Strength Score: {hit_strength:.0f}%")
            print(f"Predictive Alignment:")
            print(f" - In High Freq: {in_high_freq}/6")
            print(f" - In Pos Regression: {in_pos_regression}/6")
            print(f" - In KNN: {in_knn}/6")

            # Forward-looking insight
            if len(surprise) > 2:
                trend = "Volatility (cold breakouts likely)"
            else:
                trend = "Stability (hot continuation likely)"
            freq_band = f"{min_freq}-{max_freq}"
            print(f"Forward Insight: Favor {trend}, expected freq band {freq_band}")
            print("")

            return actual_freqs  # For aggregate

    num_predictor = NumberPredictor(historical_data)
    num_predictor.display_number_frequency()
    means = num_predictor.display_position_frequencies()
    # Position regression
    time_steps_pos = list(range(1, len(all_sets) + 1))
    predicted_pos_means = []
    for pos in range(6):
        pos_vals = [s[pos] for s in all_sets]
        reg = CustomLinearRegression()
        reg.fit(time_steps_pos, pos_vals)
        pred = round(reg.predict(len(all_sets) + 1))
        predicted_pos_means.append(pred)
    print("\nPredicted Pos Means (Regression):", predicted_pos_means)
    prediction_AN = sorted(predicted_pos_means)
    next_set_number = int(num_predictor.sorted_keys[-1].split('_')[1]) + 1
    print(f"\n✅ Predicted SET_{next_set_number} Based on KNN logic => {prediction_AN}")

   

    # ===========================
    # POSITION FREQUENCIES
    # ===========================
    class PositionFrequencyAnalyzer:
        def __init__(self, data):
            self.data = data
            self.position_frequencies = self._analyze()
        def _analyze(self):
            position_data = {i + 1: [] for i in range(6)}
            for numbers in self.data.values():
                for i, num in enumerate(numbers):
                    position_data[i + 1].append(num)
            return position_data
        def get_frequencies(self):
            position_counters = {}
            for pos, nums in self.position_frequencies.items():
                position_counters[pos] = Counter(nums)
            return position_counters
        def get_Bmost_frequent_numbers(self, top_n=1):
            freqs = self.get_frequencies()
            most_frequent = []
            for pos in sorted(freqs):
                top_numbers = [num for num, count in freqs[pos].most_common(top_n)]
                most_frequent.append(top_numbers)
            return most_frequent
        def display_frequencies(self):
            print("\n", "-" * 40)
            print("POSITION FREQUENCIES")
            print("-" * 40)
            freqs = self.get_frequencies()
            for pos, counter in sorted(freqs.items()):
                print(f"Ball #{pos}: {dict(counter)}")
    analyzer = PositionFrequencyAnalyzer(historical_data)
    top_1 = analyzer.get_Bmost_frequent_numbers(top_n=1)
    Bmost_frequent_numbers = [nums[0] for nums in top_1]
    print(f"\n✅ Ball Frequency Position => {Bmost_frequent_numbers}")
    analyzer.display_frequencies()

    # ===========================
    # COLUMN ANALYSIS
    # ===========================
    class HistoricalDataset:
        @staticmethod
        def get_values():
            return historical_data
        @staticmethod
        def column_means():
            columns = list(zip(*historical_data.values()))
            return [round(sum(col)/len(col), 2) for col in columns]
        @staticmethod
        def most_frequent_in_columns():
            columns = list(zip(*historical_data.values()))
            return [Counter(col).most_common(1)[0] for col in columns]
    print("\n" + "-" * 40)
    print("SET OF ELEMENTS COLUMN ANALYSIS")
    print("-" * 40)
    means_col = HistoricalDataset.column_means()
    freqs_col = HistoricalDataset.most_frequent_in_columns()
    print("\n{:<12}{}".format("", "\t".join([f"B{i+1}" for i in range(6)])))
    print("{:<12}{}".format("MEAN", "\t".join(map(str, means_col))))
    print("{:<12}{}".format("Frequent", "\t".join([f"{val}:{cnt}" for val, cnt in freqs_col])))
    column_avg = [round(m) for m in means_col]
    print(f"Average when checking from Column: {column_avg}")
    print(f"\n✅Average when checking from Column: {column_avg}")

    # ===========================
    # FIRST ANTICIPATION SET
    # ===========================
    class SimpleMLModel:
        def __init__(self, number_range=58):
            self.number_range = number_range
            self.number_scores = {i: 0 for i in range(1, number_range + 1)}
        def fit(self, data):
            for entry in data.values():
                for num in entry:
                    self.number_scores[num] += 1
                for i in entry:
                    for j in entry:
                        if i != j:
                            self.number_scores[i] += 0.1
        def predict(self, top_n=6):
            sorted_scores = sorted(self.number_scores.items(), key=lambda x: x[1], reverse=True)
            top_numbers = [num for num, _ in sorted_scores[:top_n]]
            return sorted(top_numbers)
    print("\n" + "-" * 40)
    print("FIRST ANTICIPATION SET")
    print("-" * 40)
    model = SimpleMLModel()
    model.fit(historical_data)
    prediction_1st = model.predict()
    print(f"Anticipated Elements: {prediction_1st}\n")
    print(f"✅Reuse 1st Anticipated here: {prediction_1st}")

    # ===========================
    # PAIRS AND TRIPLETS
    # ===========================
    def analyze_combinations(historical_data_dict):
        all_pairs = []
        all_triplets = []
        for draw in historical_data_dict.values():
            draw_sorted = sorted(draw)
            pairs = itertools.combinations(draw_sorted, 2)
            triplets = itertools.combinations(draw_sorted, 3)
            all_pairs.extend(pairs)
            all_triplets.extend(triplets)
        return Counter(all_pairs), Counter(all_triplets)
    print("-" * 40)
    print("ELEMETS PAIRS & TRIPLETS")
    print("-" * 40)
    pairs, triplets = analyze_combinations(historical_data)
    print("GROUP OF 2:")
    for pair, count in pairs.most_common(10):
        print(f"{pair} appeared {count} times")
    print("\nGROUP OF 3:")
    for triplet, count in triplets.most_common(10):
        print(f"{triplet} appeared {count} times")

    # ML prediction for combinations
    top_pairs_ml = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    common_nums_ml = [num for pair, _ in top_pairs_ml for num in pair]
    predicted_ml = sorted(list(set(common_nums_ml)))[:6]
    print(f"\n✅Anticipated Group Combination (ML-like) => {predicted_ml}")
    predictedPair = predicted_ml
    print(f"\nYou can now reuse predictedPair: {predictedPair}")

    # ===========================
    # HOT AND COLD NUMBERS
    # ===========================
    class FrequencyAnalyzer:
        def __init__(self, data):
            self.flat_list = [num for value in data.values() for num in value]
            self.freq_dict = Counter(self.flat_list)
        def hot_numbers(self):
            return [num for num, count in self.freq_dict.items() if count > 1]
        def cold_numbers(self):
            return [num for num, count in self.freq_dict.items() if count == 1]
        def top_hot_numbers(self, top_n=6):
            sorted_items = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)
            return [num for num, _ in sorted_items[:top_n]]
    freq_analyzer = FrequencyAnalyzer(historical_data)
    hot_numbers = freq_analyzer.hot_numbers()
    cold_numbers = freq_analyzer.cold_numbers()
    top_hot = freq_analyzer.top_hot_numbers()
    print("\n" + "-" * 40)
    print("HOT AND COLD NUMBERS")
    print("-" * 40)
    print("HOT NUMBERS (Appeared >1 times):", hot_numbers)
    print("COLD NUMBERS (Appeared only once):", cold_numbers)
    print("\n✅TOP HOT ANTICIPATED:")
    print(top_hot)

    # Store all predictions for return
    results = {
        'pre_Deff_set': pre_Deff_set,
        'pre_EandO_set': pre_EandO_set,
        'prediction_AN': prediction_AN,
        'Bmost_frequent_numbers': Bmost_frequent_numbers,
        'prediction_1st': prediction_1st,
        'predictedPair': predictedPair,
        'top_hot': top_hot,
        'column_avg': column_avg,
        'frequency_count': num_predictor.frequency_count,
        'max_number': num_predictor.max_number,
        'means': means,
        'predicted_pos_means': predicted_pos_means,
        'pre_EandO_set_knn': pre_EandO_set_knn,
        'num_predictor': num_predictor  # To call validate
    }

    return results

# ===========================
# MAIN SLIDING WINDOW EXECUTION
# ===========================
print("\n" + "="*80)
print("🚀 COMPREHENSIVE LOTTERY PREDICTION WITH SLIDING WINDOW ANALYSIS")
print("="*80)
window_size = 5
keys = sorted(COMPLETE_HISTORICAL_DATA.keys(), key=lambda x: int(x.split("_")[1]))
num_windows = len(keys) - window_size
print(f"\nTotal Sets: {len(keys)}")
print(f"Window Size: {window_size}")
print(f"Number of Windows: {num_windows}\n")
# Store ALL predictions from each window
all_window_results = []
all_predictions_by_method = {
    'difference': [],
    'even_odd': [],
    'position_regression': [],
    'ball_frequency': [],
    'first_anticipation': [],
    'pair_combination': [],
    'hot_numbers': [],
    'column_avg': []
}
next_counts_list = []
# Run complete analysis for each window
for window_idx in range(num_windows):
    window_keys = keys[window_idx : window_idx + window_size]  # Adjust to 0-based for actual next
    window_data = {}
    for i, k in enumerate(window_keys, 1):
        window_data[f"SET_{i}"] = COMPLETE_HISTORICAL_DATA[k]
    # Run YOUR complete analysis script
    results = run_complete_analysis(window_data, window_idx + 1)
    # Store results
    all_window_results.append(results)
    all_predictions_by_method['difference'].append(results['pre_Deff_set'])
    all_predictions_by_method['even_odd'].append(results['pre_EandO_set'])
    all_predictions_by_method['position_regression'].append(results['prediction_AN'])
    all_predictions_by_method['ball_frequency'].append(results['Bmost_frequent_numbers'])
    all_predictions_by_method['first_anticipation'].append(results['prediction_1st'])
    all_predictions_by_method['pair_combination'].append(results['predictedPair'])
    all_predictions_by_method['hot_numbers'].append(results['top_hot'])
    all_predictions_by_method['column_avg'].append(results['column_avg'])
    if window_idx + window_size < len(keys):
        actual_next_key = keys[window_idx + window_size]
        actual_next_set = COMPLETE_HISTORICAL_DATA[actual_next_key]
        actual_freqs = results['num_predictor'].validate_actual_set(actual_next_key, actual_next_set, window_idx + 1, results['predicted_pos_means'], results['pre_EandO_set_knn'])
        next_counts_list.append(actual_freqs)
# ===========================
# SUMMARY OF FREQUENCY COUNTS
# ===========================
print("\n" + "="*80)
print("🔍 SUMMARY OF FREQUENCY COUNTS IN ACTUAL NEXT SETS")
print("="*80)
all_next_counts = [c for sublist in next_counts_list for c in sublist]
count_distribution = Counter(all_next_counts)
print("Distribution of frequency counts for numbers in actual next sets:")
for count_val, freq in sorted(count_distribution.items()):
    print(f"Frequency {count_val}: {freq} occurrences (across {len(all_next_counts)} numbers)")
common_freqs = [val for val, _ in count_distribution.most_common()]
print("\nMost common frequency levels for next set numbers:", common_freqs[:3])

avg_surprises = sum(1 for c in all_next_counts if c == 0) / len(next_counts_list) if next_counts_list else 0
print("\nForward-Looking Insight:")
print(f"Expected frequency band for next set: {', '.join(map(str, common_freqs[:3]))}")
if avg_surprises > 1:
    print("Favor volatility (cold numbers emerging)")
else:
    print("Favor stability (hot numbers continuing)")

# ===========================
# AGGREGATE LEARNING ACROSS ALL WINDOWS
# ===========================
print("\n" + "="*80)
print("🎓 LEARNING SUMMARY FROM ALL SLIDING WINDOWS")
print("="*80)
print(f"\nTotal Windows Analyzed: {num_windows}")
# Aggregate all predicted numbers
all_predicted_numbers = []
for method_name, predictions in all_predictions_by_method.items():
    for prediction in predictions:
        if prediction:
            all_predicted_numbers.extend(prediction)
learned_frequency = Counter(all_predicted_numbers)
print("\n" + "-"*60)
print("📈 LEARNED NUMBER FREQUENCY ACROSS ALL WINDOWS")
print("-"*60)
print(f"Numbers appearing most frequently across all window predictions:")
for num, count in learned_frequency.most_common(15):
    percentage = (count / len(all_predicted_numbers)) * 100
    print(f" {num:2d}: appeared {count:2d} times ({percentage:5.1f}%)")
# ===========================
# FINAL PREDICTIONS
# ===========================
print("\n" + "="*80)
print("🏆 FINAL PREDICTION (Based on All Window Learning)")
print("="*80)
learned_prediction = sorted([num for num, _ in learned_frequency.most_common(6)])
print(f"\n✨ LEARNED CONSENSUS PREDICTION:")
print(f" {learned_prediction}")
print(f"\n This prediction is based on analyzing {num_windows} sliding windows")
print(f" and aggregating patterns from {len(all_predictions_by_method)} different methods.")
# Global First Anticipation
print("\n" + "-"*60)
print("FIRST ANTICIPATION SET (Global Historical Analysis)")
print("-"*60)
all_nums = []
for nums in COMPLETE_HISTORICAL_DATA.values():
    all_nums.extend(nums)
global_freq = Counter(all_nums)
global_top_6 = sorted([num for num, _ in global_freq.most_common(6)])
print(f"Anticipated Elements: {global_top_6}")
print(f"\n✅ Reuse 1st Anticipated here: {global_top_6}")
# ===========================
# FINAL COMPARISON
# ===========================
print("\n" + "="*80)
print("📊 FINAL PREDICTIONS COMPARISON")
print("="*80)
last_window_predictions = {
    'Difference': all_predictions_by_method['difference'][-1],
    'Column': all_predictions_by_method['column_avg'][-1],
    'Even Odd': all_predictions_by_method['even_odd'][-1],
    'Position Regression': all_predictions_by_method['position_regression'][-1],
    'Ball Frequency': all_predictions_by_method['ball_frequency'][-1],
    'First Anticipation': all_predictions_by_method['first_anticipation'][-1],
    'Pair Combination': all_predictions_by_method['pair_combination'][-1],
    'Hot Numbers': all_predictions_by_method['hot_numbers'][-1],
    'Learned': learned_prediction,
    'Global First Anticipation': global_top_6
}
print(f"\n{'Method':<30} {'Prediction'}")
print("-"*60)
for method, pred in last_window_predictions.items():
    print(f"{method:<30} {pred}")
# Consensus strength
all_final_nums = []
for pred in last_window_predictions.values():
    if pred:
        all_final_nums.extend(pred)
final_consensus_freq = Counter(all_final_nums)
print("\n" + "-"*60)
print("🎯 NUMBERS BY CONSENSUS STRENGTH")
print("-"*60)
for num, count in final_consensus_freq.most_common(12):
    methods = []
    for method, pred in last_window_predictions.items():
        if pred and num in pred:
            methods.append(method[:4])
    confidence = (count / len([p for p in last_window_predictions.values() if p])) * 100
    print(f" {num:2d}: in {count}/{len([p for p in last_window_predictions.values() if p])} methods ({confidence:.0f}%) - {', '.join(methods)}")
final_consensus = sorted([n for n, c in final_consensus_freq.most_common(6)])
print("\n" + "="*80)
print(f"🌟 ULTIMATE CONSENSUS PREDICTION FOR SET_{len(keys) + 1}")
print("="*80)
print(f"\n {final_consensus}")
print(f"\n Based on:")
print(f" • {num_windows} sliding window analyses")
print(f" • {len(all_predictions_by_method)} prediction methods")
print(f" • Pattern learning across all windows")
print("\n" + "="*80)
print("✅ Multi-window sliding analysis complete! 🍀")
print("="*80)
