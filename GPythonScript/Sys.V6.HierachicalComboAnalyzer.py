from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import math
import itertools
import statistics
import random
from typing import Dict, List, Tuple, Set, Any
import json
import scipy.stats
from functools import reduce
from math import gcd
from scipy.stats import pearsonr
# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
# -------------------------------
# HISTORICAL DATA
# -------------------------------
historical_data = {
"SET_1": [5, 11, 21, 41, 42, 45],
"SET_2": [2, 20, 30, 48, 49, 51],
"SET_3": [3, 7, 21, 32, 37, 42],
"SET_4": [7, 24, 36, 39, 48, 53],
"SET_5": [5, 10, 12, 13, 16, 36],
"SET_6": [7, 11, 14, 21, 30, 52],
"SET_7": [5, 12, 22, 40, 44, 50],
"SET_8": [8, 12, 28, 32, 38, 50],
"SET_9": [3, 11, 26, 39, 40, 53],
"SET_10": [11, 36, 41, 43, 44, 45],
}
set_names = list(historical_data.keys())
all_sets = list(historical_data.values())
# -------------------------------
# Custom Linear Regression class
#(updated to return scalar for single prediction)
# -------------------------------
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
# -------------------------------
# DIFFERENCES BETWEEN CONSECUTIVE SETS
# (Enhanced with per-column regression on diffs)
# -------------------------------
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

        # ----------------------------
        # ML Prediction for next difference
        # ----------------------------
        predicted_diff = self.predict_diffs_regression()

        # Add ML row instead of average
        self.diff_df.loc['ML PREDICTION'] = predicted_diff

        # Add highest count per column
        # ----------------------------
        # Highest possible count (mode) per column
        # ----------------------------
        highest_counts = []
        for col in self.diff_df.columns:
            mode_val = self.diff_df[col].mode()
            if not mode_val.empty:
               highest_counts.append(mode_val.iloc[0])  # take the first mode if multiple
            else:
                highest_counts.append(np.nan)  # fallback if no mode
        self.diff_df.loc['HIGHEST COUNT'] = highest_counts

        print(self.diff_df.to_string())

        # ----------------------------
        # Compute predicted next set
        # ----------------------------
        last_set = [int(x) for x in self.df.iloc[-1].tolist()]
        pre_Deff_set = sorted([max(1, min(58, last_set[i] + predicted_diff[i])) for i in range(6)])  # Bound 1-58

        print("\nPredicted Difference from Last SET (ML Regression) =>", ' '.join(map(str, predicted_diff)))
        print("Predicted SET_N =>", pre_Deff_set)
        print(f"✅Reusable pre_Deff_set: {pre_Deff_set}")

        return pre_Deff_set  # Return for global use

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

# ----------------------------
# Example usage
# ----------------------------
df = prepare_dataframe(historical_data)  # your historical_data dictionary
predictor_model = DifferencePredictor(df)
pre_Deff_set = predictor_model.display_differences()

# -------------------------------
# EVEN/ODD ANALYSIS (Enhanced with regression on counts)
# -------------------------------
print("\n" + "-" * 40)
print("EVEN And ODD Details")
print("-" * 40)
print("Set Name Even# Odd# ET# OT#")
total_even = total_odd = total_et = total_ot = 0
even_odd_profiles = []
even_odd_sets = []
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
    profile = [len(evens), len(odds), even_sum, odd_sum]
    even_odd_profiles.append(profile)
    even_odd_sets.append(numbers)
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
# Generate prediction based on predicted counts
all_nums_flat = [num for nums in all_sets for num in nums]
num_freq = Counter(all_nums_flat)
hot_evens = [n for n, _ in num_freq.most_common() if n % 2 == 0][:10]
hot_odds = [n for n, _ in num_freq.most_common() if n % 2 != 0][:10]
pre_EandO_set = sorted(random.sample(hot_evens, min(pred_even_count, len(hot_evens))) + random.sample(hot_odds, min(pred_odd_count, len(hot_odds))))
print(f"✅Predicted E & O => {pre_EandO_set}")
# -------------------------------
# CUSTOM KNN-LIKE PREDICTOR
# (Enhanced with position-based averaging and weighting)
# -------------------------------
class HistoricalKNNPredictor:
    def __init__(self, k=3):
        self.k = k
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
            distances.append((dist, self.sets[i]))
        distances.sort(key=lambda x: x[0])
        return [s for _, s in distances[:self.k]]
    def predict(self, input_profile):
        neighbors = self._get_nearest_neighbors(input_profile)
        all_numbers = []
        for s in neighbors:
            all_numbers.extend(s)
        number_counts = Counter(all_numbers)
        most_common = number_counts.most_common(6)
        predicted_set = sorted([num for num, _ in most_common])
        return predicted_set
# -------------------------------
# Enhanced version with recency weighting and position averaging
# -------------------------------
class EnhancedKNNPredictor(HistoricalKNNPredictor):
    def __init__(self, k=5, recency_weight=0.8):
        super().__init__(k)
        self.recency_weight = recency_weight
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
                predicted.append(0)  # Fallback
        predicted = sorted(list(set(predicted)))  # Remove duplicates and sort
        while len(predicted) < 6:
            predicted.append(predicted[-1] + 1 if predicted else 1)  # Fill if needed
        return predicted[:6]

last_profile = even_odd_profiles[-1]
enhanced_knn = EnhancedKNNPredictor(k=5, recency_weight=0.8)
enhanced_knn.fit(even_odd_profiles, all_sets)
pre_EandO_set = enhanced_knn.predict(last_profile)
print(f"✅KNN-Like Predicted E & O => {pre_EandO_set}")
# -------------------------------
# AVAILABLE NUMBERS (Enhanced with regressed position means)
# -------------------------------
class NumberPredictor:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.sorted_keys = sorted(historical_data.keys(), key=lambda x: int(x.split('_')[1]))
        self.max_number = max(num for nums in historical_data.values() for num in nums)
        self.frequency_count = self._compute_frequency()
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
    def predict_using_knn(self, k=5):
        last_k_keys = self.sorted_keys[-k:]
        recent_numbers = []
        for key in last_k_keys:
            recent_numbers.extend(self.historical_data[key])
        recent_counter = Counter(recent_numbers)
        top_6 = [num for num, count in recent_counter.most_common(6)]
        return top_6
num_predictor = NumberPredictor(historical_data)
num_predictor.display_number_frequency()
means = num_predictor.display_position_frequencies()
# -------------------------------
# Enhance means with regression
# -------------------------------
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
# -------------------------------
# POSITION FREQUENCIES
# -------------------------------
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
    def get_position_frequency_sums(self):
        value_sums = {}
        for pos, nums in self.position_frequencies.items():
            value_sums[pos] = sum(nums)
        return value_sums
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
    def display_sums(self):
        print("\n", "-" * 40)
        print("POSITION FREQUENCY SUMS")
        print("-" * 40)
        sums = self.get_position_frequency_sums()
        for pos, total in sorted(sums.items()):
            print(f"Ball #{pos}: {total}")
analyzer = PositionFrequencyAnalyzer(historical_data)
top_1 = analyzer.get_Bmost_frequent_numbers(top_n=1)
Bmost_frequent_numbers = [nums[0] for nums in top_1]
print(f"\n✅ Ball Frequency Position => {Bmost_frequent_numbers}")
analyzer.display_frequencies()
analyzer.display_sums()
freqs = analyzer.get_frequencies()
class KNNFrequencyPredictor:
    def __init__(self, position_freqs):
        self.freqs = position_freqs
    def predict_top_k(self, k=3):
        pre_Fre_Prox = {}
        for pos, counter in self.freqs.items():
            most_common = counter.most_common(k)
            pre_Fre_Prox[pos] = [num for num, _ in most_common]
        return pre_Fre_Prox
    def display_predictions(self, pre_Fre_Prox):
        print("\n", "-" * 40)
        print(f"PREDICTIONS (Top {len(next(iter(pre_Fre_Prox.values())))} for each Ball Position)")
        print("-" * 40)
        for pos, values in pre_Fre_Prox.items():
            print(f"Ball #{pos}: {values}")
freq_predictor = KNNFrequencyPredictor(freqs)
pre_Fre_Prox = freq_predictor.predict_top_k(k=3)
freq_predictor.display_predictions(pre_Fre_Prox)
# -------------------------------
# COLUMN ANALYSIS
# -------------------------------
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
    @staticmethod
    def get_mean_test_point():
        columns = list(zip(*historical_data.values()))
        return [sum(col) / len(col) for col in columns]
class CustomKNN:
    def __init__(self, k=3):
        self.k = k
    def euclidean_distance(self, point1, point2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    def predict(self, historical_data_dict, test_point):
        distances = []
        for key, numbers in historical_data_dict.items():
            dist = self.euclidean_distance(numbers, test_point)
            distances.append((dist, numbers))
        distances.sort(key=lambda x: x[0])
        nearest = [numbers for _, numbers in distances[:self.k]]
        flat_numbers = [num for subset in nearest for num in subset]
        prediction_CA = [num for num, _ in Counter(flat_numbers).most_common(6)]
        return prediction_CA
# -------------------------------
# SET OF ELEMENTS COLUMN ANALYSIS
# -------------------------------
class LotteryAnalyzer:
    def __init__(self):
        self.dataset = HistoricalDataset()
        self.knn_model = CustomKNN(k=3)
    def analyze(self):
        data = self.dataset.get_values()
        print("\n" + "-" * 40)
        print("SET OF ELEMENTS COLUMN ANALYSIS")
        print("-" * 40)
        means = self.dataset.column_means()
        freqs = self.dataset.most_frequent_in_columns()
        print("\n{:<12}{}".format("", "\t".join([f"B{i+1}" for i in range(6)])))
        print("{:<12}{}".format("MEAN", "\t".join(map(str, means))))
        print("{:<12}{}".format("Frequent", "\t".join([f"{val}:{cnt}" for val, cnt in freqs])))
        test_point = self.dataset.get_mean_test_point()
        prediction_CA = self.knn_model.predict(data, test_point)
        print("{:<12}{}".format("KNN", "\t".join(map(str, prediction_CA))))
        # Additional: Desired column average line - computed as rounded means
        column_avg = [round(m) for m in means]
        print(f"Average when checking from Column: {column_avg}")
lottery_analyzer = LotteryAnalyzer()
lottery_analyzer.analyze()
# -------------------------------
# FIRST ANTICIPATION SET
# -------------------------------
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
class PredictionPresenter:
    @staticmethod
    def show(prediction_1st):
        print("\n" + "-" * 40)
        print("FIRST ANTICIPATION SET")
        print("-" * 40)
        print(f"Anticipated Elements: {prediction_1st}\n")
model = SimpleMLModel()
model.fit(historical_data)
prediction_1st = model.predict()
PredictionPresenter.show(prediction_1st)
print(f"✅Reuse 1st Anticipated here: {prediction_1st}")
# -------------------------------
# PAIRS AND TRIPLETS
# -------------------------------
def analyze_combinations(historical_data_dict):
    all_pairs = []
    all_triplets = []
    for draw in historical_data_dict.values():
        draw_sorted = sorted(draw)
        pairs = itertools.combinations(draw_sorted, 2)
        triplets = itertools.combinations(draw_sorted, 3)
        all_pairs.extend(pairs)
        all_triplets.extend(triplets)
    pair_counter = Counter(all_pairs)
    triplet_counter = Counter(all_triplets)
    return pair_counter, triplet_counter

def display_combo_results(pair_counter, triplet_counter, top_n=10):
    print("-" * 40)
    print("ELEMETS PAIRS & TRIPLETS")
    print("-" * 40)
    print("GROUP OF 2:")
    for pair, count in pair_counter.most_common(top_n):
        print(f"{pair} appeared {count} times")
    print("\nGROUP OF 3:")
    for triplet, count in triplet_counter.most_common(top_n):
        print(f"{triplet} appeared {count} times")

# Define euclidean for knn_predict_combination
def euclidean(x, y):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

# Enhanced KNN-like prediction for combinations
def knn_predict_combination(historical_data_dict, k=3):
    # Flatten sets to vectors for distance calculation (e.g., sorted tuple as vector)
    sets_list = list(historical_data_dict.values())
    last_set = sorted(sets_list[-1])

    # Compute distances to previous sets
    distances = []
    for s in sets_list[:-1]:
        s_sorted = sorted(s)
        dist = euclidean(last_set, s_sorted)  # Euclidean distance between sorted sets
        distances.append((dist, s_sorted))

    # Sort and get top k nearest
    distances.sort(key=lambda x: x[0])
    neighbors = [s for _, s in distances[:k]]

    # Collect common pairs from neighbors
    neighbor_pairs = []
    for s in neighbors:
        pairs = list(itertools.combinations(s, 2))
        neighbor_pairs.extend(pairs)

    pair_counter_knn = Counter(neighbor_pairs)
    common_pairs_knn = [num for pair, _ in pair_counter_knn.most_common(5) for num in pair]
    predicted_knn = sorted(list(set(common_pairs_knn)))[:6]
    return predicted_knn

# Enhanced ML-like prediction using regression on pair frequencies
def ml_predict_combination(pair_counter):
    # Get frequencies over time (assume sets are ordered)
    times = list(range(1, len(historical_data) + 1))
    freq_trends = {}

    for pair in pair_counter:
        # Simulate frequency trend for each pair (cumulative count)
        cum_count = []
        count = 0
        for s in historical_data.values():
            s_sorted = sorted(s)
            if tuple(sorted(pair)) in list(itertools.combinations(s_sorted, 2)):
                count += 1
            cum_count.append(count)

        # Fit regression
        reg = CustomLinearRegression()
        reg.fit(times, cum_count)
        next_freq = reg.predict(len(times) + 1)
        freq_trends[pair] = next_freq

    # Top pairs based on predicted frequency
    top_pairs_ml = sorted(freq_trends.items(), key=lambda x: x[1], reverse=True)[:5]
    common_nums_ml = [num for pair, _ in top_pairs_ml for num in pair]
    predicted_ml = sorted(list(set(common_nums_ml)))[:6]
    return predicted_ml

# Usage
pairs, triplets = analyze_combinations(historical_data)
display_combo_results(pairs, triplets)

# KNN-like prediction
predicted_knn = knn_predict_combination(historical_data)
print(f"\n✅Anticipated Group Combination (KNN-like) => {predicted_knn}")

# ML-like prediction
predicted_ml = ml_predict_combination(pairs)
print(f"\n✅Anticipated Group Combination (ML-like) => {predicted_ml}")

# Reuse one of them, e.g., ML-like
predictedPair = predicted_ml
print(f"\nReuse AnticipatedPair: {predictedPair}")
# -------------------------------
# HOT AND COLD NUMBERS
# -------------------------------
class FrequencyAnalyzer:
    def __init__(self, data):
        self.data = data
        self.flat_list = self.flatten()
        self.freq_dict = self.count_frequency()
    def flatten(self):
        return [num for value in self.data.values() for num in value]
    def count_frequency(self):
        freq = {}
        for num in self.flat_list:
            freq[num] = freq.get(num, 0) + 1
        return freq
    def hot_numbers(self):
        return [num for num, count in self.freq_dict.items() if count > 1]
    def cold_numbers(self):
        return [num for num, count in self.freq_dict.items() if count == 1]
    def top_hot_numbers(self, top_n=6):
        sorted_items = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_items[:top_n]]
class SimpleKNN:
    def __init__(self, data, k=3):
        self.k = k
        self.data = list(data.values())
    def distance(self, set1, set2):
        return len(set(set1) ^ set(set2))
    def predict_next(self):
        test_set = self.data[-1]
        distances = []
        for i in range(len(self.data) - 1):
            dist = self.distance(test_set, self.data[i])
            distances.append((dist, self.data[i]))
        neighbors = [data for _, data in sorted(distances)[:self.k]]
        combined = [num for neighbor in neighbors for num in neighbor]
        pred_freq = {}
        for num in combined:
            pred_freq[num] = pred_freq.get(num, 0) + 1
        sorted_pred = sorted(pred_freq.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_pred[:6]]
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
possible_mix = top_hot[:3] + [num for num in cold_numbers if num not in top_hot][:3]
print("\n✅POSSIBLE HOT AND COLD ANTICIPATED COMBINATION:")
print(possible_mix)
knn_hot_cold = SimpleKNN(historical_data, k=3)
knn_prediction_HaC = knn_hot_cold.predict_next()
print("\n✅KNN ANTICIPATED COMBINATION BASED ON LAST SET:")
print(knn_prediction_HaC)
# -------------------------------
# EVERY 5 SETS MEAN CHECK (Enhanced with regression on group means)
# -------------------------------
def group_sets(data, group_size=5):
    keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    return [keys[i:i+group_size] for i in range(0, len(keys), group_size)]
def calculate_means(grouped_keys, data):
    results = []
    for group in grouped_keys:
        position_data = [[] for _ in range(6)]
        for key in group:
            for i in range(6):
                position_data[i].append(data[key][i])
        means = [round(statistics.mean(pos), 1) for pos in position_data]
        results.append((group[0], group[-1], means))
    return results
def print_mean_results(mean_results):
    print("\n", "-" * 40)
    print("EVERY 5 SETS MEAN CHECK")
    print("-" * 40)
    for start, end, means in mean_results:
        print(f"Mean values of {start} to {end}: {means}")
grouped_keys = group_sets(historical_data)
mean_results = calculate_means(grouped_keys, historical_data)
print_mean_results(mean_results)
# Enhance: Regression on group means per position
group_means = [means for _, _, means in mean_results]
group_time = list(range(1, len(group_means) + 1))
predicted_group_mean = []
for pos in range(6):
    pos_means = [g[pos] for g in group_means]
    reg = CustomLinearRegression()
    reg.fit(group_time, pos_means)
    pred = round(reg.predict(len(group_means) + 1))
    predicted_group_mean.append(pred)
print(f"✅Anticipated EVERY 5 SETS ML: {predicted_group_mean}")
# -------------------------------
# COMBINATION COUNT PER SEGMENT (Enhanced with weighted frequencies)
# -------------------------------
def get_segments(numbers):
    segments = []
    for number in numbers:
        if 1 <= number <= 9:
            segments.append("1-9")
        elif 10 <= number <= 19:
            segments.append("10-19")
        elif 20 <= number <= 29:
            segments.append("20-29")
        elif 30 <= number <= 39:
            segments.append("30-39")
        elif 40 <= number <= 49:
            segments.append("40-49")
        elif 50 <= number <= 58:
            segments.append("50-58")
    return segments
segments_data = {}
for set_name, numbers in historical_data.items():
    segments_data[set_name] = get_segments(numbers)
def count_combinations_segments(segments_data_dict):
    combination_counts = Counter()
    for set_name, segments in segments_data_dict.items():
        for comb in itertools.combinations(sorted(segments), 6):
            combination_counts[comb] += 1
    return combination_counts
combination_counts = count_combinations_segments(segments_data)
print("\n", "-" * 40)
print("COMBINATION COUNT PER SEGMENT")
print("-" * 40)
for set_name, segments in segments_data.items():
    print(f'"{set_name}": {segments}')
top_5_combinations = combination_counts.most_common(5)
print("\n", "-" * 40)
print("Top 5 Most Frequent Segment Patterns:")
print("-" * 40)
for comb, count in top_5_combinations:
    print(f"{', '.join(comb)} => {count} times")
def get_number_frequencies():
    segment_frequencies = {
        "1-9": Counter(),
        "10-19": Counter(),
        "20-29": Counter(),
        "30-39": Counter(),
        "40-49": Counter(),
        "50-58": Counter(),
    }
    for set_name, numbers in historical_data.items():
        segments = get_segments(numbers)
        for segment, number in zip(segments, numbers):
            segment_frequencies[segment][number] += 1
    return segment_frequencies
segment_frequencies = get_number_frequencies()
print("\n", "-" * 40)
print("Number Frequencies per Segment:")
print("-" * 40)
for segment, freq in segment_frequencies.items():
    print(f"{segment}: {dict(freq)}")
# Enhance: Weighted by recency (exponential decay)
def weighted_segment_freq(segment_frequencies, decay=0.9):
    weighted = {seg: {} for seg in segment_frequencies}
    recent_sets = list(reversed(all_sets))
    for i, s in enumerate(recent_sets):
        segs = get_segments(s)
        for seg, num in zip(segs, s):
            if num not in weighted[seg]:
                weighted[seg][num] = 0
            weighted[seg][num] += (decay ** i)
    return weighted
weighted_segments = weighted_segment_freq(segment_frequencies)
predicted_combination_seg = []
if top_5_combinations:
    for segment in top_5_combinations[0][0]:
        d = weighted_segments.get(segment, {})
        if d:
            most_frequent_number = max(d, key=d.get)
            predicted_combination_seg.append(most_frequent_number)
print("\n✅Anticipated Combination Segment (based on weighted frequency):")
print(predicted_combination_seg)
# -------------------------------
# BASED ON POSITION FREQUENCY
# -------------------------------
def recency_weight(number, historical_data_dict):
    recent_sets = list(historical_data_dict.values())[-3:]
    recent_count = sum(1 for s in recent_sets if number in s)
    return recent_count / 3.0 if len(recent_sets) > 0 else 0
def enhanced_generate_prediction(position_frequencies, historical_data_dict, weight_factor=0.7):
    pre_set_Bsd_Fre = []
    for freq in position_frequencies.values():
        if not freq:
            continue
        numbers, counts = zip(*freq.items())
        probabilities = np.array(counts) / np.sum(counts) if np.sum(counts) > 0 else np.zeros(len(counts))
        weighted_probs = probabilities * weight_factor + (1 - weight_factor) * np.array([recency_weight(number, historical_data_dict) for number in numbers])
        if np.sum(weighted_probs) > 0:
            weighted_probs /= np.sum(weighted_probs)
            prediction_fre = np.random.choice(numbers, p=weighted_probs)
            pre_set_Bsd_Fre.append(int(prediction_fre))
    return sorted(pre_set_Bsd_Fre)
print("\n", "-" * 40)
print("BASED ON POSITION FREQUENCY")
print("-" * 40)
pre_set_Bsd_Fre = enhanced_generate_prediction(analyzer.get_frequencies(), historical_data)
print(pre_set_Bsd_Fre)
# -------------------------------
# ANTICIPATED PROBABILITY
# -------------------------------
all_values = [value for sublist in historical_data.values() for value in sublist]
unique_values, counts = np.unique(all_values, return_counts=True)
probabilities = counts / len(all_values) * 100
prob_dict = {val: prob for val, prob in zip(unique_values, probabilities)}
sorted_probabilities = {k: v for k, v in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)}
for idx, (val, prob) in enumerate(sorted_probabilities.items()):
    if idx < 6:
        print(f"{val}\t{prob:.2f}%")
predicted_numbers2 = []
for val, _ in list(sorted_probabilities.items())[:6]:
    predicted_numbers2.append(int(val))
print("\n", "-" * 40)
print("ANTICIPATED PROBABILITY")
print("-" * 40)
print(f"\n✅Anticioated Probability Element:\n {predicted_numbers2}")
# -------------------------------
# SPEED DRAWN (Enhanced with recency weighting)
# -------------------------------
def recency_weighted_speed(data, decay=0.8):
    weighted = Counter()
    for i, (name, numbers) in enumerate(reversed(data.items())):
        w = decay ** i
        for num in numbers:
            weighted[num] += w
    return [num for num, _ in weighted.most_common(6)]
recommended_numbers = recency_weighted_speed(historical_data)
print("\n", "-" * 40)
print("SPEED DRAWN BASED ON HISTORICAL VALUE")
print("-" * 40)
print("✅ SPEED DRAWN:\n", recommended_numbers)
# -------------------------------
# MAPPING OF SETS
# -------------------------------
def map_sets(historical_data_dict):
    mapping = {}
    for set_name, numbers in historical_data_dict.items():
        for i, number in enumerate(numbers):
            if number not in mapping:
                mapping[number] = []
            mapping[number].append((set_name, f"B{i+1}"))
    mapped_sets = {}
    last_five_sets = list(historical_data_dict.keys())[-5:]
    for set_name in last_five_sets:
        mapped_set = []
        mapped_set_copy = []
        current_set_index = list(historical_data_dict.keys()).index(set_name)
        for number in historical_data_dict[set_name]:
            number_mapping = mapping.get(number, [])
            if number_mapping:
                earliest_mapping = None
                for mapped_set_name, bucket in number_mapping:
                    mapped_set_index = list(historical_data_dict.keys()).index(mapped_set_name)
                    if mapped_set_index < current_set_index:
                        if earliest_mapping is None or mapped_set_index > list(historical_data_dict.keys()).index(earliest_mapping[0]):
                            earliest_mapping = (mapped_set_name, bucket)
                if earliest_mapping:
                    mapped_set.append(f"{earliest_mapping[0]} {earliest_mapping[1]}")
                    earliest_set_name, bucket = earliest_mapping
                    bucket_index = int(bucket[1:]) - 1
                    next_set_index = list(historical_data_dict.keys()).index(earliest_set_name) + 1
                    if next_set_index < len(historical_data_dict):
                        next_set_name = list(historical_data_dict.keys())[next_set_index]
                        mapped_set_copy.append(historical_data_dict[next_set_name][bucket_index])
                    else:
                        mapped_set_copy.append("N/A")
                else:
                    mapped_set.append("N/A")
                    mapped_set_copy.append("N/A")
            else:
                mapped_set.append("N/A")
                mapped_set_copy.append("N/A")
        mapped_sets[set_name] = {
            "MAP_SET": mapped_set,
            "MAP_Copy": mapped_set_copy
        }
    return mapped_sets
print("\n", "-" * 40)
print("MAPPING LAST 5 SETS OF NUMBER")
print("-" * 40)
mapped_sets = map_sets(historical_data)
for set_name, mapped_set in mapped_sets.items():
    print(f'"{set_name}": {historical_data[set_name]}')
    print(f'MAP_{set_name}: {mapped_set["MAP_SET"]}')
    print(f'MAP_Copy_{set_name}: {mapped_set["MAP_Copy"]}')
    print()
# -------------------------------
# PREDICTING OPTIMAL LOTTERY SET
# -------------------------------
candidate_sets = [
    pre_Deff_set,
    pre_EandO_set,
    prediction_AN,
    Bmost_frequent_numbers,
    prediction_1st,
    predictedPair,
    knn_prediction_HaC,
    predicted_combination_seg
]
expanded_candidate_sets = []
for candidate in candidate_sets:
    candidate = list(set(candidate))
    expanded_candidate_sets.append(candidate)
    half_size = len(candidate) // 2
    for _ in range(2):
        if len(candidate) >= half_size and half_size > 0:
            subset = random.sample(candidate, half_size)
            expanded_candidate_sets.append(subset)
all_numbers_flat = [num for draw in historical_data.values() for num in draw]
number_counts = Counter(all_numbers_flat)
hot_numbers_list = [num for num, _ in number_counts.most_common(10)]
cold_numbers_list = [num for num, _ in number_counts.most_common()[-10:]]
hot_numbers_set = set(hot_numbers_list)
cold_numbers_set = set(cold_numbers_list)
last_set_key = sorted(historical_data.keys(), key=lambda k: int(k.split('_')[1]))[-1]
last_draw = historical_data[last_set_key]
last_draw_set = set(last_draw)
# ====================================
# Ranked Anticipation Sets
# ====================================
def score_candidate(candidate, last_draw_list, hot_set, cold_set):
    score = 0
    seen = set()
    dup_penalty = 0
    hot_score = 0
    cold_penalty = 0
    repeat_penalty = 0
    segment_spread = set()
    even, odd = 0, 0
    for num in candidate:
        if num in seen:
            dup_penalty += 10
        seen.add(num)
        if num in hot_set:
            hot_score += 2
        if num in cold_set:
            cold_penalty += 1
        if num in last_draw_list:
            repeat_penalty += 2
        if num <= 9:
            segment_spread.add('1-9')
        elif num <= 19:
            segment_spread.add('10-19')
        elif num <= 29:
            segment_spread.add('20-29')
        elif num <= 39:
            segment_spread.add('30-39')
        elif num <= 49:
            segment_spread.add('40-49')
        else:
            segment_spread.add('50-58')
        if num % 2 == 0:
            even += 1
        else:
            odd += 1
    even_odd_score = -abs(even - 3)
    segment_score = len(segment_spread)
    score += hot_score * 2
    score -= cold_penalty
    score -= repeat_penalty
    score -= dup_penalty
    score += even_odd_score * 1.5
    score += segment_score * 2
    return score
scored_candidates = [(score_candidate(s, last_draw, hot_numbers_set, cold_numbers_set), s)
                    for s in expanded_candidate_sets if len(s) >= 6]  # Filter to full sets
scored_candidates.sort(reverse=True)
print(f"Last Draw ({last_set_key}): {last_draw}")
print(f"Hot Numbers: {hot_numbers_list}")
print(f"Cold Numbers: {cold_numbers_list}\n")
print("\n", "-" * 40)
print("Ranked Anticipated Sets (including half-choices):")
print("-" * 40)
for rank, (score, s) in enumerate(scored_candidates, 1):
    print(f"{rank}. Set: {s}, Score: {score}")
best_predicted_set = scored_candidates[0][1] if scored_candidates else []
print(f"\n✅ BEST ANTICIPATED SET: {' '.join(map(str, best_predicted_set))}")
# ====================================
# Unified Lottery Predictor
# ====================================
class RNGCracker:
    def __init__(self, historical_dict):
        self.historical = historical_dict
        self.flatten_seq = [num for s in historical_dict.values() for num in s]
        self.positional_seqs = [[] for _ in range(6)]
        for s in historical_dict.values():
            for i, num in enumerate(s):
                self.positional_seqs[i].append(num)
    def crack_lcg_flat(self):
        seq = self.flatten_seq
        if len(seq) < 3: return None
        s = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        z = [abs(s[i+2]*s[i] - s[i+1]**2) for i in range(len(s)-2)]
        if not z: return None
        gcd_z = reduce(gcd, z)
        return None if gcd_z <= 1 else None
    def check_lfsr(self):
        seq = np.array(self.flatten_seq)
        if len(seq) < 2: return None
        corr, _ = pearsonr(seq[:-1], seq[1:])
        return None if abs(corr) < 0.5 else None
    def crack_mt(self):
        if len(self.flatten_seq) >= 624:
            return None
        np.random.seed(20251110)
        candidate = sorted(np.random.choice(range(1, 59), 6, replace=False))
        return candidate, 0.2
    def predict(self):
        lcg = self.crack_lcg_flat()
        if lcg: return lcg, "LCG"
        lfsr = self.check_lfsr()
        if lfsr: return lfsr, "LFSR"
        mt = self.crack_mt()
        if mt: return mt[0], "MT", mt[1]
        return None, "Fail"
class Utility:
    @staticmethod
    def extract_columns(d):
        cols = [[] for _ in range(6)]
        for key in sorted(d.keys(), key=lambda x: int(x.split('_')[1])):
            for i, n in enumerate(d[key]):
                cols[i].append(n)
        return cols
    @staticmethod
    def calculate_mean(s): return sum(s) / len(s) if s else 0
    @staticmethod
    def get_gaps(s): return [s[i+1] - s[i] for i in range(5)] if len(s) >= 6 else []
    @staticmethod
    def calculate_distance(s1, s2):
        return math.sqrt(sum((a-b)**2 for a,b in zip(s1,s2)))
class FunctionalClassifier:
    def __init__(self, historical_data_sets, k=5):
        self.historical_data_sets = historical_data_sets
        self.k = k
        self.all_numbers = list(range(1, 59))
    def predict_score(self, target_position):
        scores = {n: 0.0 for n in self.all_numbers}
        last_key = sorted(self.historical_data_sets.keys())[-1]
        last_set = self.historical_data_sets[last_key]
        distances = []
        for key, cur in self.historical_data_sets.items():
            if cur == last_set: continue
            d = Utility.calculate_distance(last_set, cur)
            distances.append((d, cur[target_position]))
        distances.sort()
        neighbors = distances[:self.k]
        for _, num in neighbors:
            scores[num] += 1.0 / self.k
        max_s = max(scores.values()) if scores.values() else 1.0
        return {n: s/max_s for n,s in scores.items()}
class FunctionalRegressor:
    def __init__(self, hist):
        self.hist = hist
    def predict(self):
        return sum(self.hist)/len(self.hist) if self.hist else 0.0
def score_thematic(all_historical_data_dict):
    scores = {n: 0.0 for n in range(1,59)}
    sets = list(all_historical_data_dict.values())
    flat = [n for s in sets for n in s]
    for n in flat:
        scores[n] += (flat.count(n) / len(flat)) * 0.5
    if len(sets) >= 3:
        set_n2 = sets[-3]
        for n in set_n2:
            if n not in sets[-1] and n not in sets[-2]:
                scores[n] += 0.5
    max_s = max(scores.values()) if scores.values() else 1.0
    return {n: s/max_s for n,s in scores.items()}
def calculate_u_score(s_pos, s_gap, s_theme,
                   w={'pos':0.35, 'gap':0.35, 'theme':0.30}):
    u = {}
    for n in range(1,59):
        avg_pos = sum(s_pos[i].get(n,0) for i in range(6))/6
        gap_score = s_gap.get(n,0)
        u[n] = (w['pos']*avg_pos) + (w['gap']*gap_score) + (w['theme']*s_theme.get(n,0))
    return sorted(u.items(), key=lambda x: x[1], reverse=True)
def predict_set_24(historical_data_dict):
    print("\n", "-" * 40)
    print("--- UPS PROTOCOL – Phase I: Pre-processing ---")
    print("-" * 40)
    columns = Utility.extract_columns(historical_data_dict)
    all_sets_ups = list(historical_data_dict.values())
    means = [Utility.calculate_mean(s) for s in columns]
    mu_pred = FunctionalRegressor(means).predict()
    print(f"mu_pred = {mu_pred:.2f}")
    gaps = [Utility.get_gaps(s) for s in all_sets_ups]
    pred_gaps = [FunctionalRegressor([g[i] for g in gaps if len(g) > i]).predict() for i in range(5)]
    print(f"Anticipated gaps = {[round(g,1) for g in pred_gaps]}")
    s_pos = [FunctionalClassifier(historical_data_dict).predict_score(i) for i in range(6)]
    s_theme = score_thematic(historical_data_dict)
    s_gap = {}
    for n in range(1,59):
        cnt = sum(1 for s in all_sets_ups for i in range(1,6)
                  if i < len(s) and s[i]==n and (s[i]-s[i-1])<=5)
        s_gap[n] = cnt / len(all_sets_ups)
    print("\n", "-" * 40)
    print("--- Phase II & III: Synthesis ---")
    print("-" * 40)
    u_rank = calculate_u_score(s_pos, s_gap, s_theme)
    core_4 = sorted([n for n,_ in u_rank[:4]])
    cand_last2 = [n for n,_ in u_rank[4:10]]
    best_set = []
    best_local = -1
    best_mean_err = float('inf')
    for i in range(len(cand_last2)):
        for j in range(i+1, len(cand_last2)):
            pair = [cand_last2[i], cand_last2[j]]
            cur = sorted(core_4 + pair)
            cur_gaps = Utility.get_gaps(cur)
            local = sum(1 for g in cur_gaps if g<=5)
            mean_err = abs(Utility.calculate_mean(cur) - mu_pred)
            if local > best_local or (local==best_local and mean_err<best_mean_err):
                best_local, best_mean_err = local, mean_err
                best_set = cur
    odds = sum(1 for n in best_set if n%2)
    if odds not in (2,3,4):
        random.shuffle(best_set)
        best_set.sort()
    print(f"Core 4 : {core_4}")
    print(f"Best local gaps: {best_local}")
    print(f"Mean error : {best_mean_err:.2f}")
    return best_set
ups_pred = predict_set_24(historical_data)
print("\n" + "="*55)
print("UPS PROTOCOL RESULT – SET_24")
print("="*55)
print(f"Anticipated numbers : {ups_pred}")
print("="*55)
class LotteryAnalyzerUnified:
    def __init__(self, sets_dict):
        self.sets_dict = sets_dict
        self.sets_list = []
        self.positions = {f'B{i}': [] for i in range(1,7)}
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.total_sets = len(sets_dict)
        self.features = None
        self.validate_and_organize()
    def validate_and_organize(self):
        all_nums = []
        for i in range(1, self.total_sets+1):
            key = f"SET_{i}"
            s = self.sets_dict[key]
            if len(s) != 6: raise ValueError(f"{key} must have 6 numbers")
            if len(set(s)) != 6: raise ValueError(f"{key} has duplicates")
            if s != sorted(s): raise ValueError(f"{key} not sorted")
            self.sets_list.append(s)
            all_nums.extend(s)
        self.global_min = min(all_nums)
        self.global_max = max(all_nums)
        for s in self.sets_list:
            for idx in range(6):
                self.positions[f'B{idx+1}'].append(s[idx])
    def calculate_features(self):
        self.features = {
            'frequency': Counter(),
            'positional_freq': {f'B{i}': Counter() for i in range(1,7)},
            'gaps': {},
            'pairs': Counter(),
            'sums': [],
            'sequential_shifts': []
        }
        for s in self.sets_list:
            self.features['frequency'].update(s)
        for pos in self.positions:
            self.features['positional_freq'][pos].update(self.positions[pos])
        for s in self.sets_list:
            self.features['sums'].append(sum(s))
            for i in range(5):
                for j in range(i+1,6):
                    pair = tuple(sorted((s[i], s[j])))
                    self.features['pairs'][pair] += 1
        cur_idx = self.total_sets + 1
        for n in range(1, self.global_max+1):
            last = next((i+1 for i,s in enumerate(reversed(self.sets_list)) if n in s), 0)
            self.features['gaps'][n] = cur_idx - last - 1 if last else cur_idx
        for i in range(1, self.total_sets):
            sh = [self.sets_list[i][j] - self.sets_list[i-1][j] for j in range(6)]
            self.features['sequential_shifts'].append(sh)
        return self.features
    def genetic_algorithm_prediction(self):
        def gen(): return sorted(random.sample(range(self.global_min, self.global_max+1), 6))
        def fit(ind):
            sc = sum(self.features['frequency'].get(n,0)*2 for n in ind)
            for n in ind:
                g = self.features['gaps'].get(n, self.total_sets+1)
                sc += 3 if g>=10 else 1 if g in (1,2) else 0
            seg_sz = self.global_max // 3
            bins = [(1,seg_sz),(seg_sz+1,2*seg_sz),(2*seg_sz+1,self.global_max)]
            cnt = [0]*3
            for n in ind:
                for i,(lo,hi) in enumerate(bins):
                    if lo<=n<=hi: cnt[i] += 1
            if max(cnt)<=3: sc += 5
            return sc
        pop = [gen() for _ in range(50)]
        for _ in range(20):
            pop = sorted(pop, key=fit, reverse=True)[:10]
            while len(pop)<50:
                p1,p2 = random.sample(pop[:20] if len(pop) >= 20 else pop,2)
                cp = random.randint(1,5)
                child = sorted(list(set(p1[:cp] + p2[cp:])))
                need = 6-len(child)
                if need>0:
                    available = [n for n in range(1,self.global_max+1) if n not in child]
                    if len(available) >= need:
                        child.extend(random.sample(available, need))
                elif need<0:
                    child = child[:6]
                if random.random()<0.1 and len(child) == 6:
                    idx = random.randint(0,5)
                    available = [n for n in range(1,self.global_max+1) if n not in child]
                    if available:
                        child[idx] = random.choice(available)
                        child.sort()
                pop.append(child)
        return pop[0]
    def _component_scores(self, num, ga_pred):
        freq = self.features['frequency'].get(num,0) * 2
        pos = sum(self.features['positional_freq'][p].get(num,0) for p in self.features['positional_freq']) * 1.5
        pair = sum(self.features['pairs'].get(p,0) for p in self.features['pairs'] if num in p and self.features['pairs'][p]>1)
        gap = self.features['gaps'].get(num, self.total_sets+1)
        prng = (self.features['frequency'].get(num,0) / (gap+1)) * 1.5 if gap>0 else 0
        ga = 5 if num in ga_pred else 0
        return freq, pos, pair, prng, ga
    def predict_next_set(self):
        scores = {n:0 for n in range(1,self.global_max+1)}
        for n,c in self.features['frequency'].items(): scores[n] += c*2
        for pos in self.features['positional_freq']:
            for n,c in self.features['positional_freq'][pos].items(): scores[n] += c*1.5
        for n,g in self.features['gaps'].items():
            if g>=10: scores[n] += 3
            elif g in (1,2): scores[n] += 1
        for (a,b),c in self.features['pairs'].items():
            if c>1:
                scores[a] += c
                scores[b] += c
        for n in range(1,self.global_max+1):
            f = self.features['frequency'].get(n,0)
            g = self.features['gaps'].get(n,self.total_sets+1)
            scores[n] += (f/(g+1))*1.5 if g>0 else 0
        ga_pred = self.genetic_algorithm_prediction()
        for n in ga_pred: scores[n] += 5
        last3 = set(n for s in self.sets_list[-3:] for n in s) if self.total_sets>=3 else set()
        cand = [n for n,_ in sorted(scores.items(), key=lambda x:x[1], reverse=True)[:12]]
        final = []
        rep = 0
        for n in cand:
            if n in last3:
                if rep<2:
                    final.append(n); rep+=1
            else:
                final.append(n)
            if len(final)==6: break
        if len(final)<6:
            for n in cand:
                if n not in final:
                    final.append(n)
                if len(final)==6: break
        final = sorted(final)
        seg_sz = self.global_max // 3
        bins = [(1,seg_sz),(seg_sz+1,2*seg_sz),(2*seg_sz+1,self.global_max)]
        cnt = [0]*3
        for n in final:
            for i,(lo,hi) in enumerate(bins):
                if lo<=n<=hi: cnt[i]+=1
        while max(cnt)>3 and len(final) == 6:
            over = cnt.index(max(cnt))
            lo,hi = bins[over]
            in_range = [n for n in final if lo<=n<=hi]
            if in_range:
                to_rem = random.choice(in_range)
                final.remove(to_rem)
                available = [n for n in range(1,self.global_max+1) if n not in final]
                if available:
                    new = random.choice(available)
                    final.append(new)
            cnt = [0]*3
            for n in final:
                for i,(lo,hi) in enumerate(bins):
                    if lo<=n<=hi: cnt[i]+=1
        final.sort()
        return final, scores, ga_pred
    def generate_report(self, pred, scores, ga_pred):
        lines = []
        lines.append("\n"+"-"*40)
        lines.append("ANTICIPATION COMPONENT BREAKDOWN")
        lines.append("-"*40)
        lines.append("num | Freq | Pos | Pair | PRNGx1.5 | GA | TOTAL")
        lines.append("-"*44)
        comp = [(n, *self._component_scores(n, ga_pred), sum(self._component_scores(n, ga_pred)))
                for n in pred]
        comp.sort(key=lambda x:x[-1], reverse=True)
        for n,f,p,pr,prng,ga,tot in comp:
            lines.append(f"{n:<2} | {f:<2} | {p:<2.0f} | {pr:<2} | {prng:<5.1f} {ga:<2} | {tot:.1f}")
        lines.append("\n✅ FINAL ANTICIPATION")
        lines.append(f"SET_{self.total_sets+1}: {pred}")
        return "\n".join(lines)
def calculate_set_stats(name, values):
    odd = sum(n for n in values if n%2)
    even = sum(n for n in values if not n%2)
    codd = len([n for n in values if n % 2 != 0])
    ceven = len([n for n in values if n % 2 == 0])
    a1 = values[0] + values[2] + values[4] if len(values) >= 5 else 0
    a2 = values[1] + values[3] + values[5] if len(values) >= 6 else 0
    total = odd + even
    return {"name":name, "values":values, "1st_alt":a1, "2nd_alt":a2,
            "odd":odd, "even":even, "codd":codd, "ceven":ceven, "total":total}
def print_set_analysis(lst):
    if not lst: return
    header = f"{'Set Name':<2} | {'Set Values':<3} | {'1st Alt':>2} | {'2nd Alt':>2} | {'Odd':>2} | {'Even':>2} | {'COdd':>2} | {'CEven':>2} | {'Total':>2}"
    sep = "="*len(header)
    print(sep); print(header); print(sep)
    for d in lst:
        vs = ", ".join(map(str,d['values']))
        print(f"{d['name']:<2} | {vs:<3} | {d['1st_alt']:>2} | {d['2nd_alt']:>2} | {d['odd']:>2} | {d['even']:>2} | {d['codd']:>2} | {d['ceven']:>2} | {d['total']:>2}")
    # Add predicted row instead of average
    if lst:
        pred_name = "Predicted"
        pred_values = ""
        time_steps = list(range(1, len(lst) + 1))
        stats = { '1st_alt': [d['1st_alt'] for d in lst], '2nd_alt': [d['2nd_alt'] for d in lst],
                  'odd': [d['odd'] for d in lst], 'even': [d['even'] for d in lst],
                  'codd': [d['codd'] for d in lst], 'ceven': [d['ceven'] for d in lst], 'total': [d['total'] for d in lst] }
        pred_stats = {}
        for key, vals in stats.items():
            reg = CustomLinearRegression()
            reg.fit(time_steps, vals)
            pred_stats[key] = round(reg.predict(len(lst) + 1))
        print(f"{pred_name:<2} | {pred_values:<3} | {pred_stats['1st_alt']:>2} | {pred_stats['2nd_alt']:>2} | {pred_stats['odd']:>2} | {pred_stats['even']:>2} | {pred_stats['codd']:>2} | {pred_stats['ceven']:>2} | {pred_stats['total']:>2}")
    print(sep)
analyzer_unified = LotteryAnalyzerUnified(historical_data)
analyzer_unified.calculate_features()
pred_set, final_scores, ga_pred = analyzer_unified.predict_next_set()
report = analyzer_unified.generate_report(pred_set, final_scores, ga_pred)
print("\n" + report)
processed = [calculate_set_stats(k,v) for k,v in historical_data.items()]
print("\n" + "="*30)
print("--- Detailed Set Analysis Report ---")
print("="*30)
print_set_analysis(processed)
# -------------------------------
# ENSEMBLE PREDICTOR (Combining all)
# -------------------------------
class EnsemblePredictor:
    def __init__(self, predictors):
        self.predictors = predictors
    def predict(self, top_n=6, method='majority_vote'):
        all_preds = [num for pred in self.predictors for num in pred if pred]
        if method == 'majority_vote':
            counts = Counter(all_preds)
            ensemble_set = sorted([num for num, _ in counts.most_common(top_n)])
        return ensemble_set
all_existing_preds = [
    pre_Deff_set, pre_EandO_set, prediction_AN, Bmost_frequent_numbers,
    prediction_1st, predictedPair, knn_prediction_HaC, predicted_combination_seg,
    ups_pred, pred_set, best_predicted_set, pre_set_Bsd_Fre, recommended_numbers, predicted_group_mean
]
ensemble = EnsemblePredictor(all_existing_preds)
ensemble_vote = ensemble.predict()
print("\n" + "-" * 40)
print("ENSEMBLE ANTICIPATED (Combining All Methods)")
print("-" * 40)
print(f"✅ G for 20 SET3️⃣⭐Ensemble Majority Vote: {ensemble_vote}")


# ============================================
# ADDITION 1: DUE ELEMENTS ANTICIPATOR
# ============================================
class DueElementAnticipator:
    def __init__(self, historical_data_dict, gap_threshold=5):
        self.data = historical_data_dict
        self.gap_threshold = gap_threshold
        self.all_elements = set(range(1, 59))
       
    def calculate_gaps(self):
        """Calculate how many sets since each element last appeared"""
        gaps = {}
        total_sets = len(self.data)
       
        for element in self.all_elements:
            last_seen = 0
            for idx, (_, elements) in enumerate(self.data.items(), 1):
                if element in elements:
                    last_seen = idx
           
            gaps[element] = total_sets - last_seen
       
        return gaps
   
    def anticipate_due_elements(self, top_n=6):
        """Anticipate elements that are 'due' based on gap analysis"""
        gaps = self.calculate_gaps()
       
        scores = {}
        for element, gap in gaps.items():
            if gap >= self.gap_threshold:
                scores[element] = gap * 2
            elif gap == 0:
                scores[element] = -5
            else:
                scores[element] = gap * 0.5
       
        sorted_due = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [element for element, _ in sorted_due[:top_n]]

due_anticipator = DueElementAnticipator(historical_data, gap_threshold=5)
anticipated_due = due_anticipator.anticipate_due_elements(top_n=6)
print(f"\n✅ DUE ELEMENTS ANTICIPATION: {anticipated_due}")

# ============================================
# ADDITION 2: ZERO-FREQUENCY ELEMENT ANTICIPATOR
# ============================================
class ZeroFrequencyAnticipator:
    def __init__(self, historical_data_dict, range_max=58):
        self.data = historical_data_dict
        self.range_max = range_max
       
    def find_never_drawn(self):
        """Find elements that have NEVER appeared in current window"""
        all_drawn = set()
        for elements in self.data.values():
            all_drawn.update(elements)
       
        all_possible = set(range(1, self.range_max + 1))
        never_drawn = sorted(all_possible - all_drawn)
        return never_drawn
   
    def anticipate_mixed_rare(self, n_never=2, n_rare=4):
        """Mix never-drawn with rarely-drawn elements"""
        never = self.find_never_drawn()
       
        freq = Counter()
        for elements in self.data.values():
            freq.update(elements)
       
        rare = [e for e, count in freq.items() if 1 <= count <= 2]
       
        anticipation = []
        anticipation.extend(random.sample(never, min(n_never, len(never))))
        anticipation.extend(random.sample(rare, min(n_rare, len(rare))))
       
        return sorted(anticipation[:6])

zero_freq_anticipator = ZeroFrequencyAnticipator(historical_data, range_max=58)
never_drawn = zero_freq_anticipator.find_never_drawn()
print(f"✅ NEVER DRAWN ELEMENTS: {never_drawn}")

anticipated_rare_mix = zero_freq_anticipator.anticipate_mixed_rare()
print(f"✅ RARE MIX ANTICIPATION: {anticipated_rare_mix}")

# ============================================
# ADDITION 3: BALANCED FREQUENCY ANTICIPATION
# ============================================
def balanced_frequency_anticipation(historical_data_dict):
    freq = Counter()
    for elements in historical_data_dict.values():
        freq.update(elements)
   
    never = [e for e in range(1, 59) if e not in freq]
    rare = [e for e, c in freq.items() if 1 <= c <= 2]
    medium = [e for e, c in freq.items() if 3 <= c <= 4]
    hot = [e for e, _ in freq.most_common(10)]
   
    anticipation = []
    anticipation.extend(random.sample(never, min(1, len(never))))
    anticipation.extend(random.sample(rare, min(2, len(rare))))
    anticipation.extend(random.sample(medium, min(2, len(medium))))
    anticipation.extend(random.sample(hot, min(1, len(hot))))
   
    return sorted(anticipation[:6])

anticipated_balanced = balanced_frequency_anticipation(historical_data)
print(f"✅ BALANCED ANTICIPATION: {anticipated_balanced}")

def improved_score_candidate(candidate, last_draw_list, hot_set, cold_set, never_drawn_set):
    """Improved scoring for element-based anticipation"""
    score = 0
   
    if len(set(candidate)) != len(candidate):
        return -1000
   
    hot_score = sum(1.5 for element in candidate if element in hot_set)
    never_bonus = sum(3 for element in candidate if element in never_drawn_set)
    cold_bonus = sum(1 for element in candidate if element in cold_set)
    repeat_penalty = sum(1 for element in candidate if element in last_draw_list)
   
    even = sum(1 for element in candidate if element % 2 == 0)
    even_odd_score = -abs(even - 3) * 0.5
   
    segments = set()
    for element in candidate:
        if element <= 9: segments.add(1)
        elif element <= 19: segments.add(2)
        elif element <= 29: segments.add(3)
        elif element <= 39: segments.add(4)
        elif element <= 49: segments.add(5)
        else: segments.add(6)
   
    segment_score = len(segments) * 1.5
   
    score = (
        hot_score
        + never_bonus
        + cold_bonus
        - repeat_penalty
        + even_odd_score
        + segment_score
    )
   
    return score

# -------------------------------
# SUMMARY OF ALL ANTICIPATIONS
# -------------------------------
print("\n" + "="*50)
print("SUMMARY OF ANTICIPATIONS")
print("="*50)
print(f"✅📝Reusable pre_Deff_set: {pre_Deff_set}")
print(f"✅📝Anticipated E & O => {pre_EandO_set}")
print(f"✅📝Anticipated SET_{next_set_number} Based on KNN logic => {prediction_AN}")
print(f"✅📝Ball Frequency Position => {Bmost_frequent_numbers}")
print(f"✅G for 60 SET3️⃣🎯Reuse 1st Anticipated: {prediction_1st}")
print(f"✅📝Anticipated Group Combination => {predictedPair}")
print(f"✅📝Anticipated elements : {ups_pred}")
print(f"✅G for 20/60 SET3️⃣🎯⭐FINAL ANTICIPATED SET_N: {pred_set}")
print(f"✅G for 60 SET3️⃣🎯Best Anticipated Set: {' '.join(map(str, best_predicted_set))}")
# Compute column_avg for summary
means = HistoricalDataset.column_means()
column_avg = [round(m) for m in means]
print(f"\n✅Average when checking from Column: {column_avg}")
# Export for another script (e.g., JSON)
all_predictions = {
    "pre_Deff_set": pre_Deff_set,
    "pre_EandO_set": pre_EandO_set,
    "prediction_AN": prediction_AN,
    "Bmost_frequent_numbers": Bmost_frequent_numbers,
    "prediction_1st": prediction_1st,
    "predictedPair": predictedPair,
    "knn_prediction_HaC": knn_prediction_HaC,
    "predicted_combination_seg": predicted_combination_seg,
    "ups_pred": ups_pred,
    "pred_set": pred_set,
    "best_predicted_set": best_predicted_set,
    "column_avg": column_avg
}
print("\n✅ Script completed successfully with all results!✅")
