from collections import defaultdict, Counter

# Rule-Based Analysis
def analyze_sets(sets):
    occurrences = defaultdict(list)
    patterns = []
    frequency = Counter()
    
    for idx, (set_name, s) in enumerate(sets.items()):
        for num in s:
            occurrences[num].append(idx)
            frequency[num] += 1

    for num, indices in occurrences.items():
        if len(indices) >= 2:
            for i in range(2, len(indices)):
                if indices[i] - indices[i - 1] > 1 and indices[i - 1] - indices[i - 2] > 1:
                    patterns.append(f"Pattern 2: Num# {num} reappears after 2 absences @ pos {indices[i-2]}, {indices[i-1]}, {indices[i]}")
                if indices[i] - indices[i - 1] == 1 and indices[i - 1] - indices[i - 2] == 2:
                    patterns.append(f"Pattern 3: Num# {num} reappears after skipping one set & then appears consecutively @ pos {indices[i-2]}, {indices[i-1]}, {indices[i]}")
        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] == 1:
                patterns.append(f"Pattern 4: Num# {num} appears consecutively @ pos {indices[i-1]}, {indices[i]}")

    predicted_set = predict_next_set(frequency, occurrences, sets)
    return patterns, predicted_set

def predict_next_set(frequency, occurrences, sets, num_to_predict=6, apply_rules=None):
    if apply_rules is None:
        apply_rules = {}
    pattern_candidates = get_pattern_based_candidates(sets, apply_rules)
    most_frequent = [num for num, _ in frequency.most_common(20)]
    candidate_numbers = set(most_frequent)

    for num, indices in occurrences.items():
        if len(indices) > 2:
            for i in range(2, len(indices)):
                if indices[i] - indices[i - 1] > 1 and indices[i - 1] - indices[i - 2] > 1:
                    candidate_numbers.add(num)
                if indices[i] - indices[i - 1] == 1 and indices[i - 1] - indices[i - 2] == 2:
                    candidate_numbers.add(num)

    recent_set = sets[list(sets.keys())[-1]]
    candidate_numbers.update(recent_set)
    candidate_numbers |= pattern_candidates  # merge patterns
    
    candidate_numbers = list(candidate_numbers)
    if len(candidate_numbers) < num_to_predict:
        candidate_numbers += [num for num, _ in frequency.most_common(num_to_predict - len(candidate_numbers))]

    predicted = sorted(candidate_numbers, key=lambda x: frequency[x], reverse=True)[:num_to_predict]
    return predicted


# Frequency-Based Prediction from Last 3 Sets
def predict_from_last_3_sets(sets, num_to_predict3=6):
    frequency = Counter()
    keys = list(sets.keys())
    last_3_sets = [sets[k] for k in keys[-3:]]

    for nums3 in last_3_sets:
        frequency.update(nums3)

    return [num for num, _ in frequency.most_common(num_to_predict3)]


# Frequency-Based Prediction from Last 5 Sets
def predict_from_last_5_sets(sets, num_to_predict5=6):
    frequency = Counter()
    keys = list(sets.keys())
    last_5_sets = [sets[k] for k in keys[-5:]]

    for nums5 in last_5_sets:
        frequency.update(nums5)

    return [num for num, _ in frequency.most_common(num_to_predict5)]


# Custom KNN Prediction Logic
def set_to_vector(numbers, all_numbers):
    return [1 if n in numbers else 0 for n in all_numbers]

def vector_distance(v1, v2):
    return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

def create_dataset(sets, all_numbers, window=3):
    keys = list(sets.keys())
    X, y = [], []
    for i in range(len(keys) - window):
        past = [num for j in range(window) for num in sets[keys[i + j]]]
        input_vec = set_to_vector(past, all_numbers)
        output_vec = set_to_vector(sets[keys[i + window]], all_numbers)
        X.append(input_vec)
        y.append(output_vec)
    return X, y

def knn_predict_next_set_custom(sets, all_numbers, window=3, k=3):
    X, y = create_dataset(sets, all_numbers, window)
    if not X:
        return []
    keys = list(sets.keys())
    recent = [num for i in range(window) for num in sets[keys[-window + i]]]
    test_vec = set_to_vector(recent, all_numbers)
    distances = [(vector_distance(test_vec, vec), i) for i, vec in enumerate(X)]
    distances.sort()
    neighbors = [y[i] for _, i in distances[:k]]
    avg_vec = [sum(vec[i] for vec in neighbors) / k for i in range(len(all_numbers))]
    top_indices = sorted(range(len(avg_vec)), key=lambda i: avg_vec[i], reverse=True)[:6]
    return [all_numbers[i] for i in top_indices]

# === Pattern Matching Function Definitions ===

def filter_fourth_set_excludes_prev_three(sets, window=5):
    keys = list(sets.keys())
    valid_numbers = set()
    for i in range(len(keys) - window + 1):
        s1, s2, s3, s4 = sets[keys[i]], sets[keys[i+1]], sets[keys[i+2]], sets[keys[i+3]]
        previous = set(s1 + s2 + s3)
        fourth = set(s4)
        non_repeats = fourth - previous
        valid_numbers.update(non_repeats)
    return valid_numbers

def find_numbers_reappearing_after_2_absences(sets):
    keys = list(sets.keys())
    history = [sets[k] for k in keys]
    reappeared = set()
    for i in range(2, len(history) - 1):
        for num in history[i - 2]:
            if num not in history[i - 1] and num not in history[i] and num in history[i + 1]:
                reappeared.add(num)
    return reappeared

def pattern_skips_2_then_consecutive(sets):
    keys = list(sets.keys())
    history = [sets[k] for k in keys]
    matched = set()
    for i in range(2, len(history) - 2):
        for num in history[i - 2]:
            if num not in history[i - 1] and num not in history[i] and num in history[i + 1] and num in history[i + 2]:
                matched.add(num)
    return matched

def pattern_skips_1_then_consecutive(sets):
    keys = list(sets.keys())
    history = [sets[k] for k in keys]
    matched = set()
    for i in range(1, len(history) - 2):
        for num in history[i - 1]:
            if num not in history[i] and num in history[i + 1] and num in history[i + 2]:
                matched.add(num)
    return matched

def pattern_two_then_skip_then_return(sets):
    keys = list(sets.keys())
    history = [sets[k] for k in keys]
    matched = set()
    for i in range(3, len(history)):
        for num in history[i - 3]:
            if num in history[i - 2] and num not in history[i - 1] and num in history[i]:
                matched.add(num)
    return matched

def pattern_skip_one_appearance(sets):
    keys = list(sets.keys())
    history = [sets[k] for k in keys]
    matched = set()
    for i in range(2, len(history)):
        for num in history[i - 2]:
            if num not in history[i - 1] and num in history[i]:
                matched.add(num)
    return matched

def pattern_two_and_three_occurrences(sets):
    keys = list(sets.keys())
    history = [sets[k] for k in keys]
    matched = set()
    for i in range(len(history) - 4):
        window_sets = history[i:i + 5]
        all_nums = [n for s in window_sets for n in s]
        counts = Counter(all_nums)
        twos = {n for n, c in counts.items() if c == 2}
        threes = {n for n, c in counts.items() if c == 3}
        if twos and threes:
            matched.update(twos)
            matched.update(threes)
    return matched

def pattern_repeat_after_skip_consistent(sets):
    keys = list(sets.keys())
    history = [sets[k] for k in keys]
    repeated = defaultdict(list)
    for i in range(2, len(history)):
        for num in history[i - 2]:
            if num not in history[i - 1] and num in history[i]:
                repeated[num].append(i)
    result = {num for num, indices in repeated.items() if len(indices) >= 3}
    return result

# Pattern Matching Functions

def get_pattern_based_candidates(sets, apply_rules):
    candidates = set()
    if apply_rules.get("rule1"): candidates |= filter_fourth_set_excludes_prev_three(sets)
    if apply_rules.get("rule2"): candidates |= find_numbers_reappearing_after_2_absences(sets)
    if apply_rules.get("rule3"): candidates |= pattern_skips_2_then_consecutive(sets)
    if apply_rules.get("rule4"): candidates |= pattern_skips_1_then_consecutive(sets)
    if apply_rules.get("rule5"): candidates |= pattern_two_then_skip_then_return(sets)
    if apply_rules.get("rule6"): candidates |= pattern_skip_one_appearance(sets)
    if apply_rules.get("rule7"): candidates |= pattern_two_and_three_occurrences(sets)
    if apply_rules.get("rule8"): candidates |= pattern_repeat_after_skip_consistent(sets)
    return candidates


# Apply Pattern Rules to Predict the Next Set
sets = {
    "SET_1": [1, 13, 17, 20, 34, 46],
"SET_2": [1, 13, 25, 28, 42, 50],
"SET_3": [14, 19, 21, 27, 51, 53],
"SET_4": [28, 31, 42, 44, 45, 54],
"SET_5": [6, 15, 17, 26, 37, 40],
"SET_6": [1, 2, 12, 31, 39, 55],
"SET_7": [4, 22, 23, 24, 38, 46],
"SET_8": [5, 27, 28, 35, 39, 52],
"SET_9": [12, 15, 18, 23, 50, 51],
"SET_10": [2, 22, 26, 39, 42, 44],
"SET_11": [8, 12, 18, 22, 29, 49],
"SET_12": [14, 25, 30, 35, 39, 42],
"SET_13": [3, 13, 19, 41, 48, 49],
"SET_14": [5, 7, 12, 27, 47, 50],
"SET_15": [5, 11, 21, 41, 42, 45],
"SET_16": [2, 20, 30, 48, 49, 51],
"SET_17": [3, 7, 21, 32, 37, 42],
"SET_18": [7, 24, 36, 39, 48, 53],
"SET_19": [5, 10, 12, 13, 16, 36],
"SET_20": [7, 11, 14, 21, 30, 52],
}

selected_rules = {
    "rule1": True,
    "rule2": True,
    "rule3": False,
    "rule4": True,
    "rule5": True,
    "rule6": False,
    "rule7": True,
    "rule8": True,
}

# Step 1: Get Pattern-Based Predictions
patterns, rule_based_pred = analyze_sets(sets)

# Step 2: Predict the next set using patterns
predicted_with_patterns = predict_next_set(Counter([num for s in sets.values() for num in s]),
                                           defaultdict(list), sets, apply_rules=selected_rules)

# Step 3: Predictions Based on Last 3 and Last 5 Sets
predicted_3set = predict_from_last_3_sets(sets)
predicted_5set = predict_from_last_5_sets(sets)

# Step 4: KNN Predictions
all_numbers = sorted(set(num for s in sets.values() for num in s))
knn_3set = knn_predict_next_set_custom(sets, all_numbers, window=3)
knn_5set = knn_predict_next_set_custom(sets, all_numbers, window=5)

# Print Results
print("Detected Patterns:")
for p in patterns:
    print("-", p)

print("\nPredicted Set (Rule-Based):", rule_based_pred)
print("Predicted Set Based on Last 3 Sets:", predicted_3set)
print("Predicted Set Based on Last 5 Sets:", predicted_5set)
print("Predicted Set (KNN - 3 Sets):", knn_3set)
print("Predicted Set (KNN - 5 Sets):", knn_5set)
print("\nPattern-Based Enhanced Prediction:", predicted_with_patterns)
