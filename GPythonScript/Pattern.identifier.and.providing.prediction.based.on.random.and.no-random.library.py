from collections import defaultdict, Counter
import itertools

def discover_patterns(sets, min_occurrences=3, max_gap=6):
    from collections import defaultdict, Counter
    import itertools

    keys = list(sets.keys())
    number_positions = defaultdict(list)

    # Track index positions of each number
    for idx, key in enumerate(keys):
        for num in sets[key]:
            number_positions[num].append(idx)

    gap_patterns = defaultdict(list)
    frequency_patterns = {}
    pair_cooccurrences = Counter()
    full_gap_tracking = {}

    for num, positions in number_positions.items():
        if len(positions) >= 2:
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            full_gap_tracking[num] = gaps

            if len(positions) >= min_occurrences:
                # Count consistent gaps (e.g., every 2 sets, every 3 sets)
                gap_counts = Counter(gaps)
                most_common_gap, freq = gap_counts.most_common(1)[0]
                if freq >= min_occurrences - 1 and most_common_gap <= max_gap:
                    gap_patterns[most_common_gap].append((num, positions))

        frequency_patterns[num] = {
            'occurrences': len(positions),
            'first_seen': positions[0],
            'last_seen': positions[-1],
            'avg_gap': sum(gaps)/len(gaps) if len(positions) >= 2 else None
        }

    # Detect frequently co-occurring pairs
    all_sets = list(sets.values())
    for draw in all_sets:
        for pair in itertools.combinations(sorted(draw), 2):
            pair_cooccurrences[pair] += 1

    frequent_pairs = {pair: count for pair, count in pair_cooccurrences.items() if count >= min_occurrences}

    return {
        'gap_patterns': gap_patterns,
        'frequency_patterns': frequency_patterns,
        'frequent_pairs': frequent_pairs,
        'full_gap_tracking': full_gap_tracking  # 👈 now included
    }

# === Optional: Pretty print the results ===
def print_patterns_summary(patterns):
    print("== Gap-Based Patterns ==")
    for gap, nums in patterns['gap_patterns'].items():
        print(f"Gap of {gap}:")
        for num, indices in nums:
            print(f"  - Number {num} at positions {indices}")

    print("\n== Frequent Number Stats ==")
    for num, stats in patterns['frequency_patterns'].items():
        print(f"Number {num}:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    print("\n== Full Gap Tracking ==")
    for num, gaps in patterns['full_gap_tracking'].items():
        print(f"Number {num} gaps: {gaps}")

    print("\n== Frequent Pairs ==")
    for pair, count in patterns['frequent_pairs'].items():
        print(f"Pair {pair}: {count} times")


# === Your Data Sets ===
sets = {
    ### G55 ###
"SET_1": [10, 20, 22, 24, 28, 41],
"SET_2": [5, 9, 14, 27, 31, 55],
"SET_3": [5, 7, 28, 32, 40, 53],
"SET_4": [18, 20, 34, 35, 45, 49],
"SET_5": [36, 40, 42, 47, 51, 54],
"SET_6": [4, 24, 33, 40, 52, 55],
"SET_7": [5, 12, 37, 47, 48, 50],
"SET_8": [13, 16, 17, 26, 28, 47],
"SET_9": [12, 21, 23, 39, 46, 52],
"SET_10": [2, 21, 27, 28, 38, 44],
}

# === Run Analysis ===
patterns = discover_patterns(sets)
print_patterns_summary(patterns)


def discover_patterns(sets, min_occurrences=3, max_gap=6):
    from collections import defaultdict, Counter
    import itertools
    import statistics

    keys = list(sets.keys())
    number_positions = defaultdict(list)

    all_numbers = []
    set_statistics = []
    reused_counts = []

    prev_set = set()

    for idx, key in enumerate(keys):
        current_set = sets[key]
        current_stats = {
            'set': key,
            'min': min(current_set),
            'max': max(current_set),
            'average': sum(current_set) / len(current_set),
            'median': statistics.median(current_set),
            'odd_count': sum(1 for n in current_set if n % 2 == 1),
            'even_count': sum(1 for n in current_set if n % 2 == 0),
            'high_count': sum(1 for n in current_set if n > 30),
            'low_count': sum(1 for n in current_set if n <= 30),
            'repeats_from_prev': len(set(current_set) & prev_set)
        }
        set_statistics.append(current_stats)
        reused_counts.append(current_stats['repeats_from_prev'])
        all_numbers.extend(current_set)
        prev_set = set(current_set)

        for num in current_set:
            number_positions[num].append(idx)

    # Continue with previous logic
    gap_patterns = defaultdict(list)
    frequency_patterns = {}
    pair_cooccurrences = Counter()
    full_gap_tracking = {}

    for num, positions in number_positions.items():
        if len(positions) >= 2:
            gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            full_gap_tracking[num] = gaps

            if len(positions) >= min_occurrences:
                gap_counts = Counter(gaps)
                most_common_gap, freq = gap_counts.most_common(1)[0]
                if freq >= min_occurrences - 1 and most_common_gap <= max_gap:
                    gap_patterns[most_common_gap].append((num, positions))

        frequency_patterns[num] = {
            'occurrences': len(positions),
            'first_seen': positions[0],
            'last_seen': positions[-1],
            'avg_gap': sum(gaps)/len(gaps) if len(positions) >= 2 else None
        }

    for draw in sets.values():
        for pair in itertools.combinations(sorted(draw), 2):
            pair_cooccurrences[pair] += 1

    frequent_pairs = {pair: count for pair, count in pair_cooccurrences.items() if count >= min_occurrences}

    overall_stats = {
        'total_draws': len(sets),
        'unique_numbers': len(set(all_numbers)),
        'most_common_numbers': Counter(all_numbers).most_common(10),
        'average_repeats_from_previous': sum(reused_counts) / len(reused_counts)
    }

    return {
        'gap_patterns': gap_patterns,
        'frequency_patterns': frequency_patterns,
        'frequent_pairs': frequent_pairs,
        'full_gap_tracking': full_gap_tracking,
        'set_statistics': set_statistics,
        'overall_stats': overall_stats
    }

patterns = discover_patterns(sets)

print("=== Overall Statistics ===")
for k, v in patterns['overall_stats'].items():
    print(f"{k}: {v}")

print("\n=== Set-wise Stats ===")
for s in patterns['set_statistics']:
    print(s)


#===== PREDICTING STAGE ======
import random

def predict_next_set(patterns, num_to_pick=6):
    # Extract useful data
    freq_stats = patterns['frequency_patterns']
    gap_patterns = patterns['gap_patterns']
    overall_stats = patterns['overall_stats']
    recent_draw = patterns['set_statistics'][-1]
    frequent_pairs = patterns['frequent_pairs']

    # 1. Get top frequent numbers
    top_freq_numbers = sorted(freq_stats.items(), key=lambda x: x[1]['occurrences'], reverse=True)
    top_numbers = [num for num, _ in top_freq_numbers[:20]]

    # 2. Include numbers from stable gap patterns
    gap_candidates = []
    for nums in gap_patterns.values():
        for num, _ in nums:
            gap_candidates.append(num)

    # 3. Look at recent numbers (but not too recent to avoid exact repetition)
    recent_numbers = set()
    if recent_draw:
        recent_numbers = {n for n in freq_stats if freq_stats[n]['last_seen'] == recent_draw['set'][-1:]}

    # 4. Combine all sources and rank
    candidate_pool = list(set(top_numbers + gap_candidates))
    if recent_numbers:
        candidate_pool = [n for n in candidate_pool if n not in recent_numbers]

    if len(candidate_pool) < num_to_pick:
        # Fallback to most common
        candidate_pool = [num for num, _ in overall_stats['most_common_numbers']]

    # 5. Randomly choose from candidate pool
    prediction = random.sample(candidate_pool, num_to_pick)

    # 6. Optionally add one or two frequent pairs if available
    sorted_pairs = sorted(frequent_pairs.items(), key=lambda x: x[1], reverse=True)
    for (n1, n2), _ in sorted_pairs[:5]:
        if n1 in prediction and n2 not in prediction and len(prediction) < num_to_pick:
            prediction.append(n2)
        elif n2 in prediction and n1 not in prediction and len(prediction) < num_to_pick:
            prediction.append(n1)

    return sorted(prediction[:num_to_pick])

predicted_set = predict_next_set(patterns)
print("\n=== Predicted Next Set ===")
print(predicted_set)


#===== Predicting without random module or library =====
def predict_deterministic_next_set_with_patterns(patterns, sets, num_to_pick=6):
    freq_stats = patterns['frequency_patterns']
    full_gaps = patterns['full_gap_tracking']
    gap_patterns = patterns['gap_patterns']
    frequent_pairs = patterns['frequent_pairs']
    set_stats = patterns['set_statistics']

    # === 1. HOT NUMBERS (Top frequent)
    hot_numbers = sorted(freq_stats.items(), key=lambda x: x[1]['occurrences'], reverse=True)
    hot = [n for n, _ in hot_numbers[:15]]

    # === 2. STABLE GAPS
    stable_gap_nums = set()
    for gap, nums in gap_patterns.items():
        for n, _ in nums:
            stable_gap_nums.add(n)

    # === 3. RECENT NUMBERS (Last 3 sets)
    recent_sets = set()
    for s in set_stats[-3:]:
        recent_sets.update(sets[s['set']])

    # === 4. FREQUENT PAIRS: get individual numbers
    pair_counts = Counter()
    for (n1, n2), count in frequent_pairs.items():
        pair_counts[n1] += count
        pair_counts[n2] += count
    top_pair_members = [n for n, _ in pair_counts.most_common(20)]

    # === 5. COLD NUMBERS: numbers never seen
    all_drawn = set(freq_stats.keys())
    all_possible = set(range(1, 60))  # Adjust as needed
    cold_numbers = all_possible - all_drawn

    # === Combine Sources (Prioritized)
    candidates = []
    selected_sources = defaultdict(list)  # Track source per number

    for n in hot:
        reasons = []
        if n in stable_gap_nums:
            reasons.append('stable_gap')
        if n in recent_sets:
            reasons.append('recent')
        if n in top_pair_members:
            reasons.append('pair_linked')

        if reasons:
            candidates.append(n)
            selected_sources[n].extend(reasons)
        if len(candidates) >= num_to_pick:
            break

    # Fill remaining if needed
    if len(candidates) < num_to_pick:
        backup = list((set(hot + top_pair_members + list(stable_gap_nums)) - cold_numbers) - set(candidates))
        backup_sorted = sorted(backup, key=lambda x: freq_stats.get(x, {}).get('occurrences', 0), reverse=True)
        for n in backup_sorted:
            if n not in candidates and len(candidates) < num_to_pick:
                candidates.append(n)
                selected_sources[n].append('backup_frequent')

    # Sort the final pick
    prediction = sorted(candidates[:num_to_pick])

    # === Print Explanation ===
    print("\n=== Data-Driven Prediction (No Randomness) ===")
    print("Predicted numbers:", prediction)
    print("\nPattern Justification per Number:")
    for n in prediction:
        reasons = selected_sources.get(n, ['unknown'])
        print(f"  {n}: {', '.join(reasons)}")

    return prediction

predicted = predict_deterministic_next_set_with_patterns(patterns, sets)

#===== Predicting by accuracy, speed, and interpretability =====
 
from collections import defaultdict, Counter
import itertools

# === Data use is on the main script ===

# === Pattern Discovery (Frequency, Recency, Co-occurrence) ===
number_positions = defaultdict(list)
pair_counter = Counter()
all_numbers = []

keys = list(sets.keys())

for idx, key in enumerate(keys):
    for num in sets[key]:
        number_positions[num].append(idx)
    for pair in itertools.combinations(sorted(sets[key]), 2):
        pair_counter[pair] += 1
    all_numbers.extend(sets[key])

frequency = Counter(all_numbers)
recency_score = {num: len(keys) - max(positions) for num, positions in number_positions.items()}
avg_gap = {}
for num, positions in number_positions.items():
    if len(positions) > 1:
        gaps = [positions[i+1] - positions[i] for i in range(len(positions) - 1)]
        avg_gap[num] = sum(gaps) / len(gaps)
    else:
        avg_gap[num] = len(keys)  # Treat as long-unseen

# === Combined Scoring ===
score_board = {}
for num in set(all_numbers):
    score = (
        frequency[num] * 2 +         # Frequency weight
        recency_score[num] * 1.5 -   # Recency reward
        avg_gap.get(num, 0) * 1.0    # Penalize large gaps
    )
    score_board[num] = score

# === Select Top-Scored Numbers ===
sorted_candidates = sorted(score_board.items(), key=lambda x: x[1], reverse=True)
predicted_numbers = [num for num, score in sorted_candidates[:6]]

# === Display Prediction ===
print("\n🎯 Predicted Set Based on Data-Driven Patterns:")
print("Predicted Numbers:", sorted(predicted_numbers))

print("\n📊 Top Scores Breakdown:")
for num, score in sorted_candidates[:10]:
    print(f"Number {num}: Score = {score:.2f}, Freq = {frequency[num]}, Recency = {recency_score[num]}, AvgGap = {avg_gap.get(num):.2f}")
