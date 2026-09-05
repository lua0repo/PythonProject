print('##### Game XX -> 20 SET #####')

# Historical data
historical_sets = {
    "SET_1": [8, 12, 28, 32, 38, 50],
    "SET_2": [3, 11, 26, 39, 40, 53],
    "SET_3": [11, 36, 41, 43, 44, 45],
    "SET_4": [6, 19, 20, 24, 33, 42],
    "SET_5": [10, 15, 17, 25, 27, 41],
    "SET_6": [2, 5, 38, 41, 42, 49],
    "SET_7": [2, 14, 25, 38, 50, 54],
}

def derive_set(numbers, mid_index=None):
    """Derive a new set based on the 6-step logic."""
    lowest = min(numbers)

    if mid_index is not None:
        mid = numbers[mid_index]
    else:
        sorted_nums = sorted(numbers)
        mid = sorted_nums[2]

    highest_plus_one = max(numbers) + 1
    half_high = highest_plus_one / 2
    half_again_plus_low = (half_high / 2) + lowest

    half_again_plus_low_r = round(half_again_plus_low)
    sum_smallest3 = round(lowest + mid + half_again_plus_low)

    return [lowest, mid, highest_plus_one, half_high,
            half_again_plus_low_r, sum_smallest3]

def count_hits_seq(derived, next_historical):
    """Count matches between derived set and NEXT historical set."""
    derived_ints = {int(x) for x in derived}
    next_hist_set = set(next_historical)
    return len(derived_ints & next_hist_set)


# Process sequential comparison
keys = list(historical_sets.keys())
final_results = {}

for i, key in enumerate(keys):
    derived = derive_set(historical_sets[key])

    # Determine next set (sequential)
    if i < len(keys) - 1:
        next_key = keys[i + 1]
        next_hist = historical_sets[next_key]
    else:
        next_key = None
        next_hist = []

    hit_count = count_hits_seq(derived, next_hist)

    final_results[key] = {
        "derived": derived,
        "next_compare": next_key,
        "hit_count": hit_count
    }


# Output
for key, data in final_results.items():
    derived = data["derived"]
    next_key = data["next_compare"]
    hits = data["hit_count"]

    if next_key:
        print(f"{key}: {derived}  -> Compared w/ {next_key} | H Cnt: {hits}")
    else:
        print(f"{key}: {derived}  -> No next set to compare")
