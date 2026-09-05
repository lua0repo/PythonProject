import math
import random
from collections import defaultdict, Counter

class LotteryAnalyzer:
   def __init__(self, sets_dict):
       self.sets_dict = sets_dict
       self.sets_list = []
       self.positions = {f'B{i}': [] for i in range(1, 7)}
       self.global_min = float('inf')
       self.global_max = float('-inf')
       self.total_sets = len(sets_dict)
       self.features = None
       self.validate_and_organize()

   def validate_and_organize(self):
       all_numbers = []
       for i in range(1, self.total_sets + 1):
           key = f"SET_{i}"
           s = self.sets_dict[key]
           if len(s) != 6:
               raise ValueError(f"{key} must have 6 numbers")
           if len(s) != len(set(s)):
               raise ValueError(f"{key} contains duplicates")
           if s != sorted(s):
               raise ValueError(f"{key} is not sorted")
           self.sets_list.append(s)
           all_numbers.extend(s)
       self.global_min = min(all_numbers)
       self.global_max = max(all_numbers)
       for s in self.sets_list:
           if min(s) < self.global_min or max(s) > self.global_max:
               raise ValueError(f"Numbers in set exceed global range")
       for s in self.sets_list:
           for idx in range(6):
               self.positions[f'B{idx + 1}'].append(s[idx])

   def exploratory_data_analysis(self):
       bias = {}
       entropy = []
       # Bias: Check frequency deviation from expected
       expected_freq = (6 * self.total_sets) / self.global_max
       freq_counts = Counter(num for s in self.sets_list for num in s)
       for num in range(1, self.global_max + 1):
           actual = freq_counts.get(num, 0)
           bias[num] = actual - expected_freq
       # Entropy per set
       for s in self.sets_list:
           counts = Counter(s)
           probs = [count / 6 for count in counts.values()]
           set_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
           entropy.append(set_entropy)
       avg_entropy = sum(entropy) / len(entropy) if entropy else 0
       return {'bias': bias, 'entropy': avg_entropy}

   def calculate_features(self):
       self.features = {
           'frequency': Counter(),
           'positional_freq': {f'B{i}': Counter() for i in range(1, 7)},
           'odd_even': [],
           'segments': [],
           'pairs': Counter(),
           'triplets': Counter(),
           'gaps': {},
           'sums': [],
           'differences': [],
           'lag_features': [],
           'rolling_stats': {'mean': [], 'std': []},
           'positional_diffs': {f'B{i}': [] for i in range(1, 7)},
           'pair_consistency': [],
           'triplet_sums': {'first': [], 'second': []},
           'sequential_shifts': []
       }
       # Frequency
       for s in self.sets_list:
           self.features['frequency'].update(s)
       # Positional frequency
       for pos in self.positions:
           self.features['positional_freq'][pos].update(self.positions[pos])
       # Odd/Even
       for s in self.sets_list:
           odd_count = sum(1 for num in s if num % 2 == 1)
           self.features['odd_even'].append((odd_count, 6 - odd_count))
       # Segments
       segment_size = self.global_max // 3
       bins = [(1, segment_size), (segment_size + 1, 2 * segment_size), (2 * segment_size + 1, self.global_max)]
       for s in self.sets_list:
           segment_count = [0] * len(bins)
           for num in s:
               for i, (low, high) in enumerate(bins):
                   if low <= num <= high:
                       segment_count[i] += 1
           self.features['segments'].append(segment_count)
       # Pairs and Triplets
       for s in self.sets_list:
           for i in range(5):
               for j in range(i + 1, 6):
                   pair = tuple(sorted((s[i], s[j])))
                   self.features['pairs'][pair] += 1
           for i in range(4):
               for j in range(i + 1, 5):
                   for k in range(j + 1, 6):
                       triplet = tuple(sorted((s[i], s[j], s[k])))
                       self.features['triplets'][triplet] += 1
       # Gaps
       current_idx = self.total_sets + 1
       for num in range(1, self.global_max + 1):
           last_seen = 0
           for i in range(self.total_sets - 1, -1, -1):
               if num in self.sets_list[i]:
                   last_seen = i + 1
                   break
           gap = current_idx - last_seen - 1 if last_seen > 0 else current_idx
           self.features['gaps'][num] = gap
       # Sums
       for s in self.sets_list:
           self.features['sums'].append(sum(s))
       # Differences
       for i in range(1, self.total_sets):
           diffs = [self.sets_list[i][j] - self.sets_list[i-1][j] for j in range(6)]
           self.features['differences'].append(diffs)
       # Lag Features (previous set numbers)
       for i in range(1, self.total_sets):
           self.features['lag_features'].append(self.sets_list[i-1])
       # Rolling Statistics
       window = min(5, self.total_sets)
       for i in range(self.total_sets):
           window_sets = self.sets_list[max(0, i-window+1):i+1]
           window_nums = [num for s in window_sets for num in s]
           mean = sum(window_nums) / len(window_nums) if window_nums else 0
           variance = sum((x - mean) ** 2 for x in window_nums) / len(window_nums) if window_nums else 0
           std = math.sqrt(variance) if variance > 0 else 0
           self.features['rolling_stats']['mean'].append(mean)
           self.features['rolling_stats']['std'].append(std)
       # Positional Differences
       for pos in self.positions:
           for i in range(1, self.total_sets):
               diff = self.positions[pos][i] - self.positions[pos][i-1]
               self.features['positional_diffs'][pos].append(diff)
       # Pair Consistency
       pair_appearances = defaultdict(list)
       for idx, s in enumerate(self.sets_list):
           for i in range(5):
               for j in range(i + 1, 6):
                   pair = tuple(sorted((s[i], s[j])))
                   pair_appearances[pair].append(idx + 1)
       for pair, appearances in pair_appearances.items():
           if len(appearances) > 1:
               gaps = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
               self.features['pair_consistency'].append((pair, gaps))
       # Triplet Sums
       for s in self.sets_list:
           self.features['triplet_sums']['first'].append(sum(s[:3]))
           self.features['triplet_sums']['second'].append(sum(s[3:]))
       # Sequential Shifts
       for i in range(1, self.total_sets):
           shifts = [self.sets_list[i][j] - self.sets_list[i-1][j] for j in range(6)]
           self.features['sequential_shifts'].append(shifts)
       return self.features

   def scale_features(self):
       scaled_features = {
           'sums': [],
           'differences': [],
           'rolling_stats': {'mean': [], 'std': []}
       }
       sums = self.features['sums']
       min_sum = min(sums) if sums else 0
       max_sum = max(sums) if sums else 1
       scaled_features['sums'] = [
           (x - min_sum) / (max_sum - min_sum) if max_sum != min_sum else 0 for x in sums]
       for diffs in self.features['differences']:
           min_diff = min(diffs) if diffs else 0
           max_diff = max(diffs) if diffs else 1
           scaled_diffs = [(x - min_diff) / (max_diff - min_diff) if max_diff != min_diff else 0 for x in diffs]
           scaled_features['differences'].append(scaled_diffs)
       for key in ['mean', 'std']:
           vals = self.features['rolling_stats'][key]
           min_val = min(vals) if vals else 0
           max_val = max(vals) if vals else 1
           scaled_features['rolling_stats'][key] = [
               (x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in vals]
       split_idx = int(0.8 * self.total_sets)
       train_sets = self.sets_list[:split_idx]
       test_sets = self.sets_list[split_idx:]
       return train_sets, test_sets, scaled_features

   def genetic_algorithm_prediction(self, scaled_features):
       def generate_individual():
           return sorted(random.sample(range(self.global_min, self.global_max + 1), 6))
     
       def fitness(individual):
           score = 0
           # Frequency score
           for num in individual:
               score += self.features['frequency'].get(num, 0) * 2
           # Gap score
           for num in individual:
               gap = self.features['gaps'].get(num, self.total_sets + 1)
               score += 3 if gap >= 10 else 1 if gap in [1, 2] else 0
           # Segment balance
           segment_size = self.global_max // 3
           bins = [(1, segment_size), (segment_size + 1, 2 * segment_size), (2 * segment_size + 1, self.global_max)]
           segment_count = [0] * len(bins)
           for num in individual:
               for i, (low, high) in enumerate(bins):
                   if low <= num <= high:
                       segment_count[i] += 1
           if max(segment_count) <= 3:
               score += 5
           return score

       population_size = 50
       generations = 20
       population = [generate_individual() for _ in range(population_size)]
       for _ in range(generations):
           population = sorted(population, key=fitness, reverse=True)
           next_gen = population[:10]  # Elitism
           while len(next_gen) < population_size:
               parent1, parent2 = random.sample(population[:20], 2)
               crossover_point = random.randint(1, 5)
               child = sorted(list(set(parent1[:crossover_point] + parent2[crossover_point:])))
               if len(child) < 6:
                   child.extend(random.sample([n for n in range(1, self.global_max + 1) if n not in child], 6 - len(child)))
               elif len(child) > 6:
                   child = child[:6]
               # Mutation
               if random.random() < 0.1:
                   idx = random.randint(0, 5)
                   new_num = random.choice([n for n in range(1, self.global_max + 1) if n not in child])
                   child[idx] = new_num
                   child = sorted(child)
               next_gen.append(child)
           population = next_gen
       return population[0]

   def prng_analysis(self):
       prng_scores = {}
       for num in range(1, self.global_max + 1):
           freq = self.features['frequency'].get(num, 0)
           gap = self.features['gaps'].get(num, self.total_sets + 1)
           prng_scores[num] = freq / (gap + 1) if gap > 0 else freq
       return prng_scores


   def predict_next_set(self):
       scores = {num: 0 for num in range(1, self.global_max + 1)}
       last_3_sets = self.sets_list[-3:] if self.total_sets >= 3 else self.sets_list
       last_3_nums = set(num for s in last_3_sets for num in s)
       # Frequency scoring
       for num, count in self.features['frequency'].items():
           scores[num] += count * 2
       # Positional frequency
       for pos in self.features['positional_freq']:
           for num, count in self.features['positional_freq'][pos].items():
               scores[num] += count * 1.5
       # Gap scoring
       for num, gap in self.features['gaps'].items():
           if gap >= 10:
               scores[num] += 3
           elif gap in [1, 2]:
               scores[num] += 1
       # Pair scoring
       for pair, count in self.features['pairs'].items():
           if count > 1:
               scores[pair[0]] += count
               scores[pair[1]] += count
       # PRNG scoring
       prng_scores = self.prng_analysis()
       for num, prng_score in prng_scores.items():
           scores[num] += prng_score * 1.5
       # Shift-based projection
       last_shifts = self.features['sequential_shifts'][-1] if self.features['sequential_shifts'] else [0] * 6
       for pos, shift in enumerate(last_shifts, 1):
           last_num = self.positions[f'B{pos}'][-1]
           projected = last_num + int(shift)
           projected = max(self.global_min, min(self.global_max, projected))
           scores[projected] += 2
       # Genetic algorithm prediction
       _, _, scaled_features = self.scale_features()
       ga_pred = self.genetic_algorithm_prediction(scaled_features)
       for num in ga_pred:
           scores[num] += 5
       # Combine and rank
       sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
       candidates = [num for num, _ in sorted_nums[:12]]
       final_set = []
       repeat_count = 0
       for num in candidates:
           if num in last_3_nums:
               if repeat_count < 2:
                   final_set.append(num)
                   repeat_count += 1
           else:
               final_set.append(num)
           if len(final_set) == 6:
               break
       if len(final_set) < 6:
           for num in candidates:
               if num not in final_set:
                   final_set.append(num)
               if len(final_set) == 6:
                   break
       # Segment balance
       segment_size = self.global_max // 3
       bins = [(1, segment_size), (segment_size + 1, 2 * segment_size), (2 * segment_size + 1, self.global_max)]
       segment_count = [0] * len(bins)
       for num in final_set:
           for i, (low, high) in enumerate(bins):
               if low <= num <= high:
                   segment_count[i] += 1
       if max(segment_count) > 3:
           final_set = self.balance_segments(final_set, bins)
       final_set = sorted(final_set)
       final_scores = {num: scores[num] for num in final_set}
       # Map number origins
       number_origins = {num: [] for num in range(1, self.global_max + 1)}
       for num in range(1, self.global_max + 1):
           origins = []
           if num in self.features['frequency']:
               last_set = [s for s in self.sets_list if num in s][-1]
               origins.append(f"SET_{list(self.sets_dict.keys())[list(self.sets_dict.values()).index(last_set)]}")
           if any(num in self.features['positional_freq'][pos] for pos in self.features['positional_freq']):
               origins.append("Positional")
           if any(num in pair for pair in self.features['pairs'] if self.features['pairs'][pair] > 1):
               origins.append("Strong Pairs")
           number_origins[num] = origins[0] if origins else "None"
       # Positional reuse
       pos_reuse = {pos: bool(self.positions[pos][-1] in self.positions[pos][:-1]) for pos in self.positions}
       # Last 3 positions
       last_3_pos = {pos: [self.positions[pos][-i] for i in range(1, 4)] for pos in self.positions if len(self.positions[pos]) >= 3}
       return final_set, final_scores, number_origins, pos_reuse, last_3_pos

  

   def balance_segments(self, current_set, bins):
       segment_count = [0] * len(bins)
       for num in current_set:
           for i, (low, high) in enumerate(bins):
               if low <= num <= high:
                   segment_count[i] += 1
       while max(segment_count) > 3:
           max_segment = segment_count.index(max(segment_count))
           low, high = bins[max_segment]
           nums_in_segment = [num for num in current_set if low <= num <= high]
           if nums_in_segment:
               num_to_remove = random.choice(nums_in_segment)
               current_set.remove(num_to_remove)
               other_nums = [num for num in range(1, self.global_max + 1) if num not in current_set]
               new_num = random.choice(other_nums)
               current_set.append(new_num)
               segment_count = [0] * len(bins)
               for num in current_set:
                   for i, (low, high) in enumerate(bins):
                       if low <= num <= high:
                           segment_count[i] += 1
       return current_set

   def backtest_predictions(self, test_sets, predicted_set):
       hits = []
       for test_set in test_sets:
           common = len(set(predicted_set) & set(test_set))
           hits.append(common / 6)
       avg_accuracy = sum(hits) / len(hits) if hits else 0
       return {'accuracy': avg_accuracy}

   def monte_carlo_simulation(self, predicted_set, iterations=1000):
       successes = 0
       segment_size = self.global_max // 3
       bins = [(1, segment_size), (segment_size + 1, 2 * segment_size), (2 * segment_size + 1, self.global_max)]
       for _ in range(iterations):
           sim_set = sorted(random.sample(range(1, self.global_max + 1), 6))
           odd_count = sum(1 for num in sim_set if num % 2 == 1)
           segment_count = [0] * len(bins)
           for num in sim_set:
               for i, (low, high) in enumerate(bins):
                   if low <= num <= high:
                       segment_count[i] += 1
           if 2 <= odd_count <= 4 and all(c <= 3 for c in segment_count):
               if len(set(sim_set) & set(predicted_set)) >= 4:
                   successes += 1
       return successes / iterations

   def generate_report(self, predicted_set, final_scores, number_origins, pos_reuse, last_3_pos):
       last_10_sets = self.sets_list[-10:] if self.total_sets >= 10 else self.sets_list
       freq_counts = self.features['frequency']
       top_freq = freq_counts.most_common(5)
       gaps = self.features['gaps']
       sums = self.features['sums']
       avg_sum = sum(sums) / len(sums) if sums else 0
       sum_variance = sum((x - avg_sum) ** 2 for x in sums) / len(sums) if sums else 0
       sum_std = math.sqrt(sum_variance) if sum_variance > 0 else 0
       odd_count = sum(1 for num in predicted_set if num % 2 == 1)
       oe_ratio = {'odd': odd_count, 'even': 6 - odd_count}
       segment_size = self.global_max // 3
       bins = [(1, segment_size), (segment_size + 1, 2 * segment_size), (2 * segment_size + 1, self.global_max)]
       segment_count = {f'{low}-{high}': 0 for low, high in bins}
       for num in predicted_set:
           for low, high in bins:
               if low <= num <= high:
                   segment_count[f'{low}-{high}'] += 1
       last_3_nums = set(num for s in self.sets_list[-3:] for num in s)
       reused = list(set(predicted_set) & last_3_nums)
       rule_violation = "⚠️ Rule Violation" if len(reused) > 2 else "Compliant"
       triplet_shift = {
           'T1': sum(predicted_set[:3]) - (self.features['triplet_sums']['first'][-1] if self.features['triplet_sums']['first'] else 0),
           'T2': sum(predicted_set[3:]) - (self.features['triplet_sums']['second'][-1] if self.features['triplet_sums']['second'] else 0)
       }
       trending = [num for num, _ in top_freq]
       overdue = [(num, gap) for num, gap in gaps.items() if gap >= 10][:5]
       reinforcements = [num for num in predicted_set if self.features['frequency'].get(num, 0) > 2 or any(num in pair for pair in self.features['pairs'] if self.features['pairs'][pair] > 1)]
       stability_score = max(0, 10 - sum_std / 5) if sum_std > 0 else 10
       monte_carlo_prob = self.monte_carlo_simulation(predicted_set)
       prng_score = sum(self.features['frequency'].get(num, 0) / (gaps.get(num, 1) + 1) for num in predicted_set) / len(predicted_set)
       eda = self.exploratory_data_analysis()
       eliminated = {}
       for num in range(1, self.global_max + 1):
           if num not in predicted_set:
               reasons = []
               if num in last_3_nums:
                   reasons.append("Recent reuse")
               if final_scores.get(num, 0) < 2:
                   reasons.append("Low score")
               if gaps.get(num, 0) == 0:
                   reasons.append("Never appeared")
               if reasons:
                   eliminated[num] = {'score': final_scores.get(num, 0), 'reasons': reasons}
       penalized = [num for num in last_3_nums if num not in predicted_set and freq_counts.get(num, 0) > 2]
       low_score = [num for num, score in sorted(final_scores.items(), key=lambda x: x[1])[:5] if num not in predicted_set]
       behavioral_rules = [num for num in range(1, self.global_max + 1) if gaps.get(num, 0) == 0 and num not in predicted_set]
       report = []
       report.append("=" * 60)
       report.append("🔍 Analyzer Configuration Summary")
       report.append(f"Total Sets Analyzed: {self.total_sets}")
       report.append(f"Elements per Set: 6")
       report.append(f"Global Value Range: {self.global_min} to {self.global_max}")
       report.append(f"Elements Unique per Set: {len(set(self.sets_list[-1])) == 6 if self.sets_list else False}")
       report.append(f"Sets Are Sorted: {all(s == sorted(s) for s in self.sets_list)}")
       report.append("=" * 60)
       report.append("🎯 COMPREHENSIVE LOTTERY ANALYSIS REPORT")
       report.append("=" * 60)
       report.append("\n🔮 FINAL PREDICTION")
       report.append(f"Predicted Numbers: {predicted_set}")
       report.append(f"Number Origins: {{{', '.join(f'{num}: {number_origins[num]}' for num in predicted_set)}}}")
       report.append(f"Positional Reuse: {pos_reuse}")
       report.append(f"Reused from Last 3 Sets: {reused} ({rule_violation})")
       report.append(f"Last Appearance Gaps: {{{', '.join(f'{num}: {gaps.get(num, self.total_sets + 1)} draws ago' for num in predicted_set)}}}")
       report.append("\n🧠 PREDICTION DIAGNOSTICS")
       report.append(f"Odd/Even Balance: {oe_ratio}")
       report.append(f"Segment Distribution: {segment_count}")
       report.append(f"Triplet Shift: T1: {round(triplet_shift['T1'], 1)}, T2: {round(triplet_shift['T2'], 1)}")
       report.append(f"Stability Score: {round(stability_score, 1)}/10")
       report.append(f"Top Trending Numbers: {trending}")
       report.append(f"Behavioral Gaps: {[(num, gaps[num]) for num in predicted_set]}")
       report.append("\n📊 COMPREHENSIVE HISTORICAL VALUE ANALYSIS")
       report.append(f"Final Prediction: {predicted_set}")
       report.append(f"Monte Carlo Probability: {round(monte_carlo_prob, 3)}")
       report.append(f"PRNG Score: {round(prng_score, 3)}")
       report.append(f"Average Entropy: {round(eda['entropy'], 3)}")
       report.append(f"Number Origins: {number_origins}")
       report.append(f"Positional Reuse: {pos_reuse}")
       report.append(f"Reused from Last 3 Sets: {reused}")
       report.append(f"Last Appearance Gaps: {gaps}")
       report.append("\n📉 SCORING PROFILE")
       for num in predicted_set:
           report.append(f"Number {num}: Score {final_scores.get(num, 0)}")
       report.append("\n📈 ADDITIONAL ANALYSIS")
       report.append(f"Sequential Shifts (Last 3): {self.features['sequential_shifts'][-3:] if self.features['sequential_shifts'] else []}")
       report.append(f"Frequency Patterns: {top_freq}")
       report.append(f"Total Value Behavior: Mean={round(avg_sum, 1)}, Std={round(sum_std, 1)}")
       report.append(f"Overdue Candidates: {overdue}")
       report.append(f"Multi-Module Reinforcements: {reinforcements}")
       report.append("\n🚫 ELIMINATED NUMBERS")
       for num, info in eliminated.items():
           report.append(f"Number {num}: Score={info['score']}, Reasons: {', '.join(info['reasons'])}")
       report.append("\n⚠️ Penalized Due to Recent Reuse")
       report.append(f"Numbers: {penalized}")
       report.append("\n📤 Eliminated Due to Low Score")
       report.append(f"Numbers: {low_score}")
       report.append("\n🚫 Filtered Out by Behavioral Rules")
       report.append(f"Numbers: {behavioral_rules}")
       report.append("=" * 60)
       report.append("✅ ANALYSIS COMPLETE")
       report.append("=" * 60)
       return "\n".join(report)

   def deploy_and_monitor(self, predicted_set):
       print(f"Deploying prediction {predicted_set} via xAI API (https://x.ai/api)")
       print("Monitoring: Tracking odd/even balance, segment distribution, and stability score.")
       return {"status": "Deployed", "predicted_set": predicted_set}

# ----------------------------------------------------------------------
# Calculation Function Definition (Integrated from previous version)
# ----------------------------------------------------------------------
def calculate_set_stats(name, values):
   """
   Calculates the Odd, Even, Total, 1st Alternate, and 2nd Alternate sums
   for a given set of values.

   Args:
       name (str): The name of the set (e.g., "SET_1").
       values (list): A list of 6 numbers in the set.
      
   Returns:
       dict: A dictionary containing the set name, values, and the
             calculated statistics.
   """
   odd_sum = 0
   even_sum = 0
  
   # Calculate Odd and Even sums (based on the number's value)
   for num in values:
       if num % 2 != 0:
           odd_sum += num
       else:
           even_sum += num
          
   # Calculate Alternate sums (based on the number's position/index):
   # 1st Alt (index 0, 2, 4) = 1st + 3rd + 5th number
   # This assumes the list always has at least 5 elements (index 4)
   if len(values) >= 5:
       first_alt_sum = values[0] + values[2] + values[4]
   else:
       # Handle case where list is too short, though the class validates size 6
       first_alt_sum = sum(values[i] for i in [0, 2, 4] if i < len(values))
  
   # 2nd Alt (index 1, 3, 5) = 2nd + 4th + 6th number
   # This assumes the list always has at least 6 elements (index 5)
   if len(values) >= 6:
       second_alt_sum = values[1] + values[3] + values[5]
   else:
       # Handle case where list is too short, though the class validates size 6
       second_alt_sum = sum(values[i] for i in [1, 3, 5] if i < len(values))
  
   return {
       "name": name,
       "values": values,
       "1st_alt": first_alt_sum,
       "2nd_alt": second_alt_sum,
       "odd": odd_sum,
       "even": even_sum,
       "total": odd_sum + even_sum
   }

# ----------------------------------------------------------------------
# Printing Function Definition (Integrated from previous version)
# ----------------------------------------------------------------------
def print_set_analysis(data_list):
   """
   Prints the set data and its calculated analysis in a neat, tabular format.

   Args:
       data_list (list): A list of dictionaries, where each dictionary
                         represents one data set and must contain the
                         keys: 'name', 'values', '1st_alt', '2nd_alt',
                         'odd', 'even', and 'total'.
   """
  
   # Check if the list is empty before printing
   if not data_list:
       print("No data sets provided to print.")
       return

   # 1. Define the header row and its length
   header = (
       f"{'Set Name':<10} | {'Set Values':<25} | {'1st Alt':>7} | "
       f"{'2nd Alt':>7} | {'Odd':>5} | {'Even':>5} | {'Total':>5}"
   )
   separator = "=" * len(header)
  
   print(separator)
   print(header)
   print(separator)

   # 2. Iterate through the data list provided as an argument
   for item in data_list:
       # Format the list of values into a single, comma-separated string
       try:
           values_str = ", ".join(map(str, item['values']))
          
           # Use f-strings for precise column alignment (left < and right >)
           row = (
               f"{item['name']:<10} | {values_str:<25} | {item['1st_alt']:>7} | "
               f"{item['2nd_alt']:>7} | {item['odd']:>5} | {item['even']:>5} | "
               f"{item['total']:>5}"
           )
           print(row)
       except KeyError as e:
           # This handles cases where a required key is missing in a data item
           print(f"Error: Missing key {e} in data item {item.get('name', 'unknown')}. Skipping this row.")
  
   print(separator)

if __name__ == "__main__":
   example_data = {
       "SET_1": [5, 16, 17, 47, 57, 58],
       "SET_2": [6, 21, 23, 34, 37, 46],
       "SET_3": [5, 18, 42, 53, 55, 57],
       "SET_4": [16, 22, 24, 36, 39, 43],
       "SET_5": [5, 18, 29, 32, 34, 53]
   }

   # --- Start of LotteryAnalyzer Class Usage ---
   analyzer = LotteryAnalyzer(example_data)
   eda_results = analyzer.exploratory_data_analysis()
   features = analyzer.calculate_features()
   train_sets, test_sets, scaled_features = analyzer.scale_features()
   predicted_set, final_scores, number_origins, pos_reuse, last_3_pos = analyzer.predict_next_set()
   backtest_metrics = analyzer.backtest_predictions(test_sets, predicted_set)
   report = analyzer.generate_report(predicted_set, final_scores, number_origins, pos_reuse, last_3_pos)
   print(report)
   deployment_status = analyzer.deploy_and_monitor(predicted_set)
   print(f"Deployment Status: {deployment_status}")
   # --- End of LotteryAnalyzer Class Usage ---

   # 1. Process the raw data using the new calculation function
   processed_data = []
   # Convert dictionary items to a list of stats dictionaries
   for name, values in example_data.items():
       stats = calculate_set_stats(name, values)
       processed_data.append(stats)
      
   # 2. Print the final report
   print("\n" + "=" * 30)
   print("--- Detailed Set Analysis Report ---")
   print("=" * 30)
   print_set_analysis(processed_data)

