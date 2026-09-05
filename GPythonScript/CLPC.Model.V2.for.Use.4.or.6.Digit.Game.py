import numpy as np
from itertools import combinations
from collections import defaultdict
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

# ======================================================================
# --- 1. DATA SETUP (Select ONE dataset) ---
# ======================================================================

# --- Four-Digit Dataset (Numbers 0-9) ---
FOUR_DIGIT_DATA = {
   "SET_1": [1, 9, 7, 1], "SET_2": [7, 3, 2, 4],
   "SET_3": [1, 7, 8, 9], "SET_4": [6, 9, 2, 3],
   "SET_5": [4, 3, 3, 5], "SET_6": [1, 0, 3, 1],
   "SET_7": [7, 6, 1, 7], "SET_8": [5, 1, 9, 6],
   "SET_9": [0, 6, 9, 3], "SET_10": [1, 1, 4, 6],
}

# --- Six-Digit Dataset (Numbers 0-9) ---
SIX_DIGIT_DATA = {
   "SET_1": [1, 1, 6, 2, 9, 0], "SET_2": [0, 3, 1, 8, 5, 2],
   "SET_3": [1, 0, 4, 0, 3, 6], "SET_4": [2, 0, 5, 1, 3, 9],
   "SET_5": [4, 7, 1, 4, 2, 4], "SET_6": [1, 0, 1, 7, 5, 6],
   "SET_7": [9, 5, 7, 7, 3, 4], "SET_8": [5, 6, 4, 1, 0, 2],
   "SET_9": [9, 9, 9, 0, 9, 8], "SET_10": [1, 9, 6, 1, 7, 6],
}

# --- CONFIGURATION SWITCH ---
# ⚠️ CHOOSE YOUR DATASET HERE:
CURRENT_DATASET = FOUR_DIGIT_DATA
# CURRENT_DATASET = SIX_DIGIT_DATA # Uncomment this to switch

# --- 2. GLOBAL CONFIGURATION & DATA CLASSES (Constants/Defaults) ---
ABSENCE_WINDOW_SIZE = 3 # For a small historical set of 10
PREDICTION_ACCURACY_HISTORY = [0.0, 0.0, 0.0, 0.0]

# Note: MAX_NUMBER and SET_SIZE are now determined dynamically in HybridPredictor.__init__

@dataclass
class AnalysisQuality:
  data_completeness: float
  model_reliability: float
  prediction_accuracy: float
  confidence_calibration: float
  overall_quality: float
  quality_flags: List[str]
@dataclass
class HybridMetrics:
   a_score: float # Anti-Contagion Score (Absence)
   structural_confidence: float # New: Based on Frequency and Pair Strength
   final_score: float
   historical_frequency: int
   last_seen_set: int

# --- 3. HELPER FUNCTIONS ---
def preprocess_data(dataset):
  """Processes the data: ensures unique, sorted numbers, and correct key format."""
  cleaned_dataset = {}
  for key, values in dataset.items():
      if not values: continue
      unique_sorted = sorted(set(values))
      set_num = int(key.split('_')[-1]) if '_' in key else 0
      if set_num > 0:
          cleaned_dataset[f"SET_{set_num}"] = unique_sorted
  return cleaned_dataset
 
# --- Helper function to determine maximum number in the dataset ---
def find_max_number(dataset):
   """Finds the largest number in the raw dataset."""
   all_numbers = [num for values in dataset.values() for num in values]
   return max(all_numbers) if all_numbers else 0

def assess_data_quality(dataset, pred_history, max_number):
   total_sets = len(dataset)
   ORIGINAL_MAX_SETS = 10
   recent_accuracy = np.mean(pred_history) if pred_history else 0.5
  
   volatile_sets = {k:v for k,v in dataset.items() if int(k.split('_')[1]) >= total_sets - 2}
   volatile_numbers = set(n for s in volatile_sets.values() for n in s)
   # Use max_number + 1 for the total count of possible numbers (since numbers start at 0)
   num_range = max_number + 1
   volatility = len(volatile_numbers) / (num_range * 0.8)
  
   confidence_calibration = max(0.1, 1.0 - volatility)
  
   quality_flags = []
   if recent_accuracy < 0.2: quality_flags.append("CRITICAL: Recent model failure (Low Accuracy)")
   return AnalysisQuality(
       data_completeness=min(1.0, total_sets / ORIGINAL_MAX_SETS),
       model_reliability=max(0.1, 1.0 - volatility),
       prediction_accuracy=recent_accuracy,
       confidence_calibration=confidence_calibration,
       overall_quality=np.mean([min(1.0, total_sets / ORIGINAL_MAX_SETS), max(0.1, 1.0 - volatility), recent_accuracy, confidence_calibration]),
       quality_flags=quality_flags
   )

def calculate_pair_strength(dataset, max_number):
   """Calculates frequency of all 2-element pairs (Hierarchical Support)"""
   pair_freq = defaultdict(int)
   for set_values in dataset.values():
       for combo in combinations(set_values, 2):
           pair_freq[tuple(sorted(combo))] += 1
  
   number_pair_strength = defaultdict(lambda: {'total_strength': 0, 'pair_count': 0})
   for pair, freq in pair_freq.items():
       strength = freq / len(dataset)
       for num in pair:
           number_pair_strength[num]['total_strength'] += strength
           number_pair_strength[num]['pair_count'] += 1
          
   final_strength = {}
   # Iterate from 0 to MAX_NUMBER
   for num in range(0, max_number + 1):
       data = number_pair_strength[num]
       if data['pair_count'] > 0:
           final_strength[num] = data['total_strength'] / data['pair_count']
       else:
           final_strength[num] = 0.0
          
   return final_strength

def calculate_convergence(scores: List[float]) -> float:
   """Calculates Convergence (similarity) of scores in a candidate set."""
   if len(scores) < 2:
       return 0.0
   avg_score = np.mean(scores)
   std_dev = np.std(scores)
  
   cv = std_dev / avg_score if avg_score > 0 else float('inf')
  
   convergence_score = max(0.0, 1.0 - min(cv, 1.0))
   return convergence_score

# --- 4. CORE ANALYZER CLASS (Hybrid Logic) ---
class HybridPredictor:
  
   def __init__(self, raw_dataset, pred_history):
       self.raw_dataset = raw_dataset
       self.dataset = preprocess_data(raw_dataset)
      
       # --- FLEXIBILITY: AUTO-CONFIGURE PARAMETERS ---
       if not self.dataset:
            raise ValueError("Historical dataset is empty or invalid.")

       # 1. Determine SET_SIZE from the first set
       first_set_key = sorted(self.dataset.keys())[0]
       self.set_size = len(raw_dataset[first_set_key])
      
       # 2. Determine MAX_NUMBER from all historical data
       self.max_number = find_max_number(raw_dataset)
       # -----------------------------------------------
      
       self.max_historical_set = max(int(k.split('_')[1]) for k in self.dataset.keys()) if self.dataset else 0
       self.pair_strengths = calculate_pair_strength(self.dataset, self.max_number)
       self.metrics = self._calculate_hybrid_metrics()
       self.quality = assess_data_quality(self.dataset, pred_history, self.max_number)
      
   def _calculate_hybrid_metrics(self):
       """Calculates A-Score and the NEW Structural Confidence (Freq + Hierarchical Pair Strength)."""
       metrics = defaultdict(lambda: HybridMetrics(0.0, 0.0, 0.0, 0, 0))
      
       hist_freq = defaultdict(int)
       set_count = len(self.dataset)
       for idx, values in self.dataset.items():
           for num in values: hist_freq[num] += 1
              
       start_set = self.max_historical_set - ABSENCE_WINDOW_SIZE + 1
       absence_weights = {self.max_historical_set - i: i + 1 for i in range(ABSENCE_WINDOW_SIZE)}
      
       # Iterate from 0 to self.max_number
       for num in range(0, self.max_number + 1):
           a_score = 0.0
           last_seen = 0
          
           # A-Score Calculation
           for set_idx in range(start_set, self.max_historical_set + 1):
               set_key = f"SET_{set_idx}"
               is_present = num in self.dataset.get(set_key, [])
               if not is_present:
                   a_score += absence_weights.get(set_idx, 0)
               else:
                   last_seen = set_idx
                  
           freq = hist_freq[num]
           norm_freq = freq / set_count if set_count > 0 else 0.0
           pair_strength = self.pair_strengths.get(num, 0.0)
          
           structural_confidence = min(1.0, (norm_freq * 0.7 + pair_strength * 0.3) * 3.0)
          
           metrics[num] = HybridMetrics(
               a_score=a_score,
               structural_confidence=structural_confidence,
               final_score=0.0,
               historical_frequency=freq,
               last_seen_set=last_seen
           )
          
       return metrics
      
   def _blend_scores(self):
       """Dynamically blends A-Score and Structural Confidence."""
       recent_accuracy = self.quality.prediction_accuracy
       chaos_weight = max(0.1, 1.0 - recent_accuracy)
       structure_weight = 1.0 - chaos_weight
      
       for num, data in self.metrics.items():
           final_score = (data.a_score * chaos_weight) + (data.structural_confidence * structure_weight)
           data.final_score = final_score
      
       return chaos_weight, structure_weight
      
   def _apply_convergence_check(self, ranked_numbers: List[int]) -> List[int]:
       """Selects the final SET_SIZE numbers using the highest convergence score."""
      
       best_set = None
       best_score = -1.0
      
       # Test all combinations of SET_SIZE from the top N ranked numbers
       # Use SET_SIZE + 4 as the pool size, minimum 6
       top_n = max(self.set_size + 4, 6)
       top_candidates = ranked_numbers[:top_n]
      
       for candidate_set in combinations(top_candidates, self.set_size):
           a_scores = [self.metrics[num].a_score for num in candidate_set]
           convergence = calculate_convergence(a_scores)
          
           final_scores = [self.metrics[num].final_score for num in candidate_set]
           set_score = np.mean(final_scores) + (convergence * 0.5)
          
           if set_score > best_score:
               best_score = set_score
               best_set = candidate_set
              
       # Fallback to simply the top SET_SIZE numbers
       return sorted(list(best_set)) if best_set else sorted(ranked_numbers[:self.set_size])
      
   def generate_prediction(self):
       """Generates the final blended prediction set for the next set."""
      
       chaos_weight, structure_weight = self._blend_scores()
      
       # 1. Initial Ranking by Blended Score
       # Ensure only numbers that have appeared or are "due" are considered.
       ranked_numbers = sorted([num for num in self.metrics.keys()
                                if self.metrics[num].historical_frequency > 0 or self.metrics[num].a_score > 0],
                               key=lambda num: self.metrics[num].final_score,
                               reverse=True)
      
       # If there aren't enough unique ranked numbers, we can't make a full set
       if len(ranked_numbers) < self.set_size:
            print(f"WARNING: Not enough unique numbers ({len(ranked_numbers)}) available to form a full set of {self.set_size}.")
            # Use all available unique numbers
            prediction_set = sorted(ranked_numbers)
       else:
           # 2. Final Selection using CONVERGENCE CHECK on the Top N
           prediction_set = self._apply_convergence_check(ranked_numbers)
      
       # 3. Final Confidence Calculation (dynamic normalization)
       top_scores = [self.metrics[num].final_score for num in prediction_set]
       collective_score = sum(top_scores)
      
       # Max Possible Score based on dynamic SET_SIZE
       # Max A-Score for 3-set window is 3+2+1=6. Max Structural Conf is 1.0.
       max_score_per_num = (ABSENCE_WINDOW_SIZE * (ABSENCE_WINDOW_SIZE + 1) / 2) + 1.0
       max_possible_score = max_score_per_num * self.set_size
      
       base_confidence = collective_score / max_possible_score
      
       penalty_factor = structure_weight * self.quality.model_reliability * self.quality.confidence_calibration
      
       final_confidence = min(0.65, base_confidence * penalty_factor * 1.5)
      
       # Get the final selected numbers for the breakdown (sorted by final score)
       top_data = sorted([num for num in prediction_set],
                            key=lambda num: self.metrics[num].final_score,
                            reverse=True)
                           
       return prediction_set, final_confidence, top_data

# ======================================================================
# --- 5. EXECUTION ---
# ======================================================================
# The predictor will automatically configure itself based on the CURRENT_DATASET.
predictor = HybridPredictor(CURRENT_DATASET, PREDICTION_ACCURACY_HISTORY)
PRED_NEW, CONF_NEW, TOP_DATA = predictor.generate_prediction()

# --- 6. FINAL REPORT ---
set_size = predictor.set_size
print("="*80)
print(f"🚀 FINAL PREDICTED {set_size}-ELEMENT SET FOR SET {predictor.max_historical_set + 1} (Hybrid Model)".center(80))
print(f"Data Range: 0 to {predictor.max_number}".center(80))
print("="*80)
print(f"\n### Model Quality and Dynamic Weights (Using {len(CURRENT_DATASET)} Historical Sets)")
weights = predictor._blend_scores()
print(f"| Metric | Value | Rationale |")
print("| :--- | :--- | :--- |")
print(f"| Recent Model Accuracy | {predictor.quality.prediction_accuracy:.2%} | Failure rate dictates caution.")
print(f"| Chaos/Anti-Contagion Weight | {weights[0]:.1f} | High due to low recent accuracy.")
print(f"| Structure/Frequency Weight | {weights[1]:.1f} | Low due to model instability.")
print(f"| Overall Model Quality | {predictor.quality.overall_quality:.2f} | Comprehensive measure of data health.")
print("\n" + "="*80)
print("\n⭐ RECOMMENDED NEXT SET (Selected via Convergence Check):")
print(f"   SET: **{', '.join(map(str, PRED_NEW))}**")
print(f"   Confidence: **{CONF_NEW:.1%}** (Capped at 65% for chaotic data)")
print(f"   Rationale: This set was selected from the top candidates because its members demonstrated the highest **Convergence** (similarity in Overdue status) for the required {set_size} size.")
print(f"\n### Top {set_size} Number Breakdown (Selected by Final Score & Convergence)")
print(f"| Rank | Number | Final Score | A-Score (Chaos) | Structural Conf. (Pair/Freq) | Freq. | Last Seen |")
print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for i, num in enumerate(TOP_DATA):
   metrics = predictor.metrics[num]
   final_score_str = f"**{metrics.final_score:.2f}**"
   a_score_str = f"**{metrics.a_score:.0f}**"
      
   print(f"| {i+1} | **{num}** | {final_score_str} | {a_score_str} | {metrics.structural_confidence:.2f} | {metrics.historical_frequency} | {metrics.last_seen_set} |")
print("\n" + "="*80)
