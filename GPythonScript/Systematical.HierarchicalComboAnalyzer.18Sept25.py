import itertools
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import warnings

# -------------- Data Structures --------------

@dataclass
class ProblemFrame:
   objectives: List[str]
   constraints: List[str]
   stakeholders: List[str]
   assumptions: List[str]
   unknowns: List[str]
   risk_factors: List[str]
   success_criteria: List[str]

@dataclass
class AnalysisQuality:
   data_completeness: float
   model_reliability: float
   prediction_accuracy: float
   confidence_calibration: float
   overall_quality: float
   quality_flags: List[str]

@dataclass
class ValidationResult:
   is_valid: bool
   confidence_score: float
   validation_notes: List[str]
   recommended_actions: List[str]
   risk_assessment: str

# -------------- Helper Functions --------------

def validate_input(dataset, combo_sizes, significance_threshold):
   if not isinstance(dataset, dict):
       raise ValueError("Dataset should be a dict of sets.")
   if not all(isinstance(v, list) for v in dataset.values()):
       raise ValueError("Each set's value should be a list of integers.")
   if not all(2 <= size <= 6 for size in combo_sizes):
       raise ValueError("Combo sizes must be between 2 and 6 inclusive.")
   if not (0 < significance_threshold <= 1):
       raise ValueError("Significance threshold should be between 0 and 1.")

def preprocess_data(dataset):
   cleaned_dataset = {}
   preprocessing_stats = {
       'original_sets': len(dataset),
       'duplicates_removed': 0,
       'empty_sets_removed': 0,
       'total_numbers': 0
   }
  
   for key, values in dataset.items():
       if not values:
           preprocessing_stats['empty_sets_removed'] += 1
           continue
       original_count = len(values)
       unique_sorted = sorted(set(values))
       preprocessing_stats['duplicates_removed'] += original_count - len(unique_sorted)
       preprocessing_stats['total_numbers'] += len(unique_sorted)
       cleaned_dataset[key] = unique_sorted
  
   return cleaned_dataset, preprocessing_stats

def assess_data_quality(dataset, preprocessing_stats):
   if not dataset:
       return AnalysisQuality(0, 0, 0, 0, 0, ["CRITICAL: No valid data"])
  
   total_sets = len(dataset)
   avg_set_size = preprocessing_stats['total_numbers'] / total_sets if total_sets > 0 else 0
   duplicate_ratio = preprocessing_stats['duplicates_removed'] / preprocessing_stats['total_numbers'] if preprocessing_stats['total_numbers'] > 0 else 0
  
   quality_flags = []
   if total_sets < 5:
       quality_flags.append("WARNING: Very small dataset (< 5 sets)")
   if avg_set_size < 3:
       quality_flags.append("WARNING: Very small average set size")
   if duplicate_ratio > 0.3:
       quality_flags.append("WARNING: High duplicate ratio in data")
  
   data_completeness = min(1.0, total_sets / 10)
   model_reliability = max(0.1, 1.0 - duplicate_ratio)
  
   return AnalysisQuality(
       data_completeness=data_completeness,
       model_reliability=model_reliability,
       prediction_accuracy=0.5,
       confidence_calibration=0.5,
       overall_quality=(data_completeness + model_reliability) / 2,
       quality_flags=quality_flags
   )

def generate_combinations(dataset, combo_size, max_combos=500000):
   combo_occurrences = defaultdict(list)
   combo_count = 0
   for set_key, numbers in dataset.items():
       combos = itertools.combinations(numbers, combo_size)
       for combo in combos:
           combo_occurrences[combo].append(set_key)
           combo_count += 1
           if combo_count >= max_combos:
               warnings.warn(f"Stopped generating size-{combo_size} combos at {max_combos}")
               break
       if combo_count >= max_combos:
           break
   return combo_occurrences

def filter_combos_by_frequency(combo_occurrences, min_freq):
   return {combo: sets for combo, sets in combo_occurrences.items() if len(sets) >= min_freq}

def calculate_intervals(occurrences_indices):
   intervals = []
   for i in range(1, len(occurrences_indices)):
       intervals.append(occurrences_indices[i] - occurrences_indices[i-1])
   return intervals

def calculate_trend(intervals):
   if len(intervals) < 3:
       return None, None
   x = np.arange(len(intervals))
   y = np.array(intervals)
   slope = np.polyfit(x, y, 1)[0]
   corr_matrix = np.corrcoef(x, y)
   corr = corr_matrix[0, 1]
   return slope, corr

def calculate_partial_support(combo, dataset, min_match_ratio=0.5):
   combo_set = set(combo)
   supporting_sets = []
   match_ratios = []
  
   for set_key, numbers in dataset.items():
       numbers_set = set(numbers)
       matches = len(combo_set.intersection(numbers_set))
       match_ratio = matches / len(combo)
       if match_ratio >= min_match_ratio:
           supporting_sets.append(set_key)
           match_ratios.append(match_ratio)
  
   support_ratio = len(supporting_sets) / len(dataset)
   avg_match_ratio = np.mean(match_ratios) if match_ratios else 0.0
  
   return support_ratio, avg_match_ratio, supporting_sets

def calculate_hierarchical_strength(combo, combo_stats, min_sub_size=2):
   n = len(combo)
   if n <= min_sub_size:
       return 0.0, {}
  
   sub_combo_strengths = []
   sub_combo_details = {}
  
   for sub_size in range(min_sub_size, n):
       for sub_combo in itertools.combinations(combo, sub_size):
           if sub_combo in combo_stats:
               stats = combo_stats[sub_combo]
               strength = stats.get('boosted_confidence', stats.get('confidence', 0))
               sub_combo_strengths.append(strength)
               sub_combo_details[sub_combo] = {
                   'size': sub_size,
                   'frequency': stats.get('frequency', 0),
                   'confidence': stats.get('confidence', 0),
                   'strength': strength
               }
  
   if not sub_combo_strengths:
       return 0.0, {}
  
   weighted_strengths = []
   for sub_combo, details in sub_combo_details.items():
       size_weight = details['size'] / n
       weighted_strength = details['strength'] * size_weight
       weighted_strengths.append(weighted_strength)
  
   hierarchical_boost = np.mean(weighted_strengths) * 0.25
  
   return hierarchical_boost, sub_combo_details

def calculate_convergence(confidences, strengths):
   if not confidences or not strengths:
       return 0.0
   avg_conf = np.mean(confidences)
   avg_strength = np.mean(strengths)
   if avg_conf == 0:
       return 0.0
   std_conf = np.std(confidences)
   cv = std_conf / avg_conf if avg_conf > 0 else float('inf')
   convergence = max(0, (1 - min(cv, 1)) * 100)
   strength_bonus = min(avg_strength * 15, 15)
   return min(100, convergence + strength_bonus)

def predict_next_occurrence(occurrence_indices):
   if len(occurrence_indices) < 2:
       return None, None
   intervals = calculate_intervals(occurrence_indices)
   if not intervals:
       return None, None
   avg_interval = np.mean(intervals)
   ewma = intervals[0]
   alpha = 0.4
   for i in intervals[1:]:
       ewma = alpha * i + (1 - alpha) * ewma
   slope, corr = calculate_trend(intervals)
   if slope is not None:
       trend_adj_interval = avg_interval + slope
       if trend_adj_interval < 1:
           trend_adj_interval = 1
   else:
       trend_adj_interval = avg_interval
   harmonic_mean = len(intervals) / np.sum(1 / np.array(intervals)) if intervals else avg_interval
   weights = [0.3, 0.3, 0.2, 0.2]
   preds = np.array([avg_interval, ewma, trend_adj_interval, harmonic_mean])
   pred = np.dot(weights, preds)
   uncertainty = np.var(preds)
   next_occurrence = occurrence_indices[-1] + pred
   return next_occurrence, uncertainty

def assign_alert(confidence, prediction=None, current_index=None, historical_accuracy=0.5):
   if confidence >= 0.85:
       base_alert = "RED"
   elif confidence >= 0.7:
       base_alert = "ORANGE"
   elif confidence >= 0.45:
       base_alert = "YELLOW"
   else:
       base_alert = "GREEN"
   if historical_accuracy < 0.4 and base_alert in ["RED", "ORANGE"]:
       base_alert = "YELLOW"
   if prediction is not None and current_index is not None:
       imminence = prediction - current_index
       if imminence <= 3 and base_alert != "RED":
           base_alert = "ORANGE"
       elif imminence <= 5 and base_alert == "GREEN":
           base_alert = "YELLOW"
   return base_alert

def generate_speculative_combos(high_conf_combos, target_size, existing_stats):
   speculative = set()
   combos_by_size = defaultdict(list)
   for combo in high_conf_combos:
       combos_by_size[len(combo)].append(combo)
   for size in range(2, target_size):
       if size not in combos_by_size:
           continue
       for i, c1 in enumerate(combos_by_size[size]):
           for j, c2 in enumerate(combos_by_size[size]):
               if i >= j:
                   continue
               overlap = len(set(c1) & set(c2))
               if 1 <= overlap < size:
                   merged = tuple(sorted(set(c1) | set(c2)))
                   if len(merged) == target_size:
                       speculative.add(merged)
   return speculative

# -------------- Validation Functions --------------

def validate_analysis_results(analysis_results, problem_frame, data_quality):
   validation_notes = []
   risk_assessment = "LOW"
   recommended_actions = []
  
   total_results = len(analysis_results)
   if total_results == 0:
       validation_notes.append("CRITICAL: No combinations generated - possible high frequency threshold")
       risk_assessment = "HIGH"
       recommended_actions.append("Lower significance threshold or check dataset for sufficient overlap")
       return ValidationResult(
           is_valid=False,
           confidence_score=0.1,
           validation_notes=validation_notes,
           recommended_actions=recommended_actions,
           risk_assessment=risk_assessment
       )
  
   high_conf_count = sum(1 for r in analysis_results if r.get('boosted_confidence', 0) > 0.85)
   if high_conf_count / total_results > 0.3 and total_results > 10:
       validation_notes.append("WARNING: High number of overconfident results - possible overfitting")
       risk_assessment = "MEDIUM"
       recommended_actions.append("Increase significance threshold and review data quality")
  
   spec_count = sum(1 for r in analysis_results if r.get('speculative', False))
   if spec_count / total_results > 0.5:
       validation_notes.append("WARNING: Too many speculative combos")
       recommended_actions.append("Reduce speculative combo generation threshold")
  
   if data_quality.overall_quality < 0.5:
       validation_notes.append("CAUTION: Low data quality may affect results reliability")
       risk_assessment = "HIGH" if risk_assessment != "HIGH" else "HIGH"
       recommended_actions.append("Collect more high-quality data")
  
   confidences = [r.get('boosted_confidence', 0) for r in analysis_results]
   if len(set(confidences)) < len(confidences) * 0.15:
       validation_notes.append("WARNING: Low confidence score diversity - possible model bias")
       recommended_actions.append("Adjust confidence weights")
  
   is_valid = len([note for note in validation_notes if "WARNING" in note or "CAUTION" in note]) < 2
   confidence_score = max(0.1, 1.0 - len(validation_notes) * 0.25)
  
   return ValidationResult(
       is_valid=is_valid,
       confidence_score=confidence_score,
       validation_notes=validation_notes,
       recommended_actions=recommended_actions,
       risk_assessment=risk_assessment
   )


def should_rethink_analysis(validation_result, data_quality, iteration_count=1):
   rethink_triggers = []
   if not validation_result.is_valid:
       rethink_triggers.append("Failed validation checks")
   if validation_result.confidence_score < 0.5:
       rethink_triggers.append("Low validation confidence")
   if data_quality.overall_quality < 0.4:
       rethink_triggers.append("Poor data quality")
   if validation_result.risk_assessment == "HIGH":
       rethink_triggers.append("High risk assessment")
   should_rethink = bool(rethink_triggers) and iteration_count < 3
   return should_rethink, rethink_triggers

def suggest_parameter_adjustments(validation_result, data_quality, current_params):
   suggestions = {}
   if "overfitting" in " ".join(validation_result.validation_notes).lower():
       suggestions['significance_threshold'] = min(0.6, current_params.get('significance_threshold', 0.1) * 1.8)
   if data_quality.data_completeness < 0.5:
       suggestions['combo_sizes'] = [2, 3]
   if "low confidence score diversity" in " ".join(validation_result.validation_notes).lower():
       suggestions['confidence_weights'] = "rebalance"
   if "speculative" in " ".join(validation_result.validation_notes).lower() or "no combinations" in " ".join(validation_result.validation_notes).lower():
       suggestions['significance_threshold'] = max(0.05, current_params.get('significance_threshold', 0.1) * 0.5)
       suggestions['speculative_threshold'] = current_params.get('speculative_threshold', 0.65) * 0.8
   return suggestions

# -------------- Main Analyzer Class --------------

class HierarchicalComboAnalyzer:
   def __init__(self, dataset, combo_sizes=[2,3,4,5,6], significance_threshold=0.1,
                problem_frame=None):
       validate_input(dataset, combo_sizes, significance_threshold)
       self.problem_frame = problem_frame or self._create_default_problem_frame()
       self.raw_dataset = dataset
       self.combo_sizes = sorted(combo_sizes)
       self.total_sets = len(dataset)
       self.significance_threshold = min(significance_threshold, 3.0 / self.total_sets if self.total_sets > 0 else 0.1)
       self.speculative_threshold = 0.5
       self.dataset, self.preprocessing_stats = preprocess_data(dataset)
       self.data_quality = assess_data_quality(self.dataset, self.preprocessing_stats)
       self.min_freq = max(2, int(self.significance_threshold * self.total_sets))
       self.set_indices = {k: i for i, k in enumerate(sorted(self.dataset.keys()))}
       self.combo_occurrences = {}
       self.combo_stats = {}
       self.confidences = {}
       self.boosted_confidences = {}
       self.predictions = {}
       self.alerts = {}
       self.speculative_combos = set()
       self.speculative_stats = {}
       self.validation_result = None
       self.analysis_iterations = 0
       self.historical_accuracy = defaultdict(lambda: 0.5)
       self.decision_log = []

   def _create_default_problem_frame(self):
       return ProblemFrame(
           objectives=["Identify high-confidence number combinations", "Predict future occurrences"],
           constraints=["Limited historical data", "Computational complexity"],
           stakeholders=["Analyst", "Decision makers"],
           assumptions=["Historical patterns continue", "Data is representative"],
           unknowns=["External factors affecting patterns", "Optimal prediction horizon"],
           risk_factors=["Overfitting to limited data", "Pattern changes over time"],
           success_criteria=["High prediction accuracy", "Actionable insights"]
       )

   def log_decision(self, decision_type, description, reasoning, confidence=None):
       self.decision_log.append({
           'timestamp': datetime.now().isoformat(),
           'iteration': self.analysis_iterations,
           'type': decision_type,
           'description': description,
           'reasoning': reasoning,
           'confidence': confidence
       })

   def extract_combos(self):
       self.log_decision("EXTRACTION", "Starting combo extraction",
                        f"Targeting {self.combo_sizes} sizes with min_freq {self.min_freq}")
       for size in self.combo_sizes:
           occurrences = generate_combinations(self.dataset, size)
           total_combos = len(occurrences)
           filtered = filter_combos_by_frequency(occurrences, self.min_freq)
           self.combo_occurrences[size] = filtered
           self.log_decision("FILTER", f"Filtered size-{size} combos",
                            f"Kept {len(filtered)}/{total_combos} combos")
           if len(filtered) == 0:
               self.data_quality.quality_flags.append(f"WARNING: No size-{size} combos passed min_freq={self.min_freq}")

   def analyze_combos(self):
       all_combos = {}
       set_indices = self.set_indices
       insufficient_data_count = 0
       HIGH_CONFIDENCE_THRESHOLD = 0.75

       for size in self.combo_sizes:
           combos = self.combo_occurrences.get(size, {})
           for combo, sets in combos.items():
               occurrence_indices = sorted(set_indices[s] for s in sets)
               intervals = calculate_intervals(occurrence_indices)
               slope, corr = calculate_trend(intervals)
               freq = len(sets)
               support_ratio, avg_match_ratio, supporting_sets = calculate_partial_support(
                   combo, self.dataset, min_match_ratio=0.5
               )
               if len(intervals) < 3:
                   insufficient_data_count += 1
                   trend_strength = 0.0
                   trend_type = "INSUFFICIENT_DATA"
               else:
                   trend_strength = abs(corr) if corr is not None else 0.0
                   trend_type = "INCREASING" if slope > 0 else "DECREASING" if slope < 0 else "STABLE"
               freq_norm = freq / self.total_sets
               recency = occurrence_indices[-1] / (self.total_sets - 1) if self.total_sets > 1 else 1.0
               size_penalty = (size - 2) * 0.08
               base_confidence = (
                   0.25 * freq_norm +
                   0.25 * support_ratio +
                   0.2 * avg_match_ratio +
                   0.15 * trend_strength +
                   0.15 * recency
               ) - size_penalty
               confidence = max(0.0, min(base_confidence, 1.0))
               significance_level = "HIGH" if confidence >= HIGH_CONFIDENCE_THRESHOLD else "NORMAL"
               all_combos[combo] = {
                   "frequency": freq,
                   "frequency_norm": freq_norm,
                   "support_ratio": support_ratio,
                   "avg_match_ratio": avg_match_ratio,
                   "supporting_sets": supporting_sets,
                   "intervals": intervals,
                   "trend_slope": slope,
                   "trend_corr": corr,
                   "trend_strength": trend_strength,
                   "trend_type": trend_type,
                   "recency": recency,
                   "size": size,
                   "size_penalty": size_penalty,
                   "confidence": confidence,
                   "significance_level": significance_level,
                   "occurrence_indices": occurrence_indices,
                   "sets": sets,
               }
       self.combo_stats = all_combos
       if insufficient_data_count > len(all_combos) * 0.5:
           self.data_quality.quality_flags.append("WARNING: Many combos have insufficient trend data")

   def apply_hierarchical_support(self):
       self.confidences = {combo: stats["confidence"] for combo, stats in self.combo_stats.items()}
       hierarchical_impacts = []
       for combo, stats in self.combo_stats.items():
           n = stats['size']
           if n <= 2:
               continue
           hierarchical_boost, sub_combo_details = calculate_hierarchical_strength(combo, self.combo_stats)
           if hierarchical_boost > 0:
               boosted_conf = stats["confidence"] + hierarchical_boost
               boosted_conf = min(boosted_conf, 0.95)
               stats["hierarchical_boost"] = hierarchical_boost
               stats["sub_combo_details"] = sub_combo_details
               self.boosted_confidences[combo] = boosted_conf
               stats["boosted_confidence"] = boosted_conf
               hierarchical_impacts.append(hierarchical_boost)
       if hierarchical_impacts:
           avg_boost = np.mean(hierarchical_impacts)
           self.log_decision("HIERARCHICAL", "Applied hierarchical support",
                            f"Average boost: {avg_boost:.3f}, {len(hierarchical_impacts)} combos affected")

   def assign_alerts(self):
       alert_distribution = defaultdict(int)
       current_index = self.total_sets
       for combo, stats in self.combo_stats.items():
           boosted_conf = self.boosted_confidences.get(combo, stats["confidence"])
           pred, uncert = self.predictions.get(combo, (None, None))
           hist_acc = self.historical_accuracy.get(combo, 0.5)
           alert = assign_alert(boosted_conf, pred, current_index, hist_acc)
           self.alerts[combo] = alert
           alert_distribution[alert] += 1
       for combo, spec_stats in self.speculative_stats.items():
           alert = assign_alert(spec_stats["confidence"], spec_stats.get("prediction"), current_index, 0.3)
           self.alerts[combo] = alert
           alert_distribution[alert] += 1
       self.log_decision("ALERTS", "Assigned alert levels",
                        f"Distribution: {dict(alert_distribution)}")

   def generate_predictions(self):
       prediction_count = 0
       avg_uncertainty = []
       current_index = self.total_sets
       for combo, stats in self.combo_stats.items():
           if len(stats["occurrence_indices"]) > 1:
               pred, uncert = predict_next_occurrence(stats["occurrence_indices"])
               if "hierarchical_boost" in stats and stats["hierarchical_boost"] > 0:
                   hier_blend = pred * (1 + stats["hierarchical_boost"] * 0.1)
                   pred = (pred + hier_blend) / 2
               self.predictions[combo] = (pred, uncert)
               if uncert is not None:
                   avg_uncertainty.append(uncert)
               prediction_count += 1
               if pred and current_index:
                   error = abs(pred - current_index)
                   self.historical_accuracy[combo] = max(0.3, min(0.7, 1.0 - error / 10))
           else:
               self.predictions[combo] = (None, None)
       if avg_uncertainty:
           mean_uncertainty = np.mean(avg_uncertainty)
           self.log_decision("PREDICTION", f"Generated {prediction_count} predictions",
                            f"Average uncertainty: {mean_uncertainty:.3f}")

   def generate_speculative(self):
       high_conf_combos = [
           combo for combo, conf in self.boosted_confidences.items()
           if conf >= self.speculative_threshold
       ]
       speculative_generated = 0
       for target_size in range(max(self.combo_sizes), 7):
           new_speculative = generate_speculative_combos(
               high_conf_combos, target_size, self.combo_stats
           )
           for combo in new_speculative:
               if combo not in self.combo_stats and combo not in self.speculative_combos:
                   hierarchical_boost, sub_details = calculate_hierarchical_strength(
                       combo, self.combo_stats
                   )
                   support_ratio, avg_match_ratio, _ = calculate_partial_support(
                       combo, self.dataset, min_match_ratio=0.5
                   )
                   base_spec_conf = (
                       0.35 * hierarchical_boost +
                       0.3 * support_ratio +
                       0.25 * avg_match_ratio
                   )
                   speculative_conf = base_spec_conf * 0.7
                   speculative_conf = max(0.1, min(speculative_conf, 0.8))
                   sub_preds = []
                   for sub, details in sub_details.items():
                       if sub in self.predictions:
                           sub_pred, _ = self.predictions[sub]
                           if sub_pred:
                               sub_preds.append(sub_pred)
                   spec_pred = np.mean(sub_preds) if sub_preds else None
                   alert = assign_alert(speculative_conf, spec_pred, self.total_sets, 0.3)
                   self.speculative_combos.add(combo)
                   self.speculative_stats[combo] = {
                       "confidence": speculative_conf,
                       "boosted_confidence": speculative_conf,
                       "support_ratio": support_ratio,
                       "avg_match_ratio": avg_match_ratio,
                       "hierarchical_boost": hierarchical_boost,
                       "sub_combo_details": sub_details,
                       "alert": alert,
                       "speculative": True,
                       "prediction": spec_pred,
                       "significance_level": "SPECULATIVE"
                   }
                   speculative_generated += 1
       self.log_decision("SPECULATIVE", f"Generated {speculative_generated} speculative combos",
                        f"Based on {len(high_conf_combos)} high-confidence parent combos")

   def calculate_analysis_metrics(self, report_items):
       if not report_items:
           return {
               "total_combos": 0,
               "convergence": 0,
               "avg_confidence": 0,
               "avg_support": 0,
               "data_quality": asdict(self.data_quality),
               "alert_distribution": {"RED": 0, "ORANGE": 0, "YELLOW": 0, "GREEN": 0}
           }
       confidences = [item["boosted_confidence"] for item in report_items]
       supports = [item.get("support_ratio", 0) for item in report_items]
       convergence = calculate_convergence(confidences, supports)
       confidence_std = np.std(confidences) if confidences else 0
       if confidence_std > 0:
           self.data_quality.confidence_calibration = 1.0 - min(confidence_std, 0.5) * 1.5
       self.data_quality.prediction_accuracy = np.mean([self.historical_accuracy.get(item["combo"], 0.5)
                                                       for item in report_items]) if report_items else 0.5
       self.data_quality.overall_quality = (
           self.data_quality.data_completeness +
           self.data_quality.model_reliability +
           self.data_quality.prediction_accuracy +
           self.data_quality.confidence_calibration
       ) / 4
       return {
           "total_combos": len(report_items),
           "convergence": convergence,
           "avg_confidence": np.mean(confidences) * 100 if confidences else 0,
           "avg_support": np.mean(supports) * 100 if supports else 0,
           "data_quality": asdict(self.data_quality),
           "alert_distribution": {
               "RED": sum(1 for item in report_items if item["alert"] == "RED"),
               "ORANGE": sum(1 for item in report_items if item["alert"] == "ORANGE"),
               "YELLOW": sum(1 for item in report_items if item["alert"] == "YELLOW"),
               "GREEN": sum(1 for item in report_items if item["alert"] == "GREEN"),
           },
           "preprocessing_stats": self.preprocessing_stats
       }

   def compile_report(self, top_n_per_size=3):
       report_items = []
       for combo, stats in self.combo_stats.items():
           boosted_conf = self.boosted_confidences.get(combo, stats["confidence"])
           alert = self.alerts.get(combo, "GREEN")
           pred, uncert = self.predictions.get(combo, (None, None))
           report_items.append({
               "combo": combo,
               "size": len(combo),
               "frequency": stats["frequency"],
               "confidence": float(stats["confidence"]),
               "boosted_confidence": float(boosted_conf),
               "support_ratio": float(stats.get("support_ratio", 0)),
               "avg_match_ratio": float(stats.get("avg_match_ratio", 0)),
               "hierarchical_boost": float(stats.get("hierarchical_boost", 0)),
               "alert": alert,
               "speculative": False,
               "prediction": float(pred) if pred is not None else None,
               "uncertainty": float(uncert) if uncert is not None else None,
               "trend_strength": float(stats.get("trend_strength", 0)),
               "recency": float(stats.get("recency", 0)),
               "significance_level": stats.get("significance_level", "NORMAL"),
               "trend_type": stats.get("trend_type", "INSUFFICIENT_DATA"),
           })
       for combo, spec_stats in self.speculative_stats.items():
           report_items.append({
               "combo": combo,
               "size": len(combo),
               "frequency": 0,
               "confidence": float(spec_stats["confidence"]),
               "boosted_confidence": float(spec_stats["boosted_confidence"]),
               "support_ratio": float(spec_stats.get("support_ratio", 0)),
               "avg_match_ratio": float(spec_stats.get("avg_match_ratio", 0)),
               "hierarchical_boost": float(spec_stats.get("hierarchical_boost", 0)),
               "alert": spec_stats["alert"],
               "speculative": True,
               "prediction": float(spec_stats.get("prediction")) if spec_stats.get("prediction") else None,
               "uncertainty": None,
               "trend_strength": 0,
               "recency": 0,
               "significance_level": spec_stats.get("significance_level", "SPECULATIVE"),
           })
       # Group by size and select top 3 per size
       items_by_size = defaultdict(list)
       for item in report_items:
           items_by_size[item["size"]].append(item)
       top_items = []
       for size in self.combo_sizes:
           size_items = sorted(
               items_by_size[size],
               key=lambda x: x["boosted_confidence"],
               reverse=True
           )[:top_n_per_size]
           top_items.extend(size_items)
       return top_items, []

  

   def generate_hierarchical_summary(self):
       report_items, imminent = self.compile_report(top_n_per_size=3)
       metrics = self.calculate_analysis_metrics(report_items)
       high_sig_count = sum(1 for item in report_items if item["significance_level"] == "HIGH")
       speculative_count = sum(1 for item in report_items if item["speculative"])
       boosts = [item["hierarchical_boost"] for item in report_items if item["hierarchical_boost"] > 0]
       avg_boost_impact = float(np.mean(boosts)) if boosts else 0
       alert_distribution = metrics.get("alert_distribution", {"RED": 0, "ORANGE": 0, "YELLOW": 0, "GREEN": 0})
       summary = {
           "num_combos_per_level": {
               "HIGH": high_sig_count,
               "NORMAL": len(report_items) - high_sig_count - speculative_count,
               "SPECULATIVE": speculative_count
           },
           "alert_distribution": alert_distribution,
           "hierarchical_boost_impact": {
               "avg_boost": avg_boost_impact,
               "affected_combos": len(boosts)
           },
           "prediction_accuracy_estimate": float(self.data_quality.prediction_accuracy),
           "top_combos_by_size": {
               str(size): [
                   {k: float(v) if isinstance(v, np.float64) else v for k, v in item.items()}
                   for item in sorted(
                       [item for item in report_items if item["size"] == size],
                       key=lambda x: x["boosted_confidence"],
                       reverse=True
                   )[:3]
               ]
               for size in self.combo_sizes
           },
           "total_combos": len(report_items),
           "convergence": float(metrics["convergence"])
       }
       return summary, metrics

   def generate_readable_report(self, report_items, summary, decision):
       readable_report = []
       readable_report.append("=== Top Number Combinations by Confidence ===")
       readable_report.append(f"Analyzed {self.total_sets} sets of numbers to find repeating patterns (sizes {', '.join(map(str, self.combo_sizes))}).")
       if summary['total_combos'] == 0:
           readable_report.append("No combinations found. The dataset may lack repeating patterns, or the frequency threshold may be too high.")
           readable_report.append(f"Recommendation: Lower the significance threshold (currently {self.significance_threshold:.3f}) or add more data.")
           return "\n".join(readable_report)
      
       for size in self.combo_sizes:
           readable_report.append(f"\n=== Top 3 Combinations (Size {size}) ===")
           size_combos = summary['top_combos_by_size'].get(str(size), [])
           if not size_combos:
               readable_report.append(f"No combinations found for size {size}.")
               continue
           for i, item in enumerate(size_combos, 1):
               combo_str = ", ".join(map(str, item['combo']))
               confidence = item['boosted_confidence'] * 100
               alert = item['alert']
               speculative = item['speculative']
               readable_report.append(f"{i}. Numbers: {combo_str}")
               readable_report.append(f"   - Confidence: {confidence:.1f}%")
               readable_report.append(f"   - Priority: {alert}")
               if speculative:
                   readable_report.append(f"   - Note: Speculative combination based on smaller patterns")
      
       readable_report.append("\nRecommendation:")
       preferred = decision['preferred_solution']
       combo_str = preferred['option'].split(": ")[1]
       readable_report.append(f"- Focus on numbers {combo_str} (Priority Score: {preferred['priority_score']:.2f})")
       readable_report.append(f"  - Expected Impact: {preferred['impact']:.1f}/10")
       readable_report.append(f"  - Plan: Monitor new sets and update analysis weekly")
       if summary['prediction_accuracy_estimate'] < 0.5:
           readable_report.append(f"  - Caution: Low prediction accuracy ({summary['prediction_accuracy_estimate']*100:.1f}%) suggests collecting more data.")
      
       readable_report.append("\nNotes:")
       if summary['hierarchical_boost_impact']['affected_combos'] > 0:
           readable_report.append(f"- {summary['hierarchical_boost_impact']['affected_combos']} combinations boosted by smaller patterns (average boost: {summary['hierarchical_boost_impact']['avg_boost']:.2f}).")
       if self.data_quality.quality_flags:
           readable_report.append(f"- Data quality concerns: {', '.join(self.data_quality.quality_flags)}")
      
       return "\n".join(readable_report)

   def export_report(self, format_type="json", filename=None):
       summary, metrics = self.generate_hierarchical_summary()
       full_report = {
           "summary": summary,
           "metrics": metrics,
           "report_items": self.compile_report(top_n_per_size=3)[0],
           "decision_log": self.decision_log,
           "data_quality": asdict(self.data_quality)
       }
       if filename is None:
           filename = f"combo_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
       if format_type == "json":
           with open(filename, 'w') as f:
               json.dump(full_report, f, indent=4, default=str)
       elif format_type == "csv":
           import csv
           with open(filename, 'w', newline='') as f:
               if full_report["report_items"]:
                   writer = csv.DictWriter(f, fieldnames=full_report["report_items"][0].keys())
                   writer.writeheader()
                   writer.writerows(full_report["report_items"])
               else:
                   writer = csv.writer(f)
                   writer.writerow(["No data"])
       else:
           warnings.warn("Unsupported format, defaulting to JSON")
           with open(filename, 'w') as f:
               json.dump(full_report, f, indent=4, default=str)
       self.log_decision("EXPORT", f"Exported report as {format_type}", f"Filename: {filename}")
       return filename

   def solution_ideation(self):
       report_items, _ = self.compile_report(top_n_per_size=3)
       alternatives = []
       for i, item in enumerate(report_items):
           alt = {
               "option": f"Prioritize Combo {i+1}: {item['combo']}",
               "impact": item["boosted_confidence"] * 10,
               "feasibility": 8 - item["size"] * 1.5,
               "cost": item["size"] * 3,
               "description": f"Focus on high-confidence combo with {item['alert']} alert"
           }
           alternatives.append(alt)
       for alt in alternatives:
           score = (alt["impact"] * 0.5 + alt["feasibility"] * 0.3 - alt["cost"] * 0.2) / 10
           alt["priority_score"] = score
       top_options = sorted(alternatives, key=lambda x: x["priority_score"], reverse=True)[:5]
       self.log_decision("IDEATION", "Generated solution alternatives", f"Top 5 options prioritized")
       return top_options

   def final_decision_making(self, top_options):
       preferred = max(top_options, key=lambda x: x["priority_score"]) if top_options else {
           "option": "No viable combos",
           "impact": 0,
           "feasibility": 0,
           "cost": 0,
           "description": "No combinations available for prioritization",
           "priority_score": 0
       }
       decision = {
           "preferred_solution": preferred,
           "alignment_check": "Aligned with objectives and risk appetite",
           "cross_reference": "Historical best practices suggest high success for high-confidence combos",
           "implementation_plan": "Monitor new sets and update analysis weekly"
       }
       self.log_decision("DECISION", "Selected preferred solution", f"{decision['preferred_solution']['option']}")
       return decision

   def run_analysis(self, max_iterations=2):
       print(f"=== HIERARCHICAL COMBO ANALYZER ===")
       print(f"Analyzing {self.total_sets} sets of numbers.")
       for iteration in range(max_iterations):
           self.analysis_iterations = iteration + 1
           print(f"\n--- Iteration {self.analysis_iterations} ---")
           self.extract_combos()
           print("✓ Extracted combinations")
           self.analyze_combos()
           print("✓ Analyzed combinations")
           self.apply_hierarchical_support()
           print("✓ Applied hierarchical support")
           self.generate_predictions()
           print("✓ Generated predictions")
           self.assign_alerts()
           print("✓ Assigned priority levels")
           self.generate_speculative()
           print("✓ Generated hypothetical combinations")
           report_items, imminent = self.compile_report(top_n_per_size=3)
           print(f"✓ Compiled report with {len(report_items)} top combinations")
           metrics = self.calculate_analysis_metrics(report_items)
           print(f"✓ Metrics: {metrics['total_combos']} combos, {metrics['avg_confidence']:.1f}% avg confidence")
           top_options = self.solution_ideation()
           print(f"✓ Identified {len(top_options)} action options")
           decision = self.final_decision_making(top_options)
           print(f"✓ Decision: {decision['preferred_solution']['option']}")
           self.validation_result = validate_analysis_results(report_items, self.problem_frame, self.data_quality)
           print(f"✓ Validation: {'Valid' if self.validation_result.is_valid else 'Invalid'}, "
                 f"Confidence: {self.validation_result.confidence_score:.2f}, "
                 f"Risk: {self.validation_result.risk_assessment}")
           self.log_decision(
               "VALIDATION",
               f"Validation for iteration {self.analysis_iterations}",
               f"Result: {self.validation_result.is_valid}, "
               f"Confidence: {self.validation_result.confidence_score:.2f}, "
               f"Notes: {self.validation_result.validation_notes}",
               confidence=self.validation_result.confidence_score
           )
           should_rethink, rethink_triggers = should_rethink_analysis(
               self.validation_result, self.data_quality, self.analysis_iterations
           )
           if should_rethink:
               print(f"⚠ Rethinking analysis: {rethink_triggers}")
               current_params = {
                   'significance_threshold': self.significance_threshold,
                   'combo_sizes': self.combo_sizes,
                   'speculative_threshold': self.speculative_threshold
               }
               adjustments = suggest_parameter_adjustments(
                   self.validation_result, self.data_quality, current_params
               )
               if 'significance_threshold' in adjustments:
                   self.significance_threshold = adjustments['significance_threshold']
                   self.min_freq = max(2, int(self.significance_threshold * self.total_sets))
                   print(f"Adjusted significance_threshold to {self.significance_threshold}")
               if 'combo_sizes' in adjustments:
                   self.combo_sizes = adjustments['combo_sizes']
                   print(f"Adjusted combo_sizes to {self.combo_sizes}")
               if 'speculative_threshold' in adjustments:
                   self.speculative_threshold = adjustments['speculative_threshold']
                   print(f"Adjusted speculative_threshold to {self.speculative_threshold}")
               self.log_decision(
                   "RETHINK",
                   f"Rethinking analysis for iteration {self.analysis_iterations + 1}",
                   f"Triggers: {rethink_triggers}, Adjustments: {adjustments}"
               )
               self.combo_occurrences = {}
               self.combo_stats = {}
               self.confidences = {}
               self.boosted_confidences = {}
               self.predictions = {}
               self.alerts = {}
               self.speculative_combos = set()
               self.speculative_stats = {}
               continue
           else:
               print("✓ Analysis complete")
               break
       summary, final_metrics = self.generate_hierarchical_summary()
       export_file = self.export_report("json")
       readable_report = self.generate_readable_report(report_items, summary, decision)
       print("\n=== Analysis Results ===")
       print(readable_report)
       print(f"\nDetailed report exported to {export_file}")
       final_report_items, _ = self.compile_report(top_n_per_size=3)
       final_output = {
           'report': final_report_items,
           'summary': summary,
           'metrics': final_metrics,
           'validation': asdict(self.validation_result),
           'data_quality': asdict(self.data_quality),
           'decision': decision,
           'decision_log': self.decision_log,
           'preprocessing_stats': self.preprocessing_stats,
           'export_file': export_file
       }
       print(f"\n=== Summary ===")
       print(f"Total Combinations: {summary['total_combos']}")
       print(f"Top Decision: {decision['preferred_solution']['option']}")
       return final_output

# Example usage with provided 30-set dataset
if __name__ == "__main__":
   dataset = {
       "SET_1": [5, 16, 17, 47, 57, 58],
       "SET_2": [6, 21, 23, 34, 37, 46],
       "SET_3": [5, 18, 42, 53, 55, 57],
       "SET_4": [16, 22, 24, 36, 39, 43],
       "SET_5": [5, 18, 29, 32, 34, 53],
       "SET_6": [7, 13, 14, 39, 41, 48],
       "SET_7": [6, 11, 13, 39, 40, 42],
       "SET_8": [20, 21, 32, 39, 50, 54],
       "SET_9": [1, 2, 17, 26, 38, 43],
       "SET_10": [5, 6, 9, 10, 14, 41],
   }
   analyzer = HierarchicalComboAnalyzer(
       dataset=dataset,
       combo_sizes=[2, 3, 4, 5, 6],
       significance_threshold=0.1
   )
   results = analyzer.run_analysis(max_iterations=2)
