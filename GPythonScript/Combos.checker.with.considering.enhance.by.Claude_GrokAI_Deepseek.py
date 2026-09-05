import itertools
import statistics
import math
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any
import json

class HierarchicalComboAnalyzer:
   def __init__(self, dataset: Dict[str, List[int]], combo_size: int = 2):
       """
       Enhanced Hierarchical combo pattern analyzer with advanced support checking and predictions
    
       Args:
           dataset: Dictionary with set names as keys and number lists as values
           combo_size: Size of combinations to analyze (2=pairs, 3=triplets, 4=quadruplets, 5=quintuplets, 6=sextuplets)
       """
       # EXPLICIT VALIDATION
       if combo_size not in [2, 3, 4, 5, 6]:
           raise ValueError(f"combo_size must be in [2, 3, 4, 5, 6], got {combo_size}")
         
       self.dataset = dataset
       self.combo_size = combo_size
       self.combo_data = {}
       self.set_indices = {name: idx for idx, name in enumerate(dataset.keys())}
       self.dataset_size = len(dataset)
     
       # Dataset momentum tracking
       self.dataset_momentum = self._calculate_dataset_momentum()
    
       # Store all analyzed combo sizes for hierarchical analysis
       self.all_combo_analyzers = {}
     
       # Prediction models for reliability tracking
       self.prediction_models = ['simple_avg', 'weighted_avg', 'trend_adjusted', 'harmonic_mean', 'hierarchical']
    
       # Enhanced significance thresholds
       if combo_size == 2:
           self.significance_threshold = max(2, int(self.dataset_size * 0.01))
           self.high_significance_threshold = max(3, int(self.dataset_size * 0.015))
       elif combo_size == 3:
           self.significance_threshold = max(2, int(self.dataset_size * 0.003))
           self.high_significance_threshold = max(3, int(self.dataset_size * 0.005))
       elif combo_size == 4:
           self.significance_threshold = 2
           self.high_significance_threshold = 3
       elif combo_size == 5:
           self.significance_threshold = 2
           self.high_significance_threshold = 2
       else: # combo_size == 6
           self.significance_threshold = 1
           self.high_significance_threshold = 2
    
       combo_names = {2: "pairs", 3: "triplets", 4: "quadruplets", 5: "quintuplets", 6: "sextuplets"}
       print(f"Dataset size: {self.dataset_size} sets")
       print(f"Analyzing: {combo_names.get(combo_size, f'{combo_size}-combos')}")
       print(f"Significance threshold: {self.significance_threshold} occurrences")
       print(f"High significance threshold: {self.high_significance_threshold} occurrences")
       print(f"Dataset momentum: {self.dataset_momentum}")
    
   def _calculate_dataset_momentum(self) -> str:
       """Calculate overall dataset momentum"""
       if self.dataset_size < 10:
           return "INSUFFICIENT_DATA"
         
       recent_third = self.dataset_size // 3
       early_sets = list(self.dataset.keys())[:recent_third]
       recent_sets = list(self.dataset.keys())[-recent_third:]
     
       early_numbers = set()
       recent_numbers = set()
     
       for set_name in early_sets:
           early_numbers.update(self.dataset[set_name])
       for set_name in recent_sets:
           recent_numbers.update(self.dataset[set_name])
         
       if len(recent_numbers) > len(early_numbers) * 1.2:
           return "ACCELERATING"
       elif len(recent_numbers) < len(early_numbers) * 0.8:
           return "DECELERATING"
       else:
           return "STABLE"
    
   def set_hierarchical_analyzers(self, analyzers_dict: Dict[int, 'HierarchicalComboAnalyzer']):
       """Set references to other combo size analyzers for hierarchical analysis"""
       self.all_combo_analyzers = analyzers_dict
    
   def extract_combos(self) -> Dict[Tuple, Dict]:
       """Extract all combinations and build comprehensive occurrence database with seasonal analysis"""
       print("\n=== EXTRACTING COMBINATIONS ===")
    
       combo_occurrences = defaultdict(list)
    
       for set_name, numbers in self.dataset.items():
           set_idx = self.set_indices[set_name]
           combos = list(itertools.combinations(sorted(numbers), self.combo_size))
        
           for combo in combos:
               combo_occurrences[combo].append(set_idx)
    
       print(f"Found {len(combo_occurrences)} unique {self.combo_size}-combinations")
    
       significant_count = 0
       for combo, occurrences in combo_occurrences.items():
           if len(occurrences) >= self.significance_threshold:
               intervals = []
               if len(occurrences) > 1:
                   intervals = [occurrences[i] - occurrences[i-1] for i in range(1, len(occurrences))]
             
               seasonal_score = self._calculate_seasonal_score(occurrences)
             
               trend = self._calculate_trend(occurrences, seasonal_score)
             
               confidence = self._calculate_confidence(occurrences, intervals, seasonal_score)
            
               self.combo_data[combo] = {
                   'occurrences': occurrences,
                   'frequency': len(occurrences),
                   'intervals': intervals,
                   'trend': trend,
                   'confidence': confidence,
                   'seasonal_score': seasonal_score,
                   'prediction_accuracy': 0.0,
                   'model_reliability': {model: 0.5 for model in self.prediction_models},
                   'predictions': {},
                   'alert_level': 'GREEN',
                   'combo_size': self.combo_size,
                   'significance_level': 'HIGH' if len(occurrences) >= self.high_significance_threshold else 'NORMAL',
                   'hierarchical_support': {},
                   'boosted_confidence': 0.0,
                   'support_score': 0.0,
                   'support_rank': 0,
                   'cascade_strength': 0.0,
                   'convergence_score': 0.0
               }
               significant_count += 1
    
       print(f"Significant {self.combo_size}-combinations (≥{self.significance_threshold} occurrences): {significant_count}")
       print(f"High significance (≥{self.high_significance_threshold} occurrences): {sum(1 for data in self.combo_data.values() if data['significance_level'] == 'HIGH')}")
    
       return self.combo_data
 
   def _calculate_seasonal_score(self, occurrences: List[int]) -> float:
       """Calculate seasonal pattern score using Fourier analysis"""
       if len(occurrences) < 6:
           return 0.0
         
       try:
           time_series = np.zeros(self.dataset_size)
           for occ in occurrences:
               if occ < len(time_series):
                   time_series[occ] = 1
         
           fft = np.fft.fft(time_series)
           power_spectrum = np.abs(fft) ** 2
         
           if len(power_spectrum) > 2:
               dominant_freq_power = np.max(power_spectrum[1:len(power_spectrum)//2])
               total_power = np.sum(power_spectrum[1:len(power_spectrum)//2])
             
               if total_power > 0:
                   seasonal_strength = dominant_freq_power / total_power
                   return min(1.0, seasonal_strength * 2)
         
           return 0.0
       except:
           return 0.0
   def analyze_hierarchical_dependencies(self):
       """Enhanced hierarchical dependency analysis with cross-level validation"""
       if self.combo_size <= 2:
           print("Skipping hierarchical analysis for pairs (base level)")
           return
        
       print(f"\n=== ENHANCED HIERARCHICAL DEPENDENCY ANALYSIS FOR {self.combo_size}-COMBOS ===")
    
       analyzed_count = 0
       max_support_score = 0.0
     
       for combo, data in self.combo_data.items():
           support_analysis = self._analyze_combo_support_enhanced(combo)
           data['hierarchical_support'] = support_analysis
           data['support_score'] = support_analysis.get('overall_score', 0.0)
           data['cascade_strength'] = support_analysis.get('cascade_strength', 0.0)
           max_support_score = max(max_support_score, data['support_score'])
           analyzed_count += 1
     
       for combo, data in self.combo_data.items():
           if max_support_score > 0:
               data['support_score'] = data['support_score'] / max_support_score
         
           size_adjustment = max(0.3, 1.0 - (self.combo_size - 2) * 0.15)
           data['support_score'] *= size_adjustment
         
       sorted_by_support = sorted(self.combo_data.items(), key=lambda x: x[1]['support_score'], reverse=True)
       for rank, (combo, data) in enumerate(sorted_by_support, 1):
           data['support_rank'] = rank
         
           original_confidence = data['confidence']
           hierarchical_boost = self._calculate_hierarchical_boost_enhanced(
               data['hierarchical_support'], original_confidence, data['cascade_strength'], data['prediction_accuracy']
           )
           data['boosted_confidence'] = min(0.98, original_confidence + hierarchical_boost)
    
       print(f"Analyzed hierarchical dependencies for {analyzed_count} combinations")
       print(f"Average support score: {statistics.mean([d['support_score'] for d in self.combo_data.values()]):.2%}")
   def _analyze_combo_support_enhanced(self, combo: Tuple) -> Dict:
       """Enhanced analysis with new support factors"""
       support_analysis = {
           'sub_combos_analyzed': {},
           'support_strengths': [],
           'overall_score': 0.0,
           'strong_supporters': 0,
           'weak_supporters': 0,
           'cascade_strength': 0.0,
           'mutual_reinforcement': 0.0,
           'contradiction_level': 0.0,
           'temporal_clustering': 0.0,
           'sequential_dependencies': 0.0,
           'gap_pattern_score': 0.0
       }
     
       all_sub_predictions = []
       all_sub_confidences = []
    
       for sub_size in range(2, self.combo_size):
           if sub_size in self.all_combo_analyzers:
               sub_analyzer = self.all_combo_analyzers[sub_size]
               sub_support = self._analyze_sub_combo_support_enhanced(combo, sub_size, sub_analyzer)
               support_analysis['sub_combos_analyzed'][sub_size] = sub_support
             
               for detail in sub_support['support_details']:
                   sub_combo = detail['sub_combo']
                   if sub_combo in sub_analyzer.combo_data:
                       sub_data = sub_analyzer.combo_data[sub_combo]
                       if 'predictions' in sub_data and 'ensemble' in sub_data['predictions']:
                           all_sub_predictions.append(sub_data['predictions']['ensemble'])
                           all_sub_confidences.append(sub_data.get('boosted_confidence', sub_data['confidence']))
            
               if sub_support['average_strength'] > 0:
                   support_analysis['support_strengths'].append(sub_support['average_strength'])
                   if sub_support['average_strength'] >= 0.7:
                       support_analysis['strong_supporters'] += sub_support['found_count']
                   elif sub_support['average_strength'] >= 0.4:
                       support_analysis['weak_supporters'] += sub_support['found_count']
     
       if len(all_sub_predictions) >= 2:
           pred_std = np.std(all_sub_predictions)
           pred_mean = np.mean(all_sub_predictions)
           if pred_mean > 0:
               prediction_consistency = 1.0 - min(1.0, pred_std / (pred_mean * 0.1))
               support_analysis['mutual_reinforcement'] = prediction_consistency
             
               conflicts = sum(1 for p in all_sub_predictions if abs(p - pred_mean) > pred_mean * 0.3)
               conflict_ratio = conflicts / len(all_sub_predictions)
               support_analysis['contradiction_level'] = conflict_ratio
     
       cascade_scores = []
       for sub_size in range(2, self.combo_size):
           if sub_size in support_analysis['sub_combos_analyzed']:
               sub_analysis = support_analysis['sub_combos_analyzed'][sub_size]
               if sub_analysis['found_count'] > 0:
                   cascade_score = sub_analysis['average_strength'] * (sub_analysis['found_count'] / sub_analysis['total_sub_combos'])
                   cascade_scores.append(cascade_score)
     
       if cascade_scores:
           support_analysis['cascade_strength'] = statistics.mean(cascade_scores)
     
       if support_analysis['support_strengths']:
           weights = []
           for size in range(2, self.combo_size):
               base_weight = 1.0 / (size - 1)
               if size in support_analysis['sub_combos_analyzed']:
                   sub_analysis = support_analysis['sub_combos_analyzed'][size]
                   avg_accuracy = 0.5
                   if sub_analysis['support_details']:
                       accuracies = [0.5 for _ in sub_analysis['support_details']]
                       avg_accuracy = statistics.mean(accuracies)
                   adjusted_weight = base_weight * (0.5 + avg_accuracy)
                   weights.append(adjusted_weight)
               else:
                   weights.append(base_weight)
         
           if sum(weights) > 0:
               weighted_avg = sum(strength * weight for strength, weight in zip(support_analysis['support_strengths'], weights)) / sum(weights)
               support_analysis['overall_score'] = max(0.0, min(1.0, weighted_avg))
             
               if support_analysis['mutual_reinforcement'] > 0.7:
                   support_analysis['overall_score'] = min(1.0, support_analysis['overall_score'] * 1.1)
               if support_analysis['contradiction_level'] > 0.3:
                   support_analysis['overall_score'] = max(0.0, support_analysis['overall_score'] * 0.8)
    
       return support_analysis
   def _analyze_sub_combo_support_enhanced(self, combo: Tuple, sub_size: int, sub_analyzer: 'HierarchicalComboAnalyzer') -> Dict:
       """Enhanced sub-combo support analysis with new factors"""
       sub_combos = list(itertools.combinations(combo, sub_size))
    
       analysis = {
           'total_sub_combos': len(sub_combos),
           'found_count': 0,
           'support_details': [],
           'average_strength': 0.0,
           'best_supporter': None,
           'weakest_supporter': None,
           'clustering_coefficient': 0.0,
           'sequential_dependencies': 0.0,
           'gap_pattern_score': 0.0
       }
    
       strengths = []
       all_occurrences = []
     
       for sub_combo in sub_combos:
           if sub_combo in sub_analyzer.combo_data:
               sub_data = sub_analyzer.combo_data[sub_combo]
             
               frequency_strength = min(1.0, sub_data['frequency'] / max(1, self.dataset_size * 0.05))
               confidence_strength = sub_data.get('boosted_confidence', sub_data['confidence'])
               trend_strength = {'INCREASING': 1.0, 'STABLE': 0.8, 'DECREASING': 0.4, 'INSUFFICIENT_DATA': 0.6}.get(sub_data['trend'], 0.5)
             
               last_occurrence = sub_data['occurrences'][-1] if sub_data['occurrences'] else 0
               time_decay = math.exp(-0.01 * (self.dataset_size - last_occurrence))
               recency_strength = time_decay
             
               if len(sub_data['intervals']) > 1:
                   interval_cv = np.std(sub_data['intervals']) / max(1, np.mean(sub_data['intervals']))
                   interval_consistency = max(0.0, 1.0 - interval_cv)
               else:
                   interval_consistency = 0.5
             
               clustering_score = self._calculate_temporal_clustering(sub_data['occurrences'])
             
               dependency_score = self._calculate_sequential_dependency(sub_combo, sub_analyzer)
             
               gap_score = self._calculate_gap_pattern_score(sub_data['occurrences'])
             
               support_strength = (
                   frequency_strength * 0.25 +
                   confidence_strength * 0.25 +
                   trend_strength * 0.20 +
                   recency_strength * 0.15 +
                   interval_consistency * 0.10 +
                   clustering_score * 0.05 +
                   dependency_score * 0.05 -
                   gap_score * 0.05
               )
             
               support_strength = max(0.0, min(1.0, support_strength))
            
               detail = {
                   'sub_combo': sub_combo,
                   'frequency': sub_data['frequency'],
                   'confidence': confidence_strength,
                   'trend': sub_data['trend'],
                   'support_strength': support_strength,
                   'recency_strength': recency_strength,
                   'interval_consistency': interval_consistency,
                   'clustering_score': clustering_score,
                   'gap_score': gap_score
               }
            
               analysis['support_details'].append(detail)
               strengths.append(support_strength)
               all_occurrences.extend(sub_data['occurrences'])
               analysis['found_count'] += 1
            
               if analysis['best_supporter'] is None or support_strength > analysis['best_supporter']['support_strength']:
                   analysis['best_supporter'] = detail
               if analysis['weakest_supporter'] is None or support_strength < analysis['weakest_supporter']['support_strength']:
                   analysis['weakest_supporter'] = detail
     
       if strengths:
           analysis['average_strength'] = statistics.mean(strengths)
         
       if all_occurrences:
           analysis['clustering_coefficient'] = self._calculate_temporal_clustering(sorted(set(all_occurrences)))
           analysis['gap_pattern_score'] = self._calculate_gap_pattern_score(sorted(set(all_occurrences)))
           analysis['sequential_dependencies'] = self._calculate_sequential_dependency_aggregate(all_occurrences)
    
       return analysis
 
   def _calculate_temporal_clustering(self, occurrences: List[int]) -> float:
       """Calculate temporal clustering coefficient"""
       if len(occurrences) < 3:
           return 0.0
         
       total_span = occurrences[-1] - occurrences[0] + 1
       if total_span == 0:
           return 1.0
         
       cluster_threshold = max(3, total_span * 0.1)
       clusters = []
       current_cluster = [occurrences[0]]
     
       for i in range(1, len(occurrences)):
           if occurrences[i] - occurrences[i-1] <= cluster_threshold:
               current_cluster.append(occurrences[i])
           else:
               if len(current_cluster) > 1:
                   clusters.append(current_cluster)
               current_cluster = [occurrences[i]]
     
       if len(current_cluster) > 1:
           clusters.append(current_cluster)
         
       if not clusters:
           return 0.0
         
       max_cluster_size = max(len(cluster) for cluster in clusters)
       return min(1.0, max_cluster_size / len(occurrences))
 
   def _calculate_sequential_dependency(self, sub_combo: Tuple, sub_analyzer: 'HierarchicalComboAnalyzer') -> float:
       """Calculate sequential dependency score"""
       if sub_combo not in sub_analyzer.combo_data:
           return 0.0
         
       occurrences = sub_analyzer.combo_data[sub_combo]['occurrences']
       if len(occurrences) < 2:
           return 0.5
         
       dependency_score = 0.0
       for i in range(len(occurrences) - 1):
           gap = occurrences[i + 1] - occurrences[i]
           if gap <= 3:  # Assuming short gaps indicate dependency
               dependency_score += 1.0
     
       return min(1.0, dependency_score / max(1, len(occurrences) - 1))
 
  
 
   def _calculate_sequential_dependency_aggregate(self, occurrences: List[int]) -> float:
       """Calculate aggregate sequential dependency score"""
       if len(occurrences) < 2:
           return 0.0
         
       gaps = [occurrences[i + 1] - occurrences[i] for i in range(len(occurrences) - 1)]
       short_gaps = sum(1 for gap in gaps if gap <= 3)
       return min(1.0, short_gaps / max(1, len(gaps)))
 
   def _calculate_gap_pattern_score(self, occurrences: List[int]) -> float:
       """Calculate gap pattern penalty score"""
       if len(occurrences) < 3:
           return 0.0
         
       intervals = [occurrences[i] - occurrences[i-1] for i in range(1, len(occurrences))]
       if not intervals:
           return 0.0
         
       avg_interval = statistics.mean(intervals)
       large_gaps = [gap for gap in intervals if gap > avg_interval * 2]
     
       gap_penalty = len(large_gaps) / len(intervals)
       return min(1.0, gap_penalty)
   def _calculate_hierarchical_boost_enhanced(self, support_analysis: Dict, original_confidence: float, cascade_strength: float, prediction_accuracy: float) -> float:
       """Enhanced hierarchical boost calculation with new factors"""
       if not support_analysis or support_analysis['overall_score'] == 0:
           return 0.0
    
       base_boost = support_analysis['overall_score'] * 0.3
       strong_support_bonus = min(0.15, support_analysis['strong_supporters'] * 0.03)
       weak_support_penalty = support_analysis['weak_supporters'] * 0.01
       cascade_bonus = cascade_strength * 0.1
       accuracy_bonus = prediction_accuracy * 0.05
    
       total_boost = base_boost + strong_support_bonus + cascade_bonus + accuracy_bonus - weak_support_penalty
       max_boost = (1.0 - original_confidence) * 0.4
    
       return min(total_boost, max_boost, 0.35)
   def _calculate_trend(self, occurrences: List[int], seasonal_score: float = 0.0) -> str:
       """Enhanced trend calculation with seasonal adjustment"""
       if len(occurrences) < 3:
           return 'INSUFFICIENT_DATA'
    
       third = len(occurrences) // 3
       if third == 0:
           return 'INSUFFICIENT_DATA'
        
       first_third_indices = occurrences[:third]
       last_third_indices = occurrences[-third:]

       # Handle case where first or last third might be empty
       if not first_third_indices or not last_third_indices:
           return 'INSUFFICIENT_DATA'
    
       first_span = occurrences[third-1] - occurrences[0] + 1 if third > 0 else 1
       last_span = occurrences[-1] - occurrences[-third] + 1

       # Prevent division by zero
       if first_span == 0 or last_span == 0:
         return 'INSUFFICIENT_DATA'
    
       first_density = len(first_third_indices) / first_span
       last_density = len(last_third_indices) / last_span
     
       seasonal_adjustment = 1.0 + seasonal_score * 0.2
     
       increasing_threshold = 1.3 / seasonal_adjustment
       decreasing_threshold = 0.7 * seasonal_adjustment
    
       if last_density > first_density * increasing_threshold:
           return 'INCREASING'
       elif last_density < first_density * decreasing_threshold:
           return 'DECREASING'
       else:
           return 'STABLE'
   def _calculate_confidence(self, occurrences: List[int], intervals: List[int], seasonal_score: float = 0.0) -> float:
       """Enhanced confidence calculation with seasonal bonus"""
       if len(occurrences) < 2:
           return 0.2
    
       max_expected_freq = {2: 30, 3: 8, 4: 4, 5: 3, 6: 2}.get(self.combo_size, 2)
       frequency_score = min(len(occurrences) / max_expected_freq, 1.0)
    
       if len(intervals) >= 2:
           mean_interval = statistics.mean(intervals)
           if mean_interval > 0:
               cv = statistics.stdev(intervals) / mean_interval
               regularity_score = max(0, 1.0 - cv)
           else:
               regularity_score = 0.5
       else:
           regularity_score = 0.3
    
       last_occurrence_position = occurrences[-1] / (self.dataset_size - 1)
       recent_bonus = 1.0 if last_occurrence_position >= 0.8 else 0.9 if last_occurrence_position >= 0.6 else 0.8
     
       seasonal_bonus = seasonal_score
    
       size_penalty = max(0.6, 1.0 - (self.combo_size - 2) * 0.1)
    
       confidence = (
           frequency_score * 0.35 +
           regularity_score * 0.35 +
           recent_bonus * 0.20 +
           seasonal_bonus * 0.10
       ) * size_penalty
    
       return round(min(confidence, 0.95), 3)
   def generate_predictions(self) -> Dict:
       """Enhanced prediction generation with dynamic weighting and momentum adjustment"""
       print("\n=== GENERATING ENHANCED PREDICTIONS ===")
    
       current_set = self.dataset_size - 1
       predictions_generated = 0
    
       for combo, data in self.combo_data.items():
           if len(data['intervals']) > 0:
               predictions = {}
            
               avg_interval = statistics.mean(data['intervals'])
               predictions['simple_avg'] = data['occurrences'][-1] + avg_interval
            
               if len(data['intervals']) >= 2:
                   weights = [1.5 ** i for i in range(len(data['intervals']))]
                   weighted_avg = sum(interval * weight for interval, weight in zip(data['intervals'], weights)) / sum(weights)
                   predictions['weighted_avg'] = data['occurrences'][-1] + weighted_avg
               else:
                   predictions['weighted_avg'] = predictions['simple_avg']
            
               if len(data['occurrences']) >= 3:
                   recent_intervals = data['intervals'][-3:] if len(data['intervals']) >= 3 else data['intervals']
                   if len(recent_intervals) > 1:
                       trend = (recent_intervals[-1] - recent_intervals[0]) / len(recent_intervals)
                       trend_adjusted = avg_interval + trend
                       predictions['trend_adjusted'] = data['occurrences'][-1] + max(1, trend_adjusted)
                   else:
                       predictions['trend_adjusted'] = predictions['simple_avg']
               else:
                   predictions['trend_adjusted'] = predictions['simple_avg']
            
               if len(data['intervals']) >= 2:
                   positive_intervals = [i for i in data['intervals'] if i > 0]
                   if positive_intervals:
                       harmonic_mean = len(positive_intervals) / sum(1/i for i in positive_intervals)
                       predictions['harmonic_mean'] = data['occurrences'][-1] + harmonic_mean
                   else:
                       predictions['harmonic_mean'] = predictions['simple_avg']
               else:
                   predictions['harmonic_mean'] = predictions['simple_avg']
            
               if self.combo_size > 2 and data.get('hierarchical_support'):
                   hierarchical_pred = self._generate_hierarchical_prediction_enhanced(combo, data, predictions)
                   if hierarchical_pred:
                       predictions['hierarchical'] = hierarchical_pred
            
               model_reliability = data.get('model_reliability', {model: 0.5 for model in self.prediction_models})
             
               if 'hierarchical' in predictions:
                   support_strength = data.get('support_score', 0.0)
                   cascade_strength = data.get('cascade_strength', 0.0)
                 
                   hierarchical_weight = min(0.5, (support_strength + cascade_strength) * 0.3)
                 
                   remaining_weight = 1.0 - hierarchical_weight
                   weights = {
                       'hierarchical': hierarchical_weight,
                       'simple_avg': remaining_weight * 0.20 * model_reliability.get('simple_avg', 0.5),
                       'weighted_avg': remaining_weight * 0.35 * model_reliability.get('weighted_avg', 0.5),
                       'trend_adjusted': remaining_weight * 0.25 * model_reliability.get('trend_adjusted', 0.5),
                       'harmonic_mean': remaining_weight * 0.20 * model_reliability.get('harmonic_mean', 0.5)
                   }
                 
                   total_weight = sum(weights.values())
                   if total_weight > 0:
                       weights = {k: v/total_weight for k, v in weights.items()}
               else:
                   base_weights = {'simple_avg': 0.25, 'weighted_avg': 0.35, 'trend_adjusted': 0.25, 'harmonic_mean': 0.15}
                   weights = {model: weight * model_reliability.get(model, 0.5) for model, weight in base_weights.items() if model in predictions}
                 
                   total_weight = sum(weights.values())
                   if total_weight > 0:
                       weights = {k: v/total_weight for k, v in weights.items()}
            
               ensemble_pred = sum(predictions[model] * weight for model, weight in weights.items() if model in predictions)
             
               momentum_adjustment = self._calculate_momentum_adjustment(ensemble_pred, data)
               ensemble_pred += momentum_adjustment
             
               predictions['ensemble'] = ensemble_pred
            
               pred_values = [predictions[model] for model in ['simple_avg', 'weighted_avg', 'trend_adjusted', 'harmonic_mean'] if model in predictions]
               if len(pred_values) > 1:
                   pred_weights = [model_reliability.get(model, 0.5) for model in ['simple_avg', 'weighted_avg', 'trend_adjusted', 'harmonic_mean'] if model in predictions]
                   if sum(pred_weights) > 0:
                       pred_weights = [w/sum(pred_weights) for w in pred_weights]
                       weighted_mean = sum(p*w for p, w in zip(pred_values, pred_weights))
                       weighted_variance = sum(w * (p - weighted_mean)**2 for p, w in zip(pred_values, pred_weights))
                       predictions['std_dev'] = math.sqrt(weighted_variance)
                   else:
                       predictions['std_dev'] = statistics.stdev(pred_values)
               else:
                   predictions['std_dev'] = avg_interval * 0.2
             
               if len(pred_values) > 1:
                   pred_range = max(pred_values) - min(pred_values)
                   mean_pred = statistics.mean(pred_values)
                   if mean_pred > 0:
                       convergence_score = max(0.0, 1.0 - (pred_range / (mean_pred * 0.05)))
                       data['convergence_score'] = convergence_score
                     
                       if convergence_score > 0.8:
                           data['boosted_confidence'] = min(0.98, data.get('boosted_confidence', data['confidence']) * 1.1)
             
               predictions['min_prediction'] = min(pred_values) if pred_values else ensemble_pred - predictions['std_dev']
               predictions['max_prediction'] = max(pred_values) if pred_values else ensemble_pred + predictions['std_dev']
            
               data['predictions'] = predictions
             
               self._update_model_reliability(data, pred_values, ensemble_pred)
            
               alert_confidence = data.get('boosted_confidence', data['confidence'])
               support_score = data.get('support_score', 0.0)
               trend_momentum = self._calculate_trend_momentum(data)
               prediction_accuracy = data.get('prediction_accuracy', 0.0)
               convergence_score = data.get('convergence_score', 0.0)
             
               data['alert_level'] = self._calculate_alert_level_enhanced(
                   ensemble_pred, current_set, alert_confidence, support_score,
                   trend_momentum, prediction_accuracy, convergence_score, pred_values
               )
               predictions_generated += 1
    
       print(f"Generated predictions for {predictions_generated} combinations")
       if self.combo_size > 2:
           hierarchical_count = sum(1 for data in self.combo_data.values() if 'hierarchical' in data.get('predictions', {}))
           enhanced_count = sum(1 for data in self.combo_data.values() if data.get('boosted_confidence', 0) > data['confidence'])
           print(f"Enhanced {hierarchical_count} predictions with hierarchical analysis")
           print(f"Boosted confidence for {enhanced_count} combinations")
    
       return self.combo_data
 
   def _calculate_momentum_adjustment(self, prediction: float, data: Dict) -> float:
       """Calculate momentum adjustment based on dataset trends"""
       base_adjustment = 0.0
     
       if self.dataset_momentum == "ACCELERATING":
           base_adjustment = -1.0
       elif self.dataset_momentum == "DECELERATING":
           base_adjustment = 1.0
     
       confidence = data.get('boosted_confidence', data['confidence'])
       trend_multiplier = {'INCREASING': 1.2, 'STABLE': 1.0, 'DECREASING': 0.8, 'INSUFFICIENT_DATA': 0.9}.get(data['trend'], 1.0)
     
       return base_adjustment * confidence * trend_multiplier
 
   def _calculate_trend_momentum(self, data: Dict) -> float:
       """Calculate trend momentum score for alert calculation"""
       trend = data.get('trend', 'INSUFFICIENT_DATA')
       intervals = data.get('intervals', [])
     
       base_score = {'INCREASING': 0.8, 'STABLE': 0.5, 'DECREASING': 0.2, 'INSUFFICIENT_DATA': 0.4}.get(trend, 0.4)
     
       if len(intervals) >= 3:
           recent_trend = intervals[-1] - intervals[0] if len(intervals) > 1 else 0
           if recent_trend < 0:
               base_score += 0.2
           elif recent_trend > 0:
               base_score -= 0.1
     
       return max(0.0, min(1.0, base_score))
 
   def _update_model_reliability(self, data: Dict, pred_values: List[float], ensemble_pred: float):
       """Update model reliability based on prediction performance"""
       if not pred_values or len(pred_values) < 2:
           return
         
       model_names = ['simple_avg', 'weighted_avg', 'trend_adjusted', 'harmonic_mean']
       model_reliability = data.get('model_reliability', {})
     
       for i, model in enumerate(model_names):
           if i < len(pred_values) and model in data.get('predictions', {}):
               prediction_error = abs(pred_values[i] - ensemble_pred)
               max_error = max(abs(p - ensemble_pred) for p in pred_values) or 1
             
               accuracy = 1.0 - (prediction_error / max_error) if max_error > 0 else 1.0
             
               learning_rate = 0.1
               current_reliability = model_reliability.get(model, 0.5)
               model_reliability[model] = current_reliability * (1 - learning_rate) + accuracy * learning_rate
             
       data['model_reliability'] = model_reliability
 
   def _calculate_alert_level_enhanced(self, prediction: float, current_set: int, confidence: float,
                                     support_score: float, trend_momentum: float, prediction_accuracy: float,
                                     convergence_score: float, pred_values: List[float]) -> str:
       """Enhanced alert level calculation with multi-factor scoring"""
       sets_until_prediction = prediction - current_set
     
       proximity_score = max(0.0, 1.0 - (sets_until_prediction / 20))
       support_factor = support_score if self.combo_size > 2 else 0.5
       trend_factor = trend_momentum
       accuracy_factor = prediction_accuracy
       convergence_factor = convergence_score
     
       multi_factor_score = (
           proximity_score * 0.3 +
           confidence * 0.25 +
           support_factor * 0.2 +
           trend_factor * 0.15 +
           (accuracy_factor + convergence_factor) * 0.1
       )
     
       has_consensus = False
       if len(pred_values) >= 2:
           pred_mean = statistics.mean(pred_values)
           consensus_count = sum(1 for p in pred_values if abs(p - pred_mean) <= pred_mean * 0.1)
           has_consensus = consensus_count >= max(2, len(pred_values) * 0.6)
     
       if multi_factor_score > 0.8 and (has_consensus or self.combo_size == 2):
           return 'RED'
       elif multi_factor_score > 0.6 and (has_consensus or self.combo_size == 2):
           return 'ORANGE'
       elif multi_factor_score > 0.4:
           return 'YELLOW'
       else:
           return 'GREEN'
   def _generate_hierarchical_prediction_enhanced(self, combo: Tuple, combo_data: Dict, base_predictions: Dict) -> float:
       """Enhanced hierarchical prediction with recency weighting and accuracy tracking"""
       support_analysis = combo_data.get('hierarchical_support', {})
       if not support_analysis or not support_analysis.get('sub_combos_analyzed'):
           return None
    
       hierarchical_predictions = []
       confidence_weights = []
    
       for sub_size, sub_analysis in support_analysis['sub_combos_analyzed'].items():
           if sub_size in self.all_combo_analyzers and sub_analysis['found_count'] > 0:
               sub_analyzer = self.all_combo_analyzers[sub_size]
               for detail in sub_analysis['support_details']:
                   sub_combo = detail['sub_combo']
                   if sub_combo in sub_analyzer.combo_data:
                       sub_data = sub_analyzer.combo_data[sub_combo]
                       if 'predictions' in sub_data and 'ensemble' in sub_data['predictions']:
                           sub_prediction = sub_data['predictions']['ensemble']
                           sub_confidence = sub_data.get('boosted_confidence', sub_data['confidence'])
                           support_strength = detail['support_strength']
                           prediction_accuracy = sub_data.get('prediction_accuracy', 0.5)
                         
                           last_occurrence = sub_data['occurrences'][-1] if sub_data['occurrences'] else 0
                           recency_weight = math.exp(-0.01 * (self.dataset_size - last_occurrence))
                         
                           weight = max(0.001, sub_confidence * support_strength *
                                      (0.5 + prediction_accuracy * 0.3) * (0.7 + recency_weight * 0.3))
                         
                           if weight > 0:
                               hierarchical_predictions.append(sub_prediction)
                               confidence_weights.append(weight)
    
       if hierarchical_predictions and confidence_weights and sum(confidence_weights) > 0:
           total_weight = sum(confidence_weights)
           weighted_prediction = sum(pred * weight for pred, weight in zip(hierarchical_predictions, confidence_weights)) / total_weight
        
           base_ensemble = base_predictions.get('weighted_avg', base_predictions.get('simple_avg', 0))
           support_strength = support_analysis.get('overall_score', 0.0)
           cascade_strength = support_analysis.get('cascade_strength', 0.0)
         
           hierarchical_weight = min(0.7, max(0.1, (support_strength + cascade_strength * 0.5)))
           blended_prediction = (hierarchical_weight * weighted_prediction +
                               (1 - hierarchical_weight) * base_ensemble)
        
           return blended_prediction
    
       return None
   def generate_speculative_combos(self, target_size: int, min_support_score: float = 0.5, max_candidates: int = 50) -> Dict[Tuple, Dict]:
       """Enhanced speculative combo generation with adaptive thresholds and validation"""
       if target_size not in [4, 5, 6]:
           print(f"Speculative combos only supported for quadruplets (4), quintuplets (5), or sextuplets (6), got {target_size}")
           return {}
       print(f"\n=== GENERATING ENHANCED SPECULATIVE {target_size}-COMBOS ===")
     
      
     
       total_significant = sum(len(analyzer.combo_data) for analyzer in self.all_combo_analyzers.values())
       total_possible = sum(len(list(itertools.combinations(range(60), size))) for size in range(2, 7))
     
       if total_significant > total_possible * 0.1:
           adjusted_min_support = min_support_score + 0.1
           print(f"High combo density detected, increasing min_support_score to {adjusted_min_support:.1%}")
       else:
           adjusted_min_support = max(0.3, min_support_score - 0.1)
           print(f"Low combo density detected, decreasing min_support_score to {adjusted_min_support:.1%}")
     
       speculative_combos = {}
    
       sub_combos = {}
       for sub_size in range(2, min(target_size, 6)):
           if sub_size in self.all_combo_analyzers:
               analyzer = self.all_combo_analyzers[sub_size]
               sub_combos[sub_size] = [
                   (combo, data) for combo, data in analyzer.combo_data.items()
                   if data.get('boosted_confidence', data['confidence']) >= 0.6 or
                      data['frequency'] >= analyzer.high_significance_threshold
               ]
               print(f"Found {len(sub_combos[sub_size])} significant {sub_size}-combos for speculative analysis")
       if not sub_combos.get(2):
           print("No significant pairs found for speculative analysis")
           return {}
     
       candidate_combos = set()
       all_numbers = set()
       for numbers in self.dataset.values():
           all_numbers.update(numbers)
    
       if target_size == 4:
           for (pair1, pair1_data), (pair2, pair2_data) in itertools.combinations(sub_combos[2], 2):
               candidate = tuple(sorted(set(pair1 + pair2)))
               if len(candidate) == 4 and candidate not in self.combo_data:
                   candidate_combos.add(candidate)
    
       elif target_size == 5:
           if sub_combos.get(3):
               for (triplet, triplet_data) in sub_combos[3]:
                   for (pair, pair_data) in sub_combos[2]:
                       candidate = tuple(sorted(set(triplet + pair)))
                       if len(candidate) == 5 and candidate not in self.combo_data:
                           candidate_combos.add(candidate)
           for (pair1, _), (pair2, _), (pair3, _) in itertools.combinations(sub_combos[2], 3):
               candidate = tuple(sorted(set(pair1 + pair2 + pair3)))
               if len(candidate) == 5 and candidate not in self.combo_data:
                   candidate_combos.add(candidate)
    
       elif target_size == 6:
           if sub_combos.get(5):
               for (quint, quint_data) in sub_combos[5]:
                   for num in all_numbers:
                       candidate = tuple(sorted(set(quint + (num,))))
                       if len(candidate) == 6 and candidate not in self.combo_data:
                           candidate_combos.add(candidate)
           if sub_combos.get(4):
               for (quad, quad_data) in sub_combos[4]:
                   for (pair, pair_data) in sub_combos[2]:
                       candidate = tuple(sorted(set(quad + pair)))
                       if len(candidate) == 6 and candidate not in self.combo_data:
                           candidate_combos.add(candidate)
           if sub_combos.get(3):
               for (trip1, trip1_data), (trip2, trip2_data) in itertools.combinations(sub_combos[3], 2):
                   candidate = tuple(sorted(set(trip1 + trip2)))
                   if len(candidate) == 6 and candidate not in self.combo_data:
                       candidate_combos.add(candidate)
       print(f"Generated {len(candidate_combos)} speculative {target_size}-combo candidates")
     
       analyzed_count = 0
       for combo in list(candidate_combos)[:max_candidates]:
           support_analysis = self._analyze_combo_support_enhanced(combo)
           support_score = support_analysis.get('overall_score', 0.0)
        
           if support_score >= adjusted_min_support:
               speculative_pred = self._generate_speculative_prediction_enhanced(combo, support_analysis)
             
               historical_accuracy = self._validate_speculative_accuracy(target_size)
               confidence = min(0.8, support_score * 0.7 * historical_accuracy)
             
               support_rank = analyzed_count + 1
             
               speculative_combos[combo] = {
                   'occurrences': [],
                   'frequency': 0,
                   'intervals': [],
                   'trend': 'SPECULATIVE',
                   'confidence': confidence,
                   'predictions': {'ensemble': speculative_pred} if speculative_pred else {},
                   'alert_level': 'YELLOW' if speculative_pred and speculative_pred - (self.dataset_size - 1) <= 12 else 'GREEN',
                   'combo_size': target_size,
                   'significance_level': 'SPECULATIVE',
                   'hierarchical_support': support_analysis,
                   'support_score': support_score,
                   'support_rank': support_rank,
                   'boosted_confidence': confidence,
                   'convergence_score': 0.7,
                   'prediction_accuracy': historical_accuracy
               }
               analyzed_count += 1
     
       sorted_speculative = sorted(speculative_combos.items(), key=lambda x: x[1]['support_score'], reverse=True)
       for rank, (combo, data) in enumerate(sorted_speculative, 1):
           data['support_rank'] = rank
       print(f"Analyzed {analyzed_count} speculative {target_size}-combos with sufficient support (≥{adjusted_min_support:.1%})")
       return speculative_combos
 
   def _generate_speculative_prediction_enhanced(self, combo: Tuple, support_analysis: Dict) -> float:
       """Enhanced speculative prediction with accuracy and recency weighting"""
       predictions = []
       weights = []
    
       for sub_size, sub_analysis in support_analysis['sub_combos_analyzed'].items():
           if sub_size in self.all_combo_analyzers and sub_analysis['found_count'] > 0:
               sub_analyzer = self.all_combo_analyzers[sub_size]
               for detail in sub_analysis['support_details']:
                   sub_combo = detail['sub_combo']
                   if sub_combo in sub_analyzer.combo_data:
                       sub_data = sub_analyzer.combo_data[sub_combo]
                       if 'predictions' in sub_data and 'ensemble' in sub_data['predictions']:
                           prediction_accuracy = sub_data.get('prediction_accuracy', 0.5)
                           recency_weight = detail.get('recency_strength', 0.5)
                         
                           predictions.append(sub_data['predictions']['ensemble'])
                           weights.append(detail['support_strength'] * sub_data.get('boosted_confidence', sub_data['confidence']) *
                                        (0.5 + prediction_accuracy * 0.3) * (0.7 + recency_weight * 0.3))
       if predictions and weights and sum(weights) > 0:
           return sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
       return None
 
   def _validate_speculative_accuracy(self, target_size: int) -> float:
       """Validate against historical speculative accuracy"""
       base_accuracy = {4: 0.7, 5: 0.6, 6: 0.5}.get(target_size, 0.5)
     
       if self.dataset_momentum == "ACCELERATING":
           base_accuracy *= 1.1
       elif self.dataset_momentum == "DECELERATING":
           base_accuracy *= 0.9
         
       return min(1.0, base_accuracy)
   def display_results(self, show_all: bool = False, max_display: int = 20, speculative_combos: Dict = None):
       """Enhanced display with new metrics"""
       combo_names = {2: "PAIRS", 3: "TRIPLETS", 4: "QUADRUPLETS", 5: "QUINTUPLETS", 6: "SEXTUPLETS"}
       combo_name = combo_names.get(self.combo_size, f"{self.combo_size}-COMBOS")
    
       print("\n" + "="*90)
       print(f"ENHANCED {combo_name} ANALYSIS RESULTS")
       print("="*90)
    
       if not self.combo_data:
           print(f"No significant {combo_name.lower()} found!")
       else:
           alert_priority = {'RED': 4, 'ORANGE': 3, 'YELLOW': 2, 'GREEN': 1}
           significance_priority = {'HIGH': 2, 'NORMAL': 1}
        
           sorted_combos = sorted(
               self.combo_data.items(),
               key=lambda x: (
                   alert_priority.get(x[1]['alert_level'], 0),
                   significance_priority.get(x[1]['significance_level'], 0),
                   x[1].get('support_score', 0.0),
                   -x[1].get('support_rank', float('inf')),
                   x[1].get('boosted_confidence', x[1]['confidence']),
                   x[1].get('convergence_score', 0.0),
                   x[1]['frequency']
               ),
               reverse=True
           )
        
           if not show_all:
               high_priority = [item for item in sorted_combos if item[1]['alert_level'] in ['RED', 'ORANGE']]
               other_items = [item for item in sorted_combos if item[1]['alert_level'] not in ['RED', 'ORANGE']]
               display_items = high_priority + other_items[:max_display - len(high_priority)]
           else:
               display_items = sorted_combos[:max_display]
        
           for combo, data in display_items:
               alert_icon = {'RED': '🚨', 'ORANGE': '⚠️', 'YELLOW': '⚡', 'GREEN': '✅'}
               significance_icon = {'HIGH': '⭐', 'NORMAL': '📊'}
            
               support_score = data.get('support_score', 0.0)
               support_rank = data.get('support_rank', 0)
               support_icon = '🔥' if support_score >= 0.8 else '💪' if support_score >= 0.6 else '👍' if support_score >= 0.4 else '👌'
            
               print(f"\n{alert_icon.get(data['alert_level'], '📊')} {significance_icon.get(data['significance_level'], '')} {support_icon} {combo_name[:-1]}: {combo}")
             
               convergence = data.get('convergence_score', 0.0)
               prediction_accuracy = data.get('prediction_accuracy', 0.0)
               print(f" Alert: {data['alert_level']} | Significance: {data['significance_level']} | Support: {support_score:.1%} (#{support_rank})")
               print(f" Frequency: {data['frequency']} occurrences ({data['frequency']/self.dataset_size*100:.2f}%)")
            
               original_conf = data['confidence']
               boosted_conf = data.get('boosted_confidence', original_conf)
               if boosted_conf > original_conf:
                   boost_amount = boosted_conf - original_conf
                   print(f" Confidence: {original_conf:.1%} → {boosted_conf:.1%} (+{boost_amount:.1%} boost)")
               else:
                   print(f" Confidence: {original_conf:.1%}")
             
               if convergence > 0:
                   print(f" Convergence: {convergence:.1%} | Prediction Accuracy: {prediction_accuracy:.1%}")
            
               if self.combo_size > 2 and data.get('hierarchical_support'):
                   self._display_hierarchical_support_enhanced(data['hierarchical_support'])
            
               recent_sets = [f'SET_{i+1}' for i in data['occurrences'][-5:]]
               print(f" Recent sets: {', '.join(recent_sets)}")
            
               if data['intervals']:
                   print(f" Intervals: {data['intervals'][-5:]} (last 5)")
                   print(f" Avg interval: {statistics.mean(data['intervals']):.1f} sets")
                   print(f" Trend: {data['trend']}")
                 
                   seasonal_score = data.get('seasonal_score', 0.0)
                   if seasonal_score > 0.3:
                       print(f" Seasonal pattern: {seasonal_score:.1%}")
            
               if 'predictions' in data and data['predictions']:
                   pred = data['predictions']
                   next_set = int(pred['ensemble']) + 1
                   uncertainty = pred['std_dev']
                   range_min = max(self.dataset_size + 1, int(pred['min_prediction']) + 1)
                   range_max = int(pred['max_prediction']) + 1
                
                   print(f" 📈 Next predicted: SET_{next_set} (±{uncertainty:.1f})")
                   print(f" 📊 Range: SET_{range_min} to SET_{range_max}")
                
                   if 'hierarchical' in pred:
                       hierarchical_set = int(pred['hierarchical']) + 1
                       print(f" 🔗 Hierarchical model: SET_{hierarchical_set}")
    
       if speculative_combos:
           print(f"\n{'-'*60}")
           print(f"ENHANCED SPECULATIVE {combo_name} (NEVER OBSERVED)")
           print(f"{'-'*60}")
        
           sorted_speculative = sorted(
               speculative_combos.items(),
               key=lambda x: (-x[1]['support_score'], -x[1].get('support_rank', 0), -x[1]['confidence']),
               reverse=False
           )[:max_display]
        
           for combo, data in sorted_speculative:
               support_score = data['support_score']
               support_rank = data.get('support_rank', 0)
               convergence = data.get('convergence_score', 0.0)
             
               support_icon = '🔥' if support_score >= 0.8 else '💪' if support_score >= 0.6 else '👍' if support_score >= 0.4 else '👌'
               print(f"\n{support_icon} SPECULATIVE {combo_name[:-1]}: {combo}")
               print(f" Support: {support_score:.1%} (#{support_rank}) | Confidence: {data['confidence']:.1%} | Convergence: {convergence:.1%}")
             
               self._display_hierarchical_support_enhanced(data['hierarchical_support'])
             
               if 'ensemble' in data['predictions']:
                   next_set = int(data['predictions']['ensemble']) + 1
                   print(f" 📈 Predicted: SET_{next_set}")
    
       self._display_summary_enhanced()
 
   def _display_hierarchical_support_enhanced(self, support_analysis: Dict):
       """Enhanced hierarchical support display with new metrics"""
       if not support_analysis.get('sub_combos_analyzed'):
           return
    
       print(f" 🔗 Enhanced Hierarchical Support:")
       for size, sub_analysis in support_analysis['sub_combos_analyzed'].items():
           combo_name = {2: "Pairs", 3: "Triplets", 4: "Quadruplets", 5: "Quintuplets"}.get(size, f"{size}-combos")
           found = sub_analysis['found_count']
           total = sub_analysis['total_sub_combos']
           avg_strength = sub_analysis['average_strength']
           clustering = sub_analysis.get('clustering_coefficient', 0.0)
           seq_deps = sub_analysis.get('sequential_dependencies', 0.0)
           gap_score = sub_analysis.get('gap_pattern_score', 0.0)
         
           print(f"   {combo_name}: {found}/{total} found, avg strength: {avg_strength:.1%}")
           print(f"   Clustering: {clustering:.1%} | Seq Dependencies: {seq_deps:.1%} | Gap Penalty: {gap_score:.1%}")
         
           if found > 0:
               best = sub_analysis.get('best_supporter')
               if best:
                   print(f"   Best Supporter: {best['sub_combo']} (strength: {best['support_strength']:.1%})")
               weakest = sub_analysis.get('weakest_supporter')
               if weakest:
                   print(f"   Weakest Supporter: {weakest['sub_combo']} (strength: {weakest['support_strength']:.1%})")
    
       overall_score = support_analysis.get('overall_score', 0.0)
       cascade = support_analysis.get('cascade_strength', 0.0)
       mutual = support_analysis.get('mutual_reinforcement', 0.0)
       contradiction = support_analysis.get('contradiction_level', 0.0)
       print(f"   Overall Support: {overall_score:.1%}")
       print(f"   Cascade Strength: {cascade:.1%} | Mutual Reinforcement: {mutual:.1%} | Contradiction: {contradiction:.1%}")
   def _display_summary_enhanced(self):
       """Enhanced summary with new metrics"""
       print("\n" + "="*60)
       print(f"ENHANCED SUMMARY FOR {self.combo_size}-COMBOS")
       print("="*60)
    
       alert_counts = Counter(data['alert_level'] for data in self.combo_data.values())
       high_significance = sum(1 for data in self.combo_data.values() if data['significance_level'] == 'HIGH')

       # Handle empty lists for statistics
       support_scores = [data['support_score'] for data in self.combo_data.values()]
       convergence_scores = [data['convergence_score'] for data in self.combo_data.values()]
       accuracy_scores = [data['prediction_accuracy'] for data in self.combo_data.values()]

       avg_support = statistics.mean([data['support_score'] for data in self.combo_data.values()] or [0])
       avg_convergence = statistics.mean([data['convergence_score'] for data in self.combo_data.values()] or [0])
       avg_accuracy = statistics.mean([data['prediction_accuracy'] for data in self.combo_data.values()] or [0])
    
       print(f"Total significant combos: {len(self.combo_data)}")
       print(f"High significance: {high_significance}")
       print(f"Alert distribution: {dict(alert_counts)}")
       print(f"Average support score: {avg_support:.1%}")
       print(f"Average convergence score: {avg_convergence:.1%}")
       print(f"Average prediction accuracy: {avg_accuracy:.1%}")
    
       if self.combo_size > 2:
           enhanced_count = sum(1 for data in self.combo_data.values() if data.get('boosted_confidence', 0) > data['confidence'])
           strong_support = sum(1 for data in self.combo_data.values() if data.get('support_score', 0) >= 0.7)
           print(f"Hierarchically enhanced: {enhanced_count}")
           print(f"Strong support (≥70%): {strong_support}")
   
   def get_imminent_combos(self, max_sets_ahead: int = 10) -> List[Dict]:
       """Get combos predicted to occur soon with enhanced metrics"""
       imminent = []
    
       for combo, data in self.combo_data.items():
           if 'predictions' in data and 'ensemble' in data['predictions']:
               sets_ahead = data['predictions']['ensemble'] - (self.dataset_size - 1)
               if 0 < sets_ahead <= max_sets_ahead:
                   imminent.append({
                       'combo': combo,
                       'sets_ahead': sets_ahead,
                       'confidence': data.get('boosted_confidence', data['confidence']),
                       'alert_level': data['alert_level'],
                       'support_score': data.get('support_score', 0.0),
                       'support_rank': data.get('support_rank', 0),
                       'convergence_score': data.get('convergence_score', 0.0)
                   })
    
       imminent.sort(key=lambda x: (x['sets_ahead'], -x['convergence_score'], -x['support_score'], x['support_rank'], -x['confidence']))
    
       print(f"\nFound {len(imminent)} imminent {self.combo_size}-combos within {max_sets_ahead} sets")
       return imminent

def main():
   """Main function to demonstrate the analyzer"""
   sample_data = {
"SET_1": [1, 7, 9, 12, 33, 38],
"SET_2": [4, 21, 22, 32, 33, 34],
"SET_3": [2, 6, 16, 22, 25, 36],
"SET_4": [11, 16, 33, 38, 41, 42],
"SET_5": [5, 13, 17, 22, 23, 25],
"SET_6": [21, 22, 27, 33, 35, 36],
"SET_7": [7, 8, 24, 26, 27, 31],
"SET_8": [6, 9, 11, 28, 30, 37],
"SET_9": [7, 13, 14, 16, 22, 36],
"SET_10": [3, 6, 19, 29, 37, 38],
"SET_11": [7, 12, 15, 16, 18, 35],
"SET_12": [7, 11, 25, 27, 30, 40],
"SET_13": [1, 6, 11, 29, 31, 40],
"SET_14": [18, 20, 21, 25, 27, 33],
"SET_15": [4, 19, 28, 34, 40, 42],
"SET_16": [2, 3, 20, 21, 30, 41],
"SET_17": [3, 12, 16, 17, 30, 35],
"SET_18": [3, 8, 12, 29, 32, 33],
"SET_19": [12, 16, 32, 35, 40, 41],
"SET_20": [7, 8, 11, 23, 36, 41],
"SET_21": [8, 12, 20, 23, 30, 41],
"SET_22": [3, 6, 17, 19, 32, 34],
"SET_23": [11, 18, 19, 25, 27, 40],
"SET_24": [15, 17, 23, 27, 29, 41],
"SET_25": [7, 13, 19, 20, 25, 41],
"SET_26": [12, 16, 18, 29, 31, 33],
"SET_27": [8, 21, 28, 35, 40, 42],
"SET_28": [3, 5, 33, 34, 38, 42],
"SET_29": [3, 11, 13, 15, 17, 19],
"SET_30": [5, 7, 11, 23, 32, 35],
"SET_31": [4, 11, 14, 16, 29, 31],
"SET_32": [8, 11, 12, 17, 37, 41],
"SET_33": [6, 8, 23, 31, 40, 41],
"SET_34": [3, 13, 19, 21, 32, 33],
"SET_35": [3, 11, 16, 19, 21, 30],
"SET_36": [6, 7, 11, 24, 41, 42],
"SET_37": [3, 24, 26, 30, 40, 41],
"SET_38": [2, 6, 16, 24, 25, 28],
"SET_39": [17, 19, 20, 26, 32, 40],
"SET_40": [3, 9, 12, 23, 31, 41],
"SET_41": [4, 9, 17, 21, 28, 36],
"SET_42": [8, 9, 22, 24, 27, 28],
"SET_43": [11, 19, 20, 27, 37, 39],
"SET_44": [7, 11, 13, 21, 23, 30],
"SET_45": [5, 10, 29, 30, 36, 40],
"SET_46": [14, 18, 26, 32, 33, 42],
"SET_47": [7, 17, 28, 38, 41, 42],
"SET_48": [9, 15, 25, 26, 29, 38],
"SET_49": [3, 16, 23, 25, 40, 42],
"SET_50": [2, 5, 8, 21, 26, 27],
"SET_51": [3, 10, 14, 20, 23, 38],
"SET_52": [4, 8, 16, 19, 34, 38],
"SET_53": [11, 13, 32, 34, 39, 41],
"SET_54": [12, 18, 30, 33, 37, 42],
"SET_55": [23, 25, 29, 34, 39, 41],
"SET_56": [3, 19, 25, 28, 35, 41],
"SET_57": [1, 7, 9, 20, 22, 39],
"SET_58": [8, 17, 18, 38, 40, 41],
"SET_59": [2, 5, 6, 25, 38, 42],
"SET_60": [12, 14, 21, 22, 28, 36],
"SET_61": [6, 14, 18, 20, 23, 24],
"SET_62": [1, 4, 7, 9, 20, 30],
"SET_63": [3, 4, 10, 11, 19, 29],
"SET_64": [4, 9, 10, 11, 25, 29],
"SET_65": [2, 6, 18, 25, 31, 33],
"SET_66": [1, 14, 22, 31, 33, 40],
"SET_67": [18, 23, 24, 35, 37, 40],
"SET_68": [23, 24, 25, 35, 39, 40],
"SET_69": [2, 10, 14, 15, 21, 22],
"SET_70": [2, 10, 22, 24, 34, 35],
"SET_71": [2, 10, 22, 24, 34, 35],
"SET_72": [2, 14, 21, 33, 36, 40],
"SET_73": [6, 7, 9, 18, 27, 37],
"SET_74": [8, 9, 17, 27, 29, 32],
"SET_75": [5, 7, 10, 12, 19, 37],
"SET_76": [2, 15, 16, 25, 34, 41],
"SET_77": [2, 6, 7, 9, 12, 21],
"SET_78": [3, 17, 20, 27, 29, 30],
"SET_79": [2, 14, 18, 26, 40, 42],
"SET_80": [6, 19, 20, 28, 31, 32],
"SET_81": [8, 15, 21, 24, 26, 39],
"SET_82": [1, 3, 24, 33, 35, 39],
"SET_83": [4, 19, 26, 27, 38, 39],
"SET_84": [13, 17, 21, 26, 35, 41],
"SET_85": [5, 6, 13, 28, 29, 37],
"SET_86": [4, 5, 10, 24, 29, 33],
"SET_87": [18, 24, 31, 33, 36, 40],
"SET_88": [5, 19, 22, 24, 25, 33],
"SET_89": [14, 20, 21, 33, 37, 41],
"SET_90": [4, 16, 31, 33, 38, 39],
"SET_91": [8, 12, 18, 27, 38, 42],
"SET_92": [7, 16, 19, 29, 31, 32],
"SET_93": [5, 12, 19, 23, 25, 34],
"SET_94": [12, 13, 15, 16, 23, 41],
"SET_95": [9, 19, 20, 26, 27, 41],
"SET_96": [5, 33, 37, 40, 41, 42],
"SET_97": [2, 5, 6, 24, 36, 41],
"SET_98": [9, 20, 25, 34, 37, 41],
"SET_99": [4, 16, 22, 25, 32, 34],
"SET_100": [9, 12, 23, 29, 32, 38],
"SET_101": [3, 12, 16, 31, 34, 37],
"SET_102": [25, 32, 34, 38, 40, 42],
"SET_103": [9, 13, 14, 33, 40, 42],
"SET_104": [12, 15, 19, 29, 30, 35],
"SET_105": [12, 16, 19, 20, 27, 40],
"SET_106": [10, 11, 12, 15, 26, 27],
"SET_107": [10, 11, 14, 22, 28, 31],
"SET_108": [4, 12, 22, 28, 31, 42],
"SET_109": [8, 15, 18, 27, 29, 38],
"SET_110": [4, 19, 21, 22, 29, 33],
"SET_111": [6, 12, 13, 23, 36, 37],
"SET_112": [5, 17, 28, 34, 41, 42],
"SET_113": [9, 13, 17, 19, 24, 36],
"SET_114": [2, 5, 18, 25, 31, 37],
"SET_115": [2, 8, 12, 14, 32, 33],
"SET_116": [1, 9, 18, 19, 27, 31],
"SET_117": [1, 11, 26, 29, 31, 36],
"SET_118": [3, 9, 20, 27, 35, 41],
"SET_119": [17, 22, 29, 31, 34, 35],
"SET_120": [10, 12, 25, 26, 36, 37],
"SET_121": [4, 7, 22, 31, 33, 36],
"SET_122": [12, 16, 25, 35, 40, 41],
"SET_123": [3, 7, 34, 35, 41, 42],
"SET_124": [1, 10, 22, 30, 35, 40],
"SET_125": [1, 12, 21, 30, 35, 42],
"SET_126": [1, 4, 17, 21, 22, 41],
"SET_127": [2, 13, 16, 19, 21, 22],
"SET_128": [2, 3, 14, 15, 33, 36],
"SET_129": [7, 10, 12, 18, 27, 32],
"SET_130": [10, 14, 21, 35, 36, 40],
"SET_131": [5, 16, 30, 31, 35, 37],
"SET_132": [2, 5, 7, 8, 21, 37],
"SET_133": [2, 5, 7, 11, 23, 39],
"SET_134": [14, 15, 16, 27, 39, 42],
"SET_135": [5, 10, 11, 24, 39, 42],
"SET_136": [3, 20, 25, 28, 35, 41],
"SET_137": [6, 16, 25, 29, 38, 41],
"SET_138": [6, 10, 18, 22, 25, 27],
"SET_139": [6, 10, 18, 22, 25, 27],
"SET_140": [1, 6, 7, 24, 29, 41],
"SET_141": [12, 13, 14, 35, 36, 40],
"SET_142": [11, 12, 13, 26, 34, 37],
"SET_143": [3, 4, 18, 30, 38, 40],
"SET_144": [13, 17, 25, 28, 31, 35],
"SET_145": [8, 11, 19, 34, 40, 42],
"SET_146": [11, 15, 18, 30, 31, 34],
"SET_147": [6, 9, 14, 24, 39, 40],
"SET_148": [15, 16, 21, 27, 39, 40],
"SET_149": [1, 11, 12, 32, 38, 42],
"SET_150": [9, 12, 15, 16, 20, 24],
"SET_151": [6, 8, 14, 21, 22, 24],
"SET_152": [8, 13, 14, 15, 28, 32],
"SET_153": [3, 13, 16, 20, 23, 34],
"SET_154": [17, 18, 29, 35, 38, 39],
"SET_155": [10, 22, 24, 35, 40, 42],
"SET_156": [2, 7, 15, 21, 23, 42],
"SET_157": [10, 12, 14, 24, 27, 31],
"SET_158": [3, 17, 33, 36, 41, 42],
"SET_159": [3, 5, 11, 14, 22, 23],
"SET_160": [10, 12, 16, 34, 36, 40],
"SET_161": [3, 21, 22, 31, 32, 33],
"SET_162": [9, 10, 20, 25, 30, 33],
"SET_163": [5, 13, 14, 25, 30, 42],
"SET_164": [1, 5, 9, 20, 26, 37],
"SET_165": [4, 6, 17, 18, 30, 41],
"SET_166": [2, 14, 16, 22, 28, 30],
"SET_167": [4, 11, 17, 30, 31, 33],
"SET_168": [9, 13, 22, 24, 31, 37],
"SET_169": [2, 5, 18, 23, 24, 25],
"SET_170": [3, 11, 20, 26, 29, 36],
"SET_171": [11, 17, 22, 28, 34, 37],
"SET_172": [2, 5, 17, 20, 34, 35],
"SET_173": [3, 16, 18, 26, 34, 35],
"SET_174": [5, 9, 12, 18, 23, 34],
"SET_175": [7, 8, 15, 21, 22, 23],
"SET_176": [10, 20, 22, 33, 39, 40],
"SET_177": [1, 9, 15, 27, 39, 42],
"SET_178": [19, 21, 31, 33, 37, 41],
"SET_179": [1, 6, 11, 27, 38, 40],
"SET_180": [4, 14, 24, 31, 35, 39],
"SET_181": [9, 16, 21, 24, 32, 34],
"SET_182": [10, 11, 17, 23, 33, 37],
"SET_183": [3, 5, 6, 21, 25, 33],
"SET_184": [3, 7, 9, 24, 36, 40],
"SET_185": [6, 21, 23, 29, 31, 37],
"SET_186": [3, 4, 6, 23, 28, 31],
"SET_187": [1, 5, 25, 33, 36, 41],
"SET_188": [9, 13, 18, 27, 34, 36],
"SET_189": [11, 13, 27, 30, 36, 42],
"SET_190": [16, 20, 24, 30, 32, 38],
"SET_191": [3, 8, 13, 29, 36, 41],
   }
  
   combo_sizes_to_analyze = [2, 3, 4, 5, 6]
   all_analyzers = {}
   speculative_results = {}
  
   # Phase 1: Initialize and extract combos for all sizes
   for size in combo_sizes_to_analyze:
       analyzer = HierarchicalComboAnalyzer(sample_data, size)
       all_analyzers[size] = analyzer
       analyzer.extract_combos()
  
   # Phase 2: Set hierarchical analyzers and analyze dependencies
   for size, analyzer in all_analyzers.items():
       analyzer.set_hierarchical_analyzers(all_analyzers)
       if len(analyzer.combo_data) > 0:  # Only analyze if there are significant combos
           analyzer.analyze_hierarchical_dependencies()
       else:
           print(f"Skipping hierarchical analysis for {size}-combos (no significant combinations)")
  
   # Phase 3: Generate predictions
   for size, analyzer in all_analyzers.items():
       if len(analyzer.combo_data) > 0:  # Only generate predictions if there are significant combos
           analyzer.generate_predictions()
       else:
           print(f"Skipping prediction generation for {size}-combos (no significant combinations)")


  
   # Phase 4: Generate speculative combos
   for size in [4, 5, 6]:
       if size in all_analyzers:
           speculative_results[size] = all_analyzers[size].generate_speculative_combos(size)
  
   # Phase 5: Display results
   for size, analyzer in all_analyzers.items():
       analyzer.display_results(show_all=True, speculative_combos=speculative_results.get(size, {}))
  
   # Phase 6: Get imminent combos
   print("\n=== IMMINENT COMBINATION PREDICTIONS ===")
   for size, analyzer in all_analyzers.items():
       imminent = analyzer.get_imminent_combos(max_sets_ahead=10)
       for item in imminent:
           print(f"{size}-Combo {item['combo']}: Predicted in {item['sets_ahead']:.1f} sets "
                 f"(Confidence: {item['confidence']:.1%}, Alert: {item['alert_level']}, "
                 f"Support: {item['support_score']:.1%}, Rank: {item['support_rank']}, "
                 f"Convergence: {item['convergence_score']:.1%})")
  
   # Phase 7: Hierarchical summary
   print("\n=== HIERARCHICAL ANALYSIS SUMMARY ===")
   total_significant = sum(len(analyzer.combo_data) for analyzer in all_analyzers.values())
   total_imminent = sum(len(analyzer.get_imminent_combos()) for analyzer in all_analyzers.values())
   total_enhanced = sum(sum(1 for data in analyzer.combo_data.values()
                           if data.get('boosted_confidence', 0) > data['confidence'])
                       for analyzer in all_analyzers.values())
  
   print(f"Total significant combos across all sizes: {total_significant}")
   print(f"Total imminent predictions: {total_imminent}")
   print(f"Total hierarchically enhanced: {total_enhanced}")
  
   for size, analyzer in all_analyzers.items():
       combo_name = {2: "Pairs", 3: "Triplets", 4: "Quadruplets", 5: "Quintuplets", 6: "Sextuplets"}.get(size, f"{size}-combos")
       alert_counts = Counter(data['alert_level'] for data in analyzer.combo_data.values())
       high_priority = sum(1 for data in analyzer.combo_data.values() if data['alert_level'] in ['RED', 'ORANGE'])
       avg_support = statistics.mean([data['support_score'] for data in analyzer.combo_data.values()] or [0])
       avg_convergence = statistics.mean([data['convergence_score'] for data in analyzer.combo_data.values()] or [0])
       avg_accuracy = statistics.mean([data['prediction_accuracy'] for data in analyzer.combo_data.values()] or [0])
      
       print(f"\n{combo_name}:")
       print(f"  Significant combos: {len(analyzer.combo_data)}")
       print(f"  High-priority alerts (RED/ORANGE): {high_priority}")
       print(f"  Alert distribution: {dict(alert_counts)}")
       if size > 2:
           enhanced_count = sum(1 for data in analyzer.combo_data.values() if data.get('boosted_confidence', 0) > data['confidence'])
           print(f"  Hierarchically enhanced: {enhanced_count}")
           print(f"  Average support score: {avg_support:.1%}")
       print(f"  Average convergence score: {avg_convergence:.1%}")
       print(f"  Average prediction accuracy: {avg_accuracy:.1%}")
  
   # Phase 8: Predict next 6-element set
   print("\n=== PREDICTED NEXT 6-ELEMENT SET ===")
   if speculative_results.get(6):
       sorted_speculative = sorted(
           speculative_results[6].items(),
           key=lambda x: (
               x[1]['predictions'].get('ensemble', float('inf')),
               -x[1]['support_score'],
               -x[1].get('support_rank', 0),
               -x[1]['confidence'],
               -x[1].get('convergence_score', 0.0)
           )
       )
       if sorted_speculative:
           top_combo, top_data = sorted_speculative[0]
           historical_accuracy = top_data['prediction_accuracy']
           print(f"Predicted Set: {top_combo}")
           print(f"Support: {top_data['support_score']:.1%} (#{top_data['support_rank']})")
           print(f"Confidence: {top_data['confidence']:.1%}")
           print(f"Convergence: {top_data['convergence_score']:.1%}")
           print(f"Historical Speculative Accuracy: {historical_accuracy:.1%}")
           if 'ensemble' in top_data['predictions']:
               print(f"Predicted for SET_{int(top_data['predictions']['ensemble']) + 1}")
           analyzer._display_hierarchical_support_enhanced(top_data['hierarchical_support'])
       else:
           print("No speculative sextuplets found")
   else:
       print("No speculative sextuplets generated")

if __name__ == "__main__":
   main()
