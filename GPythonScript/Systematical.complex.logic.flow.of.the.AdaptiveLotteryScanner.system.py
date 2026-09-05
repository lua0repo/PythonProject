import itertools
import statistics
import math
import numpy as np
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Optional, Set
import json
import random
from datetime import datetime, timedelta
import warnings

class EnhancedAdaptiveLotteryScanner:
   def __init__(self, dataset: Dict[str, List[int]], game_params: Dict = None):
       """
       Enhanced adaptive system with improved decision tree and validation
       """
       # Enhanced validation with detailed error reporting
       self.validation_errors = []
       self.game_params = game_params or {"min_num": 1, "max_num": 60, "draw_size": 6}
      
       self._validate_dataset_enhanced(dataset)
       if self.validation_errors:
           raise ValueError(f"Dataset validation failed: {'; '.join(self.validation_errors)}")
          
       self.dataset = dataset
       self.set_names = list(dataset.keys())
       self.dataset_size = len(dataset)
      
       # Enhanced learning system
       self.learning_history = []
       self.model_weights = {
           'frequency_bias': 0.15,
           'positional_bias': 0.15,
           'sequential_bias': 0.15,
           'temporal_bias': 0.15,
           'clustering_bias': 0.15,
           'statistical_anomaly': 0.125,  # New model
           'cross_validation': 0.125      # New model
       }
      
       # Enhanced tracking systems
       self.prediction_accuracy = {model: deque(maxlen=100) for model in self.model_weights}
       self.vulnerability_patterns = {}
       self.adaptation_rate = 0.08  # Reduced for more stable learning
       self.confidence_threshold = {"high": 0.35, "medium": 0.22, "low": 0.0}
      
       # New tracking systems
       self.meta_patterns = defaultdict(list)
       self.ensemble_performance = deque(maxlen=50)
       self.prediction_variance = deque(maxlen=30)
      
       # Enhanced analysis structures
       self.positional_analysis = {i: Counter() for i in range(self.game_params["draw_size"])}
       self.sequential_patterns = defaultdict(int)
       self.temporal_patterns = {}
       self.clustering_analysis = {}
       self.statistical_anomalies = {}
      
       print(f"Enhanced Adaptive Learning System initialized with {self.dataset_size} draws")
       print("Advanced scanning for systematic vulnerabilities and meta-patterns...")

   def _validate_dataset_enhanced(self, dataset: Dict[str, List[int]]) -> bool:
       """Enhanced validation with comprehensive error checking"""
       if not dataset:
           self.validation_errors.append("Dataset is empty")
           return False
          
       if len(dataset) < 3:
           self.validation_errors.append("Minimum 3 lottery draws required for analysis")
          
       draw_size = self.game_params["draw_size"]
       min_num = self.game_params["min_num"]
       max_num = self.game_params["max_num"]
      
       for set_name, numbers in dataset.items():
           # Check draw size
           if len(numbers) != draw_size:
               self.validation_errors.append(f"Set {set_name}: Expected {draw_size} numbers, got {len(numbers)}")
               continue
              
           # Check number validity
           invalid_nums = [n for n in numbers if not isinstance(n, int) or not (min_num <= n <= max_num)]
           if invalid_nums:
               self.validation_errors.append(f"Set {set_name}: Invalid numbers {invalid_nums}")
              
           # Check for duplicates within set
           if len(set(numbers)) != len(numbers):
               duplicates = [n for n in numbers if numbers.count(n) > 1]
               self.validation_errors.append(f"Set {set_name}: Duplicate numbers {duplicates}")
              
       return len(self.validation_errors) == 0

   def analyze_enhanced_biases(self):
       """Enhanced bias analysis with new detection methods"""
       if self.dataset_size < 2:
           print("Insufficient data for comprehensive analysis")
           return

       # Original analyses
       self._analyze_positional_bias()
       self._analyze_sequential_patterns()
       self._analyze_temporal_clustering()
       self._analyze_generation_biases()
      
       # New enhanced analyses
       self._analyze_statistical_anomalies()
       self._analyze_meta_patterns()
       self._analyze_variance_patterns()

   def _analyze_statistical_anomalies(self):
       """Detect statistical anomalies that suggest non-random behavior"""
       all_numbers = []
       for numbers in self.dataset.values():
           all_numbers.extend(numbers)
          
       # Chi-square test for uniform distribution
       expected_freq = len(all_numbers) / (self.game_params["max_num"] - self.game_params["min_num"] + 1)
       observed_freq = Counter(all_numbers)
      
       chi_square = 0
       for num in range(self.game_params["min_num"], self.game_params["max_num"] + 1):
           observed = observed_freq.get(num, 0)
           chi_square += ((observed - expected_freq) ** 2) / expected_freq
          
       # Detect runs and streaks
       runs_analysis = self._analyze_runs()
      
       # Detect gaps between consecutive draws
       gap_analysis = self._analyze_gaps()
      
       self.statistical_anomalies = {
           'chi_square': chi_square,
           'uniformity_p_value': self._calculate_p_value(chi_square),
           'runs_analysis': runs_analysis,
           'gap_analysis': gap_analysis,
           'entropy': self._calculate_entropy(all_numbers)
       }

   def _analyze_runs(self) -> Dict:
       """Analyze runs of consecutive or related numbers"""
       runs_data = {'consecutive_runs': [], 'pattern_runs': []}
      
       for numbers in self.dataset.values():
           sorted_nums = sorted(numbers)
          
           # Find consecutive runs
           current_run = [sorted_nums[0]]
           for i in range(1, len(sorted_nums)):
               if sorted_nums[i] == sorted_nums[i-1] + 1:
                   current_run.append(sorted_nums[i])
               else:
                   if len(current_run) >= 2:
                       runs_data['consecutive_runs'].append(current_run)
                   current_run = [sorted_nums[i]]
           if len(current_run) >= 2:
               runs_data['consecutive_runs'].append(current_run)
              
       return runs_data

   def _analyze_gaps(self) -> Dict:
       """Analyze gaps between numbers in draws"""
       gap_data = {'avg_gaps': [], 'gap_variance': []}
      
       for numbers in self.dataset.values():
           sorted_nums = sorted(numbers)
           gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
          
           gap_data['avg_gaps'].append(statistics.mean(gaps))
           gap_data['gap_variance'].append(statistics.variance(gaps) if len(gaps) > 1 else 0)
          
       return {
           'overall_avg_gap': statistics.mean(gap_data['avg_gaps']),
           'gap_consistency': 1 - (statistics.stdev(gap_data['avg_gaps']) / statistics.mean(gap_data['avg_gaps'])),
           'variance_pattern': gap_data['gap_variance']
       }

   def _calculate_entropy(self, numbers: List[int]) -> float:
       """Calculate Shannon entropy of number distribution"""
       if not numbers:
           return 0
          
       counts = Counter(numbers)
       total = len(numbers)
       entropy = 0
      
       for count in counts.values():
           if count > 0:
               p = count / total
               entropy -= p * math.log2(p)
              
       return entropy

   def _calculate_p_value(self, chi_square: float) -> str:
       """Simplified p-value calculation for chi-square"""
       # Simplified thresholds - in practice, use scipy.stats
       if chi_square > 100:
           return "< 0.001 (Highly Significant)"
       elif chi_square > 80:
           return "< 0.01 (Significant)"
       elif chi_square > 60:
           return "< 0.05 (Marginally Significant)"
       else:
           return "> 0.05 (Not Significant)"

   def _analyze_meta_patterns(self):
       """Analyze patterns between patterns (meta-analysis)"""
       # Look for cycles in pattern occurrence
       pattern_timeline = []
      
       for i, (set_name, numbers) in enumerate(self.dataset.items()):
           pattern_signature = {
               'has_consecutive': any(numbers[j+1] == numbers[j] + 1 for j in range(len(numbers)-1) if j+1 < len(numbers)),
               'has_multiples_of_5': any(n % 5 == 0 for n in numbers),
               'sum_range': sum(numbers) // 10,  # Categorize by sum ranges
               'spread': max(numbers) - min(numbers)
           }
           pattern_timeline.append(pattern_signature)
          
       # Look for repeating meta-patterns
       for window in [3, 5, 7]:
           for i in range(len(pattern_timeline) - window + 1):
               window_pattern = tuple(str(p) for p in pattern_timeline[i:i+window])
               self.meta_patterns[window].append(window_pattern)

   def _analyze_variance_patterns(self):
       """Analyze variance in prediction accuracy to detect systematic changes"""
       if len(self.ensemble_performance) < 10:
           return
          
       # Rolling variance analysis
       performances = list(self.ensemble_performance)
       for window_size in [5, 10, 15]:
           if len(performances) >= window_size:
               rolling_var = []
               for i in range(len(performances) - window_size + 1):
                   window_data = performances[i:i+window_size]
                   if len(set(window_data)) > 1:  # Avoid zero variance
                       rolling_var.append(statistics.variance(window_data))
                      
               if rolling_var:
                   self.prediction_variance.append({
                       'window': window_size,
                       'avg_variance': statistics.mean(rolling_var),
                       'variance_trend': rolling_var[-1] - rolling_var[0] if len(rolling_var) > 1 else 0
                   })

   def generate_enhanced_predictions(self, target_set_index: int) -> Dict:
       """Generate predictions using enhanced models"""
       predictions = self._generate_base_predictions(target_set_index)
      
       # New enhanced models
       predictions['statistical_anomaly'] = self._predict_anomaly_correction()
       predictions['cross_validation'] = self._predict_cross_validation()
      
       # Enhanced ensemble with volatility adjustment
       predictions['enhanced_ensemble'] = self._create_enhanced_ensemble(predictions)
      
       return predictions

   def _generate_base_predictions(self, target_set_index: int) -> Dict:
       """Generate base model predictions with improvements"""
       predictions = {}
      
       # Enhanced frequency model with decay
       freq_counter = Counter()
       for i, numbers in enumerate(self.dataset.values()):
           # Exponential decay weighting
           weight = 0.95 ** (self.dataset_size - i - 1)
           for num in numbers:
               freq_counter[num] += weight

       predictions['frequency_bias'] = self._select_adaptive_numbers(freq_counter)
      
       # Other base models (simplified for brevity)
       predictions['positional_bias'] = self._predict_positional_enhanced()
       predictions['sequential_bias'] = self._predict_sequential_enhanced()
       predictions['temporal_bias'] = self._predict_temporal_enhanced()
       predictions['clustering_bias'] = self._predict_clustering_enhanced()
      
       return predictions

   def _predict_anomaly_correction(self) -> List[int]:
       """Predict by correcting for detected statistical anomalies"""
       if not self.statistical_anomalies:
           return sorted(random.sample(range(self.game_params["min_num"], self.game_params["max_num"] + 1),
                                     self.game_params["draw_size"]))
      
       # Use entropy and chi-square to guide selection
       all_numbers = []
       for numbers in self.dataset.values():
           all_numbers.extend(numbers)
          
       freq_counter = Counter(all_numbers)
      
       # Select numbers that would improve uniformity
       underrepresented = [num for num in range(self.game_params["min_num"], self.game_params["max_num"] + 1)
                         if freq_counter.get(num, 0) < statistics.mean(freq_counter.values())]
      
       if len(underrepresented) >= self.game_params["draw_size"]:
           return sorted(random.sample(underrepresented, self.game_params["draw_size"]))
       else:
           prediction = underrepresented[:]
           remaining = self.game_params["draw_size"] - len(prediction)
           available = [n for n in range(self.game_params["min_num"], self.game_params["max_num"] + 1)
                       if n not in prediction]
           prediction.extend(random.sample(available, remaining))
           return sorted(prediction)

   def _predict_cross_validation(self) -> List[int]:
       """Use cross-validation approach across different time windows"""
       if self.dataset_size < 10:
           return sorted(random.sample(range(self.game_params["min_num"], self.game_params["max_num"] + 1),
                                     self.game_params["draw_size"]))
      
       # Split data into thirds and see what works across splits
       third = self.dataset_size // 3
       splits = [
           list(self.dataset.items())[:third],
           list(self.dataset.items())[third:2*third],
           list(self.dataset.items())[2*third:]
       ]
      
       # Find numbers that perform consistently across splits
       consistent_numbers = set(range(self.game_params["min_num"], self.game_params["max_num"] + 1))
      
       for split in splits:
           split_numbers = set()
           for _, numbers in split:
               split_numbers.update(numbers)
           consistent_numbers &= split_numbers
          
       if len(consistent_numbers) >= self.game_params["draw_size"]:
           return sorted(random.sample(list(consistent_numbers), self.game_params["draw_size"]))
       else:
           prediction = list(consistent_numbers)
           remaining = self.game_params["draw_size"] - len(prediction)
           available = [n for n in range(self.game_params["min_num"], self.game_params["max_num"] + 1)
                       if n not in prediction]
           prediction.extend(random.sample(available, remaining))
           return sorted(prediction)

   def _select_adaptive_numbers(self, freq_counter: Counter) -> List[int]:
       """Adaptive number selection based on performance history"""
       # Adjust strategy based on recent performance
       recent_performance = list(self.prediction_accuracy['frequency_bias'])[-10:] if self.prediction_accuracy['frequency_bias'] else []
      
       if recent_performance and statistics.mean(recent_performance) > 0.25:
           # Current strategy is working, be more aggressive
           hot_ratio = 0.7
       else:
           # Current strategy isn't working, be more conservative
           hot_ratio = 0.4
          
       hot_count = int(self.game_params["draw_size"] * hot_ratio)
       cold_count = self.game_params["draw_size"] - hot_count
      
       hot_numbers = [num for num, _ in freq_counter.most_common(20)]
       cold_numbers = [num for num, _ in freq_counter.most_common()[-20:]]
      
       selection = []
       if hot_numbers:
           selection.extend(random.sample(hot_numbers, min(hot_count, len(hot_numbers))))
       if cold_numbers and cold_count > 0:
           selection.extend(random.sample(cold_numbers, min(cold_count, len(cold_numbers))))
          
       # Fill remaining
       remaining = self.game_params["draw_size"] - len(selection)
       available = [i for i in range(self.game_params["min_num"], self.game_params["max_num"] + 1)
                   if i not in selection]
       if remaining > 0 and available:
           selection.extend(random.sample(available, min(remaining, len(available))))
          
       return sorted(selection[:self.game_params["draw_size"]])

   def _predict_positional_enhanced(self) -> List[int]:
       """Enhanced positional prediction with confidence weighting"""
       prediction = []
       for pos in range(self.game_params["draw_size"]):
           pos_freq = self.positional_analysis[pos]
           if pos_freq and pos_freq.most_common(1)[0][1] > 1:  # At least 2 occurrences
               # Weight by confidence (frequency of occurrence)
               candidates = [num for num, count in pos_freq.most_common(10) if count > 1]
               if candidates:
                   prediction.append(random.choice(candidates))
               else:
                   prediction.append(random.randint(self.game_params["min_num"], self.game_params["max_num"]))
           else:
               prediction.append(random.randint(self.game_params["min_num"], self.game_params["max_num"]))
              
       # Remove duplicates and fill
       prediction = list(set(prediction))
       while len(prediction) < self.game_params["draw_size"]:
           candidate = random.randint(self.game_params["min_num"], self.game_params["max_num"])
           if candidate not in prediction:
               prediction.append(candidate)
              
       return sorted(prediction[:self.game_params["draw_size"]])

   def _predict_sequential_enhanced(self) -> List[int]:
       """Enhanced sequential prediction with pattern weighting"""
       if self.dataset_size < 2:
           return sorted(random.sample(range(self.game_params["min_num"], self.game_params["max_num"] + 1),
                                     self.game_params["draw_size"]))
      
       last_set = set(self.dataset[self.set_names[-1]])
      
      
      
       # Enhanced carryover analysis with weighting
       weighted_carryover = 0
       total_weight = 0
      
       for carryover, count in self.sequential_patterns.items():
           weight = count * (1 + carryover * 0.1)  # Slight bonus for higher carryover
           weighted_carryover += carryover * weight
           total_weight += weight
          
       expected_carryover = int(weighted_carryover / total_weight) if total_weight > 0 else 1
       expected_carryover = max(0, min(expected_carryover, len(last_set)))
      
       # Select carryover numbers
       carryover_nums = random.sample(list(last_set), expected_carryover)
      
       # Fill remaining
       remaining = self.game_params["draw_size"] - len(carryover_nums)
       available = [i for i in range(self.game_params["min_num"], self.game_params["max_num"] + 1)
                   if i not in carryover_nums]
       new_nums = random.sample(available, remaining)
      
       return sorted(carryover_nums + new_nums)

   def _predict_temporal_enhanced(self) -> List[int]:
       """Enhanced temporal prediction with cooling periods"""
       if not self.temporal_patterns:
           return sorted(random.sample(range(self.game_params["min_num"], self.game_params["max_num"] + 1),
                                     self.game_params["draw_size"]))
      
       # Numbers with cooling periods based on recent frequency
       recent_draws = min(8, self.dataset_size)
       recent_numbers = Counter()
      
       for i in range(self.dataset_size - recent_draws, self.dataset_size):
           for num in self.dataset[self.set_names[i]]:
               # More recent = higher weight
               weight = (recent_draws - (self.dataset_size - i - 1)) / recent_draws
               recent_numbers[num] += weight
      
       # Avoid overly frequent recent numbers
       cooling_threshold = statistics.mean(recent_numbers.values()) * 1.5 if recent_numbers else 0
       cooled_numbers = [num for num, freq in recent_numbers.items() if freq > cooling_threshold]
      
       available = [i for i in range(self.game_params["min_num"], self.game_params["max_num"] + 1)
                   if i not in cooled_numbers]
      
       if len(available) >= self.game_params["draw_size"]:
           return sorted(random.sample(available, self.game_params["draw_size"]))
       else:
           # If too many numbers are cooling, relax the constraint
           prediction = random.sample(available, len(available))
           remaining = self.game_params["draw_size"] - len(prediction)
           prediction.extend(random.sample(cooled_numbers, remaining))
           return sorted(prediction)

   def _predict_clustering_enhanced(self) -> List[int]:
       """Enhanced clustering prediction with dynamic ranges"""
       num_range = self.game_params["max_num"] - self.game_params["min_num"] + 1
       num_clusters = min(5, self.game_params["draw_size"])
       cluster_size = num_range // num_clusters
      
       prediction = []
      
       # Analyze historical cluster preferences
       cluster_prefs = defaultdict(int)
       for numbers in self.dataset.values():
           for num in numbers:
               cluster_idx = (num - self.game_params["min_num"]) // cluster_size
               cluster_prefs[cluster_idx] += 1
      
       # Select from preferred clusters with some randomness
       selected_clusters = []
       if cluster_prefs:
           # Weight clusters by historical preference
           total_weight = sum(cluster_prefs.values())
           for _ in range(min(num_clusters, self.game_params["draw_size"])):
               weights = [(cluster, weight/total_weight) for cluster, weight in cluster_prefs.items()
                         if cluster not in selected_clusters]
               if weights:
                   # Weighted random selection
                   rand_val = random.random()
                   cumulative = 0
                   for cluster, weight in weights:
                       cumulative += weight
                       if rand_val <= cumulative:
                           selected_clusters.append(cluster)
                           break
      
       # Fill if needed
       while len(selected_clusters) < min(num_clusters, self.game_params["draw_size"]):
           available_clusters = [i for i in range(num_clusters) if i not in selected_clusters]
           if available_clusters:
               selected_clusters.append(random.choice(available_clusters))
           else:
               break
      
       # Select numbers from chosen clusters
       for cluster in selected_clusters:
           cluster_start = self.game_params["min_num"] + cluster * cluster_size
           cluster_end = min(cluster_start + cluster_size - 1, self.game_params["max_num"])
           prediction.append(random.randint(cluster_start, cluster_end))
      
       # Fill remaining with random selection
       while len(prediction) < self.game_params["draw_size"]:
           candidate = random.randint(self.game_params["min_num"], self.game_params["max_num"])
           if candidate not in prediction:
               prediction.append(candidate)
      
       return sorted(prediction[:self.game_params["draw_size"]])

   def _create_enhanced_ensemble(self, predictions: Dict) -> List[int]:
       """Create enhanced ensemble with volatility and confidence adjustment"""
       number_votes = defaultdict(float)
      
       # Calculate volatility adjustment
       volatility_factor = self._calculate_volatility_factor()
      
       for model, weight in self.model_weights.items():
           if model in predictions:
               # Adjust weight by model's recent performance and volatility
               recent_performance = list(self.prediction_accuracy[model])[-5:] if self.prediction_accuracy[model] else [0]
               performance_factor = statistics.mean(recent_performance) if recent_performance else 0
              
               adjusted_weight = weight * (1 + performance_factor * 0.3) * volatility_factor
              
               for number in predictions[model]:
                   number_votes[number] += adjusted_weight
      
       # Select top numbers with some randomization to avoid over-fitting
       sorted_numbers = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
      
       # Use top candidates but with some randomization
       top_candidates = [num for num, _ in sorted_numbers[:self.game_params["draw_size"] * 2]]
      
       if len(top_candidates) >= self.game_params["draw_size"]:
           # Weighted random selection from top candidates
           selected = []
           for _ in range(self.game_params["draw_size"]):
               available = [num for num in top_candidates if num not in selected]
               if available:
                   # Higher vote weight = higher selection probability
                   weights = [number_votes[num] for num in available]
                   total_weight = sum(weights)
                   if total_weight > 0:
                       probabilities = [w/total_weight for w in weights]
                       selected_num = random.choices(available, weights=probabilities)[0]
                       selected.append(selected_num)
                   else:
                       selected.append(random.choice(available))
           return sorted(selected)
       else:
           # Fallback to simple selection
           selected = [num for num, _ in sorted_numbers[:self.game_params["draw_size"]]]
           while len(selected) < self.game_params["draw_size"]:
               available = [i for i in range(self.game_params["min_num"], self.game_params["max_num"] + 1)
                           if i not in selected]
               if available:
                   selected.append(random.choice(available))
               else:
                   break
           return sorted(selected)

   def _calculate_volatility_factor(self) -> float:
       """Calculate volatility factor to adjust ensemble weighting"""
       if len(self.ensemble_performance) < 5:
           return 1.0
      
       recent_performance = list(self.ensemble_performance)[-10:]
       if len(recent_performance) < 2:
           return 1.0
          
       volatility = statistics.stdev(recent_performance) if len(recent_performance) > 1 else 0
      
       # Lower volatility = more confident in current weights
       # Higher volatility = be more conservative
       return max(0.7, min(1.3, 1.0 - volatility * 0.5))

   def enhanced_backtest(self, test_window: int = 30) -> Dict:
       """Enhanced backtesting with more sophisticated analysis"""
       if self.dataset_size < test_window + 15:
           print(f"Insufficient data for enhanced backtesting")
           return {}

       results = []
       model_results = {model: [] for model in list(self.model_weights.keys()) + ['enhanced_ensemble']}
       confidence_tracking = []

       print(f"Starting enhanced backtesting on {test_window} predictions...")

       for i in range(self.dataset_size - test_window, self.dataset_size):
           # Create training dataset
           training_data = {name: numbers for j, (name, numbers) in enumerate(self.dataset.items()) if j < i}

           # Create temporary scanner
           temp_scanner = EnhancedAdaptiveLotteryScanner(training_data, self.game_params)
           temp_scanner.analyze_enhanced_biases()

           # Generate predictions
           predictions = temp_scanner.generate_enhanced_predictions(i)
           actual = self.dataset[self.set_names[i]]

           # Test predictions
           test_results = {}
           for model, prediction in predictions.items():
               result = self.test_prediction(prediction, actual)
               test_results[model] = result
               model_results[model].append(result['accuracy'])

           # Track confidence calibration
           confidence_data = temp_scanner._calculate_enhanced_confidence()
           confidence_tracking.append({
               'predicted_confidence': confidence_data['expected_accuracy'],
               'actual_accuracy': test_results.get('enhanced_ensemble', {}).get('accuracy', 0)
           })

           results.append({
               'set_index': i,
               'set_name': self.set_names[i],
               'predictions': test_results,
               'confidence': confidence_data
           })

           # Update ensemble performance tracking
           if 'enhanced_ensemble' in test_results:
               temp_scanner.ensemble_performance.append(test_results['enhanced_ensemble']['accuracy'])

           # Adaptive learning
           if len(results) > 1:
               temp_scanner.learn_and_adapt([test_results])

       return self._analyze_enhanced_backtest_results(results, model_results, confidence_tracking)

   def _analyze_enhanced_backtest_results(self, results: List[Dict], model_results: Dict,
                                        confidence_tracking: List[Dict]) -> Dict:
       """Enhanced analysis of backtesting results"""
       analysis = {
           'total_tests': len(results),
           'model_performance': {},
           'best_model': None,
           'confidence_calibration': {},
           'learning_progression': {},
           'statistical_significance': {}
       }

       # Enhanced model performance analysis
       for model, accuracies in model_results.items():
           if accuracies:
               analysis['model_performance'][model] = {
                   'avg_accuracy': statistics.mean(accuracies),
                   'median_accuracy': statistics.median(accuracies),
                   'max_accuracy': max(accuracies),
                   'std_dev': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
                   'success_rate_2plus': sum(1 for acc in accuracies if acc >= 0.33) / len(accuracies),
                   'success_rate_3plus': sum(1 for acc in accuracies if acc >= 0.5) / len(accuracies),
                   'consistency_score': 1 - (statistics.stdev(accuracies) / statistics.mean(accuracies)) if accuracies and statistics.mean(accuracies) > 0 else 0,
                   'trend': self._calculate_trend(accuracies)
               }

       # Find best model with enhanced criteria
       best_score = 0
       for model, metrics in analysis['model_performance'].items():
           # Combined score: accuracy + consistency + success rate
           combined_score = (metrics['avg_accuracy'] * 0.4 +
                           metrics['success_rate_2plus'] * 0.3 +
                           metrics['consistency_score'] * 0.3)
           if combined_score > best_score:
               best_score = combined_score
               analysis['best_model'] = model

       # Confidence calibration analysis
       if confidence_tracking:
           predicted_confidences = [c['predicted_confidence'] for c in confidence_tracking]
           actual_accuracies = [c['actual_accuracy'] for c in confidence_tracking]
          
           # Calculate calibration error
           calibration_error = statistics.mean([abs(pred - actual) for pred, actual in zip(predicted_confidences, actual_accuracies)])
          
           analysis['confidence_calibration'] = {
               'mean_predicted_confidence': statistics.mean(predicted_confidences),
               'mean_actual_accuracy': statistics.mean(actual_accuracies),
               'calibration_error': calibration_error,
               'overconfidence': statistics.mean(predicted_confidences) - statistics.mean(actual_accuracies),
               'correlation': self._calculate_correlation(predicted_confidences, actual_accuracies)
           }

       # Learning progression analysis
       if len(results) >= 15:
           window_size = 5
           progression = []
           for i in range(0, len(results) - window_size + 1, window_size):
               window_results = [r['predictions'].get('enhanced_ensemble', {}).get('accuracy', 0)
                               for r in results[i:i+window_size]]
               progression.append(statistics.mean(window_results))
          
           analysis['learning_progression'] = {
               'progression_values': progression,
               'overall_trend': self._calculate_trend(progression),
               'learning_rate': self._calculate_learning_rate(progression),
               'stability': 1 - (statistics.stdev(progression) / statistics.mean(progression)) if progression and statistics.mean(progression) > 0 else 0
           }

       # Statistical significance testing
       if analysis['best_model'] and len(model_results[analysis['best_model']]) >= 10:
           baseline_performance = 1/6  # Random chance baseline
           best_model_results = model_results[analysis['best_model']]
          
           # Simple t-test approximation
           mean_performance = statistics.mean(best_model_results)
           std_performance = statistics.stdev(best_model_results) if len(best_model_results) > 1 else 0.1
           n = len(best_model_results)
          
           t_statistic = (mean_performance - baseline_performance) / (std_performance / math.sqrt(n))
          
           analysis['statistical_significance'] = {
               'mean_performance': mean_performance,
               'baseline': baseline_performance,
               't_statistic': t_statistic,
               'is_significant': abs(t_statistic) > 1.96,  # Approximate 95% confidence
               'effect_size': (mean_performance - baseline_performance) / std_performance if std_performance > 0 else 0
           }

       return analysis

   def _calculate_trend(self, values: List[float]) -> Dict:
       """Calculate trend direction and strength"""
       if len(values) < 3:
           return {'direction': 'insufficient_data', 'strength': 0}
      
       # Simple linear regression slope
       x = list(range(len(values)))
       n = len(values)
      
       sum_x = sum(x)
       sum_y = sum(values)
       sum_xy = sum(x[i] * values[i] for i in range(n))
       sum_x2 = sum(xi ** 2 for xi in x)
      
       slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
      
       direction = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
       strength = abs(slope) * 100  # Scale for readability
      
       return {'direction': direction, 'strength': strength, 'slope': slope}

   def _calculate_learning_rate(self, progression: List[float]) -> float:
       """Calculate how quickly the system is learning"""
       if len(progression) < 2:
           return 0
      
       improvements = [progression[i+1] - progression[i] for i in range(len(progression)-1)]
       positive_improvements = [imp for imp in improvements if imp > 0]
      
       return len(positive_improvements) / len(improvements) if improvements else 0

   def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
       """Calculate Pearson correlation coefficient"""
       if len(x) != len(y) or len(x) < 2:
           return 0
      
       n = len(x)
       sum_x = sum(x)
       sum_y = sum(y)
       sum_xy = sum(x[i] * y[i] for i in range(n))
       sum_x2 = sum(xi ** 2 for xi in x)
       sum_y2 = sum(yi ** 2 for yi in y)
      
       numerator = n * sum_xy - sum_x * sum_y
       denominator = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
      
       return numerator / denominator if denominator != 0 else 0

   def _calculate_enhanced_confidence(self) -> Dict:
       """Enhanced confidence calculation with multiple factors"""
       base_confidence = {'confidence_level': 'LOW', 'expected_accuracy': 0.16, 'factors': {}}
      
       if not self.learning_history and not self.ensemble_performance:
           return base_confidence
      
       factors = {}
      
       # Factor 1: Recent ensemble performance
       if self.ensemble_performance:
           recent_performance = list(self.ensemble_performance)[-10:]
           avg_performance = statistics.mean(recent_performance)
           performance_consistency = 1 - (statistics.stdev(recent_performance) / avg_performance) if len(recent_performance) > 1 and avg_performance > 0 else 0
          
           factors['recent_performance'] = {
               'value': avg_performance,
               'weight': 0.4,
               'contribution': avg_performance * 0.4
           }
          
           factors['consistency'] = {
               'value': performance_consistency,
               'weight': 0.2,
               'contribution': performance_consistency * 0.2
           }
      
       # Factor 2: Data quality and quantity
       data_quality_score = min(1.0, self.dataset_size / 50)  # Optimal around 50+ samples
       factors['data_quality'] = {
           'value': data_quality_score,
           'weight': 0.2,
           'contribution': data_quality_score * 0.2
       }
      
      
      
       # Factor 3: Statistical anomaly detection confidence
       if self.statistical_anomalies:
           anomaly_confidence = 0.8 if 'Significant' in str(self.statistical_anomalies.get('uniformity_p_value', '')) else 0.4
           factors['anomaly_detection'] = {
               'value': anomaly_confidence,
               'weight': 0.1,
               'contribution': anomaly_confidence * 0.1
           }
      
       # Factor 4: Model agreement
       if len(self.model_weights) > 1:
           weight_entropy = -sum(w * math.log2(w) for w in self.model_weights.values() if w > 0)
           max_entropy = math.log2(len(self.model_weights))
           agreement_score = 1 - (weight_entropy / max_entropy) if max_entropy > 0 else 0
          
           factors['model_agreement'] = {
               'value': agreement_score,
               'weight': 0.1,
               'contribution': agreement_score * 0.1
           }
      
       # Calculate final confidence
       total_contribution = sum(factor['contribution'] for factor in factors.values())
       expected_accuracy = max(0.16, min(0.8, total_contribution))  # Bound between random and realistic max
      
       # Determine confidence level
       if expected_accuracy > self.confidence_threshold['high']:
           confidence_level = 'HIGH'
       elif expected_accuracy > self.confidence_threshold['medium']:
           confidence_level = 'MEDIUM'
       else:
           confidence_level = 'LOW'
      
       return {
           'confidence_level': confidence_level,
           'expected_accuracy': expected_accuracy,
           'factors': factors,
           'total_contribution': total_contribution
       }

   def _analyze_positional_bias(self):
       """Analyze for positional biases"""
       for set_name, numbers in self.dataset.items():
           sorted_numbers = sorted(numbers)
           for position, number in enumerate(sorted_numbers):
               self.positional_analysis[position][number] += 1

   def _analyze_sequential_patterns(self):
       """Analyze sequential patterns"""
       if self.dataset_size < 2:
           return
          
       for i in range(1, self.dataset_size):
           prev_set = set(self.dataset[self.set_names[i-1]])
           curr_set = set(self.dataset[self.set_names[i]])
           carryover = len(prev_set & curr_set)
           self.sequential_patterns[carryover] += 1

   def _analyze_temporal_clustering(self):
       """Analyze temporal clustering patterns"""
       for window_size in [3, 5, 7, 10]:
           if self.dataset_size < window_size:
               continue
              
           for i in range(len(self.set_names) - window_size + 1):
               window_sets = [self.dataset[self.set_names[i+j]] for j in range(window_size)]
               all_numbers = []
               for s in window_sets:
                   all_numbers.extend(s)

               unique_numbers = len(set(all_numbers))
               expected_unique = min(60, window_size * 6 * 0.8)

               if unique_numbers < expected_unique * 0.7:
                   self.temporal_patterns[f'cluster_{window_size}_{i}'] = {
                       'unique_count': unique_numbers,
                       'expected': expected_unique,
                       'clustering_strength': (expected_unique - unique_numbers) / expected_unique
                   }

   def _analyze_generation_biases(self):
       """Analyze generation biases"""
       biases = {
           'consecutive_pairs': 0,
           'multiples_of_5': 0,
           'multiples_of_10': 0,
           'same_digit_endings': 0,
           'arithmetic_sequences': 0
       }

       for numbers in self.dataset.values():
           sorted_nums = sorted(numbers)

           # Consecutive pairs
           for i in range(len(sorted_nums) - 1):
               if sorted_nums[i+1] - sorted_nums[i] == 1:
                   biases['consecutive_pairs'] += 1

           # Multiples
           biases['multiples_of_5'] += sum(1 for n in numbers if n % 5 == 0)
           biases['multiples_of_10'] += sum(1 for n in numbers if n % 10 == 0)

           # Same digit endings
           endings = [n % 10 for n in numbers]
           ending_counts = Counter(endings)
           biases['same_digit_endings'] += sum(1 for count in ending_counts.values() if count > 1)

           # Arithmetic sequences
           for i in range(len(sorted_nums) - 2):
               if sorted_nums[i+1] - sorted_nums[i] == sorted_nums[i+2] - sorted_nums[i+1]:
                   biases['arithmetic_sequences'] += 1

       self.vulnerability_patterns['generation_biases'] = biases

   def test_prediction(self, prediction: List[int], actual: List[int]) -> Dict:
       """Test prediction against actual result and return performance metrics"""
       matches = len(set(prediction) & set(actual))
       accuracy = matches / 6.0

       return {
           'prediction': prediction,
           'actual': actual,
           'matches': matches,
           'accuracy': accuracy,
           'partial_success': matches >= 2
       }

   def learn_and_adapt(self, prediction_results: List[Dict]):
       """Learn from prediction results and adapt model weights"""
       model_performance = {model: [] for model in self.model_weights}

       for result in prediction_results:
           for model in self.model_weights:
               if model in result:
                   model_performance[model].append(result[model]['accuracy'])

       total_performance = sum(sum(perfs) for perfs in model_performance.values() if perfs)

       if total_performance > 0:
           for model in self.model_weights:
               if model_performance[model]:
                   avg_performance = statistics.mean(model_performance[model])
                   self.model_weights[model] = (
                       self.model_weights[model] * (1 - self.adaptation_rate) +
                       (avg_performance / (total_performance / len([p for p in model_performance.values() if p]))) * self.adaptation_rate
                   )
               else:
                   self.model_weights[model] *= (1 - self.adaptation_rate)

       # Normalize weights
       total_weight = sum(self.model_weights.values())
       if total_weight > 0:
           for model in self.model_weights:
               self.model_weights[model] /= total_weight

       self.learning_history.append({
           'timestamp': datetime.now(),
           'weights': self.model_weights.copy(),
           'performance': model_performance
       })

   def get_enhanced_prediction(self) -> Dict:
       """Get comprehensive prediction with enhanced analysis"""
       # Perform all analyses
       self.analyze_enhanced_biases()
      
       # Generate predictions
       predictions = self.generate_enhanced_predictions(self.dataset_size)
      
       # Calculate enhanced confidence
       confidence = self._calculate_enhanced_confidence()
      
       # Generate risk assessment
       risk_assessment = self._assess_prediction_risks()
      
       # Generate alternative scenarios
       alternative_scenarios = self._generate_alternative_scenarios()
      
       return {
           'primary_prediction': predictions.get('enhanced_ensemble', predictions.get('ensemble', [])),
           'alternative_predictions': {k: v for k, v in predictions.items() if k != 'enhanced_ensemble'},
           'confidence': confidence,
           'risk_assessment': risk_assessment,
           'alternative_scenarios': alternative_scenarios,
           'model_weights': self.model_weights,
           'statistical_insights': {
               'anomalies': self.statistical_anomalies,
               'meta_patterns': dict(self.meta_patterns),
               'vulnerability_patterns': self.vulnerability_patterns
           }
       }

   def _assess_prediction_risks(self) -> Dict:
       """Assess risks and limitations of current predictions"""
       risks = {}
      
       # Data sufficiency risk
       if self.dataset_size < 20:
           risks['data_sufficiency'] = 'HIGH - Limited historical data may reduce accuracy'
       elif self.dataset_size < 50:
           risks['data_sufficiency'] = 'MEDIUM - Moderate data available'
       else:
           risks['data_sufficiency'] = 'LOW - Sufficient historical data'
      
       # Model overfitting risk
       recent_variance = statistics.stdev(list(self.ensemble_performance)[-10:]) if len(self.ensemble_performance) >= 10 else 0
       if recent_variance > 0.3:
           risks['overfitting'] = 'HIGH - High variance in recent predictions'
       elif recent_variance > 0.15:
           risks['overfitting'] = 'MEDIUM - Moderate prediction variance'
       else:
           risks['overfitting'] = 'LOW - Consistent prediction performance'
      
       # Statistical significance risk
       if self.statistical_anomalies and 'Not Significant' in str(self.statistical_anomalies.get('uniformity_p_value', '')):
           risks['statistical_significance'] = 'HIGH - No significant patterns detected'
       else:
           risks['statistical_significance'] = 'MEDIUM - Some patterns may exist'
      
       return risks

   def _generate_alternative_scenarios(self) -> Dict:
       """Generate alternative prediction scenarios"""
       scenarios = {}
      
       # Conservative scenario (favor cold numbers)
       all_numbers = []
       for numbers in self.dataset.values():
           all_numbers.extend(numbers)
       freq_counter = Counter(all_numbers)
      
       cold_numbers = [num for num, _ in freq_counter.most_common()[-30:]]
       if len(cold_numbers) >= self.game_params['draw_size']:
           scenarios['conservative'] = sorted(random.sample(cold_numbers, self.game_params['draw_size']))
      
       # Aggressive scenario (favor hot numbers)
       hot_numbers = [num for num, _ in freq_counter.most_common(20)]
       if len(hot_numbers) >= self.game_params['draw_size']:
           scenarios['aggressive'] = sorted(random.sample(hot_numbers, self.game_params['draw_size']))
      
       # Balanced scenario (mix of strategies)
       if hot_numbers and cold_numbers:
           balanced = []
           balanced.extend(random.sample(hot_numbers, min(3, len(hot_numbers))))
           balanced.extend(random.sample(cold_numbers, min(2, len(cold_numbers))))
          
           remaining = self.game_params['draw_size'] - len(balanced)
           if remaining > 0:
               available = [i for i in range(self.game_params['min_num'], self.game_params['max_num'] + 1)
                          if i not in balanced]
               balanced.extend(random.sample(available, min(remaining, len(available))))
          
           scenarios['balanced'] = sorted(list(set(balanced))[:self.game_params['draw_size']])
      
       return scenarios

   def display_enhanced_analysis(self):
       """Display comprehensive analysis results"""
       print("\n" + "=" * 100)
       print("ENHANCED ADAPTIVE LOTTERY ANALYSIS")
       print("=" * 100)
      
       # Statistical Anomalies
       if self.statistical_anomalies:
           print(f"\nSTATISTICAL ANOMALY ANALYSIS:")
           print(f"  Chi-square statistic: {self.statistical_anomalies.get('chi_square', 0):.2f}")
           print(f"  Uniformity p-value: {self.statistical_anomalies.get('uniformity_p_value', 'Unknown')}")
           print(f"  Data entropy: {self.statistical_anomalies.get('entropy', 0):.3f}")
          
           if 'gap_analysis' in self.statistical_anomalies:
               gap = self.statistical_anomalies['gap_analysis']
               print(f"  Average gap between numbers: {gap.get('overall_avg_gap', 0):.1f}")
               print(f"  Gap consistency: {gap.get('gap_consistency', 0):.1%}")
      
       # Meta Patterns
       if self.meta_patterns:
           print(f"\nMETA-PATTERN ANALYSIS:")
           for window, patterns in self.meta_patterns.items():
               if patterns:
                   pattern_freq = Counter(patterns)
                   most_common = pattern_freq.most_common(3)
                   print(f"  Window {window}: {len(set(patterns))} unique patterns, most common: {len(most_common)}")
      
       # Model Performance
       print(f"\nCURRENT MODEL WEIGHTS:")
       sorted_weights = sorted(self.model_weights.items(), key=lambda x: x[1], reverse=True)
       for model, weight in sorted_weights:
           print(f"  {model}: {weight:.3f}")
      
       # Confidence Assessment
       confidence = self._calculate_enhanced_confidence()
       print(f"\nCONFIDENCE ASSESSMENT:")
       print(f"  Overall Level: {confidence['confidence_level']}")
       print(f"  Expected Accuracy: {confidence['expected_accuracy']:.1%}")
      
       if 'factors' in confidence:
           print(f"  Contributing Factors:")
           for factor_name, factor_data in confidence['factors'].items():
               print(f"    {factor_name}: {factor_data['value']:.3f} (weight: {factor_data['weight']:.1f}) -> {factor_data['contribution']:.3f}")


def enhanced_main():
   """Enhanced main function with comprehensive testing"""
   # Enhanced sample data for better testing
   sample_data = {
       "SET_1": [10, 20, 22, 24, 28, 41],
       "SET_2": [5, 9, 14, 27, 31, 55],
       "SET_3": [5, 7, 28, 32, 40, 53],
       "SET_4": [18, 20, 34, 35, 45, 49],
       "SET_5": [36, 40, 42, 47, 51, 54],
       "SET_6": [2, 15, 23, 29, 44, 58],
       "SET_7": [8, 12, 19, 33, 46, 52],
       "SET_8": [3, 17, 25, 38, 41, 59],
       "SET_9": [11, 16, 21, 30, 43, 50],
       "SET_10": [6, 13, 26, 37, 48, 56],
   }
  
   # Custom game parameters (can be adjusted for different lotteries)
   game_params = {
       "min_num": 1,
       "max_num": 60,
       "draw_size": 6
   }
  
   try:
       scanner = EnhancedAdaptiveLotteryScanner(sample_data, game_params)
      
       print("=" * 80)
       print("PHASE 1: ENHANCED BIAS ANALYSIS")
       print("=" * 80)
       scanner.analyze_enhanced_biases()
      
       print("\n" + "=" * 80)
       print("PHASE 2: ENHANCED BACKTESTING")
       print("=" * 80)
       backtest_results = scanner.enhanced_backtest(test_window=6)  # Adjusted for sample size
      
       if backtest_results:
           print(f"Enhanced Backtesting Results:")
           print(f"  Total Tests: {backtest_results['total_tests']}")
           print(f"  Best Model: {backtest_results.get('best_model', 'Unknown')}")
          
           print(f"\n  Enhanced Model Performance:")
           for model, metrics in backtest_results['model_performance'].items():
               print(f"    {model}:")
               print(f"      Avg Accuracy: {metrics['avg_accuracy']:.1%}")
               print(f"      Success Rate (2+ matches): {metrics['success_rate_2plus']:.1%}")
               print(f"      Success Rate (3+ matches): {metrics['success_rate_3plus']:.1%}")
               print(f"      Consistency Score: {metrics['consistency_score']:.3f}")
               print(f"      Trend: {metrics['trend']['direction']} (strength: {metrics['trend']['strength']:.2f})")
          
           if 'confidence_calibration' in backtest_results:
               cal = backtest_results['confidence_calibration']
               print(f"\n  Confidence Calibration:")
               print(f"    Mean Predicted: {cal['mean_predicted_confidence']:.1%}")
               print(f"    Mean Actual: {cal['mean_actual_accuracy']:.1%}")
               print(f"    Calibration Error: {cal['calibration_error']:.3f}")
               print(f"    Overconfidence: {cal['overconfidence']:+.3f}")
          
           if 'statistical_significance' in backtest_results:
               sig = backtest_results['statistical_significance']
               print(f"\n  Statistical Significance:")
               print(f"    Performance vs Random: {sig['mean_performance']:.1%} vs {sig['baseline']:.1%}")
               print(f"    T-statistic: {sig['t_statistic']:.2f}")
               print(f"    Statistically Significant: {sig['is_significant']}")
               print(f"    Effect Size: {sig['effect_size']:.3f}")
      
       print("\n" + "=" * 80)
       print("PHASE 3: COMPREHENSIVE ANALYSIS")
       print("=" * 80)
       scanner.display_enhanced_analysis()
      
       print("\n" + "=" * 80)
       print("PHASE 4: ENHANCED PREDICTION")
       print("=" * 80)
       enhanced_prediction = scanner.get_enhanced_prediction()
      
       print(f"Primary Prediction: {enhanced_prediction['primary_prediction']}")
       print(f"Confidence: {enhanced_prediction['confidence']['confidence_level']} ({enhanced_prediction['confidence']['expected_accuracy']:.1%})")
      
       print(f"\nAlternative Predictions:")
       for model, prediction in enhanced_prediction['alternative_predictions'].items():
           print(f"  {model}: {prediction}")
      
       if enhanced_prediction['alternative_scenarios']:
           print(f"\nAlternative Scenarios:")
           for scenario, numbers in enhanced_prediction['alternative_scenarios'].items():
               print(f"  {scenario.title()}: {numbers}")
      
       print(f"\nRisk Assessment:")
       for risk_type, risk_level in enhanced_prediction['risk_assessment'].items():
           print(f"  {risk_type}: {risk_level}")
      
       return scanner, backtest_results
      
   except ValueError as e:
       print(f"Error initializing scanner: {e}")
       return None, None
   except Exception as e:
       print(f"Unexpected error: {e}")
       return None, None

if __name__ == "__main__":
   scanner, results = enhanced_main()
