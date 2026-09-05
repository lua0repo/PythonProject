import itertools
import statistics
import math
import numpy as np
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Optional
import json
import random
from datetime import datetime

class AdaptiveLotteryScanner:
   def __init__(self, dataset: Dict[str, List[int]]):
       """
       Adaptive system that learns from prediction failures to find systematic biases
       """
       for set_name, numbers in dataset.items():
           if len(numbers) != 6:
               raise ValueError(f"Set {set_name} does not contain exactly 6 numbers")
           if not all(isinstance(n, int) and 1 <= n <= 60 for n in numbers):
               raise ValueError(f"Set {set_name} contains invalid numbers")
       self.dataset = dataset
       self.set_names = list(dataset.keys())
       self.dataset_size = len(dataset)
       self.learning_history = []
       self.model_weights = {
           'frequency_bias': 0.2,
           'positional_bias': 0.2,
           'sequential_bias': 0.2,
           'temporal_bias': 0.2,
           'clustering_bias': 0.2
       }
       self.prediction_accuracy = {model: deque(maxlen=50) for model in self.model_weights}
       self.vulnerability_patterns = {}
       self.adaptation_rate = 0.1

       # Track systematic biases
       self.positional_analysis = {i: Counter() for i in range(6)}
       self.sequential_patterns = defaultdict(int)
       self.temporal_patterns = {}
       self.clustering_analysis = {}

       print(f"Adaptive Learning System initialized with {self.dataset_size} draws")
       print("Scanning for systematic vulnerabilities and biases...")

   def analyze_systematic_biases(self):
       """Analyze for non-random patterns that suggest system flaws"""
       if self.dataset_size < 2:
           print("Not enough data for sequential analysis")
           return

       # 1. Positional Bias Analysis
       for set_name, numbers in self.dataset.items():
           sorted_numbers = sorted(numbers)
           for position, number in enumerate(sorted_numbers):
               self.positional_analysis[position][number] += 1

       # 2. Sequential Pattern Analysis
       for i in range(1, self.dataset_size):
           prev_set = set(self.dataset[self.set_names[i-1]])
           curr_set = set(self.dataset[self.set_names[i]])

           # Analyze carry-over patterns
           carryover = len(prev_set & curr_set)
           self.sequential_patterns[carryover] += 1

       # 3. Temporal Clustering Analysis
       self._analyze_temporal_clustering()

       # 4. Number Generation Bias (consecutive numbers, multiples, etc.)
       self._analyze_generation_biases()

   def _analyze_temporal_clustering(self):
       """Look for time-based patterns that suggest algorithmic flaws"""
       for window_size in [3, 5, 7, 10]:
           for i in range(len(self.set_names) - window_size + 1):
               window_sets = [self.dataset[self.set_names[i+j]] for j in range(window_size)]
               all_numbers = []
               for s in window_sets:
                   all_numbers.extend(s)

               # Check for unusual clustering
               unique_numbers = len(set(all_numbers))
               expected_unique = min(60, window_size * 6 * 0.8)  # Expected with some overlap

               if unique_numbers < expected_unique * 0.7:  # Significant clustering
                   self.temporal_patterns[f'cluster_{window_size}_{i}'] = {
                       'unique_count': unique_numbers,
                       'expected': expected_unique,
                       'clustering_strength': (expected_unique - unique_numbers) / expected_unique
                   }

   def _analyze_generation_biases(self):
       """Analyze for algorithmic biases in number generation"""
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

   def generate_adaptive_prediction(self, target_set_index: int) -> Dict:
       """Generate prediction using current model weights"""
       predictions = {}

       # Model 1: Frequency Bias Exploitation
       freq_counter = Counter()
       recent_weight = 0.7
       for i, numbers in enumerate(self.dataset.values()):
           weight = recent_weight ** (self.dataset_size - i - 1)
           for num in numbers:
               freq_counter[num] += weight

       hot_numbers = [num for num, _ in freq_counter.most_common(20)]
       cold_numbers = [num for num, _ in freq_counter.most_common()[-20:]]

       predictions['frequency_bias'] = self._select_weighted_numbers(hot_numbers, cold_numbers)

       # Model 2: Positional Bias Exploitation
       positional_pred = []
       for pos in range(6):
           pos_freq = self.positional_analysis[pos]
           candidates = [num for num, _ in pos_freq.most_common(15)] if pos_freq else list(range(1, 61))
           positional_pred.append(random.choice(candidates))
       predictions['positional_bias'] = sorted(list(set(positional_pred)))[:6]
       if len(predictions['positional_bias']) < 6:
           available = [i for i in range(1, 61) if i not in predictions['positional_bias']]
           predictions['positional_bias'].extend(random.sample(available, 6 - len(predictions['positional_bias'])))
       predictions['positional_bias'] = sorted(predictions['positional_bias'][:6])

       # Model 3: Sequential Pattern Exploitation
       if self.dataset_size > 1:
           last_set = set(self.dataset[self.set_names[-1]])

           # Find most common carryover pattern
           most_common_carryover = max(self.sequential_patterns.items(), key=lambda x: x[1], default=(0, 0))[0]

           # Predict based on expected carryover
           sequential_pred = list(random.sample(list(last_set), min(most_common_carryover, len(last_set))))
           remaining = 6 - len(sequential_pred)
           available = [i for i in range(1, 61) if i not in sequential_pred]
           sequential_pred.extend(random.sample(available, min(remaining, len(available))))

           predictions['sequential_bias'] = sorted(sequential_pred[:6])
       else:
           predictions['sequential_bias'] = sorted(random.sample(range(1, 61), 6))

       # Model 4: Temporal Pattern Exploitation
       if self.temporal_patterns:
           # Use clustering patterns to avoid over-clustered numbers
           recent_numbers = set()
           for i in range(max(0, self.dataset_size-5), self.dataset_size):
               recent_numbers.update(self.dataset[self.set_names[i]])

           available_numbers = [i for i in range(1, 61) if i not in recent_numbers]
           if len(available_numbers) >= 6:
               predictions['temporal_bias'] = sorted(random.sample(available_numbers, 6))
           else:
               predictions['temporal_bias'] = sorted(random.sample(range(1, 61), 6))
       else:
           predictions['temporal_bias'] = sorted(random.sample(range(1, 61), 6))

       # Model 5: Clustering Bias Exploitation
       ranges = [(1, 12), (13, 24), (25, 36), (37, 48), (49, 60)]
       clustering_pred = []
       for r_start, r_end in ranges[:5]:
           clustering_pred.append(random.randint(r_start, r_end))
       clustering_pred.append(random.randint(1, 60))  # One random

       predictions['clustering_bias'] = sorted(list(set(clustering_pred)))[:6]

       # Ensemble prediction based on current weights
       ensemble = self._create_weighted_ensemble(predictions)
       predictions['ensemble'] = ensemble

       return predictions

   def _select_weighted_numbers(self, hot_numbers: List[int], cold_numbers: List[int]) -> List[int]:
       """Select numbers with bias toward hot numbers"""
       selection = []

       # Add 3-4 hot numbers
       if hot_numbers:
           selection.extend(random.sample(hot_numbers, min(4, len(hot_numbers))))

       # Add 1-2 cold numbers (contrarian play)
       if cold_numbers:
           selection.extend(random.sample(cold_numbers, min(2, len(cold_numbers))))

       # Fill remaining with random
       remaining = 6 - len(selection)
       available = [i for i in range(1, 61) if i not in selection]
       if remaining > 0 and available:
           selection.extend(random.sample(available, min(remaining, len(available))))

       return sorted(list(set(selection)))[:6]

   def _create_weighted_ensemble(self, predictions: Dict) -> List[int]:
       """Create ensemble prediction based on model weights"""
       number_votes = defaultdict(float)

       for model, weight in self.model_weights.items():
           if model in predictions:
               for number in predictions[model]:
                   number_votes[number] += weight

       # Select top 6 numbers by vote weight
       sorted_numbers = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)

       # If we don't have enough, fill with random
       selected = [num for num, _ in sorted_numbers[:6]]
       while len(selected) < 6:
           available = [i for i in range(1, 61) if i not in selected]
           if available:
               selected.append(random.choice(available))
           else:
               break

       return sorted(selected)

   def test_prediction(self, prediction: List[int], actual: List[int]) -> Dict:
       """Test prediction against actual result and return performance metrics"""
       matches = len(set(prediction) & set(actual))
       accuracy = matches / 6.0

       return {
           'prediction': prediction,
           'actual': actual,
           'matches': matches,
           'accuracy': accuracy,
           'partial_success': matches >= 2  # Consider 2+ matches as partial success
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
                   # Adaptive weight adjustment
                   self.model_weights[model] = (
                       self.model_weights[model] * (1 - self.adaptation_rate) +
                       (avg_performance / (total_performance / len([p for p in model_performance.values() if p]))) * self.adaptation_rate
                   )
               else:
                   self.model_weights[model] *= (1 - self.adaptation_rate)  # Slightly reduce weight for non-performing models

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

   def backtest_and_learn(self, test_window: int = 20):
       """Comprehensive backtesting with continuous learning"""
       if self.dataset_size < test_window + 10:
           print(f"Insufficient data for backtesting")
           return {}

       results = []
       # Initialize model_results to include 'ensemble'
       model_results = {model: [] for model in list(self.model_weights.keys()) + ['ensemble']}

       print(f"Starting adaptive backtesting on {test_window} predictions...")

       for i in range(self.dataset_size - test_window, self.dataset_size):
           # Create training dataset up to point i
           training_data = {name: numbers for j, (name, numbers) in enumerate(self.dataset.items()) if j < i}

           # Create temporary scanner with training data
           temp_scanner = AdaptiveLotteryScanner(training_data)
           temp_scanner.analyze_systematic_biases()

           # Generate predictions
           predictions = temp_scanner.generate_adaptive_prediction(i)
           actual = self.dataset[self.set_names[i]]

           # Test each model, including ensemble
           test_results = {}
           for model, prediction in predictions.items():
               result = self.test_prediction(prediction, actual)
               test_results[model] = result
               model_results[model].append(result['accuracy'])

           results.append({
               'set_index': i,
               'set_name': self.set_names[i],
               'predictions': test_results
           })

           # Learn from this result
           if len(results) > 1:
               self.learn_and_adapt([test_results])

       return self._analyze_backtest_results(results, model_results)

   def _analyze_backtest_results(self, results: List[Dict], model_results: Dict) -> Dict:
       """Analyze backtesting results and identify best performing models"""
       analysis = {
           'total_tests': len(results),
           'model_performance': {},
           'best_model': None,
           'improvement_trend': {},
           'vulnerability_exploitation': {}
       }

       # Calculate performance metrics for each model
       for model, accuracies in model_results.items():
           if accuracies:
               analysis['model_performance'][model] = {
                   'avg_accuracy': statistics.mean(accuracies),
                   'max_accuracy': max(accuracies),
                   'consistency': 1 - (statistics.stdev(accuracies) if len(accuracies) > 1 else 0),
                   'success_rate': sum(1 for acc in accuracies if acc >= 0.33) / len(accuracies),  # 2+ matches
                   'total_predictions': len(accuracies)
               }
           else:
               analysis['model_performance'][model] = {
                   'avg_accuracy': 0,
                   'max_accuracy': 0,
                   'consistency': 1,
                   'success_rate': 0,
                   'total_predictions': 0
               }

       # Find best performing model
       best_performance = 0
       for model, metrics in analysis['model_performance'].items():
           combined_score = metrics['avg_accuracy'] * 0.6 + metrics['success_rate'] * 0.4
           if combined_score > best_performance:
               best_performance = combined_score
               analysis['best_model'] = model

       # Analyze improvement trend
       if len(results) >= 10:
           first_half = model_results[analysis['best_model']][:len(results)//2]
           second_half = model_results[analysis['best_model']][len(results)//2:]

           if first_half and second_half:
               improvement = statistics.mean(second_half) - statistics.mean(first_half)
               analysis['improvement_trend'] = {
                   'first_half_avg': statistics.mean(first_half),
                   'second_half_avg': statistics.mean(second_half),
                   'improvement': improvement,
                   'is_learning': improvement > 0.05
               }

       return analysis

   def get_next_prediction(self) -> Dict:
       """Get prediction for the next draw using learned weights"""
       self.analyze_systematic_biases()
       predictions = self.generate_adaptive_prediction(self.dataset_size)

       # Add confidence based on learning history
       confidence = self._calculate_adaptive_confidence()

       return {
           'prediction': predictions['ensemble'],
           'alternative_models': predictions,
           'confidence': confidence,
           'model_weights': self.model_weights,
           'detected_vulnerabilities': self.vulnerability_patterns
       }

   def _calculate_adaptive_confidence(self) -> Dict:
       """Calculate confidence based on learning performance"""
       if not self.learning_history:
           return {'confidence_level': 'LOW', 'expected_accuracy': 0.16}

       recent_performance = self.learning_history[-5:]  # Last 5 learning cycles
       if recent_performance and 'performance' in recent_performance[-1]:
           avg_accuracies = []
           for perf_data in recent_performance:
               for model_perfs in perf_data['performance'].values():
                   if model_perfs:
                       avg_accuracies.extend(model_perfs)

           if avg_accuracies:
               expected_accuracy = statistics.mean(avg_accuracies)
               confidence_level = 'HIGH' if expected_accuracy > 0.4 else 'MEDIUM' if expected_accuracy > 0.25 else 'LOW'

               return {
                   'confidence_level': confidence_level,
                   'expected_accuracy': expected_accuracy,
                   'sample_size': len(avg_accuracies)
               }

       return {'confidence_level': 'LOW', 'expected_accuracy': 0.16}

   def display_learning_progress(self):
       """Display learning progress and model evolution"""
       print("\n" + "=" * 80)
       print("ADAPTIVE LEARNING PROGRESS")
       print("=" * 80)

       if self.learning_history:
           print(f"\nLearning Cycles: {len(self.learning_history)}")

           # Show weight evolution
           print(f"\nModel Weight Evolution:")
           initial_weights = list(self.learning_history[0]['weights'].values()) if self.learning_history else list(self.model_weights.values())
           current_weights = list(self.model_weights.values())

           for i, model in enumerate(self.model_weights.keys()):
               initial = initial_weights[i] if i < len(initial_weights) else 0.2
               current = current_weights[i]
               change = current - initial
               print(f"  {model}: {initial:.3f} → {current:.3f} ({change:+.3f})")

       print(f"\nDetected Vulnerability Patterns:")
       for pattern_type, data in self.vulnerability_patterns.items():
           print(f"  {pattern_type}: {data}")

       print(f"\nCurrent Model Weights:")
       for model, weight in sorted(self.model_weights.items(), key=lambda x: x[1], reverse=True):
           print(f"  {model}: {weight:.3f}")

def main():
   sample_data = {
       "SET_1": [10, 20, 22, 24, 28, 41],
       "SET_2": [5, 9, 14, 27, 31, 55],
       "SET_3": [5, 7, 28, 32, 40, 53],
       "SET_4": [18, 20, 34, 35, 45, 49],
       "SET_5": [36, 40, 42, 47, 51, 54],

   }

   scanner = AdaptiveLotteryScanner(sample_data)

   print("Phase 1: Initial Analysis")
   scanner.analyze_systematic_biases()

   print("Phase 2: Backtesting with Adaptive Learning")
   backtest_results = scanner.backtest_and_learn(test_window=10)

   print("Phase 3: Learning Progress Analysis")
   scanner.display_learning_progress()

   print("\n" + "=" * 80)
   print("BACKTEST RESULTS")
   print("=" * 80)

   if backtest_results:
       print(f"Total Tests: {backtest_results['total_tests']}")
       print(f"Best Model: {backtest_results.get('best_model', 'Unknown')}")

       print(f"\nModel Performance Summary:")
       for model, metrics in backtest_results['model_performance'].items():
           print(f"  {model}:")
           print(f"    Average Accuracy: {metrics['avg_accuracy']:.1%}")
           print(f"    Success Rate (2+ matches): {metrics['success_rate']:.1%}")
           print(f"    Max Accuracy: {metrics['max_accuracy']:.1%}")

       if 'improvement_trend' in backtest_results:
           trend = backtest_results['improvement_trend']
           print(f"\nLearning Trend:")
           print(f"  First Half Average: {trend['first_half_avg']:.1%}")
           print(f"  Second Half Average: {trend['second_half_avg']:.1%}")
           print(f"  Improvement: {trend['improvement']:+.1%}")
           print(f"  Is Learning: {trend['is_learning']}")

   print("Phase 4: Next Prediction")
   next_prediction = scanner.get_next_prediction()
   print(f"\nNext Prediction: {next_prediction['prediction']}")
   print(f"Confidence: {next_prediction['confidence']}")

   return scanner, backtest_results

if __name__ == "__main__":
   scanner, results = main()
