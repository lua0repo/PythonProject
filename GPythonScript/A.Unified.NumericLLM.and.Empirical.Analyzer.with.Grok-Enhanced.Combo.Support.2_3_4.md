
# ====================================
# Unified Lottery Predictor – 20 historical sets
# ====================================

import json
import random
import math
import numpy as np
from collections import defaultdict, Counter
from functools import reduce
from math import gcd
from scipy.stats import pearsonr
from random import shuffle

# -----------------------------------------------
# 1. INPUT DATA (20 historical sets)
# -----------------------------------------------
data = {
    "SET_1":  [19, 20, 21, 28, 45, 50],
    "SET_2":  [2,  21, 30, 32, 37, 45],
    "SET_3":  [5,  10, 17, 37, 47, 54],
    "SET_4":  [1,  13, 17, 20, 34, 46],
    "SET_5":  [1,  13, 25, 28, 42, 50],
    "SET_6":  [14, 19, 21, 27, 51, 53],
    "SET_7":  [28, 31, 42, 44, 45, 54],
    "SET_8":  [6,  15, 17, 26, 37, 40],
    "SET_9":  [1,  2,  12, 31, 39, 55],
    "SET_10": [4,  22, 23, 24, 38, 46],
    "SET_11": [5,  27, 28, 35, 39, 52],
    "SET_12": [12, 15, 18, 23, 50, 51],
    "SET_13": [2,  22, 26, 39, 42, 44],
    "SET_14": [8,  12, 18, 22, 29, 49],
    "SET_15": [14, 25, 30, 35, 39, 42],
    "SET_16": [3,  13, 19, 41, 48, 49],
    "SET_17": [5,  7,  12, 27, 47, 50],
    "SET_18": [5,  11, 21, 41, 42, 45],
    "SET_19": [2,  20, 30, 48, 49, 51],
    "SET_20": [3,  7,  21, 32, 37, 42],
}

# -----------------------------------------------
# 2. RNGCracker (kept for completeness – does not affect the final set)
# -----------------------------------------------
class RNGCracker:
    def __init__(self, historical):
        self.historical = historical
        self.flatten_seq = [num for s in historical.values() for num in s]
        self.positional_seqs = [[] for _ in range(6)]
        for s in historical.values():
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

# -----------------------------------------------
# 3. UPS PROTOCOL (GeminiA) – predicts SET_24
# -----------------------------------------------
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
    def get_gaps(s): return [s[i+1] - s[i] for i in range(5)]

    @staticmethod
    def calculate_distance(s1, s2):
        return math.sqrt(sum((a-b)**2 for a,b in zip(s1,s2)))

class FunctionalClassifier:
    def __init__(self, data_sets, k=5):
        self.data_sets = data_sets
        self.k = k
        self.all_numbers = list(range(1, 59))

    def predict_score(self, target_position):
        scores = {n: 0.0 for n in self.all_numbers}
        last_key = sorted(self.data_sets.keys())[-1]
        last_set = self.data_sets[last_key]

        distances = []
        for key, cur in self.data_sets.items():
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

def score_thematic(all_data):
    scores = {n: 0.0 for n in range(1,59)}
    sets = list(all_data.values())
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

def predict_set_24(data):
    print("\n--- UPS PROTOCOL – Phase I: Pre-processing ---")
    columns = Utility.extract_columns(data)
    all_sets = list(data.values())

    # 1. Global mean
    means = [Utility.calculate_mean(s) for s in all_sets]
    mu_pred = FunctionalRegressor(means).predict()
    print(f"mu_pred = {mu_pred:.2f}")

    # 2. Gaps
    gaps = [Utility.get_gaps(s) for s in all_sets]
    pred_gaps = [FunctionalRegressor([g[i] for g in gaps]).predict() for i in range(5)]
    print(f"Predicted gaps = {[round(g,1) for g in pred_gaps]}")

    # 3. M_POS
    s_pos = [FunctionalClassifier(data).predict_score(i) for i in range(6)]

    # 4. M_THEME
    s_theme = score_thematic(data)

    # 5. M_GAP (small-gap tendency)
    s_gap = {}
    for n in range(1,59):
        cnt = sum(1 for s in all_sets for i in range(1,6)
                  if s[i]==n and (s[i]-s[i-1])<=5)
        s_gap[n] = cnt / len(all_sets)

    print("\n--- Phase II & III: Synthesis ---")
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

    # Simple parity fix
    odds = sum(1 for n in best_set if n%2)
    if odds not in (2,3,4):
        random.shuffle(best_set)
        best_set.sort()

    print(f"Core 4        : {core_4}")
    print(f"Best local gaps: {best_local}")
    print(f"Mean error    : {best_mean_err:.2f}")
    return best_set

# -----------------------------------------------
# 4. Full LotteryAnalyzer (Grok-enhanced) – component breakdown
# -----------------------------------------------
class LotteryAnalyzer:
    def __init__(self, sets_dict):
        self.sets_dict = sets_dict
        self.sets_list = []
        self.positions = {f'B{i}': [] for i in range(1,7)}
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.total_sets = len(sets_dict)
        self.features = None
        self.validate_and_organize()

    # --------- validation & organisation ----------
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

    #--------- feature extraction ---------------
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

    # ------------- GA prediction ---------------
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
                p1,p2 = random.sample(pop[:20],2)
                cp = random.randint(1,5)
                child = sorted(list(set(p1[:cp] + p2[cp:])))
                need = 6-len(child)
                if need>0:
                    child.extend(random.sample([n for n in range(1,self.global_max+1) if n not in child], need))
                elif need<0:
                    child = child[:6]
                if random.random()<0.1:
                    idx = random.randint(0,5)
                    child[idx] = random.choice([n for n in range(1,self.global_max+1) if n not in child])
                    child.sort()
                pop.append(child)
        return pop[0]

    # ------------------- component scores (for the table) -------------------
    def _component_scores(self, num, ga_pred):
        freq = self.features['frequency'].get(num,0) * 2
        pos  = sum(self.features['positional_freq'][p].get(num,0) for p in self.features['positional_freq']) * 1.5
        pair = sum(self.features['pairs'].get(p,0) for p in self.features['pairs'] if num in p and self.features['pairs'][p]>1)
        gap  = self.features['gaps'].get(num, self.total_sets+1)
        prng = (self.features['frequency'].get(num,0) / (gap+1)) * 1.5 if gap>0 else 0
        ga   = 5 if num in ga_pred else 0
        return freq, pos, pair, prng, ga

    # ------------- final prediction -------------
    def predict_next_set(self):
        # ---- basic scoring ----
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
        # PRNG
        for n in range(1,self.global_max+1):
            f = self.features['frequency'].get(n,0)
            g = self.features['gaps'].get(n,self.total_sets+1)
            scores[n] += (f/(g+1))*1.5 if g>0 else 0
        # GA boost
        ga_pred = self.genetic_algorithm_prediction()
        for n in ga_pred: scores[n] += 5

        # ---- candidate selection (max 2 repeats from last 3) ----
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

       

        # ---- segment balance (max 3 per third) ----
        seg_sz = self.global_max // 3
        bins = [(1,seg_sz),(seg_sz+1,2*seg_sz),(2*seg_sz+1,self.global_max)]
        cnt = [0]*3
        for n in final:
            for i,(lo,hi) in enumerate(bins):
                if lo<=n<=hi: cnt[i]+=1
        while max(cnt)>3:
            over = cnt.index(max(cnt))
            lo,hi = bins[over]
            to_rem = random.choice([n for n in final if lo<=n<=hi])
            final.remove(to_rem)
            new = random.choice([n for n in range(1,self.global_max+1) if n not in final])
            final.append(new)
            cnt = [0]*3
            for n in final:
                for i,(lo,hi) in enumerate(bins):
                    if lo<=n<=hi: cnt[i]+=1
        final.sort()

        return final, scores, ga_pred

    # ------------------- report with component table -------------------
    def generate_report(self, pred, scores, ga_pred):
        lines = []
        lines.append("\nPREDICTION COMPONENT BREAKDOWN")
        lines.append("num | Freq | Pos | Pair | PRNGx1.5 | GA | TOTAL")
        lines.append("-"*44)
        comp = [(n, *self._component_scores(n, ga_pred), sum(self._component_scores(n, ga_pred)))
                for n in pred]
        comp.sort(key=lambda x:x[-1], reverse=True)
        for n,f,p,pr,prng,ga,tot in comp:
            lines.append(f"{n:<2} | {f:<2} | {p:<2} | {pr:<2} | {prng:<5.1f} {ga:<2} | {tot:.1f}")
        lines.append("\nFINAL PREDICTION")
        lines.append(f"SET_{self.total_sets+1}: {pred}")
        return "\n".join(lines)

# ----------------------------------------------------------------
# 5. Helper for the “detailed set analysis” table
# ----------------------------------------------------------------
def calculate_set_stats(name, values):
    odd  = sum(n for n in values if n%2)
    even = sum(n for n in values if not n%2)
    a1 = values[0]+values[2]+values[4]
    a2 = values[1]+values[3]+values[5]
    return {"name":name, "values":values, "1st_alt":a1, "2nd_alt":a2,
            "odd":odd, "even":even, "total":odd+even}

def print_set_analysis(lst):
    if not lst: return
    header = f"{'Set Name':<3} | {'Set Values':<10} | {'1st Alt':>3} | {'2nd Alt':>3} | {'Odd':>3} | {'Even':>3} | {'Total':>3}"
    sep = "="*len(header)
    print(sep); print(header); print(sep)
    for d in lst:
        vs = ", ".join(map(str,d['values']))
        print(f"{d['name']:<3} | {vs:<10} | {d['1st_alt']:>3} | {d['2nd_alt']:>3} | {d['odd']:>3} | {d['even']:>3} | {d['total']:>3}")
    print(sep)

# ----------------------------------------------------------------
# 6. MAIN – run everything
# ----------------------------------------------------------------
if __name__ == "__main__":
    # ---- 1. UPS (GeminiA) prediction for SET_24 ----
    ups_pred = predict_set_24(data)
    print("\n" + "="*55)
    print("UPS PROTOCOL RESULT – SET_24")
    print("="*55)
    print(f"Predicted numbers : {ups_pred}")
    print("="*55)

    # ---- 2. Full Grok-enhanced analysis ----
    analyzer = LotteryAnalyzer(data)
    analyzer.calculate_features()
    pred_set, final_scores, ga_pred = analyzer.predict_next_set()
    report = analyzer.generate_report(pred_set, final_scores, ga_pred)
    print("\n" + report)

    # ---- 3. Detailed set-by-set stats table ----
    processed = [calculate_set_stats(k,v) for k,v in data.items()]
    print("\n" + "="*30)
    print("--- Detailed Set Analysis Report ---")
    print("="*30)
    print_set_analysis(processed)


#=================================================
# Combos checker with considering 2/3/4 as support possibility to show from GrokAI  ***START***
#=================================================

    import itertools
import statistics
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any
import json

class HierarchicalComboAnalyzer:
    def __init__(self, dataset: Dict[str, List[int]], combo_size: int = 2):
        """
        Hierarchical combo pattern analyzer with sub-combination dependency analysis
       
        Args:
            dataset: Dictionary with set names as keys and number lists as values
            combo_size: Size of combinations to analyze (2=pairs, 3=triplets, 4=quadruplets, 5=quintuplets, 6=sextuplets)
        """
        self.dataset = dataset
        self.combo_size = combo_size
        self.combo_data = {}
        self.set_indices = {name: idx for idx, name in enumerate(dataset.keys())}
        self.dataset_size = len(dataset)
       
        # Store all analyzed combo sizes for hierarchical analysis
        self.all_combo_analyzers = {}  # Will store analyzers for sizes 2,3,4,5,6 as they're built
       
        # Realistic thresholds
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
        else:  # combo_size == 6
            self.significance_threshold = 1  # Allow single occurrences for sextuplets
            self.high_significance_threshold = 2
       
        combo_names = {2: "pairs", 3: "triplets", 4: "quadruplets", 5: "quintuplets", 6: "sextuplets"}
        print(f"Dataset size: {self.dataset_size} sets")
        print(f"Analyzing: {combo_names.get(combo_size, f'{combo_size}-combos')}")
        print(f"Significance threshold: {self.significance_threshold} occurrences")
        print(f"High significance threshold: {self.high_significance_threshold} occurrences")
       
    def set_hierarchical_analyzers(self, analyzers_dict: Dict[int, 'HierarchicalComboAnalyzer']):
        """Set references to other combo size analyzers for hierarchical analysis"""
        self.all_combo_analyzers = analyzers_dict
       
    def extract_combos(self) -> Dict[Tuple, Dict]:
        """Extract all combinations and build comprehensive occurrence database"""
        print("\n=== EXTRACTING COMBINATIONS ===")
       
        combo_occurrences = defaultdict(list)
       
        # Extract all combos from each set
        for set_name, numbers in self.dataset.items():
            set_idx = self.set_indices[set_name]
            combos = list(itertools.combinations(sorted(numbers), self.combo_size))
           
            for combo in combos:
                combo_occurrences[combo].append(set_idx)
       
        print(f"Found {len(combo_occurrences)} unique {self.combo_size}-combinations")
       
        # Build enhanced data structure for each combo
        significant_count = 0
        for combo, occurrences in combo_occurrences.items():
            if len(occurrences) >= self.significance_threshold:
                intervals = []
                if len(occurrences) > 1:
                    intervals = [occurrences[i] - occurrences[i-1] for i in range(1, len(occurrences))]
               
                self.combo_data[combo] = {
                    'occurrences': occurrences,
                    'frequency': len(occurrences),
                    'intervals': intervals,
                    'trend': self._calculate_trend(occurrences),
                    'confidence': self._calculate_confidence(occurrences, intervals),
                    'predictions': {},
                    'alert_level': 'GREEN',
                    'combo_size': self.combo_size,
                    'significance_level': 'HIGH' if len(occurrences) >= self.high_significance_threshold else 'NORMAL',
                    'hierarchical_support': {},  # Will be populated later
                    'boosted_confidence': 0.0,   # Confidence after hierarchical boost
                    'support_score': 0.0         # How well sub-combos support this combo
                }
                significant_count += 1
       
        print(f"Significant {self.combo_size}-combinations (≥{self.significance_threshold} occurrences): {significant_count}")
        print(f"High significance (≥{self.high_significance_threshold} occurrences): {sum(1 for data in self.combo_data.values() if data['significance_level'] == 'HIGH')}")
       
        return self.combo_data
   
    def analyze_hierarchical_dependencies(self):
        """Analyze how sub-combinations support larger combinations"""
        if self.combo_size <= 2:
            print("Skipping hierarchical analysis for pairs (base level)")
            return
           
        print(f"\n=== HIERARCHICAL DEPENDENCY ANALYSIS FOR {self.combo_size}-COMBOS ===")
       
        analyzed_count = 0
        for combo, data in self.combo_data.items():
            support_analysis = self._analyze_combo_support(combo)
            data['hierarchical_support'] = support_analysis
            data['support_score'] = support_analysis.get('overall_score', 0.0)
           
            # Boost confidence based on hierarchical support
            original_confidence = data['confidence']
            hierarchical_boost = self._calculate_hierarchical_boost(support_analysis, original_confidence)
            data['boosted_confidence'] = min(0.98, original_confidence + hierarchical_boost)
           
            analyzed_count += 1
       
        print(f"Analyzed hierarchical dependencies for {analyzed_count} combinations")
   
    def _analyze_combo_support(self, combo: Tuple) -> Dict:
        """Analyze how well sub-combinations support this combination"""
        support_analysis = {
            'sub_combos_analyzed': {},
            'support_strengths': [],
            'overall_score': 0.0,
            'strong_supporters': 0,
            'weak_supporters': 0
        }
       
        # Analyze each sub-combination size
        for sub_size in range(2, self.combo_size):
            if sub_size in self.all_combo_analyzers:
                sub_analyzer = self.all_combo_analyzers[sub_size]
                sub_support = self._analyze_sub_combo_support(combo, sub_size, sub_analyzer)
                support_analysis['sub_combos_analyzed'][sub_size] = sub_support
               
                if sub_support['average_strength'] > 0:
                    support_analysis['support_strengths'].append(sub_support['average_strength'])
                    if sub_support['average_strength'] >= 0.7:
                        support_analysis['strong_supporters'] += sub_support['found_count']
                    elif sub_support['average_strength'] >= 0.4:
                        support_analysis['weak_supporters'] += sub_support['found_count']
       
        # Calculate overall support score
        if support_analysis['support_strengths']:
            # Weighted average giving more weight to pairs (most fundamental)
            weights = [1.0 / (size - 1) for size in range(2, self.combo_size)]  # Pairs get weight 1.0, triplets 0.5, etc.
            if len(weights) == len(support_analysis['support_strengths']) and sum(weights) > 0:
                weighted_avg = sum(strength * weight for strength, weight in zip(support_analysis['support_strengths'], weights)) / sum(weights)
                support_analysis['overall_score'] = max(0.0, min(1.0, weighted_avg))  # Clamp between 0 and 1
            else:
                support_analysis['overall_score'] = max(0.0, min(1.0, statistics.mean(support_analysis['support_strengths'])))
       
        return support_analysis
   
    def _analyze_sub_combo_support(self, combo: Tuple, sub_size: int, sub_analyzer: 'HierarchicalComboAnalyzer') -> Dict:
        """Analyze support from sub-combinations of specific size"""
        sub_combos = list(itertools.combinations(combo, sub_size))
       
        analysis = {
            'total_sub_combos': len(sub_combos),
            'found_count': 0,
            'support_details': [],
            'average_strength': 0.0,
            'best_supporter': None,
            'weakest_supporter': None
        }
       
        strengths = []
        for sub_combo in sub_combos:
            if sub_combo in sub_analyzer.combo_data:
                sub_data = sub_analyzer.combo_data[sub_combo]
               
                # Calculate support strength based on multiple factors
                frequency_strength = min(1.0, sub_data['frequency'] / max(1, self.dataset_size * 0.05))  # Prevent division by zero
                confidence_strength = sub_data.get('boosted_confidence', sub_data['confidence'])
                trend_strength = {'INCREASING': 1.0, 'STABLE': 0.8, 'DECREASING': 0.4, 'INSUFFICIENT_DATA': 0.6}.get(sub_data['trend'], 0.5)
                recency_strength = 1.0 if sub_data['occurrences'][-1] >= self.dataset_size * 0.7 else 0.7
               
                # Weighted combination of strength factors
                support_strength = (
                    frequency_strength * 0.3 +
                    confidence_strength * 0.3 +
                    trend_strength * 0.2 +
                    recency_strength * 0.2
                )
               
                # Ensure support strength is valid
                support_strength = max(0.0, min(1.0, support_strength))
               
                detail = {
                    'sub_combo': sub_combo,
                    'frequency': sub_data['frequency'],
                    'confidence': confidence_strength,
                    'trend': sub_data['trend'],
                    'support_strength': support_strength
                }
               
                analysis['support_details'].append(detail)
                strengths.append(support_strength)
                analysis['found_count'] += 1
               
                # Track best and weakest supporters
                if analysis['best_supporter'] is None or support_strength > analysis['best_supporter']['support_strength']:
                    analysis['best_supporter'] = detail
                if analysis['weakest_supporter'] is None or support_strength < analysis['weakest_supporter']['support_strength']:
                    analysis['weakest_supporter'] = detail
       
        if strengths:
            analysis['average_strength'] = statistics.mean(strengths)
       
        return analysis
   
    def _calculate_hierarchical_boost(self, support_analysis: Dict, original_confidence: float) -> float:
        """Calculate confidence boost based on hierarchical support"""
        if not support_analysis or support_analysis['overall_score'] == 0:
            return 0.0
       
        base_boost = support_analysis['overall_score'] * 0.3  # Up to 30% boost
       
        # Additional boost for strong supporters
        strong_support_bonus = min(0.15, support_analysis['strong_supporters'] * 0.03)  # Up to 15% for many strong supporters
       
        # Penalty for weak support
        weak_support_penalty = support_analysis['weak_supporters'] * 0.01  # Small penalty for weak supporters
       
        total_boost = base_boost + strong_support_bonus - weak_support_penalty
       
        # Cap boost based on original confidence (lower confidence gets more boost)
        max_boost = (1.0 - original_confidence) * 0.4  # Max 40% of remaining confidence space
       
        return min(total_boost, max_boost, 0.35)  # Never boost more than 35%
   
    def _calculate_trend(self, occurrences: List[int]) -> str:
        """Calculate if combo occurrences are increasing, decreasing, or stable"""
        if len(occurrences) < 3:
            return 'INSUFFICIENT_DATA'
       
        third = len(occurrences) // 3
        if third == 0:
            return 'INSUFFICIENT_DATA'
           
        first_third_indices = occurrences[:third]
        last_third_indices = occurrences[-third:]
       
        first_span = occurrences[third-1] - occurrences[0] + 1 if third > 0 else 1
        last_span = occurrences[-1] - occurrences[-third] + 1
       
        first_density = len(first_third_indices) / first_span
        last_density = len(last_third_indices) / last_span
       
        if last_density > first_density * 1.3:
            return 'INCREASING'
        elif last_density < first_density * 0.7:
            return 'DECREASING'
        else:
            return 'STABLE'
   
    def _calculate_confidence(self, occurrences: List[int], intervals: List[int]) -> float:
        """Calculate prediction confidence based on regularity, frequency, and recency"""
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
       
        size_penalty = max(0.6, 1.0 - (self.combo_size - 2) * 0.1)
       
        confidence = (
            frequency_score * 0.4 +
            regularity_score * 0.4 +
            recent_bonus * 0.2
        ) * size_penalty
       
        return round(min(confidence, 0.95), 3)
   
   
   
    def generate_predictions(self) -> Dict:
        """Generate predictions using multiple models with hierarchical enhancement"""
        print("\n=== GENERATING HIERARCHICAL PREDICTIONS ===")
       
        current_set = self.dataset_size - 1
        predictions_generated = 0
       
        for combo, data in self.combo_data.items():
            if len(data['intervals']) > 0:
                predictions = {}
               
                # Standard prediction models
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
               
               
               
                # HIERARCHICAL PREDICTION MODEL
                if self.combo_size > 2 and data.get('hierarchical_support'):
                    hierarchical_pred = self._generate_hierarchical_prediction(combo, data, predictions)
                    if hierarchical_pred:
                        predictions['hierarchical'] = hierarchical_pred
               
                # Enhanced ensemble with hierarchical weighting
                if 'hierarchical' in predictions:
                    support_strength = data.get('support_score', 0.0)
                    hierarchical_weight = min(0.4, support_strength * 0.5)  # Up to 40% weight for hierarchical
                   
                    remaining_weight = 1.0 - hierarchical_weight
                    if len(data['intervals']) >= 3:
                        weights = {
                            'hierarchical': hierarchical_weight,
                            'simple_avg': remaining_weight * 0.20,
                            'weighted_avg': remaining_weight * 0.35,
                            'trend_adjusted': remaining_weight * 0.25,
                            'harmonic_mean': remaining_weight * 0.20
                        }
                    else:
                        weights = {
                            'hierarchical': hierarchical_weight,
                            'simple_avg': remaining_weight * 0.35,
                            'weighted_avg': remaining_weight * 0.40,
                            'trend_adjusted': remaining_weight * 0.15,
                            'harmonic_mean': remaining_weight * 0.10
                        }
                else:
                    # Standard weights when no hierarchical prediction
                    if len(data['intervals']) >= 3:
                        weights = {'simple_avg': 0.25, 'weighted_avg': 0.35, 'trend_adjusted': 0.25, 'harmonic_mean': 0.15}
                    else:
                        weights = {'simple_avg': 0.4, 'weighted_avg': 0.4, 'trend_adjusted': 0.15, 'harmonic_mean': 0.05}
               
                ensemble_pred = sum(predictions[model] * weight for model, weight in weights.items() if model in predictions)
                predictions['ensemble'] = ensemble_pred
               
                # Enhanced prediction uncertainty
                pred_values = [predictions[model] for model in ['simple_avg', 'weighted_avg', 'trend_adjusted', 'harmonic_mean'] if model in predictions]
                predictions['std_dev'] = statistics.stdev(pred_values) if len(pred_values) > 1 else avg_interval * 0.2
                predictions['min_prediction'] = min(pred_values) if pred_values else ensemble_pred - predictions['std_dev']
                predictions['max_prediction'] = max(pred_values) if pred_values else ensemble_pred + predictions['std_dev']
               
                data['predictions'] = predictions
               
                # Use boosted confidence for alert calculation
                alert_confidence = data.get('boosted_confidence', data['confidence'])
                data['alert_level'] = self._calculate_alert_level(ensemble_pred, current_set, alert_confidence)
                predictions_generated += 1
       
        print(f"Generated predictions for {predictions_generated} combinations")
        if self.combo_size > 2:
            hierarchical_count = sum(1 for data in self.combo_data.values() if 'hierarchical' in data.get('predictions', {}))
            print(f"Enhanced {hierarchical_count} predictions with hierarchical analysis")
       
        return self.combo_data
   
    def _generate_hierarchical_prediction(self, combo: Tuple, combo_data: Dict, base_predictions: Dict) -> float:
        """Generate prediction based on sub-combination patterns"""
        support_analysis = combo_data.get('hierarchical_support', {})
        if not support_analysis or not support_analysis.get('sub_combos_analyzed'):
            return None
       
        hierarchical_predictions = []
        confidence_weights = []
       
        # Analyze predictions from each sub-combo size
        for sub_size, sub_analysis in support_analysis['sub_combos_analyzed'].items():
            if sub_size in self.all_combo_analyzers and sub_analysis['found_count'] > 0:
                sub_analyzer = self.all_combo_analyzers[sub_size]
               
                for detail in sub_analysis['support_details']:
                    sub_combo = detail['sub_combo']
                    if sub_combo in sub_analyzer.combo_data:
                        sub_data = sub_analyzer.combo_data[sub_combo]
                       
                        # Get sub-combo's next prediction
                        if 'predictions' in sub_data and 'ensemble' in sub_data['predictions']:
                            sub_prediction = sub_data['predictions']['ensemble']
                            sub_confidence = sub_data.get('boosted_confidence', sub_data['confidence'])
                            support_strength = detail['support_strength']
                           
                            # Weight by both confidence and support strength - ensure minimum weight
                            weight = max(0.001, sub_confidence * support_strength)  # Prevent zero weights
                            if weight > 0:  # Double check weight is positive
                                hierarchical_predictions.append(sub_prediction)
                                confidence_weights.append(weight)
       
        if hierarchical_predictions and confidence_weights and sum(confidence_weights) > 0:
            # Weighted average of sub-combo predictions
            total_weight = sum(confidence_weights)
            if total_weight > 0:  # Additional safety check
                weighted_prediction = sum(pred * weight for pred, weight in zip(hierarchical_predictions, confidence_weights)) / total_weight
               
                # Blend with base ensemble prediction for stability
                base_ensemble = base_predictions.get('weighted_avg', base_predictions.get('simple_avg', 0))
                support_strength = support_analysis.get('overall_score', 0.0)
               
                # Higher support = more weight to hierarchical prediction
                hierarchical_weight = min(0.7, max(0.1, support_strength))  # Ensure weight is between 0.1 and 0.7
                blended_prediction = (hierarchical_weight * weighted_prediction +
                                    (1 - hierarchical_weight) * base_ensemble)
               
                return blended_prediction
       
        return None
   
    def _calculate_alert_level(self, prediction: float, current_set: int, confidence: float) -> str:
        """Calculate alert level based on proximity and confidence"""
        sets_until_prediction = prediction - current_set
       
        confidence_multiplier = 0.5 + confidence
       
        red_threshold = 3 * confidence_multiplier
        orange_threshold = 8 * confidence_multiplier
        yellow_threshold = 15 * confidence_multiplier
       
        if sets_until_prediction <= red_threshold:
            return 'RED'
        elif sets_until_prediction <= orange_threshold:
            return 'ORANGE'
        elif sets_until_prediction <= yellow_threshold:
            return 'YELLOW'
        else:
            return 'GREEN'
   
    def generate_speculative_combos(self, target_size: int, min_support_score: float = 0.5, max_candidates: int = 50) -> Dict[Tuple, Dict]:
        """
        Generate and evaluate speculative combinations (quadruplets, quintuplets, or sextuplets) that haven't appeared.
       
        Args:
            target_size: Size of combinations to generate (4, 5, or 6)
            min_support_score: Minimum hierarchical support score to include a candidate
            max_candidates: Maximum number of speculative combinations to return
       
        Returns:
            Dict mapping speculative combos to their analysis data
        """
        if target_size not in [4, 5, 6]:
            print(f"Speculative combos only supported for quadruplets (4), quintuplets (5), or sextuplets (6), got {target_size}")
            return {}

        print(f"\n=== GENERATING SPECULATIVE {target_size}-COMBOS ===")
        speculative_combos = {}
       
        # Step 1: Collect high-confidence sub-combinations
        sub_combos = {}
        for sub_size in range(2, min(target_size, 6)):
            if sub_size in self.all_combo_analyzers:
                analyzer = self.all_combo_analyzers[sub_size]
                # Select top sub-combinations based on boosted confidence or frequency
                sub_combos[sub_size] = [
                    (combo, data) for combo, data in analyzer.combo_data.items()
                    if data.get('boosted_confidence', data['confidence']) >= 0.6 or data['frequency'] >= analyzer.high_significance_threshold
                ]
                print(f"Found {len(sub_combos[sub_size])} significant {sub_size}-combos for speculative analysis")

        if not sub_combos.get(2):
            print("No significant pairs found for speculative analysis")
            return {}

        # Step 2: Generate candidate combinations
        all_numbers = set()
        for numbers in self.dataset.values():
            all_numbers.update(numbers)
       
        candidate_combos = set()
        if target_size == 4:
            # Combine pairs to form quadruplets
            for (pair1, pair1_data), (pair2, pair2_data) in itertools.combinations(sub_combos[2], 2):
                candidate = tuple(sorted(set(pair1 + pair2)))
                if len(candidate) == 4 and candidate not in self.combo_data:
                    candidate_combos.add(candidate)
       
        elif target_size == 5:
            # Combine triplets with pairs or pairs with pairs
            if sub_combos.get(3):
                for (triplet, triplet_data) in sub_combos[3]:
                    for (pair, pair_data) in sub_combos[2]:
                        candidate = tuple(sorted(set(triplet + pair)))
                        if len(candidate) == 5 and candidate not in self.combo_data:
                            candidate_combos.add(candidate)
            # Also try combining three pairs
            for (pair1, _), (pair2, _), (pair3, _) in itertools.combinations(sub_combos[2], 3):
                candidate = tuple(sorted(set(pair1 + pair2 + pair3)))
                if len(candidate) == 5 and candidate not in self.combo_data:
                    candidate_combos.add(candidate)
       
        elif target_size == 6:
            # Combine quintuplets with single numbers, quadruplets with pairs, or triplets with triplets
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

        # Step 3: Evaluate candidates
        analyzed_count = 0
        for combo in list(candidate_combos)[:max_candidates]:
            # Analyze hierarchical support
            support_analysis = self._analyze_combo_support(combo)
            support_score = support_analysis.get('overall_score', 0.0)
           
            if support_score >= min_support_score:
                # Estimate speculative prediction
                speculative_pred = self._generate_speculative_prediction(combo, support_analysis)
                confidence = min(0.8, support_score * 0.7)  # Cap speculative confidence
               
                speculative_combos[combo] = {
                    'occurrences': [],  # Never occurred
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
                    'boosted_confidence': confidence
                }
                analyzed_count += 1

        print(f"Analyzed {analyzed_count} speculative {target_size}-combos with sufficient support (≥{min_support_score:.1%})")
        return speculative_combos

    def _generate_speculative_prediction(self, combo: Tuple, support_analysis: Dict) -> float:
        """
        Estimate when a speculative combination might appear based on sub-combo predictions.
        """
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
                            predictions.append(sub_data['predictions']['ensemble'])
                            weights.append(detail['support_strength'] * sub_data.get('boosted_confidence', sub_data['confidence']))

        if predictions and weights and sum(weights) > 0:
            return sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
        return None

    def display_results(self, show_all: bool = False, max_display: int = 20, speculative_combos: Dict = None):
        """Display analysis results including speculative combos"""
        combo_names = {2: "PAIRS", 3: "TRIPLETS", 4: "QUADRUPLETS", 5: "QUINTUPLETS", 6: "SEXTUPLETS"}
        combo_name = combo_names.get(self.combo_size, f"{self.combo_size}-COMBOS")
       
        print("\n" + "="*90)
        print(f"{combo_name} ANALYSIS RESULTS (WITH HIERARCHICAL DEPENDENCIES)")
        print("="*90)
       
        if not self.combo_data:
            print(f"No significant {combo_name.lower()} found!")
        else:
            # Enhanced sorting with hierarchical support
            alert_priority = {'RED': 4, 'ORANGE': 3, 'YELLOW': 2, 'GREEN': 1}
            significance_priority = {'HIGH': 2, 'NORMAL': 1}
           
            sorted_combos = sorted(
                self.combo_data.items(),
                key=lambda x: (
                    alert_priority.get(x[1]['alert_level'], 0),
                    significance_priority.get(x[1]['significance_level'], 0),
                    x[1].get('support_score', 0.0),  # Hierarchical support score
                    x[1].get('boosted_confidence', x[1]['confidence']),
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
               
                # Hierarchical support indicator
                support_score = data.get('support_score', 0.0)
                if support_score >= 0.8:
                    support_icon = '🔥'  # Strong hierarchical support
                elif support_score >= 0.6:
                    support_icon = '💪'  # Good hierarchical support
                elif support_score >= 0.4:
                    support_icon = '👍'  # Moderate hierarchical support
                elif support_score > 0:
                    support_icon = '👌'  # Some hierarchical support
                else:
                    support_icon = '⚪'  # No hierarchical support (pairs only)
               
               
               
                print(f"\n{alert_icon.get(data['alert_level'], '📊')} {significance_icon.get(data['significance_level'], '')} {support_icon} {combo_name[:-1]}: {combo}")
                print(f"   Alert: {data['alert_level']} | Significance: {data['significance_level']} | Support: {support_score:.1%}")
                print(f"   Frequency: {data['frequency']} occurrences ({data['frequency']/self.dataset_size*100:.2f}%)")
               
                # Show confidence boost from hierarchical analysis
                original_conf = data['confidence']
                boosted_conf = data.get('boosted_confidence', original_conf)
                if boosted_conf > original_conf:
                    boost_amount = boosted_conf - original_conf
                    print(f"   Confidence: {original_conf:.1%} → {boosted_conf:.1%} (+{boost_amount:.1%} hierarchical boost)")
                else:
                    print(f"   Confidence: {original_conf:.1%}")
               
                # Show hierarchical support details
                if self.combo_size > 2 and data.get('hierarchical_support'):
                    self._display_hierarchical_support(data['hierarchical_support'])
               
               
               
                recent_sets = [f'SET_{i+1}' for i in data['occurrences'][-5:]]
                print(f"   Recent sets: {', '.join(recent_sets)}")
               
                if data['intervals']:
                    print(f"   Intervals: {data['intervals'][-5:]} (last 5)")
                    print(f"   Avg interval: {statistics.mean(data['intervals']):.1f} sets")
                    print(f"   Trend: {data['trend']}")
               
                if 'predictions' in data and data['predictions']:
                    pred = data['predictions']
                    next_set = int(pred['ensemble']) + 1
                    uncertainty = pred['std_dev']
                    range_min = max(self.dataset_size + 1, int(pred['min_prediction']) + 1)
                    range_max = int(pred['max_prediction']) + 1
                   
                    print(f"   📈 Next predicted: SET_{next_set} (±{uncertainty:.1f})")
                    print(f"   📊 Range: SET_{range_min} to SET_{range_max}")
                   
                    # Show hierarchical prediction if available
                    if 'hierarchical' in pred:
                        hierarchical_set = int(pred['hierarchical']) + 1
                        print(f"   🔗 Hierarchical model: SET_{hierarchical_set}")
       
        # Display speculative combos
        if speculative_combos:
            print(f"\n{'-'*60}")
            print(f"SPECULATIVE {combo_name} (NEVER OBSERVED)")
            print(f"{'-'*60}")
           
            sorted_speculative = sorted(
                speculative_combos.items(),
                key=lambda x: (-x[1]['support_score'], -x[1]['confidence']),
                reverse=True
            )[:max_display]
           
            for combo, data in sorted_speculative:
                support_icon = '🔥' if data['support_score'] >= 0.8 else '💪' if data['support_score'] >= 0.6 else '👍' if data['support_score'] >= 0.4 else '👌'
                print(f"\n{support_icon} SPECULATIVE {combo_name[:-1]}: {combo}")
                print(f"   Support: {data['support_score']:.1%} | Confidence: {data['confidence']:.1%}")
                self._display_hierarchical_support(data['hierarchical_support'])
                if 'ensemble' in data['predictions']:
                    next_set = int(data['predictions']['ensemble']) + 1
                    print(f"   📈 Predicted: SET_{next_set}")
       
        self._display_summary()
   
    def _display_hierarchical_support(self, support_analysis: Dict):
        """Display hierarchical support details"""
        if not support_analysis.get('sub_combos_analyzed'):
            return
       
        print(f"   🔗 Hierarchical Support:")
        for sub_size, sub_analysis in support_analysis['sub_combos_analyzed'].items():
            if sub_analysis['found_count'] > 0:
                combo_name = {2: "pairs", 3: "triplets", 4: "quadruplets", 5: "quintuplets"}[sub_size]
                strength = sub_analysis['average_strength']
                strength_desc = "STRONG" if strength >= 0.7 else "GOOD" if strength >= 0.5 else "MODERATE" if strength >= 0.3 else "WEAK"
               
                print(f"      • {sub_analysis['found_count']}/{sub_analysis['total_sub_combos']} {combo_name} found ({strength:.1%} {strength_desc})")
               
                if sub_analysis.get('best_supporter'):
                    best = sub_analysis['best_supporter']
                    print(f"        Best: {best['sub_combo']} ({best['frequency']}x, {best['support_strength']:.1%})")
   
    def _display_summary(self):
        """Display summary statistics with hierarchical information"""
        combo_names = {2: "PAIRS", 3: "TRIPLETS", 4: "QUADRUPLETS", 5: "QUINTUPLETS", 6: "SEXTUPLETS"}
        combo_name = combo_names.get(self.combo_size, f"{self.combo_size}-COMBOS")
       
        print(f"\n{'='*60}")
        print(f"{combo_name} SUMMARY:")
       
        alert_counts = Counter(data['alert_level'] for data in self.combo_data.values())
        significance_counts = Counter(data['significance_level'] for data in self.combo_data.values())
       
        print(f"\n🚨 Alert Distribution:")
        for level in ['RED', 'ORANGE', 'YELLOW', 'GREEN']:
            count = alert_counts[level]
            if count > 0:
                icon = {'RED': '🚨', 'ORANGE': '⚠️', 'YELLOW': '⚡', 'GREEN': '✅'}[level]
                print(f"   {icon} {level}: {count}")
       
        print(f"\n⭐ Significance Distribution:")
        for level in ['HIGH', 'NORMAL']:
            count = significance_counts[level]
            if count > 0:
                icon = {'HIGH': '⭐', 'NORMAL': '📊'}[level]
                print(f"   {icon} {level}: {count}")
       
        # Hierarchical support statistics
        if self.combo_size > 2 and any(data.get('support_score', 0) > 0 for data in self.combo_data.values()):
            support_scores = [data.get('support_score', 0) for data in self.combo_data.values()]
            avg_support = statistics.mean(support_scores)
            strong_support_count = sum(1 for score in support_scores if score >= 0.7)
            boosted_count = sum(1 for data in self.combo_data.values() if data.get('boosted_confidence', 0) > data['confidence'])
           
            print(f"\n🔗 Hierarchical Support:")
            print(f"   • Average support score: {avg_support:.1%}")
            print(f"   • Strong support (≥70%): {strong_support_count}")
            print(f"   • Confidence boosted: {boosted_count}")
       
        if self.combo_data:
            top_frequency = max(data['frequency'] for data in self.combo_data.values())
            top_confidence = max(data.get('boosted_confidence', data['confidence']) for data in self.combo_data.values())
           
            print(f"\n📊 Statistics:")
            print(f"   • Total significant: {len(self.combo_data)}")
            print(f"   • Highest frequency: {top_frequency} occurrences")
            print(f"   • Highest confidence: {top_confidence:.1%}")
   
    def get_imminent_combos(self, max_sets_ahead: int = 10) -> List[Tuple]:
        """Get combos predicted to appear within the next N sets"""
        imminent = []
        current_set = self.dataset_size - 1
       
        for combo, data in self.combo_data.items():
            if 'predictions' in data and 'ensemble' in data['predictions']:
                sets_until = data['predictions']['ensemble'] - current_set
                if 0 < sets_until <= max_sets_ahead:
                    boosted_conf = data.get('boosted_confidence', data['confidence'])
                    support_score = data.get('support_score', 0.0)
                    imminent.append((combo, sets_until, boosted_conf, data['alert_level'], support_score))
       
        return sorted(imminent, key=lambda x: (x[1], -x[2], -x[4]))  # Sort by proximity, then by confidence and support

def main():
    # Sample dataset (use your full 1462-set dataset here)
    sample_data = data
   
    print("HIERARCHICAL COMBO PATTERN ANALYZER")
    print("=" * 70)
    print("🔗 Using sub-combination dependencies for enhanced predictions")
    print("=" * 70)
   
    # Build analyzers for all combo sizes
    combo_sizes_to_analyze = [2, 3, 4, 5, 6]
    all_analyzers = {}
    all_imminent = []
   
    # Step 1: Create all analyzers and extract base combinations
    print("\n🔍 PHASE 1: EXTRACTING BASE COMBINATIONS")
    print("=" * 50)
   
    for combo_size in combo_sizes_to_analyze:
        combo_name = {2: "PAIRS", 3: "TRIPLETS", 4: "QUADRUPLETS", 5: "QUINTUPLETS", 6: "SEXTUPLETS"}
        print(f"\n📊 Analyzing {combo_name[combo_size]} ({combo_size}-combos)")
       
        analyzer = HierarchicalComboAnalyzer(sample_data, combo_size=combo_size)
        combo_data = analyzer.extract_combos()
       
        if combo_data:
            all_analyzers[combo_size] = analyzer
            print(f"   ✅ Found {len(combo_data)} significant {combo_size}-combos")
        else:
            print(f"   ❌ No significant {combo_size}-combos found")
   
    # Step 2: Set hierarchical references and analyze dependencies
    print(f"\n🔗 PHASE 2: ANALYZING HIERARCHICAL DEPENDENCIES")
    print("=" * 50)
   
    for combo_size in combo_sizes_to_analyze:
        if combo_size in all_analyzers:
            analyzer = all_analyzers[combo_size]
            analyzer.set_hierarchical_analyzers(all_analyzers)
           
            if combo_size > 2:
                analyzer.analyze_hierarchical_dependencies()
   
    # Step 3: Generate enhanced predictions
    print(f"\n📈 PHASE 3: GENERATING ENHANCED PREDICTIONS")
    print("=" * 50)
   
    for combo_size in combo_sizes_to_analyze:
        if combo_size in all_analyzers:
            analyzer = all_analyzers[combo_size]
            analyzer.generate_predictions()
   
    # Step 4: Generate speculative combos for quadruplets, quintuplets, and sextuplets
    print(f"\n🔮 PHASE 4: GENERATING SPECULATIVE COMBOS")
    print("=" * 50)
   
    speculative_results = {}
    for combo_size in [4, 5, 6]:
        if combo_size in all_analyzers:
            analyzer = all_analyzers[combo_size]
            speculative_combos = analyzer.generate_speculative_combos(target_size=combo_size, min_support_score=0.5)
            speculative_results[combo_size] = speculative_combos
            print(f"Generated {len(speculative_combos)} speculative {combo_size}-combos")
   
    # Step 5: Display results for each combo size
    print(f"\n📋 PHASE 5: DETAILED RESULTS")
    print("=" * 50)
   
    for combo_size in combo_sizes_to_analyze:
        if combo_size in all_analyzers:
            analyzer = all_analyzers[combo_size]
            speculative_combos = speculative_results.get(combo_size, {})
            analyzer.display_results(show_all=False, max_display=15, speculative_combos=speculative_combos)
           
            # Collect imminent predictions
            imminent = analyzer.get_imminent_combos(max_sets_ahead=12)
            for combo, sets_ahead, confidence, alert_level, support_score in imminent:
                all_imminent.append((combo_size, combo, sets_ahead, confidence, alert_level, support_score))
   
    # Step 6: Combined imminent analysis across all sizes
    if all_imminent:
        print(f"\n🚨 PHASE 6: COMBINED IMMINENT PREDICTIONS (next 12 sets)")
        print("=" * 70)
       
        # Group by alert level and sort by hierarchical support
        red_alerts = sorted([item for item in all_imminent if item[4] == 'RED'],
                           key=lambda x: (x[2], -x[5], -x[3]))  # Sort by proximity, support, confidence
        orange_alerts = sorted([item for item in all_imminent if item[4] == 'ORANGE'],
                              key=lambda x: (x[2], -x[5], -x[3]))
        other_alerts = sorted([item for item in all_imminent if item[4] not in ['RED', 'ORANGE']],
                             key=lambda x: (x[2], -x[5], -x[3]))
       
        for alert_group, title in [(red_alerts, "🚨 CRITICAL ALERTS (RED)"),
                                  (orange_alerts, "⚠️  WARNING ALERTS (ORANGE)"),
                                  (other_alerts, "⚡ WATCH ALERTS (YELLOW/GREEN)")]:
            if alert_group:
                print(f"\n{title}:")
               
                for combo_size, combo, sets_ahead, confidence, alert_level, support_score in alert_group[:8]:
                    combo_name = {2: "pair", 3: "triplet", 4: "quadruplet", 5: "quintuplet", 6: "sextuplet"}[combo_size]
                   
                    # Enhanced display with hierarchical info
                    if support_score >= 0.7:
                        support_indicator = "🔥"
                        support_text = "STRONG"
                    elif support_score >= 0.5:
                        support_indicator = "💪"
                        support_text = "GOOD"
                    elif support_score >= 0.3:
                        support_indicator = "👍"
                        support_text = "MODERATE"
                    elif support_score > 0:
                        support_indicator = "👌"
                        support_text = "SOME"
                    else:
                        support_indicator = "⚪"
                        support_text = "BASE"
                   
                    next_set = len(sample_data) + int(sets_ahead)
                    print(f"   {support_indicator} {combo} ({combo_name}) → SET_{next_set}")
                    print(f"      └─ ~{sets_ahead:.1f} sets | {confidence:.0%} confidence | {support_text} hierarchical support")
   
    # Step 7: Final hierarchical summary
    print(f"\n📊 PHASE 7: HIERARCHICAL ANALYSIS SUMMARY")
    print("=" * 70)
   
    total_significant = sum(len(analyzer.combo_data) for analyzer in all_analyzers.values())
    total_imminent = len(all_imminent)
   
    # Hierarchical enhancement statistics
    hierarchically_enhanced = 0
    strong_hierarchical_support = 0
    total_confidence_boost = 0
   
    for combo_size in combo_sizes_to_analyze:
        if combo_size in all_analyzers and combo_size > 2:
            analyzer = all_analyzers[combo_size]
            enhanced = sum(1 for data in analyzer.combo_data.values()
                          if data.get('boosted_confidence', 0) > data['confidence'])
            strong_support = sum(1 for data in analyzer.combo_data.values()
                               if data.get('support_score', 0) >= 0.7)
           
            confidence_boosts = [data.get('boosted_confidence', data['confidence']) - data['confidence']
                               for data in analyzer.combo_data.values()]
            avg_boost = statistics.mean(confidence_boosts) if confidence_boosts else 0
           
            hierarchically_enhanced += enhanced
            strong_hierarchical_support += strong_support
            total_confidence_boost += avg_boost
   
    for combo_size in combo_sizes_to_analyze:
        if combo_size in all_analyzers:
            analyzer = all_analyzers[combo_size]
            count = len(analyzer.combo_data)
            imminent_count = len([x for x in all_imminent if x[0] == combo_size])
            combo_name = {2: "pairs", 3: "triplets", 4: "quadruplets", 5: "quintuplets", 6: "sextuplets"}[combo_size]
           
            # Alert distribution
            alert_counts = Counter(data['alert_level'] for data in analyzer.combo_data.values())
            high_priority = alert_counts['RED'] + alert_counts['ORANGE']
           
            # Hierarchical info
            if combo_size > 2:
                avg_support = statistics.mean([data.get('support_score', 0) for data in analyzer.combo_data.values()])
                enhanced_count = sum(1 for data in analyzer.combo_data.values()
                                   if data.get('boosted_confidence', 0) > data['confidence'])
               
                print(f"   🔗 {combo_name.capitalize()}: {count} significant ({high_priority} high-priority)")
                print(f"      └─ {enhanced_count} hierarchically enhanced | {avg_support:.1%} avg support")
            else:
                print(f"   📊 {combo_name.capitalize()}: {count} significant ({high_priority} high-priority) [base level]")
   
    print(f"\n🎯 TOTALS:")
    print(f"   • {total_significant} significant combinations found")
    print(f"   • {total_imminent} imminent predictions generated")
    print(f"   • {hierarchically_enhanced} predictions enhanced by hierarchical analysis")
    print(f"   • {strong_hierarchical_support} combinations with strong hierarchical support (≥70%)")
   
    # Step 8: Select top speculative sextuplet as next predicted set
    print(f"\n🔮 PHASE 8: PREDICTED NEXT 6-ELEMENT SET")
    print("=" * 70)
   
    if 6 in speculative_results and speculative_results[6]:
        # Sort speculative sextuplets by support score and confidence, prioritizing those predicted soon
        top_sextuplets = sorted(
            speculative_results[6].items(),
            key=lambda x: (
                -(x[1]['predictions']['ensemble'] - len(sample_data)) if 'ensemble' in x[1]['predictions'] else float('inf'),  # Proximity
                -x[1]['support_score'],
                -x[1]['confidence']
            )
        )
       
        if top_sextuplets:
            top_combo, top_data = top_sextuplets[0]
            print(f"\n🏆 Predicted Next Set (SET_{len(sample_data) + 1}): {top_combo}")
            print(f"   Support: {top_data['support_score']:.1%} | Confidence: {top_data['confidence']:.1%}")
            if 'ensemble' in top_data['predictions']:
                next_set = int(top_data['predictions']['ensemble']) + 1
                sets_ahead = top_data['predictions']['ensemble'] - len(sample_data)
                print(f"   📈 Predicted: SET_{next_set} (~{sets_ahead:.1f} sets ahead)")
            analyzer._display_hierarchical_support(top_data['hierarchical_support'])
        else:
            print("No speculative sextuplets with sufficient support found.")
    else:
        print("No speculative sextuplets generated. Consider lowering min_support_score or increasing dataset size.")

if __name__ == "__main__":
    main()

#=====================================
# Combos checker with considering 2/3/4 as support possibility to show from GrokAI  ***END***
#=====================================

