# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** EntityLink AI  
**Problem:** Large-Scale Commercial Entity Resolution across 3 Independent Data Sources  
**Evaluation Metric:** Macro-averaged F_0.5 Score  

---

## 1. Executive Summary
We propose a scalable, two-stage Entity Resolution architecture capable of matching over 12 million business records across 3 noisy sources under a precision-heavy macro F_0.5 metric. Our solution integrates:
1. A country-partitioned multi-key inverted index candidate blocker achieving **93.07% candidate recall** with an average of only 25 candidates per reference entity.
2. A hybrid decision architecture combining a calibrated non-linear LightGBM Gradient Boosted Decision Tree (trained on 22 engineered pairwise features) with high-speed token-order invariant RapidFuzz string metrics.
3. Precision-prioritized singleton gating, achieving an out-of-fold **Macro F_0.5 score of 88.62%** with **92.30% precision** and **93.06% singleton accuracy**.

---

## 2. Methodology

### 2.1 Problem Analysis & Exploratory Findings
- **Intra-Country Invariance:** An exhaustive audit of all 7,638,365 training ground-truth matches confirmed 0 cross-country matches. The search space can be partitioned strictly by country (`US`, `India`, `France`), drastically reducing candidate generation space while maintaining 100% theoretical recall.
- **Unseen Geographic Domain (France):** While training data strictly covers US (~60%) and India (~40%), the test set contains ~15% French records (`SARL`, `SASU`, `SA`, `Rue`, `Bd`). All normalizations were designed language-agnostic to prevent overfitting.
- **Multilingual & Indic Script Noise:** In India, business names frequently appear in Latin English in Source 1 but Devanagari (Hindi), Tamil, Odia, or Gujarati script in Sources 2 and 3. In these instances, address keywords (e.g. apartment names, street names, PIN codes) share identical English roots, making address-token blocking essential.
- **Domain Names as Entity Names:** Businesses in Source 3 frequently appear as web domains (e.g., `painterslocal46 com`, `celestialmemorialtrust com`). We developed domain unpacking and compact root tokenization to resolve them.
- **Word Transpositions & Aliases:** Word order transpositions (`XX Nippon Apex` vs `XX Apex Nippon`) and DBA aliases (`Halodelta aka Clemons Silver Eastern Inc`) were prominent.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Key Inverted Index Blocking + Pairwise Tabular Feature Extraction + Calibrated Hybrid LightGBM / RapidFuzz GBDT Matcher.

**Core Innovation:**
1. *Multi-Key Blended Inverted Index:* Combines root name n-grams, compact domain prefixes, address numbers (with leading-zero normalization), and distinctive address vocabulary.
2. *Precision-Calibrated Singleton Gate:* Prevents false merges on singletons (5.58% of entities) where a single false positive reduces the entity score from 1.0 to 0.0.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.7 	imes 10^{13}$ pairwise comparison space down to a tractable pool:
- **Country Partitioning:** Blocks queries strictly within their declared country.
- **Blocking Keys:**
  - `n_<token>`: Significant root name tokens (length $\ge 3$, stop words excluded).
  - `n2_<tok1>_<tok2>`: Two-token prefix compound.
  - `cp_<compact>`: First 8 characters of compact name (captures domain names without spaces).
  - `num_<digits>`: Zero-stripped numeric tokens (house, shop, door numbers, PIN codes).
  - `aw_<token>`: Distinctive address landmark / locality tokens (e.g. `rahul`, `okhla`, `malad`).
- **Posting List Pruning:** Keys with posting lists $> 25,000$ entries are dynamically pruned to preserve sub-second query latency.
- **Candidate Quality:**
  - **Candidate Recall Ceiling:** **93.07%** (161,049 out of 173,044 true matches retrieved).
  - **Candidates per Entity:** 25 candidates.
  - **Space Reduction Ratio:** $> 99.98%$.

---

## 4. Matching Model

### 4.1 Features Used (22 Pairwise Features)
1. **Name Similarity Features:**
   - `n_fuzz_ratio`, `n_token_sort`, `n_token_set`, `n_partial` (RapidFuzz metrics).
   - `root_sort`: Levenshtein ratio on suffix-stripped root names.
   - `root_exact`: Binary indicator of exact root name identity.
   - `first_match`: Binary indicator of identical initial word.
   - `len_diff`, `len_ratio`: Token and character length ratios.
   - `n_jaccard`: Word-level Jaccard similarity.
   - `compact_sub`: Substring containment flag for domain names.
2. **Address Similarity Features:**
   - `has_a2`: Indicator for non-empty candidate address.
   - `a_fuzz_ratio`, `a_token_sort`, `a_token_set`, `a_jaccard`: Multi-metric address string similarities.
   - `num_overlap`, `num_exact`: Numeric token intersection and identity.
   - `word_overlap`: Distinctive address word intersection ratio.
3. **Context & Source Features:**
   - `is_source2`: Source 2 vs Source 3 binary indicator.
   - `b_rank`, `b_score`: Inverted index rank and token frequency score.

### 4.2 Model Type & Training
- **Model:** LightGBM Gradient Boosted Decision Tree (`LGBMClassifier`) with 300 estimators, depth 6, and leaf count 31.
- **Training Data:** 35,000 reference entities generating 217,916 pairwise training samples with hard negatives mined directly from the candidate blocking phase.
- **Top Predictive Features:**
  1. `n_partial` (handles domain names and prefixes)
  2. `root_sort` (core name match)
  3. `a_token_set` & `a_jaccard` (address token overlap)
  4. `len_ratio` (length consistency)

### 4.3 Threshold Selection Method
- Due to the precision weighting of the macro F_0.5 metric ($eta = 0.5$) and the presence of singletons, probabilities were calibrated via grid search on holdout data.
- The optimal threshold is $	au = 0.85$ with a singleton confidence margin $\max(P) \ge 	au + 0.02$.

---

## 5. Results & Error Analysis

### 5.1 Metric Progression
| Iteration | Blocking Recall | Macro Precision | Macro Recall | **Macro F_0.5** | Singleton Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Heuristic** | 80.34% | 84.72% | 79.30% | **81.94%** | 87.29% |
| **Enhanced Multi-Key** | 93.07% | 85.02% | 84.64% | **83.57%** | 85.35% |
| **LightGBM Hybrid (Advanced)** | **93.07%** | **92.30%** | **81.92%** | **88.62%** | **93.06%** |

### 5.2 Error Analysis
- **False Positives (Wrong Merges):** Common when two distinct franchises or branches share an identical business name (e.g., generic local stores) with partial address overlap in dense urban areas. Mitigated by requiring strict number/PIN matches.
- **False Negatives (Missed Matches):** Common when a record features complete transliteration in non-Latin script combined with completely missing address fields in Source 2/3 (~3.3% of target records have null addresses).

---

## 6. Conclusion
By decomposing the problem by country, deploying multi-key inverted index blocking, and combining a 22-feature LightGBM GBDT with calibrated singleton thresholding, our pipeline scales efficiently to 12M+ records. It achieves **88.62% Macro F_0.5** on out-of-fold validation while executing at over **390 entities/second** on standard 12-core CPU hardware.

---

## Appendix

### A. Code Artefacts
- `src/config.py`: Central path configurations.
- `src/preprocessor.py`: Text cleaning, domain unpacking, and address tokenization.
- `src/blocking.py`: Multi-key country-partitioned inverted index candidate generator.
- `src/feature_engineering.py`: 22-dimensional pairwise tabular feature extractor.
- `src/baseline_matcher.py`: Fast RapidFuzz similarity matcher.
- `src/hybrid_matcher.py`: LightGBM + RapidFuzz hybrid ensemble.
- `src/train_classifier.py`: LightGBM training and feature importance extraction.
- `src/evaluate.py`: Fast validation benchmark on 50k entities.
- `src/predict.py`: Full test set inference generator.
- `requirements.txt`: Pinned dependencies.
