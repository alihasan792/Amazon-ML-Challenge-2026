# Amazon ML Challenge 2026: Business Entity Resolution

An end-to-end, high-performance Machine Learning solution for large-scale multi-source commercial entity resolution and record linkage.

## Problem Statement
Given business records from 3 independent sources (`Source 1`, `Source 2`, `Source 3`) with noisy, missing, and inconsistent fields:
- Determine which records refer to the same real-world business entity.
- Source 1 serves as the deduplicated reference source.
- Evaluated on **Macro-Averaged F_0.5 Score** (precision-weighted, penalizing false merges $2\times$ more than missed matches).

## Performance Highlights (Validation Benchmark: 50,000 Entities)
| Model | Candidate Recall | Macro Precision | Macro Recall | **Macro F_0.5** | Singleton Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Matcher** | 80.34% | 84.72% | 79.30% | **81.94%** | 87.29% |
| **Enhanced Multi-Key Blocker** | 93.07% | 85.02% | 84.64% | **83.57%** | 85.35% |
| **LightGBM Initial Hybrid** | 93.07% | 92.30% | 81.92% | **88.62%** | 93.06% |
| **CatBoost + LightGBM Ensemble (Latest)** | **93.07%** | **94.75%** | **88.50%** | **92.54%** | **92.28%** |

### Per-Country Breakdown with Calibrated Thresholds:
- **United States:** **94.12% Macro $F_{0.5}$** ($	au_{\text{US}} = 0.94$, Precision = 95.84%)
- **India:** **90.17% Macro $F_{0.5}$** ($	au_{\text{India}} = 0.94$, Precision = 93.12%)
- **France:** $	au_{\text{France}} = 0.96$ *(Precision-Guarded against false-positive over-prediction)*
- **Overall Macro $F_{0.5}$:** **92.54%**

## Key Innovations
1. **Strict Country Partitioning:** 100% intra-country matching confirmed across all 7.64M ground truth links, safely cutting pairwise search space by >70%.
2. **Multilingual Script Transliteration (`unidecode`):**
   - Transliterates Indic scripts (Hindi/Devanagari, Tamil, Odia, Gujarati, Marathi) to Latin phonetics.
   - Cross-script Indian records (e.g. `Ss Food Private Limited` $\leftrightarrow$ `एसएस फूड प्राइवेट लिमिटेड`) now match with high similarity, eliminating false negatives.
3. **Multi-Key Inverted Index Candidate Generation:**
   - Unpacks domain names (`painterslocal46 com` $\rightarrow$ `painters`, `local`, `46`).
   - Normalizes numeric tokens with leading-zero stripping (`06446` $\leftrightarrow$ `6446`).
   - Indexes distinctive address vocabulary to bridge multilingual transliteration gaps.
4. **42 Advanced Pairwise Tabular Features:**
   - Multi-metric string distances (Jaro-Winkler, Levenshtein, Damerau-Levenshtein, Token Sort, Token Set, Partial).
   - Address numeric and landmark overlaps.
   - Relative score margins between candidate and top candidate.
5. **Multi-Model GBDT Ensemble:**
   - Blended CatBoost and LightGBM classifier with country-calibrated decision thresholds.
6. **Precision-Calibrated Singleton Gating:** Protects singletons from false positives, keeping singleton accuracy above 92%.

## Project Structure
```
├── business_entity_resolution/
│   ├── src/
│   │   ├── config.py                 # Central configurations
│   │   ├── preprocessor.py           # Multilingual transliteration & normalizer
│   │   ├── blocking.py               # Country-partitioned multi-key blocker
│   │   ├── feature_engineering.py    # 42 pairwise tabular features
│   │   ├── baseline_matcher.py       # RapidFuzz baseline matcher
│   │   ├── ensemble_matcher.py       # CatBoost + LightGBM ensemble matcher
│   │   ├── train_ensemble.py         # Model training & country calibration
│   │   ├── evaluate.py               # Benchmark evaluation on 50k entities
│   │   ├── predict.py                # Vectorized batch test inference pipeline
│   │   └── utils_metric.py           # Macro F_0.5 computation engine
│   ├── output/                       # Output TSV files
│   ├── catboost_model.cbm            # Trained CatBoost model
│   ├── lgbm_model.txt                # Trained LightGBM booster
│   ├── calibrated_thresholds.json    # Country-specific optimal thresholds
│   ├── requirements.txt              # Pinned python dependencies
│   ├── README.md                     # Reproduction guide
│   └── Documentation_template.md     # Official methodology report
├── .gitignore
└── README.md
```

## Setup & Reproduction

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r business_entity_resolution/requirements.txt
```

### 2. Train Ensemble & Calibrate Thresholds
```bash
cd business_entity_resolution
python src/train_ensemble.py
```

### 3. Generate Submission Files (Batch Vectorized Inference)
```bash
python src/predict.py
```

### 4. Validate Submission
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
