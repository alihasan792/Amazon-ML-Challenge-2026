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
| **Advanced Hybrid LightGBM** | **93.07%** | **92.30%** | **81.92%** | **88.62%** | **93.06%** |

## Key Innovations
1. **Strict Country Partitioning:** 100% intra-country matching confirmed across all 7.64M ground truth links, safely cutting pairwise search space by >70%.
2. **Multi-Key Inverted Index Candidate Generation:**
   - Unpacks domain names (`painterslocal46 com` $\rightarrow$ `painters`, `local`, `46`).
   - Normalizes numeric tokens with leading-zero stripping (`06446` $\leftrightarrow$ `6446`).
   - Indexes distinctive address vocabulary to bridge multilingual transliteration gaps (e.g. Hindi/Devanagari names sharing English address roots).
3. **Hybrid Decision Architecture:** Combines 22-dimensional pairwise tabular features with LightGBM GBDT and token-order invariant RapidFuzz metrics.
4. **Precision-Calibrated Singleton Gating:** Protects singletons from false positives, keeping singleton accuracy above 93%.

## Project Structure
```
├── business_entity_resolution/
│   ├── src/
│   │   ├── config.py                 # Central configurations
│   │   ├── preprocessor.py           # Domain unpacking & text normalizer
│   │   ├── blocking.py               # Country-partitioned multi-key blocker
│   │   ├── feature_engineering.py    # 22 pairwise tabular features
│   │   ├── baseline_matcher.py       # High-speed RapidFuzz matcher
│   │   ├── hybrid_matcher.py         # LightGBM + RapidFuzz hybrid ensemble
│   │   ├── train_classifier.py       # LightGBM model training pipeline
│   │   ├── evaluate.py               # Benchmark evaluation on 50k entities
│   │   ├── predict.py                # Full test inference pipeline
│   │   └── utils_metric.py           # Macro F_0.5 computation
│   ├── output/                       # Output TSV files
│   ├── lgbm_model.txt                # Pre-trained LightGBM booster
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

### 2. Run Local Evaluation
```bash
cd business_entity_resolution
python src/evaluate.py
```

### 3. Generate Submission Files
```bash
python src/predict.py --hybrid
```

### 4. Validate Submission
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
