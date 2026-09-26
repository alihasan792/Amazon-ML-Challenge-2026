# Business Entity Resolution Solution (ML Challenge 2026)

## System Overview
An end-to-end, high-performance Business Entity Resolution pipeline designed for large-scale multi-source commercial entity deduplication and record linkage.

- **Baseline Model:** Country-partitioned inverted-index blocking with multi-token normalization + calibrated RapidFuzz string similarity metric. **Macro F_0.5 = 83.57%** on 50k holdout.
- **Advanced Model:** Multi-key hybrid blocking + 22 pairwise tabular features + LightGBM GBDT ranking classifier with calibrated singleton thresholding. **Macro F_0.5 = 88.62%** on holdout.

## Directory Structure
```
business_entity_resolution/
├── src/
│   ├── config.py                 # Paths and constants
│   ├── preprocessor.py           # Domain unpacking, text normalization, Indic script handling
│   ├── blocking.py               # Country-partitioned inverted index candidate generator
│   ├── feature_engineering.py    # Pairwise feature extraction
│   ├── baseline_matcher.py       # High-speed calibrated baseline matcher
│   ├── hybrid_matcher.py         # LightGBM + FastMatcher ensemble matcher
│   ├── train_classifier.py       # LightGBM model training pipeline
│   ├── evaluate.py               # Benchmark evaluation on local 50k validation split
│   └── predict.py                # Full test set inference generator
├── val_data/                     # Local 50k stratified validation benchmark
├── output/
│   ├── matching_results.tsv      # Leaderboard output
│   └── candidate_pairs.tsv       # Blocking candidate pairs
├── requirements.txt              # Pinned dependencies
└── README.md                     # Reproduction guide
```

## Quick Start / Reproduction

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Local Benchmark Evaluation
To evaluate on the 50,000 entity validation benchmark:
```bash
python src/evaluate.py
```

### 3. Train LightGBM Model
To train the LightGBM classifier on mined positive and negative candidate pairs:
```bash
python src/train_classifier.py
```

### 4. Run Full Test Set Inference
To generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
# Baseline Fast Matcher (High speed)
python src/predict.py

# Or Advanced Hybrid Model (Top accuracy)
python src/predict.py --hybrid
```

### 5. Validate Output Format
Validate submission outputs against all competition constraints:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
