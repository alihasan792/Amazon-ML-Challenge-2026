import os
import time
import random
import numpy as np
import lightgbm as lgb
from collections import Counter
import config
import utils_metric
import feature_engineering
from blocking import FastInvertedIndexBlocker

def train_and_evaluate_lgbm():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    val_s1 = os.path.join(config.VAL_DIR, "val_source1.tsv")
    val_s2 = os.path.join(config.VAL_DIR, "val_source2.tsv")
    val_s3 = os.path.join(config.VAL_DIR, "val_source3.tsv")
    val_gt = os.path.join(config.VAL_DIR, "val_ground_truth.tsv")
    
    # 1. Load Ground Truth
    print("Loading Ground Truth...")
    gt = {}
    with open(val_gt, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.partition("\t")
            gt[s1.strip()] = set(m.strip() for m in rest.split(",") if m.strip())
            
    all_s1_records = []
    with open(val_s1, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 4:
                all_s1_records.append((p[0], p[1], p[2], p[3]))
                
    random.shuffle(all_s1_records)
    n_train = 35000
    train_s1 = all_s1_records[:n_train]
    holdout_s1 = all_s1_records[n_train:]
    print(f"Dataset split: {len(train_s1)} training S1 entities, {len(holdout_s1)} hold-out S1 entities.")
    
    # 2. Build Inverted Index Blocker
    blocker = FastInvertedIndexBlocker(max_candidates_per_entity=25)
    def combined_targets():
        for path in [val_s2, val_s3]:
            with open(path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 4: yield p[0], p[1], p[2], p[3]
    blocker.fit_targets(combined_targets())
    
    # 3. Extract Training Pairs & Features
    print("Extracting training features (positives and hard negatives)...")
    X_train = []
    y_train = []
    
    for i, (s1_id, name, addr, country) in enumerate(train_s1):
        cands = blocker.get_candidates_for_entity(name, addr, country)
        true_matches = gt.get(s1_id, set())
        
        pos_count = 0
        neg_count = 0
        for rank, cid in enumerate(cands):
            t_data = blocker.target_data.get(cid)
            if not t_data: continue
            c_name, c_addr, _ = t_data
            
            is_match = 1 if cid in true_matches else 0
            # Sample hard negatives: keep up to 3 negatives per entity
            if is_match == 0:
                if neg_count >= 3:
                    continue
                neg_count += 1
            else:
                pos_count += 1
                
            feats = feature_engineering.extract_pairwise_features(
                s1_id, name, addr, cid, c_name, c_addr, rank, 25 - rank
            )
            X_train.append(feats)
            y_train.append(is_match)
            
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} / {len(train_s1)} training entities ({len(X_train)} pair samples)...")
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Training dataset ready: {X_train.shape[0]} pairs, Positives={np.sum(y_train)}, Negatives={len(y_train)-np.sum(y_train)}")
    
    # 4. Train LightGBM Model
    print("Training LightGBM GBDT Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Print Feature Importances
    print("\nTop 10 Feature Importances:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx[:10]:
        print(f"  {feature_engineering.FEATURE_NAMES[idx]:<18}: {importances[idx]}")
        
    # 5. Evaluate on Hold-out S1 Entities
    print(f"\nEvaluating on {len(holdout_s1)} hold-out S1 entities...")
    holdout_gt = {s1: gt.get(s1, set()) for s1, _, _, _ in holdout_s1}
    
    # Pre-extract candidates and features for hold-out
    holdout_cands = {}
    holdout_pairs_meta = [] # (s1_id, cand_id)
    holdout_X = []
    
    for s1_id, name, addr, country in holdout_s1:
        cands = blocker.get_candidates_for_entity(name, addr, country)
        holdout_cands[s1_id] = cands
        for rank, cid in enumerate(cands):
            t_data = blocker.target_data.get(cid)
            if not t_data: continue
            c_name, c_addr, _ = t_data
            feats = feature_engineering.extract_pairwise_features(
                s1_id, name, addr, cid, c_name, c_addr, rank, 25 - rank
            )
            holdout_pairs_meta.append((s1_id, cid))
            holdout_X.append(feats)
            
    print(f"Hold-out inference on {len(holdout_X)} candidate pairs...")
    holdout_X = np.array(holdout_X, dtype=np.float32)
    probs = model.predict_proba(holdout_X)[:, 1]
    
    pair_prob_map = {}
    for (s1_id, cid), prob in zip(holdout_pairs_meta, probs):
        if s1_id not in pair_prob_map:
            pair_prob_map[s1_id] = []
        pair_prob_map[s1_id].append((cid, prob))
        
    # 6. Grid Search Optimal Threshold for Macro F_0.5
    print("\nOptimizing decision threshold for Macro F_0.5...")
    best_f05 = 0.0
    best_thresh = 0.5
    best_metrics = {}
    
    for thresh in np.arange(0.40, 0.85, 0.05):
        preds = {}
        for s1_id in holdout_gt:
            scored = pair_prob_map.get(s1_id, [])
            # Singleton gate: top probability must clear threshold + 0.05
            if scored:
                max_p = max(p for _, p in scored)
                if max_p >= thresh + 0.05:
                    preds[s1_id] = {cid for cid, p in scored if p >= thresh}
                else:
                    preds[s1_id] = set()
            else:
                preds[s1_id] = set()
                
        metrics = utils_metric.compute_macro_f05(preds, holdout_gt)
        f05 = metrics["macro_f05"]
        print(f"  Thresh {thresh:.2f} -> Prec: {metrics['macro_precision']*100:.2f}%, Rec: {metrics['macro_recall']*100:.2f}%, F_0.5: {f05*100:.2f}%, SingAcc: {metrics['singleton_accuracy']*100:.2f}%")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_metrics = metrics
            
    print("\n" + "="*50)
    print("      LIGHTGBM ADVANCED MODEL VALIDATION RESULTS")
    print("="*50)
    print(f"Hold-out Entities:       {len(holdout_s1)}")
    print(f"Optimal Threshold:       {best_thresh:.2f}")
    print(f"Macro Precision:         {best_metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:            {best_metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F_0.5 Score:       {best_metrics['macro_f05'] * 100:.2f}%")
    print(f"Singleton Accuracy:      {best_metrics['singleton_accuracy'] * 100:.2f}%")
    print(f"Total Elapsed Time:      {time.time() - t0:.2f}s")
    print("="*50 + "\n")
    
    # Save model artifact
    model_save_path = os.path.join(config.PROJECT_ROOT, "lgbm_model.txt")
    model.booster_.save_model(model_save_path)
    print(f"Saved trained LightGBM model to {model_save_path}")

if __name__ == "__main__":
    train_and_evaluate_lgbm()
