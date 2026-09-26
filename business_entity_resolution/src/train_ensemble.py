import os
import sys
import time
import json
import random
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import config
import utils_metric
import feature_engineering
from blocking import FastInvertedIndexBlocker

def train_ensemble_pipeline():
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
    
    # 3. Extract Training Pairs & 42 Features
    print("Extracting 42 advanced pairwise features...")
    X_train = []
    y_train = []
    
    for i, (s1_id, name, addr, country) in enumerate(train_s1):
        cand_tuples = blocker.get_candidates_with_scores(name, addr, country)
        true_matches = gt.get(s1_id, set())
        
        top_score = cand_tuples[0][1] if cand_tuples else 1.0
        neg_count = 0
        for rank, (cid, score) in enumerate(cand_tuples):
            t_data = blocker.target_data.get(cid)
            if not t_data: continue
            c_name, c_addr, _ = t_data
            
            is_match = 1 if cid in true_matches else 0
            if is_match == 0:
                if neg_count >= 3:
                    continue
                neg_count += 1
                
            feats = feature_engineering.extract_pairwise_features(
                s1_id, name, addr, country, cid, c_name, c_addr, rank, score, top_score
            )
            X_train.append(feats)
            y_train.append(is_match)
            
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} / {len(train_s1)} training entities ({len(X_train)} samples)...")
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Training dataset ready: {X_train.shape[0]} pairs, Positives={np.sum(y_train)}, Negatives={len(y_train)-np.sum(y_train)}")
    
    # 4. Train CatBoost Model
    print("\n--- Training CatBoost Classifier ---")
    cb_model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=100
    )
    cb_model.fit(X_train, y_train)
    
    # 5. Train LightGBM Model
    print("\n--- Training LightGBM Classifier ---")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    
    # Save Models
    cb_save_path = os.path.join(config.PROJECT_ROOT, "catboost_model.cbm")
    lgb_save_path = os.path.join(config.PROJECT_ROOT, "lgbm_model.txt")
    cb_model.save_model(cb_save_path)
    lgb_model.booster_.save_model(lgb_save_path)
    print(f"Saved CatBoost model to {cb_save_path}")
    print(f"Saved LightGBM model to {lgb_save_path}")
    
    # 6. Evaluate Ensemble on Holdout
    print(f"\nEvaluating Ensemble on {len(holdout_s1)} hold-out entities...")
    holdout_gt = {s1: gt.get(s1, set()) for s1, _, _, _ in holdout_s1}
    
    holdout_pairs_meta = []
    holdout_X = []
    holdout_entity_countries = {}
    
    for s1_id, name, addr, country in holdout_s1:
        holdout_entity_countries[s1_id] = country
        cand_tuples = blocker.get_candidates_with_scores(name, addr, country)
        top_score = cand_tuples[0][1] if cand_tuples else 1.0
        for rank, (cid, score) in enumerate(cand_tuples):
            t_data = blocker.target_data.get(cid)
            if not t_data: continue
            c_name, c_addr, _ = t_data
            feats = feature_engineering.extract_pairwise_features(
                s1_id, name, addr, country, cid, c_name, c_addr, rank, score, top_score
            )
            holdout_pairs_meta.append((s1_id, cid))
            holdout_X.append(feats)
            
    holdout_X = np.array(holdout_X, dtype=np.float32)
    print(f"Running inference on {len(holdout_X)} candidate pairs...")
    
    cb_probs = cb_model.predict_proba(holdout_X)[:, 1]
    lgb_probs = lgb_model.predict_proba(holdout_X)[:, 1]
    
    # Blended Ensemble Probabilities
    ensemble_probs = 0.5 * cb_probs + 0.5 * lgb_probs
    
    pair_prob_map = {}
    for (s1_id, cid), prob in zip(holdout_pairs_meta, ensemble_probs):
        if s1_id not in pair_prob_map:
            pair_prob_map[s1_id] = []
        pair_prob_map[s1_id].append((cid, prob))
        
    # 7. Country-by-Country Threshold Calibration
    print("\n=== CALIBRATING COUNTRY-SPECIFIC THRESHOLDS ===")
    calibrated_thresholds = {}
    
    for country in ["US", "India"]:
        country_s1 = [s for s, c in holdout_entity_countries.items() if c == country]
        country_gt = {s: holdout_gt[s] for s in country_s1}
        print(f"\nOptimizing {country} ({len(country_s1)} entities)...")
        best_f05 = 0.0
        best_th = 0.85
        best_sing = 0.0
        
        for th in np.arange(0.70, 0.96, 0.02):
            preds = {}
            for s1_id in country_s1:
                scored = pair_prob_map.get(s1_id, [])
                if scored:
                    max_p = max(p for _, p in scored)
                    if max_p >= th + 0.02:
                        preds[s1_id] = {cid for cid, p in scored if p >= th}
                    else:
                        preds[s1_id] = set()
                else:
                    preds[s1_id] = set()
                    
            metrics = utils_metric.compute_macro_f05(preds, country_gt)
            f05 = metrics["macro_f05"]
            print(f"  {country} Thresh {th:.2f} -> Prec: {metrics['macro_precision']*100:.2f}%, Rec: {metrics['macro_recall']*100:.2f}%, F_0.5: {f05*100:.2f}%, SingAcc: {metrics['singleton_accuracy']*100:.2f}%")
            if f05 > best_f05:
                best_f05 = f05
                best_th = th
                best_sing = metrics["singleton_accuracy"]
                
        calibrated_thresholds[country] = float(best_th)
        print(f"--> Best {country} Threshold: {best_th:.2f} (Macro F_0.5: {best_f05*100:.2f}%)")
        
    # For France (unseen in train, high false-positive risk): use conservative threshold = max(US, India) + 0.02
    calibrated_thresholds["France"] = round(max(calibrated_thresholds["US"], calibrated_thresholds["India"]) + 0.02, 2)
    print(f"--> Calibrated France Threshold: {calibrated_thresholds['France']:.2f} (Precision-Guarded)")
    
    # Save calibrated thresholds
    thresh_path = os.path.join(config.PROJECT_ROOT, "calibrated_thresholds.json")
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(calibrated_thresholds, f, indent=2)
    print(f"Saved calibrated thresholds to {thresh_path}")
    
    # Overall Holdout Score with Country-Specific Calibration
    overall_preds = {}
    for s1_id in holdout_gt:
        c = holdout_entity_countries[s1_id]
        th = calibrated_thresholds.get(c, 0.88)
        scored = pair_prob_map.get(s1_id, [])
        if scored:
            max_p = max(p for _, p in scored)
            if max_p >= th + 0.02:
                overall_preds[s1_id] = {cid for cid, p in scored if p >= th}
            else:
                overall_preds[s1_id] = set()
        else:
            overall_preds[s1_id] = set()
            
    overall_metrics = utils_metric.compute_macro_f05(overall_preds, holdout_gt)
    print("\n" + "="*55)
    print("   ADVANCED CATBOOST + LIGHTGBM ENSEMBLE RESULTS")
    print("="*55)
    print(f"Total Hold-out Entities:  {len(holdout_s1)}")
    print(f"Calibrated Thresholds:   {calibrated_thresholds}")
    print(f"Macro Precision:         {overall_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:            {overall_metrics['macro_recall']*100:.2f}%")
    print(f"Macro F_0.5 Score:       {overall_metrics['macro_f05']*100:.2f}%")
    print(f"Singleton Accuracy:      {overall_metrics['singleton_accuracy']*100:.2f}%")
    print(f"Total Time:              {time.time() - t0:.1f}s")
    print("="*55 + "\n")

if __name__ == "__main__":
    train_ensemble_pipeline()
