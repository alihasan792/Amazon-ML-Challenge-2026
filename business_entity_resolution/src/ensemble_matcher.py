import os
import json
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import config
import feature_engineering

class CatBoostLGBMEnsembleMatcher:
    def __init__(self, cb_path=None, lgb_path=None, thresh_path=None):
        if cb_path is None:
            cb_path = os.path.join(config.PROJECT_ROOT, "catboost_model.cbm")
        if lgb_path is None:
            lgb_path = os.path.join(config.PROJECT_ROOT, "lgbm_model.txt")
        if thresh_path is None:
            thresh_path = os.path.join(config.PROJECT_ROOT, "calibrated_thresholds.json")
            
        print(f"Loading CatBoost model from {cb_path}...")
        self.cb_model = CatBoostClassifier()
        self.cb_model.load_model(cb_path)
        
        print(f"Loading LightGBM model from {lgb_path}...")
        self.lgb_booster = lgb.Booster(model_file=lgb_path)
        
        # Default calibrated thresholds
        self.thresholds = {"US": 0.94, "India": 0.94, "France": 0.96}
        if os.path.isfile(thresh_path):
            with open(thresh_path, "r", encoding="utf-8") as f:
                self.thresholds.update(json.load(f))
        print(f"Active Calibrated Thresholds: {self.thresholds}")

    def score_batch(self, batch_pairs):
        """
        batch_pairs: list of (s1_id, name, addr, country, cid, c_name, c_addr, rank, score, top_score)
        Returns: list of ensemble probabilities
        """
        if not batch_pairs:
            return []
            
        X = [
            feature_engineering.extract_pairwise_features(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
            )
            for p in batch_pairs
        ]
        X = np.array(X, dtype=np.float32)
        
        cb_p = self.cb_model.predict_proba(X)[:, 1]
        lgb_p = self.lgb_booster.predict(X)
        
        # 50/50 blend
        return 0.5 * cb_p + 0.5 * lgb_p
