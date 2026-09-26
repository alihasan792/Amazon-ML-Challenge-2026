import os
import numpy as np
import lightgbm as lgb
from baseline_matcher import FastBaselineMatcher
import feature_engineering

class HybridEntityMatcher:
    def __init__(self, model_path: str = None, threshold: float = 0.85, singleton_margin: float = 0.02):
        self.threshold = threshold
        self.singleton_margin = singleton_margin
        self.fast_matcher = FastBaselineMatcher(threshold=0.74, singleton_threshold=0.78)
        self.booster = None
        if model_path and os.path.isfile(model_path):
            self.booster = lgb.Booster(model_file=model_path)
            
    def match_candidates(self, s1_id: str, s1_name: str, s1_addr: str,
                         candidate_ids: list, target_store: dict) -> list:
        if not candidate_ids:
            return []
            
        if self.booster is None:
            # Fallback to fast matcher if model not loaded
            return self.fast_matcher.match_candidates(s1_id, s1_name, s1_addr, candidate_ids, target_store)
            
        feats_list = []
        valid_cids = []
        fast_scores = []
        
        for rank, cid in enumerate(candidate_ids):
            t_data = target_store.get(cid)
            if not t_data: continue
            c_name, c_addr, _ = t_data
            
            f_score = self.fast_matcher.score_pair(s1_name.lower(), s1_addr.lower(), c_name, c_addr)
            feats = feature_engineering.extract_pairwise_features(
                s1_id, s1_name, s1_addr, cid, c_name, c_addr, rank, 25 - rank
            )
            feats_list.append(feats)
            valid_cids.append(cid)
            fast_scores.append(f_score)
            
        if not feats_list:
            return []
            
        X = np.array(feats_list, dtype=np.float32)
        lgb_probs = self.booster.predict(X)
        fast_scores = np.array(fast_scores, dtype=np.float32)
        
        # Blended score
        hybrid_scores = 0.5 * lgb_probs + 0.5 * fast_scores
        
        max_h = np.max(hybrid_scores)
        if max_h < (self.threshold + self.singleton_margin):
            return []
            
        matched = [cid for cid, s in zip(valid_cids, hybrid_scores) if s >= self.threshold]
        return matched
