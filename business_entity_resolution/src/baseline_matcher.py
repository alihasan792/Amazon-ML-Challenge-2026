import os
from rapidfuzz import fuzz
import preprocessor

class FastBaselineMatcher:
    def __init__(self, threshold=0.74, singleton_threshold=0.78):
        self.threshold = threshold
        self.singleton_threshold = singleton_threshold

    def score_pair(self, name1: str, addr1: str, name2: str, addr2: str) -> float:
        """
        Computes composite similarity score between query (1) and target (2).
        Returns a score in [0.0, 1.0].
        """
        # Name similarity (token sort ratio is robust to word reordering)
        n_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
        n_set = fuzz.token_set_ratio(name1, name2) / 100.0
        name_score = max(n_sort, n_set * 0.95)
        
        # Address similarity
        if not addr1 or not addr2 or addr1 == "null" or addr2 == "null":
            # Target has no address: require very high name confidence
            return name_score * 0.90
            
        a_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        a_set = fuzz.token_set_ratio(addr1, addr2) / 100.0
        addr_score = max(a_sort, a_set * 0.95)
        
        # Combined score: both name and address matter
        # When name is identical and address matches reasonably well, high score
        # When address is identical (same exact plaza/shop), can match even if name changed slightly
        if addr_score >= 0.90 and name_score >= 0.50:
            return 0.5 * name_score + 0.5 * addr_score
            
        return 0.60 * name_score + 0.40 * addr_score

    def match_candidates(self, query_id: str, query_name: str, query_addr: str,
                         candidate_ids: list, target_store: dict) -> list:
        """
        Given query and list of candidate IDs, returns list of matching entity IDs.
        """
        if not candidate_ids:
            return []
            
        q_name_clean = preprocessor.clean_text(query_name)
        q_addr_clean = preprocessor.clean_text(query_addr)
        
        scored_candidates = []
        for cid in candidate_ids:
            t_data = target_store.get(cid)
            if not t_data:
                continue
            c_name, c_addr, _ = t_data
            score = self.score_pair(q_name_clean, q_addr_clean, c_name, c_addr)
            if score >= self.threshold:
                scored_candidates.append((cid, score))
                
        if not scored_candidates:
            return []
            
        # Top score must clear singleton threshold to avoid false merges
        max_score = max(s for _, s in scored_candidates)
        if max_score < self.singleton_threshold:
            return []
            
        # Return all candidates clearing threshold
        return [cid for cid, s in scored_candidates if s >= self.threshold]
