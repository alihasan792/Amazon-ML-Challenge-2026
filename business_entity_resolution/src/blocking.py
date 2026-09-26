import os
import sys
from collections import defaultdict, Counter
import preprocessor

STOPWORDS = {
    "and", "the", "for", "with", "near", "opp", "road", "street", "avenue", "city",
    "floor", "shop", "plot", "door", "main", "cross", "colony", "nagar", "market",
    "complex", "plaza", "bhavan", "building", "delhi", "mumbai", "chennai", "bangalore"
}

def extract_blocking_tokens(clean_name: str, root_tokens: str, clean_addr: str,
                            numbers_set: set, distinctive_addr_words: list, compact_name: str) -> list:
    """Extracts high-recall distinctive tokens for inverted index blocking."""
    tokens = []
    
    # 1. Root name tokens (length >= 3, not generic stop words)
    r_tokens = root_tokens.split()
    for t in r_tokens:
        if len(t) >= 3 and t not in STOPWORDS:
            tokens.append(f"n_{t}")
            
    # 2. First 2 tokens combined if available
    name_parts = clean_name.split()
    if len(name_parts) >= 2:
        tokens.append(f"n2_{name_parts[0]}_{name_parts[1]}")
    elif len(name_parts) == 1 and len(name_parts[0]) >= 3:
        tokens.append(f"n1_{name_parts[0]}")
        
    # 3. Compact prefix (for domain names like painterslocal46, celestialmemorialtrust)
    if len(compact_name) >= 6:
        tokens.append(f"cp_{compact_name[:8]}")
        
    # 4. Address numbers (zero-normalized)
    for num in numbers_set:
        tokens.append(f"num_{num}")
        
    # 5. Distinctive address words (e.g. rahul, marve, malad, springfield, okhla)
    for aw in distinctive_addr_words[:4]:
        tokens.append(f"aw_{aw}")
        
    return tokens

class FastInvertedIndexBlocker:
    def __init__(self, max_candidates_per_entity=30):
        self.max_candidates = max_candidates_per_entity
        self.index = defaultdict(lambda: defaultdict(list))
        self.target_data = {} # entity_id -> (clean_name, clean_addr, country)

    def fit_targets(self, targets_generator):
        """Populates country-partitioned inverted index with multi-key tokens."""
        print("Indexing target records (S2 & S3)...")
        count = 0
        for eid, raw_name, raw_addr, country in targets_generator:
            count += 1
            c_name, root_tokens, _, compact = preprocessor.normalize_name(raw_name)
            c_addr, _, nums, dist_words = preprocessor.normalize_address(raw_addr)
            
            self.target_data[eid] = (c_name, c_addr, country)
            tokens = extract_blocking_tokens(c_name, root_tokens, c_addr, nums, dist_words, compact)
            
            c_index = self.index[country]
            for tok in tokens:
                c_index[tok].append(eid)
                
        print(f"Indexed {count} target records across {len(self.index)} countries.")
        
        # Prune ultra-frequent tokens (posting list > 25,000) to keep inverted queries fast
        for country, c_index in self.index.items():
            to_remove = [tok for tok, lst in c_index.items() if len(lst) > 25000]
            for tok in to_remove:
                del c_index[tok]
            print(f"Country {country}: {len(c_index)} active indexing keys after pruning.")

    def get_candidates_for_entity(self, raw_name: str, raw_addr: str, country: str) -> list:
        """Returns top candidate entity IDs for a query entity."""
        c_name, root_tokens, _, compact = preprocessor.normalize_name(raw_name)
        c_addr, _, nums, dist_words = preprocessor.normalize_address(raw_addr)
        
        c_index = self.index.get(country)
        if not c_index:
            return []
            
        tokens = extract_blocking_tokens(c_name, root_tokens, c_addr, nums, dist_words, compact)
        if not tokens:
            return []
            
        candidate_scores = Counter()
        for tok in tokens:
            postings = c_index.get(tok)
            if postings:
                if tok.startswith("n2_"):
                    w = 4
                elif tok.startswith("n_"):
                    w = 3
                elif tok.startswith("cp_"):
                    w = 3
                elif tok.startswith("aw_"):
                    w = 2
                else: # num_
                    w = 1
                for cand_id in postings:
                    candidate_scores[cand_id] += w
                    
        return [cand_id for cand_id, _ in candidate_scores.most_common(self.max_candidates)]
