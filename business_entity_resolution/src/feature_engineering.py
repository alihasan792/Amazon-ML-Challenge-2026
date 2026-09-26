import re
from rapidfuzz import fuzz
import preprocessor

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def extract_pairwise_features(s1_id: str, s1_name: str, s1_addr: str,
                              cand_id: str, cand_name: str, cand_addr: str,
                              blocking_rank: int, blocking_score: int) -> list:
    """
    Extracts 20+ rich numerical and binary features for a pair (S1, Candidate).
    """
    # 1. Preprocessed strings
    n1_clean, n1_root, n1_first, n1_compact = preprocessor.normalize_name(s1_name)
    n2_clean, n2_root, n2_first, n2_compact = preprocessor.normalize_name(cand_name)
    
    a1_clean, a1_tokens, a1_nums, a1_words = preprocessor.normalize_address(s1_addr)
    a2_clean, a2_tokens, a2_nums, a2_words = preprocessor.normalize_address(cand_addr)
    
    # 2. Name Features
    n_fuzz_ratio = fuzz.ratio(n1_clean, n2_clean) / 100.0
    n_token_sort = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0
    n_token_set = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0
    n_partial = fuzz.partial_ratio(n1_clean, n2_clean) / 100.0
    
    # Root name features
    root_sort = fuzz.token_sort_ratio(n1_root, n2_root) / 100.0
    root_exact = 1.0 if n1_root and n1_root == n2_root else 0.0
    first_match = 1.0 if n1_first and n1_first == n2_first else 0.0
    
    # Length features
    l1, l2 = len(n1_clean), len(n2_clean)
    len_diff = abs(l1 - l2)
    len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
    
    # Name tokens Jaccard
    n1_set = set(n1_clean.split())
    n2_set = set(n2_clean.split())
    n_jaccard = jaccard_similarity(n1_set, n2_set)
    
    # Compact substring match (for domain names)
    compact_sub = 1.0 if (n1_compact and n1_compact in n2_compact) or (n2_compact and n2_compact in n1_compact) else 0.0
    
    # 3. Address Features
    has_a2 = 0.0 if not cand_addr or cand_addr.lower() == "null" or len(a2_clean) < 3 else 1.0
    if has_a2 > 0:
        a_fuzz_ratio = fuzz.ratio(a1_clean, a2_clean) / 100.0
        a_token_sort = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0
        a_token_set = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0
        a_jaccard = jaccard_similarity(a1_tokens, a2_tokens)
        
        # Number matching (house number, PIN code)
        if a1_nums and a2_nums:
            num_overlap = len(a1_nums & a2_nums) / len(a1_nums)
            num_exact = 1.0 if a1_nums == a2_nums else 0.0
        else:
            num_overlap = 0.5
            num_exact = 0.0
            
        # Distinctive address words
        w1 = set(a1_words)
        w2 = set(a2_words)
        word_overlap = len(w1 & w2) / len(w1) if w1 else 0.5
    else:
        a_fuzz_ratio = 0.0
        a_token_sort = 0.0
        a_token_set = 0.0
        a_jaccard = 0.0
        num_overlap = 0.0
        num_exact = 0.0
        word_overlap = 0.0
        
    # 4. Context & Source Features
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    b_rank = float(blocking_rank)
    b_score = float(blocking_score)
    
    return [
        n_fuzz_ratio, n_token_sort, n_token_set, n_partial,
        root_sort, root_exact, first_match, len_diff, len_ratio,
        n_jaccard, compact_sub,
        has_a2, a_fuzz_ratio, a_token_sort, a_token_set, a_jaccard,
        num_overlap, num_exact, word_overlap,
        is_source2, b_rank, b_score
    ]

FEATURE_NAMES = [
    "n_fuzz_ratio", "n_token_sort", "n_token_set", "n_partial",
    "root_sort", "root_exact", "first_match", "len_diff", "len_ratio",
    "n_jaccard", "compact_sub",
    "has_a2", "a_fuzz_ratio", "a_token_sort", "a_token_set", "a_jaccard",
    "num_overlap", "num_exact", "word_overlap",
    "is_source2", "b_rank", "b_score"
]
