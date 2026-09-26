import re
from rapidfuzz import fuzz, distance
import preprocessor

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def overlap_coefficient(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))

def extract_pairwise_features(s1_id: str, s1_name: str, s1_addr: str, s1_country: str,
                              cand_id: str, cand_name: str, cand_addr: str,
                              blocking_rank: int, blocking_score: float, top_score: float) -> list:
    """
    Extracts 42 rich pairwise features between reference S1 and candidate record.
    """
    # 1. Preprocessing
    n1_clean, n1_root, n1_first, n1_compact = preprocessor.normalize_name(s1_name)
    n2_clean, n2_root, n2_first, n2_compact = preprocessor.normalize_name(cand_name)
    
    a1_clean, a1_tokens, a1_nums, a1_words = preprocessor.normalize_address(s1_addr)
    a2_clean, a2_tokens, a2_nums, a2_words = preprocessor.normalize_address(cand_addr)
    
    # 2. Name Similarity Features (8 features)
    n_fuzz = fuzz.ratio(n1_clean, n2_clean) / 100.0
    n_token_sort = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0
    n_token_set = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0
    n_partial = fuzz.partial_ratio(n1_clean, n2_clean) / 100.0
    n_partial_sort = fuzz.partial_token_sort_ratio(n1_clean, n2_clean) / 100.0
    n_jw = distance.JaroWinkler.similarity(n1_clean, n2_clean)
    n_lev = distance.Levenshtein.normalized_similarity(n1_clean, n2_clean)
    n_damerau = distance.DamerauLevenshtein.normalized_similarity(n1_clean, n2_clean)
    
    # 3. Root Name Features (7 features)
    root_fuzz = fuzz.ratio(n1_root, n2_root) / 100.0
    root_sort = fuzz.token_sort_ratio(n1_root, n2_root) / 100.0
    root_set = fuzz.token_set_ratio(n1_root, n2_root) / 100.0
    root_jw = distance.JaroWinkler.similarity(n1_root, n2_root)
    root_exact = 1.0 if n1_root and n1_root == n2_root else 0.0
    first_match = 1.0 if n1_first and n1_first == n2_first else 0.0
    first_jw = distance.JaroWinkler.similarity(n1_first, n2_first) if n1_first and n2_first else 0.0
    
    # 4. Token & Length Features (7 features)
    t1_set = set(n1_clean.split())
    t2_set = set(n2_clean.split())
    n_jaccard = jaccard_similarity(t1_set, t2_set)
    n_overlap = overlap_coefficient(t1_set, t2_set)
    n_dice = (2.0 * len(t1_set & t2_set) / (len(t1_set) + len(t2_set))) if (len(t1_set) + len(t2_set)) > 0 else 0.0
    
    compact_sub = 1.0 if (n1_compact and n1_compact in n2_compact) or (n2_compact and n2_compact in n1_compact) else 0.0
    l1, l2 = len(n1_clean), len(n2_clean)
    len_diff = abs(l1 - l2)
    len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
    w_diff = abs(len(t1_set) - len(t2_set))
    
    # 5. Address String Features (9 features)
    has_a2 = 0.0 if not cand_addr or cand_addr.lower() == "null" or len(a2_clean) < 3 else 1.0
    if has_a2 > 0:
        a_fuzz = fuzz.ratio(a1_clean, a2_clean) / 100.0
        a_token_sort = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0
        a_token_set = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0
        a_partial = fuzz.partial_ratio(a1_clean, a2_clean) / 100.0
        a_jw = distance.JaroWinkler.similarity(a1_clean, a2_clean)
        a_lev = distance.Levenshtein.normalized_similarity(a1_clean, a2_clean)
        a_jaccard = jaccard_similarity(a1_tokens, a2_tokens)
        a_overlap = overlap_coefficient(a1_tokens, a2_tokens)
        la1, la2 = len(a1_clean), len(a2_clean)
        a_len_ratio = min(la1, la2) / max(la1, la2) if max(la1, la2) > 0 else 1.0
    else:
        a_fuzz = a_token_sort = a_token_set = a_partial = a_jw = a_lev = a_jaccard = a_overlap = a_len_ratio = 0.0
        
    # 6. Address Numeric & Landmark Features (6 features)
    if has_a2 > 0:
        if a1_nums and a2_nums:
            num_overlap = len(a1_nums & a2_nums) / len(a1_nums)
            num_exact = 1.0 if a1_nums == a2_nums else 0.0
            num_jaccard = jaccard_similarity(a1_nums, a2_nums)
            has_matching_num = 1.0 if (a1_nums & a2_nums) else 0.0
        else:
            num_overlap = 0.5
            num_exact = 0.0
            num_jaccard = 0.5
            has_matching_num = 0.0
            
        w1, w2 = set(a1_words), set(a2_words)
        word_overlap = len(w1 & w2) / len(w1) if w1 else 0.5
        word_jaccard = jaccard_similarity(w1, w2)
    else:
        num_overlap = num_exact = num_jaccard = has_matching_num = word_overlap = word_jaccard = 0.0
        
    # 7. Interaction & Rank Context Features (5 features)
    name_addr_prod = n_token_sort * a_token_sort if has_a2 > 0 else (n_token_sort * 0.9)
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    b_rank = float(blocking_rank)
    b_score = float(blocking_score)
    score_margin = (top_score - blocking_score) / (top_score + 1e-5)
    
    return [
        n_fuzz, n_token_sort, n_token_set, n_partial, n_partial_sort, n_jw, n_lev, n_damerau,
        root_fuzz, root_sort, root_set, root_jw, root_exact, first_match, first_jw,
        n_jaccard, n_overlap, n_dice, compact_sub, len_diff, len_ratio, w_diff,
        has_a2, a_fuzz, a_token_sort, a_token_set, a_partial, a_jw, a_lev, a_jaccard, a_overlap, a_len_ratio,
        num_overlap, num_exact, num_jaccard, has_matching_num, word_overlap, word_jaccard,
        name_addr_prod, is_source2, b_rank, score_margin
    ]

FEATURE_NAMES = [
    "n_fuzz", "n_token_sort", "n_token_set", "n_partial", "n_partial_sort", "n_jw", "n_lev", "n_damerau",
    "root_fuzz", "root_sort", "root_set", "root_jw", "root_exact", "first_match", "first_jw",
    "n_jaccard", "n_overlap", "n_dice", "compact_sub", "len_diff", "len_ratio", "w_diff",
    "has_a2", "a_fuzz", "a_token_sort", "a_token_set", "a_partial", "a_jw", "a_lev", "a_jaccard", "a_overlap", "a_len_ratio",
    "num_overlap", "num_exact", "num_jaccard", "has_matching_num", "word_overlap", "word_jaccard",
    "name_addr_prod", "is_source2", "b_rank", "score_margin"
]
