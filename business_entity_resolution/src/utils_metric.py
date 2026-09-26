"""
Evaluation Metric: Macro-averaged F_0.5 Score.
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Special Cases:
- Both pred and true are empty (singleton correct): score = 1.0
- True is empty, pred is non-empty (singleton false positive): score = 0.0
- True is non-empty, pred is empty (missed matches): score = 0.0
"""

def compute_f05_single(pred_set: set, true_set: set) -> tuple:
    """Computes (precision, recall, f05) for a single entity."""
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not true_set and pred_set:
        return 0.0, 0.0, 0.0
    if true_set and not pred_set:
        return 0.0, 0.0, 0.0
    
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return precision, recall, f05

def compute_macro_f05(predictions: dict, ground_truth: dict) -> dict:
    """
    Computes macro-averaged Precision, Recall, and F_0.5 across all entities in ground_truth.
    predictions: {s1_id: set([matched_ids])}
    ground_truth: {s1_id: set([matched_ids])}
    """
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    n = len(ground_truth)
    
    singleton_count = 0
    singleton_correct = 0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        if not true_set:
            singleton_count += 1
            if not pred_set:
                singleton_correct += 1
        
        p, r, f = compute_f05_single(pred_set, true_set)
        total_prec += p
        total_rec += r
        total_f05 += f
        
    return {
        "macro_f05": total_f05 / n if n > 0 else 0.0,
        "macro_precision": total_prec / n if n > 0 else 0.0,
        "macro_recall": total_rec / n if n > 0 else 0.0,
        "total_entities": n,
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_correct / singleton_count if singleton_count > 0 else 1.0
    }

if __name__ == "__main__":
    # Test sample from README:
    # Pred: [S2-00047, S2-00193, S3-00812], True: [S2-00047, S3-00812]
    # Expected: P=2/3, R=1.0, F_0.5=0.714
    p, r, f = compute_f05_single({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    print(f"Sample test: Precision={p:.4f}, Recall={r:.4f}, F0.5={f:.4f}")
    assert round(f, 3) == 0.714, f"Mismatch: expected 0.714, got {f}"
    
    # Singleton tests
    assert compute_f05_single(set(), set()) == (1.0, 1.0, 1.0)
    assert compute_f05_single({"S2-1"}, set()) == (0.0, 0.0, 0.0)
    assert compute_f05_single(set(), {"S2-1"}) == (0.0, 0.0, 0.0)
    print("All metric tests passed successfully!")
