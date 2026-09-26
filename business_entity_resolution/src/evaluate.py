import os
import time
import config
import utils_metric
from blocking import FastInvertedIndexBlocker
from baseline_matcher import FastBaselineMatcher

def load_tsv_records(tsv_path):
    """Generator yielding (entity_id, name, address, country)."""
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                yield parts[0], parts[1], parts[2], parts[3]

def run_evaluation(threshold=0.74, singleton_threshold=0.78, max_candidates=25):
    t0 = time.time()
    val_s1 = os.path.join(config.VAL_DIR, "val_source1.tsv")
    val_s2 = os.path.join(config.VAL_DIR, "val_source2.tsv")
    val_s3 = os.path.join(config.VAL_DIR, "val_source3.tsv")
    val_gt = os.path.join(config.VAL_DIR, "val_ground_truth.tsv")
    
    # 1. Load Ground Truth
    print("Loading validation ground truth...")
    gt = {}
    total_true_matches = 0
    with open(val_gt, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.partition("\t")
            s1 = s1.strip()
            rest = rest.strip()
            matches = set(m.strip() for m in rest.split(",") if m.strip())
            gt[s1] = matches
            total_true_matches += len(matches)
            
    print(f"Validation Ground Truth: {len(gt)} S1 entities, {total_true_matches} total true match links.")
    
    # 2. Fit Blocker on S2 and S3
    blocker = FastInvertedIndexBlocker(max_candidates_per_entity=max_candidates)
    
    def combined_targets():
        for item in load_tsv_records(val_s2):
            yield item
        for item in load_tsv_records(val_s3):
            yield item
            
    t_index = time.time()
    blocker.fit_targets(combined_targets())
    print(f"Indexing completed in {time.time() - t_index:.2f}s")
    
    # 3. Generate Candidates & Match
    matcher = FastBaselineMatcher(threshold=threshold, singleton_threshold=singleton_threshold)
    
    print("Generating candidates and matching S1 entities...")
    t_match = time.time()
    
    predictions = {}
    candidates_dict = {}
    
    found_true_in_candidates = 0
    total_candidates_generated = 0
    
    with open(val_s1, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            p = line.strip().split("\t")
            s1_id, name, addr, country = p[0], p[1], p[2], p[3]
            
            # Blocking
            cands = blocker.get_candidates_for_entity(name, addr, country)
            candidates_dict[s1_id] = cands
            total_candidates_generated += len(cands)
            
            # Check candidate recall
            true_set = gt.get(s1_id, set())
            cand_set = set(cands)
            found_true_in_candidates += len(true_set & cand_set)
            
            # Matching
            matched = matcher.match_candidates(s1_id, name, addr, cands, blocker.target_data)
            predictions[s1_id] = set(matched)
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} / {len(gt)} entities...")
                
    match_duration = time.time() - t_match
    print(f"Matching finished in {match_duration:.2f}s ({len(gt) / match_duration:.1f} entities/sec).")
    
    # 4. Compute Metrics
    cand_recall = found_true_in_candidates / total_true_matches if total_true_matches > 0 else 0.0
    avg_cands = total_candidates_generated / len(gt) if len(gt) > 0 else 0.0
    
    eval_results = utils_metric.compute_macro_f05(predictions, gt)
    
    print("\n" + "="*50)
    print("           BASELINE EVALUATION RESULTS")
    print("="*50)
    print(f"Entities Evaluated:         {eval_results['total_entities']}")
    print(f"Blocking Candidate Recall:  {cand_recall * 100:.2f}% ({found_true_in_candidates} / {total_true_matches})")
    print(f"Avg Candidates per S1:      {avg_cands:.2f}")
    print(f"Macro Precision:            {eval_results['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:               {eval_results['macro_recall'] * 100:.2f}%")
    print(f"Macro F_0.5 Score:          {eval_results['macro_f05'] * 100:.2f}%")
    print(f"Singleton Count:            {eval_results['singleton_count']}")
    print(f"Singleton Accuracy:         {eval_results['singleton_accuracy'] * 100:.2f}%")
    print(f"Total Elapsed Time:         {time.time() - t0:.2f}s")
    print("="*50 + "\n")
    
    return eval_results

if __name__ == "__main__":
    run_evaluation(threshold=0.74, singleton_threshold=0.78, max_candidates=25)
