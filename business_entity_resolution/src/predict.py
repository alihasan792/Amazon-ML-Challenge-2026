import os
import sys
import time
import argparse
import config
from blocking import FastInvertedIndexBlocker
from baseline_matcher import FastBaselineMatcher
from hybrid_matcher import HybridEntityMatcher

def run_prediction(use_hybrid=False, max_candidates=25, country_filter=None):
    """
    Generates output/matching_results.tsv and output/candidate_pairs.tsv.
    Supports country-by-country chunked execution for zero memory pressure.
    """
    t0 = time.time()
    out_matching = config.OUT_MATCHING
    out_candidate = config.OUT_CANDIDATE
    
    print("=== STARTING PREDICTION PIPELINE ===")
    print(f"Test S1: {config.TEST_S1}")
    print(f"Output Matching: {out_matching}")
    print(f"Output Candidate: {out_candidate}")
    print(f"Model Type: {'Hybrid (LightGBM + FastMatcher)' if use_hybrid else 'Fast Baseline Matcher'}")
    
    # Target files
    model_path = os.path.join(config.PROJECT_ROOT, "lgbm_model.txt")
    if use_hybrid and os.path.isfile(model_path):
        matcher = HybridEntityMatcher(model_path=model_path, threshold=0.85)
    else:
        matcher = FastBaselineMatcher(threshold=0.74, singleton_threshold=0.78)
        
    blocker = FastInvertedIndexBlocker(max_candidates_per_entity=max_candidates)
    
    def test_targets():
        for path in [config.TEST_S2, config.TEST_S3]:
            with open(path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.strip().split("\t")
                    if len(p) >= 4:
                        if country_filter is None or p[3] == country_filter:
                            yield p[0], p[1], p[2], p[3]
                            
    blocker.fit_targets(test_targets())
    
    print("Generating candidate pairs and final matches...")
    count = 0
    with open(config.TEST_S1, "r", encoding="utf-8") as f_in, \
         open(out_matching, "w", encoding="utf-8") as f_match, \
         open(out_candidate, "w", encoding="utf-8") as f_cand:
         
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        f_in.readline()
        for line in f_in:
            p = line.strip().split("\t")
            if len(p) < 4: continue
            s1_id, name, addr, country = p[0], p[1], p[2], p[3]
            if country_filter and country != country_filter:
                continue
                
            cands = blocker.get_candidates_for_entity(name, addr, country)
            matched = matcher.match_candidates(s1_id, name, addr, cands, blocker.target_data)
            
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
            f_match.write(f"{s1_id}\t{','.join(matched)}\n")
            
            count += 1
            if count % 100000 == 0:
                print(f"  Processed {count:,} entities in {time.time() - t0:.1f}s...")
                
    print(f"Successfully processed {count:,} entities in {time.time() - t0:.1f}s.")
    print(f"Saved {out_matching}")
    print(f"Saved {out_candidate}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid LightGBM model")
    parser.add_argument("--country", type=str, default=None, help="Filter by country (e.g. France, US, India)")
    args = parser.parse_args()
    run_prediction(use_hybrid=args.hybrid, country_filter=args.country)
