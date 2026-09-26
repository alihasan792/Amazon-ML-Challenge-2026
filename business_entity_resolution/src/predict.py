import os
import sys
import time
import argparse
import config
from blocking import FastInvertedIndexBlocker
from ensemble_matcher import CatBoostLGBMEnsembleMatcher

def run_prediction_pipeline(batch_size=5000, country_filter=None):
    t0 = time.time()
    out_matching = config.OUT_MATCHING
    out_candidate = config.OUT_CANDIDATE
    
    print("=== STARTING ADVANCED ENSEMBLE PREDICTION PIPELINE ===")
    print(f"Test S1: {config.TEST_S1}")
    print(f"Output Matching: {out_matching}")
    print(f"Output Candidate: {out_candidate}")
    print(f"Country Filter: {country_filter or 'All (France, US, India)'}")
    
    matcher = CatBoostLGBMEnsembleMatcher()
    blocker = FastInvertedIndexBlocker(max_candidates_per_entity=25)
    
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
    
    print("\nGenerating candidates and scoring with CatBoost + LightGBM Ensemble in batches...")
    
    total_processed = 0
    with open(config.TEST_S1, "r", encoding="utf-8") as f_in, \
         open(out_matching, "w", encoding="utf-8") as f_match, \
         open(out_candidate, "w", encoding="utf-8") as f_cand:
         
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        f_in.readline()
        
        batch_entities = [] # [(s1_id, name, addr, country)]
        
        for line in f_in:
            p = line.strip().split("\t")
            if len(p) < 4: continue
            s1_id, name, addr, country = p[0], p[1], p[2], p[3]
            if country_filter and country != country_filter:
                continue
            batch_entities.append((s1_id, name, addr, country))
            
            if len(batch_entities) >= batch_size:
                process_batch(batch_entities, blocker, matcher, f_match, f_cand)
                total_processed += len(batch_entities)
                elapsed = time.time() - t0
                speed = total_processed / elapsed
                print(f"  Processed {total_processed:,} entities ({speed:.1f} entities/s, Elapsed: {elapsed/60:.1f} min)...")
                batch_entities = []
                
        if batch_entities:
            process_batch(batch_entities, blocker, matcher, f_match, f_cand)
            total_processed += len(batch_entities)
            
    print(f"\nPipeline finished! Total processed: {total_processed:,} entities in {(time.time() - t0)/60:.1f} minutes.")
    print(f"Saved: {out_matching}")
    print(f"Saved: {out_candidate}")

def process_batch(entities, blocker, matcher, f_match, f_cand):
    """Processes a chunk of entities with vectorized batch scoring."""
    entity_cands_map = {}
    batch_pairs = []
    
    for s1_id, name, addr, country in entities:
        cand_tuples = blocker.get_candidates_with_scores(name, addr, country)
        cands_only = [cid for cid, _ in cand_tuples]
        entity_cands_map[s1_id] = (country, cand_tuples)
        f_cand.write(f"{s1_id}\t{','.join(cands_only)}\n")
        
        top_score = cand_tuples[0][1] if cand_tuples else 1.0
        for rank, (cid, score) in enumerate(cand_tuples):
            t_data = blocker.target_data.get(cid)
            if not t_data: continue
            c_name, c_addr, _ = t_data
            batch_pairs.append((s1_id, name, addr, country, cid, c_name, c_addr, rank, score, top_score))
            
    # Batch score with Ensemble
    if batch_pairs:
        probs = matcher.score_batch(batch_pairs)
        
        # Group probabilities back to s1_id
        entity_scores = {}
        for pair_info, p in zip(batch_pairs, probs):
            s1_id = pair_info[0]
            cid = pair_info[4]
            if s1_id not in entity_scores:
                entity_scores[s1_id] = []
            entity_scores[s1_id].append((cid, p))
    else:
        entity_scores = {}
        
    # Write matching results using country-calibrated thresholds
    for s1_id, name, addr, country in entities:
        th = matcher.thresholds.get(country, 0.94)
        c_scores = entity_scores.get(s1_id, [])
        if c_scores:
            max_p = max(p for _, p in c_scores)
            if max_p >= (th + 0.02):
                matched = [cid for cid, p in c_scores if p >= th]
            else:
                matched = []
        else:
            matched = []
        f_match.write(f"{s1_id}\t{','.join(matched)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for vectorized inference")
    parser.add_argument("--country", type=str, default=None, help="Filter by country (e.g. France, US, India)")
    args = parser.parse_args()
    run_prediction_pipeline(batch_size=args.batch_size, country_filter=args.country)
