import os
import subprocess
import config
from blocking import FastInvertedIndexBlocker
from baseline_matcher import FastBaselineMatcher

def generate_and_validate():
    val_s1 = os.path.join(config.VAL_DIR, "val_source1.tsv")
    val_s2 = os.path.join(config.VAL_DIR, "val_source2.tsv")
    val_s3 = os.path.join(config.VAL_DIR, "val_source3.tsv")
    
    out_matching = os.path.join(config.OUTPUT_DIR, "val_matching_results.tsv")
    out_candidate = os.path.join(config.OUTPUT_DIR, "val_candidate_pairs.tsv")
    
    blocker = FastInvertedIndexBlocker(max_candidates_per_entity=25)
    
    def combined_targets():
        with open(val_s2, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4: yield p[0], p[1], p[2], p[3]
        with open(val_s3, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 4: yield p[0], p[1], p[2], p[3]
                
    blocker.fit_targets(combined_targets())
    matcher = FastBaselineMatcher(threshold=0.74, singleton_threshold=0.78)
    
    print("Generating validation TSVs...")
    with open(val_s1, "r", encoding="utf-8") as f_in, \
         open(out_matching, "w", encoding="utf-8") as f_match, \
         open(out_candidate, "w", encoding="utf-8") as f_cand:
         
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        f_in.readline()
        for line in f_in:
            p = line.strip().split("\t")
            s1_id, name, addr, country = p[0], p[1], p[2], p[3]
            
            cands = blocker.get_candidates_for_entity(name, addr, country)
            matched = matcher.match_candidates(s1_id, name, addr, cands, blocker.target_data)
            
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
            f_match.write(f"{s1_id}\t{','.join(matched)}\n")
            
    print(f"Generated {out_matching} and {out_candidate}")
    
    # Run official validator
    print("\nRunning official submission validator on validation outputs...")
    cmd = [
        "py", config.VALIDATOR_SCRIPT,
        "--matching", out_matching,
        "--candidate", out_candidate,
        "--test-dir", config.VAL_DIR.replace("val_data", "val_test_dummy")
    ]
    # For validator, we need a directory with test_source1.tsv, let's point to a directory with test_source1.tsv = val_source1.tsv
    val_test_dir = os.path.join(config.PROJECT_ROOT, "val_test_dir")
    os.makedirs(val_test_dir, exist_ok=True)
    import shutil
    shutil.copyfile(val_s1, os.path.join(val_test_dir, "test_source1.tsv"))
    shutil.copyfile(val_s2, os.path.join(val_test_dir, "test_source2.tsv"))
    shutil.copyfile(val_s3, os.path.join(val_test_dir, "test_source3.tsv"))
    
    res = subprocess.run([
        "py", config.VALIDATOR_SCRIPT,
        "--matching", out_matching,
        "--candidate", out_candidate,
        "--test-dir", val_test_dir,
        "--check-ids"
    ], capture_output=True, text=True, encoding="utf-8")
    
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    print("Return code:", res.returncode)

if __name__ == "__main__":
    generate_and_validate()
