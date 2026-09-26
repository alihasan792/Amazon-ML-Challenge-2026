import os
import random
from collections import defaultdict
import config

def build_validation_set(target_s1_count=50000, seed=42):
    """
    Extracts a stratified validation set from train files:
    - target_s1_count: total S1 entities to sample (stratified across countries and match counts)
    - Saves val_s1.tsv, val_s2.tsv, val_s3.tsv, val_ground_truth.tsv in val_data/
    """
    random.seed(seed)
    print(f"Creating local validation split with {target_s1_count} S1 entities...")
    
    # 1. Load S1 IDs and countries
    print("Scanning S1...")
    s1_countries = {}
    with open(config.TRAIN_S1, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            s1_countries[p[0]] = p[3]
            
    # 2. Load Ground Truth
    print("Scanning Ground Truth...")
    gt = {}
    match_count_buckets = defaultdict(list)
    with open(config.TRAIN_GT, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.partition("\t")
            s1 = s1.strip()
            rest = rest.strip()
            matches = [m.strip() for m in rest.split(",") if m.strip()]
            gt[s1] = matches
            bucket = min(len(matches), 5) # 0, 1, 2, 3, 4, 5+
            c = s1_countries.get(s1, "Unknown")
            match_count_buckets[(c, bucket)].append(s1)
            
    # Sample proportionally across (country, bucket)
    sampled_s1 = set()
    total_train = len(s1_countries)
    for (c, bucket), s1_list in match_count_buckets.items():
        sample_size = int(len(s1_list) / total_train * target_s1_count)
        if sample_size > 0:
            sampled_s1.update(random.sample(s1_list, min(sample_size, len(s1_list))))
            
    # Top off to exact target_s1_count
    if len(sampled_s1) < target_s1_count:
        remaining = [s for s in s1_countries if s not in sampled_s1]
        sampled_s1.update(random.sample(remaining, target_s1_count - len(sampled_s1)))
        
    print(f"Sampled {len(sampled_s1)} S1 entities for validation.")
    
    # Collect all needed true match IDs from S2 and S3
    needed_s2 = set()
    needed_s3 = set()
    val_gt = {}
    for s1 in sampled_s1:
        matches = gt.get(s1, [])
        val_gt[s1] = matches
        for m in matches:
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)
                
    print(f"True positive targets: S2={len(needed_s2)}, S3={len(needed_s3)}")
    
    # Also add distractor negative records from S2 and S3 (~50k random each from the same countries)
    print("Sampling distractor negatives for S2 and S3...")
    val_countries = {s1_countries[s] for s in sampled_s1}
    
    # 3. Write val_s1.tsv
    val_s1_path = os.path.join(config.VAL_DIR, "val_source1.tsv")
    with open(config.TRAIN_S1, "r", encoding="utf-8") as f_in, open(val_s1_path, "w", encoding="utf-8") as f_out:
        f_out.write(f_in.readline())
        for line in f_in:
            sid = line.split("\t", 1)[0]
            if sid in sampled_s1:
                f_out.write(line)
    print(f"Saved {val_s1_path}")
    
    # 4. Write val_ground_truth.tsv
    val_gt_path = os.path.join(config.VAL_DIR, "val_ground_truth.tsv")
    with open(val_gt_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in sorted(sampled_s1):
            f_out.write(f"{s1}\t{','.join(val_gt.get(s1, []))}\n")
    print(f"Saved {val_gt_path}")
    
    # 5. Write val_source2.tsv (all true matches + 100k distractors)
    val_s2_path = os.path.join(config.VAL_DIR, "val_source2.tsv")
    with open(config.TRAIN_S2, "r", encoding="utf-8") as f_in, open(val_s2_path, "w", encoding="utf-8") as f_out:
        f_out.write(f_in.readline())
        extra = 0
        for line in f_in:
            sid = line.split("\t", 1)[0]
            if sid in needed_s2:
                f_out.write(line)
            elif extra < 100000 and random.random() < 0.05:
                f_out.write(line)
                extra += 1
    print(f"Saved {val_s2_path} (true matches + distractors)")
    
    # 6. Write val_source3.tsv (all true matches + 100k distractors)
    val_s3_path = os.path.join(config.VAL_DIR, "val_source3.tsv")
    with open(config.TRAIN_S3, "r", encoding="utf-8") as f_in, open(val_s3_path, "w", encoding="utf-8") as f_out:
        f_out.write(f_in.readline())
        extra = 0
        for line in f_in:
            sid = line.split("\t", 1)[0]
            if sid in needed_s3:
                f_out.write(line)
            elif extra < 100000 and random.random() < 0.05:
                f_out.write(line)
                extra += 1
    print(f"Saved {val_s3_path} (true matches + distractors)")
    print("Local validation dataset successfully prepared!")

if __name__ == "__main__":
    build_validation_set(target_s1_count=50000)
