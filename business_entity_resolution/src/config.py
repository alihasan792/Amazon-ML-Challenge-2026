import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = r"C:\Users\Ali hasan\OneDrive\Desktop\6ab10eb3b23ba_student_resource\New Folder\student_resource\dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
VAL_DIR = os.path.join(PROJECT_ROOT, "val_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

OUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CANDIDATE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

VALIDATOR_SCRIPT = r"C:\Users\Ali hasan\OneDrive\Desktop\6ab10eb3b23ba_student_resource\New Folder\student_resource\utils\validate_submission.py"
