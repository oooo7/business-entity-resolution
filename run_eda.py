"""
Entry-point script to run complete Data Analysis, Ground Truth Profiling,
Validation Split Generation, and Infrastructure Verification.
"""
import os
import sys
import json
import time

from src.data.loader import load_all_datasets, get_dataframe_stats
from src.data.validation_split import load_or_create_split
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.measure_blocking_recall import (
    evaluate_blocking_recall,
    synthetic_dummy_candidate_generator
)

def run_eda_pipeline():
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION — EDA & FOUNDATION PIPELINE")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Load All Datasets
    print("\n[1/5] Loading and validating datasets...")
    datasets = load_all_datasets("dataset")
    print(f"Successfully loaded {len(datasets)} dataset files in {time.time() - t0:.2f}s")
    
    # 2. Compute Part A Data Statistics
    print("\n[2/5] Profiling datasets (Part A)...")
    dataset_stats = {}
    for name, df in datasets.items():
        st = get_dataframe_stats(df, name)
        dataset_stats[name] = st
        print(f" - {name}: {st['row_count']:,} rows, {st['unique_entity_id_count']:,} unique IDs, {len(st['column_names'])} columns")
        
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/dataset_stats_summary.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)
        
    # 3. Create/Verify Validation Split (Part E)
    print("\n[3/5] Generating/Loading Entity-Level Validation Split (Part E)...")
    train_s1 = datasets['train_source1']
    train_gt = datasets['train_ground_truth']
    
    train_ids, val_ids, split_report = load_or_create_split(
        train_s1, train_gt, cache_dir="experiments", val_ratio=0.2, seed=42
    )
    print(f"Train S1 Entities: {split_report['train_split']['number_of_s1_entities']:,}")
    print(f"Val   S1 Entities: {split_report['val_split']['number_of_s1_entities']:,}")
    
    # 4. Verify F0.5 Evaluation Infrastructure (Part G)
    print("\n[4/5] Testing F0.5 Evaluation Infrastructure (Part G)...")
    # Evaluate sample baseline on validation split
    val_s1_df = train_s1[train_s1['entity_id'].isin(val_ids)]
    val_gt_df = train_gt[train_gt['source1_entity_id'].isin(val_ids)]
    
    # Dummy empty prediction baseline (predict no matches for all entities)
    empty_preds = pd.DataFrame({
        'source1_entity_id': val_s1_df['entity_id'].values,
        'matched_entity_ids': [''] * len(val_s1_df)
    })
    
    eval_res = evaluate_predictions(val_gt_df, empty_preds, val_s1_df)
    print("Baseline Empty Predictions Evaluation:")
    print(f" - Macro Precision: {eval_res['overall']['macro_precision']:.4f}")
    print(f" - Macro Recall:    {eval_res['overall']['macro_recall']:.4f}")
    print(f" - Macro F0.5:      {eval_res['overall']['macro_f0_5']:.4f}")
    print(" Country breakdown:", json.dumps(eval_res['by_country'], indent=2))
    
    # 5. Verify Blocking Recall Infrastructure (Part F)
    print("\n[5/5] Testing Blocking Recall Infrastructure (Part F)...")
    sample_val_s1 = val_s1_df.head(1000)
    sample_val_s2 = datasets['train_source2'].head(2000)
    sample_val_s3 = datasets['train_source3'].head(2000)
    
    blocking_res = evaluate_blocking_recall(
        sample_val_s1, sample_val_s2, sample_val_s3,
        val_gt_df, synthetic_dummy_candidate_generator,
        channel_name="synthetic_test_channel"
    )
    print("Synthetic Blocking Recall Evaluation:")
    print(f" - Macro Blocking Recall: {blocking_res['macro_blocking_recall']:.4f}")
    print(f" - Micro Blocking Recall: {blocking_res['micro_blocking_recall']:.4f}")
    print(f" - Total Candidates:      {blocking_res['total_candidate_pairs']:,}")
    print(f" - Reduction Ratio:       {blocking_res['reduction_ratio']:.6f}")

    print("\n" + "=" * 60)
    print(f"EDA PIPELINE COMPLETE IN {time.time() - t0:.2f}s!")
    print("=" * 60)

if __name__ == "__main__":
    import pandas as pd
    run_eda_pipeline()
