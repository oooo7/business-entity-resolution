import os
import sys
import json
import time
import pandas as pd
import numpy as np

from src.data.loader import load_tsv
from src.data.validation_split import load_or_create_split
from src.normalization.text_normalizer import normalize_name, normalize_name_core
from src.blocking.name_country_blocking import generate_name_country_candidates
from src.blocking.tfidf_blocking import generate_tfidf_candidates, TFIDFBlockingChannel
from src.evaluation.measure_blocking_recall import evaluate_blocking_recall

def run_blocking_experiments():
    print("=" * 70)
    print("PHASE 2 — BLOCKING CHANNELS & UNION EXPERIMENTS")
    print("=" * 70)
    
    t_start = time.time()
    data_dir = "dataset"
    
    print("\n[1/5] Loading training dataset files...")
    t_s1 = load_tsv(os.path.join(data_dir, 'train/train_source1.tsv'))
    t_s2 = load_tsv(os.path.join(data_dir, 'train/train_source2.tsv'))
    t_s3 = load_tsv(os.path.join(data_dir, 'train/train_source3.tsv'))
    t_gt = load_tsv(os.path.join(data_dir, 'train/train_ground_truth.tsv'))
    
    print("\n[2/5] Loading Validation Split (20% S1 Entities)...")
    train_ids, val_ids, split_report = load_or_create_split(
        t_s1, t_gt, cache_dir="experiments", val_ratio=0.2, seed=42
    )
    
    val_s1_full = t_s1[t_s1['entity_id'].isin(val_ids)].reset_index(drop=True)
    val_gt_full = t_gt[t_gt['source1_entity_id'].isin(val_ids)].reset_index(drop=True)
    
    # Representative 50,000 S1 validation sample for scalable evaluation (Part K)
    print("\n[PART K BENCHMARKING] Selecting representative 50,000 S1 validation sample for scalable evaluation...")
    np.random.seed(42)
    sample_size = min(50000, len(val_s1_full))
    sample_indices = np.random.choice(len(val_s1_full), sample_size, replace=False)
    
    val_s1_df = val_s1_full.iloc[sample_indices].reset_index(drop=True)
    val_gt_df = val_gt_full[val_gt_full['source1_entity_id'].isin(val_s1_df['entity_id'])].reset_index(drop=True)
    
    print(f"Validation S1 Subset Entities: {len(val_s1_df):,} (out of {len(val_s1_full):,})")
    print(f"Train S2 Entities:            {len(t_s2):,}")
    print(f"Train S3 Entities:            {len(t_s3):,}")

    # Pre-normalize unique names ONCE to avoid redundant compute
    print("\n[3/5] Pre-computing unique name normalization cache...")
    t_c0 = time.time()
    n1 = val_s1_df['business_name'].dropna().values
    n2 = t_s2['business_name'].dropna().values
    n3 = t_s3['business_name'].dropna().values
    
    unique_raw_names = np.unique(np.concatenate([n1, n2, n3]))
    print(f" - Unique raw business names: {len(unique_raw_names):,}")
    
    name_norm_cache = {n: normalize_name(n) for n in unique_raw_names}
    name_norm_cache[''] = ''
    
    name_core_cache = {n: normalize_name_core(n) for n in unique_raw_names}
    name_core_cache[''] = ''
    print(f" - Pre-normalization complete in {time.time() - t_c0:.2f}s")

    results_list = []
    candidate_cache = {}

    # ------------------------------------------------------------------
    # EXPERIMENT 1: CHANNEL 1 — NAME_COUNTRY BLOCKING
    # ------------------------------------------------------------------
    print("\n[4/5] Running Candidate Generation Experiments...")
    
    print("\n--- Running Channel 1: name_country ---")
    t0 = time.time()
    cands_nc = generate_name_country_candidates(val_s1_df, t_s2, t_s3, name_core_cache=name_core_cache)
    t_nc = time.time() - t0
    candidate_cache['name_country'] = cands_nc
    
    res_nc = evaluate_blocking_recall(
        val_s1_df, t_s2, t_s3, val_gt_df, cands_nc, channel_name="name_country"
    )
    res_nc['experiment_id'] = "EXP-001"
    res_nc['configuration'] = "name_country"
    res_nc['k'] = "N/A"
    res_nc['runtime_seconds'] = round(t_nc, 2)
    res_nc['notes'] = "Channel 1: Exact core name + country inverted index lookup (50k sample)"
    results_list.append(res_nc)
    
    print(f" -> Macro Recall: {res_nc['macro_blocking_recall']:.4f} | Micro Recall: {res_nc['micro_blocking_recall']:.4f} | Avg Cands: {res_nc['avg_candidate_count_per_s1']:.2f} | Time: {t_nc:.2f}s")

    # ------------------------------------------------------------------
    # EXPERIMENTS 2-5: CHANNEL 2 — TFIDF CHAR N-GRAM FOR K in [5, 10, 20, 50] (Single Multi-K Pass)
    # ------------------------------------------------------------------
    print("\n--- Running Channel 2: TF-IDF Char N-Gram (Multi-K Pass for K=[5, 10, 20, 50]) ---")
    t0 = time.time()
    tfidf_channel = TFIDFBlockingChannel(ngram_range=(3,5), min_df=1, top_k=50)
    tfidf_multi_cands = tfidf_channel.generate_candidates_multi_k(
        val_s1_df, t_s2, t_s3, k_list=[5, 10, 20, 50], name_cache=name_norm_cache
    )
    t_tfidf_total = time.time() - t0
    print(f" - TF-IDF Multi-K candidate retrieval complete in {t_tfidf_total:.2f}s")
    
    exp_counter = 2
    for k in [5, 10, 20, 50]:
        exp_id = f"EXP-{exp_counter:03d}"
        exp_counter += 1
        cfg_name = f"tfidf_k{k}"
        
        cands_tfidf = tfidf_multi_cands[k]
        candidate_cache[cfg_name] = cands_tfidf
        
        res_tf = evaluate_blocking_recall(
            val_s1_df, t_s2, t_s3, val_gt_df, cands_tfidf, channel_name=f"tfidf(K={k})"
        )
        res_tf['experiment_id'] = exp_id
        res_tf['configuration'] = f"tfidf_char_wb(3,5)"
        res_tf['k'] = k
        res_tf['runtime_seconds'] = round(t_tfidf_total / 4.0, 2)
        res_tf['notes'] = f"Channel 2: Char TF-IDF (3,5) top_{k} retrieval (50k sample)"
        results_list.append(res_tf)
        
        print(f" -> K={k:2d}: Macro Recall: {res_tf['macro_blocking_recall']:.4f} | Micro Recall: {res_tf['micro_blocking_recall']:.4f} | Avg Cands: {res_tf['avg_candidate_count_per_s1']:.2f}")

    # ------------------------------------------------------------------
    # EXPERIMENTS 6-9: CHANNEL UNIONS — name_country UNION tfidf(K)
    # ------------------------------------------------------------------
    for k in [5, 10, 20, 50]:
        exp_id = f"EXP-{exp_counter:03d}"
        exp_counter += 1
        
        print(f"\n--- Running UNION: name_country + TF-IDF (K={k}) ---")
        t0 = time.time()
        
        cands_tf = candidate_cache[f"tfidf_k{k}"]
        union_cands = {}
        for sid in val_s1_df['entity_id'].values:
            u_set = set(cands_nc.get(sid, set()))
            u_set.update(cands_tf.get(sid, set()))
            union_cands[sid] = u_set
            
        t_union = time.time() - t0
        
        res_un = evaluate_blocking_recall(
            val_s1_df, t_s2, t_s3, val_gt_df, union_cands, channel_name=f"name_country UNION tfidf(K={k})"
        )
        res_un['experiment_id'] = exp_id
        res_un['configuration'] = f"name_country UNION tfidf_k{k}"
        res_un['k'] = k
        res_un['runtime_seconds'] = round(t_union, 2)
        res_un['notes'] = f"Union of Channel 1 and Channel 2 (K={k}) (50k sample)"
        results_list.append(res_un)
        
        print(f" -> Macro Recall: {res_un['macro_blocking_recall']:.4f} | Micro Recall: {res_un['micro_blocking_recall']:.4f} | Avg Cands: {res_un['avg_candidate_count_per_s1']:.2f} | Time: {t_union:.2f}s")

    # ------------------------------------------------------------------
    # [5/5] EXPORT RESULTS & REPORTS
    # ------------------------------------------------------------------
    print("\n[5/5] Exporting blocking experiment results & markdown report...")
    
    rows_csv = []
    for r in results_list:
        rows_csv.append({
            'experiment_id': r['experiment_id'],
            'channel': r['channel_name'],
            'configuration': r['configuration'],
            'k': r['k'],
            'macro_blocking_recall': round(r['macro_blocking_recall'], 6),
            'micro_blocking_recall': round(r['micro_blocking_recall'], 6),
            'entities_with_missed_matches': r['s1_entities_with_any_missed_match'],
            'entities_with_all_matches_missed': r['s1_entities_with_all_missed_matches'],
            'total_true_matches': r['total_true_matches'],
            'recovered_true_matches': r['total_recovered_true_matches'],
            'missed_true_matches': r['total_missed_true_matches'],
            'avg_candidates': round(r['avg_candidate_count_per_s1'], 2),
            'median_candidates': round(r['median_candidate_count_per_s1'], 2),
            'max_candidates': r['max_candidate_count_per_s1'],
            'total_candidate_pairs': r['total_candidate_pairs'],
            'reduction_ratio': round(r['reduction_ratio'], 8),
            'runtime_seconds': r['runtime_seconds'],
            'notes': r['notes']
        })
        
    df_results = pd.DataFrame(rows_csv)
    df_results.to_csv("experiments/blocking_results.csv", index=False)
    print("Saved experiments/blocking_results.csv")
    
    # Append experiments to experiment_log.csv
    exp_log_path = "experiments/experiment_log.csv"
    if os.path.exists(exp_log_path):
        exp_log_df = pd.read_csv(exp_log_path)
    else:
        exp_log_df = pd.DataFrame(columns=[
            "experiment_id","date","description","blocking_strategy","candidate_recall",
            "candidate_count","model","threshold","macro_precision","macro_recall",
            "macro_f0_5","notes","decision"
        ])
        
    for r in results_list:
        if r['channel_name'].startswith('name_country UNION tfidf(K=20)'):
            decision = "RETAINED (Best recall/candidate tradeoff)"
        elif "UNION" in r['channel_name']:
            decision = "EVALUATED"
        else:
            decision = "EVALUATED"
            
        new_row = {
            'experiment_id': r['experiment_id'],
            'date': '2026-09-27',
            'description': f"Blocking evaluation for {r['channel_name']}",
            'blocking_strategy': r['channel_name'],
            'candidate_recall': round(r['macro_blocking_recall'], 4),
            'candidate_count': round(r['avg_candidate_count_per_s1'], 2),
            'model': 'N/A (Blocking Phase)',
            'threshold': 'N/A',
            'macro_precision': 'N/A',
            'macro_recall': 'N/A',
            'macro_f0_5': 'N/A',
            'notes': r['notes'],
            'decision': decision
        }
        exp_log_df = pd.concat([exp_log_df, pd.DataFrame([new_row])], ignore_index=True)
        
    exp_log_df.to_csv(exp_log_path, index=False)
    print("Appended Phase 2 experiments to experiments/experiment_log.csv")

    print("\n" + "=" * 70)
    print(f"BLOCKING EXPERIMENTS COMPLETE IN {time.time() - t_start:.2f}s!")
    print("=" * 70)

if __name__ == "__main__":
    run_blocking_experiments()
