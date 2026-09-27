import os
import json
import time
import pandas as pd
import numpy as np
from src.normalization.text_normalizer import normalize_name, normalize_name_core
from src.normalization.address_normalizer import normalize_address

def run_normalization_analysis():
    print("=" * 60)
    print("NORMALIZATION QUALITY & COLLISION DIAGNOSTIC")
    print("=" * 60)
    
    data_dir = "dataset"
    t0 = time.time()
    
    print("\n[1/3] Loading training datasets...")
    t_gt = pd.read_csv(os.path.join(data_dir, 'train/train_ground_truth.tsv'), sep="\t")
    t_s1 = pd.read_csv(os.path.join(data_dir, 'train/train_source1.tsv'), sep="\t")
    t_s2 = pd.read_csv(os.path.join(data_dir, 'train/train_source2.tsv'), sep="\t")
    t_s3 = pd.read_csv(os.path.join(data_dir, 'train/train_source3.tsv'), sep="\t")
    print(f"Loaded datasets in {time.time() - t0:.2f}s")
    
    # ------------------------------------------------------------------
    # PART C.1: EXACT MATCH INCREMENTS ON POSITIVE PAIRS
    # ------------------------------------------------------------------
    print("\n[2/3] Analyzing Exact Match Increments on 100,000 Positive Pairs...")
    
    # Fast index lookup dicts
    s1_dict = t_s1.set_index('entity_id').to_dict(orient='index')
    s2_dict = t_s2.set_index('entity_id').to_dict(orient='index')
    s3_dict = t_s3.set_index('entity_id').to_dict(orient='index')
    
    m_series = t_gt['matched_entity_ids'].fillna('').values
    s1_ids = t_gt['source1_entity_id'].values
    
    pairs = []
    for sid, m_str in zip(s1_ids, m_series):
        if m_str:
            for mid in m_str.split(','):
                mid = mid.strip()
                if mid:
                    pairs.append((sid, mid))
                    
    print(f"Total positive pairs available: {len(pairs):,}")
    
    # Sample 100,000 positive pairs
    np.random.seed(42)
    sample_indices = np.random.choice(len(pairs), min(100000, len(pairs)), replace=False)
    
    name_raw_exact = 0
    name_norm_exact = 0
    name_core_exact = 0
    
    addr_raw_exact = 0
    addr_norm_exact = 0
    
    for idx in sample_indices:
        s1_id, m_id = pairs[idx]
        rec1 = s1_dict.get(s1_id)
        rec2 = s2_dict.get(m_id) if m_id.startswith('S2-') else s3_dict.get(m_id)
        
        if not rec1 or not rec2:
            continue
            
        n1 = str(rec1['business_name']) if pd.notna(rec1['business_name']) else ''
        n2 = str(rec2['business_name']) if pd.notna(rec2['business_name']) else ''
        a1 = str(rec1['business_address']) if pd.notna(rec1['business_address']) else ''
        a2 = str(rec2['business_address']) if pd.notna(rec2['business_address']) else ''
        
        # Name exact checks
        if n1 == n2:
            name_raw_exact += 1
            
        n1_norm = normalize_name(n1, remove_legal_suffixes=False)
        n2_norm = normalize_name(n2, remove_legal_suffixes=False)
        if n1_norm and n1_norm == n2_norm:
            name_norm_exact += 1
            
        n1_core = normalize_name(n1, remove_legal_suffixes=True)
        n2_core = normalize_name(n2, remove_legal_suffixes=True)
        if n1_core and n1_core == n2_core:
            name_core_exact += 1
            
        # Address exact checks
        if a1 == a2:
            addr_raw_exact += 1
            
        a1_norm = normalize_address(a1)
        a2_norm = normalize_address(a2)
        if a1_norm and a1_norm == a2_norm:
            addr_norm_exact += 1
            
    n_samp = len(sample_indices)
    pair_diagnostics = {
        'sampled_pairs_count': n_samp,
        'name_raw_exact_pct': float(name_raw_exact / n_samp * 100),
        'name_normalized_exact_pct': float(name_norm_exact / n_samp * 100),
        'name_normalized_core_exact_pct': float(name_core_exact / n_samp * 100),
        'addr_raw_exact_pct': float(addr_raw_exact / n_samp * 100),
        'addr_normalized_exact_pct': float(addr_norm_exact / n_samp * 100),
    }
    
    print("Exact Match Increments:")
    print(f" - Name Raw Exact:             {pair_diagnostics['name_raw_exact_pct']:.2f}%")
    print(f" - Name Normalized Exact:      {pair_diagnostics['name_normalized_exact_pct']:.2f}%")
    print(f" - Name Normalized-Core Exact: {pair_diagnostics['name_normalized_core_exact_pct']:.2f}%")
    print(f" - Addr Raw Exact:             {pair_diagnostics['addr_raw_exact_pct']:.2f}%")
    print(f" - Addr Normalized Exact:      {pair_diagnostics['addr_normalized_exact_pct']:.2f}%")

    # ------------------------------------------------------------------
    # PART C.2: FALSE-COLLISION RISK ANALYSIS
    # ------------------------------------------------------------------
    print("\n[3/3] Analyzing False-Collision Risk on Source 1 Dataset...")
    
    s1_names = t_s1['business_name'].dropna().astype(str).values
    s1_core_keys = [normalize_name_core(n) for n in s1_names]
    
    key_counts = pd.Series(s1_core_keys).value_counts()
    n_unique_keys = len(key_counts)
    
    # Collisions = keys that appear more than once
    collided_keys = key_counts[key_counts > 1]
    total_collided_records = int(collided_keys.sum())
    collision_count = len(collided_keys)
    
    top_collision_groups = [
        {"normalized_key": key, "record_count": int(count)}
        for key, count in key_counts.head(10).items()
    ]
    
    collision_diagnostics = {
        'total_s1_records': len(t_s1),
        'unique_normalized_core_keys': n_unique_keys,
        'collided_unique_keys_count': collision_count,
        'records_involved_in_collisions': total_collided_records,
        'collision_record_pct': float(total_collided_records / len(t_s1) * 100),
        'largest_collision_groups': top_collision_groups
    }
    
    print("Collision Risk Statistics:")
    print(f" - Total S1 Records:             {collision_diagnostics['total_s1_records']:,}")
    print(f" - Unique Core Keys:             {collision_diagnostics['unique_normalized_core_keys']:,}")
    print(f" - Records in Collisions:        {collision_diagnostics['records_involved_in_collisions']:,} ({collision_diagnostics['collision_record_pct']:.2f}%)")
    print(" Top 5 Largest Collision Groups:")
    for group in top_collision_groups[:5]:
        print(f"    * '{group['normalized_key']}': {group['record_count']:,} records")
        
    full_report = {
        'pair_diagnostics': pair_diagnostics,
        'collision_diagnostics': collision_diagnostics
    }
    
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/normalization_analysis.json", "w") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\nAnalysis complete in {time.time() - t0:.2f}s!")

if __name__ == "__main__":
    run_normalization_analysis()
