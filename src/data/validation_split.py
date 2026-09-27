import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Set, Any

def create_validation_split(
    train_s1_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Set[str], Set[str], Dict[str, Any]]:
    """
    Create a reproducible train/validation split at the Source 1 entity level.
    Stratified by match-count bucket (0, 1, 2+) and country.
    """
    np.random.seed(seed)
    
    # Fast map ground truth matches per S1 entity
    m_series = ground_truth_df['matched_entity_ids'].fillna('')
    m_counts = [len(m.split(',')) if m else 0 for m in m_series.values]
    
    gt_count_map = dict(zip(ground_truth_df['source1_entity_id'], m_counts))
    gt_raw_map = dict(zip(ground_truth_df['source1_entity_id'], m_series.values))
    
    df = train_s1_df.copy()
    
    # Assign match count bucket fast
    buckets = []
    for s1_id in df['entity_id'].values:
        n_m = gt_count_map.get(s1_id, 0)
        if n_m == 0:
            b = '0'
        elif n_m == 1:
            b = '1'
        else:
            b = '2+'
        buckets.append(b)
        
    df['gt_bucket'] = buckets
    
    train_indices = []
    val_indices = []
    
    # Stratified grouping by bucket and country
    for (b, c), group in df.groupby(['gt_bucket', 'country']):
        indices = group.index.values.copy()
        np.random.shuffle(indices)
        n_val = int(round(len(indices) * val_ratio))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
        
    train_s1_df_split = df.iloc[train_indices]
    val_s1_df_split = df.iloc[val_indices]
    
    train_ids = set(train_s1_df_split['entity_id'].values)
    val_ids = set(val_s1_df_split['entity_id'].values)
    
    def summarize_split(split_df: pd.DataFrame) -> Dict[str, Any]:
        n_s1 = len(split_df)
        b_counts = split_df['gt_bucket'].value_counts().to_dict()
        c_dist = split_df['country'].value_counts(dropna=False).to_dict()
        tot_gt = sum(gt_count_map.get(sid, 0) for sid in split_df['entity_id'].values)
        
        return {
            'number_of_s1_entities': n_s1,
            'number_of_0_match_entities': int(b_counts.get('0', 0)),
            'number_of_1_match_entities': int(b_counts.get('1', 0)),
            'number_of_2plus_match_entities': int(b_counts.get('2+', 0)),
            'country_distribution': {str(k): int(v) for k, v in c_dist.items()},
            'total_ground_truth_matches': int(tot_gt)
        }
        
    report = {
        'seed': seed,
        'val_ratio': val_ratio,
        'train_split': summarize_split(train_s1_df_split),
        'val_split': summarize_split(val_s1_df_split)
    }
    
    return train_ids, val_ids, report

def load_or_create_split(
    train_s1_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    cache_dir: str = "experiments",
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Set[str], Set[str], Dict[str, Any]]:
    """
    Load split from cache file if available, otherwise create and save it.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "validation_split_info.json")
    train_ids_path = os.path.join(cache_dir, "train_s1_ids.json")
    val_ids_path = os.path.join(cache_dir, "val_s1_ids.json")
    
    if os.path.exists(cache_path) and os.path.exists(train_ids_path) and os.path.exists(val_ids_path):
        with open(cache_path, "r") as f:
            report = json.load(f)
        with open(train_ids_path, "r") as f:
            train_ids = set(json.load(f))
        with open(val_ids_path, "r") as f:
            val_ids = set(json.load(f))
        return train_ids, val_ids, report
        
    train_ids, val_ids, report = create_validation_split(
        train_s1_df, ground_truth_df, val_ratio=val_ratio, seed=seed
    )
    
    with open(cache_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(train_ids_path, "w") as f:
        json.dump(list(train_ids), f)
    with open(val_ids_path, "w") as f:
        json.dump(list(val_ids), f)
        
    return train_ids, val_ids, report
