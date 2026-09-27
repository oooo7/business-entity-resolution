from typing import Callable, Dict, List, Set, Any, Union, Optional
import numpy as np
import pandas as pd

# Type alias for a candidate generation function
CandidateGenerator = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], Dict[str, Set[str]]]

def evaluate_blocking_recall(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    candidate_generator: Union[CandidateGenerator, Dict[str, Set[str]]],
    channel_name: str = "default_channel"
) -> Dict[str, Any]:
    """
    Evaluate blocking recall and candidate set metrics for a candidate generation function or precomputed candidates.
    Fast vectorized implementation.
    """
    # Parse ground truth into dict s1_id -> Set[match_ids] fast
    if isinstance(ground_truth, pd.DataFrame):
        gt_s1_ids = ground_truth['source1_entity_id'].values
        gt_matches_str = ground_truth['matched_entity_ids'].fillna('').values
        gt_dict = {
            sid: set(x.strip() for x in m.split(',') if x.strip()) if m else set()
            for sid, m in zip(gt_s1_ids, gt_matches_str)
        }
    else:
        gt_dict = {k: set(v) for k, v in ground_truth.items()}
        
    # Generate or retrieve candidate pairs
    if callable(candidate_generator):
        candidates_dict = candidate_generator(s1_df, s2_df, s3_df)
    else:
        candidates_dict = {k: set(v) for k, v in candidate_generator.items()}
        
    s1_ids = list(s1_df['entity_id'].values)
    
    total_true_matches = 0
    total_recovered_true_matches = 0
    total_missed_true_matches = 0
    
    entity_recalls = []
    
    s1_with_any_missed = 0
    s1_with_all_missed = 0
    s1_zero_matches = 0
    
    candidate_counts = []
    
    for s1_id in s1_ids:
        true_matches = gt_dict.get(s1_id, set())
        cand_matches = candidates_dict.get(s1_id, set())
        
        cand_matches = set(cand_matches)
        
        n_cands = len(cand_matches)
        candidate_counts.append(n_cands)
        
        n_true = len(true_matches)
        
        if n_true == 0:
            s1_zero_matches += 1
            continue
            
        found = true_matches.intersection(cand_matches)
        n_found = len(found)
        n_missed = n_true - n_found
        
        rec = float(n_found / n_true)
        entity_recalls.append(rec)
        
        total_true_matches += n_true
        total_recovered_true_matches += n_found
        total_missed_true_matches += n_missed
        
        if n_missed > 0:
            s1_with_any_missed += 1
        if n_found == 0:
            s1_with_all_missed += 1
            
    num_s2 = len(s2_df)
    num_s3 = len(s3_df)
    num_s1 = len(s1_ids)
    
    total_candidates = sum(candidate_counts)
    total_possible_pairs = num_s1 * (num_s2 + num_s3)
    reduction_ratio = 1.0 - (total_candidates / total_possible_pairs) if total_possible_pairs > 0 else 1.0
    
    macro_recall = float(np.mean(entity_recalls)) if entity_recalls else 1.0
    micro_recall = float(total_recovered_true_matches / total_true_matches) if total_true_matches > 0 else 1.0
    
    return {
        'channel_name': channel_name,
        'total_s1_entities': num_s1,
        's1_entities_with_gt_matches': len(entity_recalls),
        's1_entities_with_zero_gt_matches': s1_zero_matches,
        'macro_blocking_recall': macro_recall,
        'micro_blocking_recall': micro_recall,
        's1_entities_with_any_missed_match': s1_with_any_missed,
        's1_entities_with_all_missed_matches': s1_with_all_missed,
        'total_true_matches': total_true_matches,
        'total_recovered_true_matches': total_recovered_true_matches,
        'total_missed_true_matches': total_missed_true_matches,
        'avg_candidate_count_per_s1': float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        'median_candidate_count_per_s1': float(np.median(candidate_counts)) if candidate_counts else 0.0,
        'max_candidate_count_per_s1': int(np.max(candidate_counts)) if candidate_counts else 0,
        'total_candidate_pairs': total_candidates,
        'reduction_ratio': float(reduction_ratio)
    }

def evaluate_blocking_channels(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    channels: Dict[str, Union[CandidateGenerator, Dict[str, Set[str]]]]
) -> Dict[str, Any]:
    """
    Evaluate multiple blocking channels independently AND evaluate the UNION of all channels.
    """
    channel_results = {}
    union_candidates: Dict[str, Set[str]] = {sid: set() for sid in s1_df['entity_id']}
    
    for name, gen in channels.items():
        res = evaluate_blocking_recall(s1_df, s2_df, s3_df, ground_truth, gen, channel_name=name)
        channel_results[name] = res
        
        cands = gen(s1_df, s2_df, s3_df) if callable(gen) else gen
        for s1_id, cand_set in cands.items():
            if s1_id not in union_candidates:
                union_candidates[s1_id] = set()
            union_candidates[s1_id].update(cand_set)
            
    union_result = evaluate_blocking_recall(
        s1_df, s2_df, s3_df, ground_truth, union_candidates, channel_name="UNION"
    )
    
    return {
        'individual_channels': channel_results,
        'union_channel': union_result
    }

def synthetic_dummy_candidate_generator(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    k_dummy: int = 2
) -> Dict[str, Set[str]]:
    """
    Tiny synthetic candidate generator for unit-testing the blocking recall infrastructure.
    """
    s2_ids = list(s2_df['entity_id'].values[:k_dummy]) if len(s2_df) > 0 else []
    s3_ids = list(s3_df['entity_id'].values[:k_dummy]) if len(s3_df) > 0 else []
    dummy_set = set(s2_ids + s3_ids)
    
    return {s1_id: set(dummy_set) for s1_id in s1_df['entity_id']}
