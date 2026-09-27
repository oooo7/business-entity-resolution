from typing import Set, Dict, List, Any, Union
import numpy as np
import pandas as pd

def per_entity_precision(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Calculate precision for a single Source 1 entity.
    - True empty & Pred empty -> 1.0
    - True empty & Pred non-empty -> 0.0
    - True non-empty & Pred empty -> 0.0
    - True non-empty & Pred non-empty -> |True ^ Pred| / |Pred|
    """
    true_ids = set(true_ids) if true_ids else set()
    pred_ids = set(pred_ids) if pred_ids else set()
    
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    
    if len(pred_ids) == 0:
        return 0.0
        
    tp = len(true_ids.intersection(pred_ids))
    return float(tp / len(pred_ids))

def per_entity_recall(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Calculate recall for a single Source 1 entity.
    - True empty & Pred empty -> 1.0
    - True empty & Pred non-empty -> 1.0 (or 0.0 precision penalizes F0.5 to 0.0)
    - True non-empty & Pred empty -> 0.0
    - True non-empty & Pred non-empty -> |True ^ Pred| / |True|
    """
    true_ids = set(true_ids) if true_ids else set()
    pred_ids = set(pred_ids) if pred_ids else set()
    
    if len(true_ids) == 0:
        return 1.0
        
    if len(pred_ids) == 0:
        return 0.0
        
    tp = len(true_ids.intersection(pred_ids))
    return float(tp / len(true_ids))

def per_entity_f0_5(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """
    Calculate F0.5 score for a single Source 1 entity.
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Competition rules:
    - True empty & Pred empty -> F0.5 = 1.0 (Full credit singleton)
    - True empty & Pred non-empty -> F0.5 = 0.0 (Penalty for false merge on singleton)
    """
    true_ids = set(true_ids) if true_ids else set()
    pred_ids = set(pred_ids) if pred_ids else set()
    
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
        
    if len(pred_ids) == 0:
        return 0.0
        
    prec = per_entity_precision(true_ids, pred_ids)
    rec = per_entity_recall(true_ids, pred_ids)
    
    denom = 0.25 * prec + rec
    if denom == 0.0:
        return 0.0
        
    return float((1.25 * prec * rec) / denom)

def macro_precision(true_dict: Dict[str, Set[str]], pred_dict: Dict[str, Set[str]]) -> float:
    if not true_dict:
        return 0.0
    scores = [per_entity_precision(true_dict[s1_id], pred_dict.get(s1_id, set())) for s1_id in true_dict]
    return float(np.mean(scores))

def macro_recall(true_dict: Dict[str, Set[str]], pred_dict: Dict[str, Set[str]]) -> float:
    if not true_dict:
        return 0.0
    scores = [per_entity_recall(true_dict[s1_id], pred_dict.get(s1_id, set())) for s1_id in true_dict]
    return float(np.mean(scores))

def macro_f0_5(true_dict: Dict[str, Set[str]], pred_dict: Dict[str, Set[str]]) -> float:
    if not true_dict:
        return 0.0
    scores = [per_entity_f0_5(true_dict[s1_id], pred_dict.get(s1_id, set())) for s1_id in true_dict]
    return float(np.mean(scores))

def evaluate_predictions(
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    predictions: Union[pd.DataFrame, Dict[str, Set[str]]],
    s1_data: Union[pd.DataFrame, Dict[str, str]]
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth and produce overall & country-level breakdown.
    Fast vectorized zip-loop version.
    """
    # Convert ground truth to dict fast
    if isinstance(ground_truth, pd.DataFrame):
        gt_ids = ground_truth['source1_entity_id'].values
        gt_m = ground_truth['matched_entity_ids'].fillna('').values
        gt_dict = {
            sid: set(m.split(',')) if m else set()
            for sid, m in zip(gt_ids, gt_m)
        }
    else:
        gt_dict = {k: set(v) for k, v in ground_truth.items()}
        
    # Convert predictions to dict fast
    if isinstance(predictions, pd.DataFrame):
        p_ids = predictions['source1_entity_id'].values
        p_m = predictions['matched_entity_ids'].fillna('').values
        pred_dict = {
            sid: set(m.split(',')) if m else set()
            for sid, m in zip(p_ids, p_m)
        }
    else:
        pred_dict = {k: set(v) for k, v in predictions.items()}
        
    # Convert s1_data to country map
    if isinstance(s1_data, pd.DataFrame):
        country_map = dict(zip(s1_data['entity_id'].values, s1_data['country'].values))
    else:
        country_map = dict(s1_data)
        
    all_s1_ids = list(gt_dict.keys())
    
    precisions = []
    recalls = []
    f0_5s = []
    
    country_scores = {}
    
    for s1_id in all_s1_ids:
        t_set = gt_dict[s1_id]
        p_set = pred_dict.get(s1_id, set())
        
        p = per_entity_precision(t_set, p_set)
        r = per_entity_recall(t_set, p_set)
        f = per_entity_f0_5(t_set, p_set)
        
        precisions.append(p)
        recalls.append(r)
        f0_5s.append(f)
        
        country = country_map.get(s1_id, "UNKNOWN")
        if country not in country_scores:
            country_scores[country] = {'prec': [], 'rec': [], 'f0_5': []}
            
        country_scores[country]['prec'].append(p)
        country_scores[country]['rec'].append(r)
        country_scores[country]['f0_5'].append(f)
        
    overall = {
        'number_of_s1_entities': len(all_s1_ids),
        'macro_precision': float(np.mean(precisions)) if precisions else 0.0,
        'macro_recall': float(np.mean(recalls)) if recalls else 0.0,
        'macro_f0_5': float(np.mean(f0_5s)) if f0_5s else 0.0,
    }
    
    by_country = {}
    for country, sc in country_scores.items():
        by_country[country] = {
            'number_of_s1_entities': len(sc['f0_5']),
            'macro_precision': float(np.mean(sc['prec'])),
            'macro_recall': float(np.mean(sc['rec'])),
            'macro_f0_5': float(np.mean(sc['f0_5'])),
        }
        
    return {
        'overall': overall,
        'by_country': by_country
    }
