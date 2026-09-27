import pytest
import pandas as pd
from src.evaluation.metrics import (
    per_entity_precision,
    per_entity_recall,
    per_entity_f0_5,
    macro_f0_5,
    macro_precision,
    macro_recall,
    evaluate_predictions
)

def test_singleton_empty_prediction():
    # True empty set, predicted empty set -> full credit (1.0)
    true_ids = set()
    pred_ids = set()
    
    assert per_entity_precision(true_ids, pred_ids) == 1.0
    assert per_entity_recall(true_ids, pred_ids) == 1.0
    assert per_entity_f0_5(true_ids, pred_ids) == 1.0

def test_singleton_false_match_penalty():
    # True empty set, predicted false match -> penalized to 0.0
    true_ids = set()
    pred_ids = {"S2-00001"}
    
    assert per_entity_precision(true_ids, pred_ids) == 0.0
    assert per_entity_f0_5(true_ids, pred_ids) == 0.0

def test_true_match_empty_prediction():
    # True matches exist, predicted empty -> 0.0
    true_ids = {"S2-00001", "S3-00002"}
    pred_ids = set()
    
    assert per_entity_precision(true_ids, pred_ids) == 0.0
    assert per_entity_recall(true_ids, pred_ids) == 0.0
    assert per_entity_f0_5(true_ids, pred_ids) == 0.0

def test_exact_match():
    true_ids = {"S2-00001", "S3-00002"}
    pred_ids = {"S2-00001", "S3-00002"}
    
    assert per_entity_precision(true_ids, pred_ids) == 1.0
    assert per_entity_recall(true_ids, pred_ids) == 1.0
    assert per_entity_f0_5(true_ids, pred_ids) == 1.0

def test_partial_match():
    # True = {S2-1, S3-2}, Pred = {S2-1, S2-99}
    true_ids = {"S2-1", "S3-2"}
    pred_ids = {"S2-1", "S2-99"}
    
    prec = 0.5  # 1/2
    rec = 0.5   # 1/2
    expected_f05 = (1.25 * 0.5 * 0.5) / (0.25 * 0.5 + 0.5)  # 0.3125 / 0.625 = 0.5
    
    assert per_entity_precision(true_ids, pred_ids) == pytest.approx(0.5)
    assert per_entity_recall(true_ids, pred_ids) == pytest.approx(0.5)
    assert per_entity_f0_5(true_ids, pred_ids) == pytest.approx(expected_f05)

def test_f05_formula_precision_weight():
    # Precision=1.0, Recall=0.5
    true_ids = {"S2-1", "S3-2"}
    pred_ids = {"S2-1"}
    
    # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.83333...
    expected_f05 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
    assert per_entity_f0_5(true_ids, pred_ids) == pytest.approx(expected_f05)

def test_macro_and_country_evaluation():
    gt_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_ids": ["S2-10,S3-20", "", "S2-30"]
    })
    pred_df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_ids": ["S2-10,S3-20", "", "S2-99"] # S1-3 is incorrect
    })
    s1_df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "country": ["US", "US", "France"]
    })
    
    res = evaluate_predictions(gt_df, pred_df, s1_df)
    
    # S1-1: F0.5=1.0, S1-2: F0.5=1.0 (singleton), S1-3: F0.5=0.0
    # Macro F0.5 overall = (1.0 + 1.0 + 0.0) / 3 = 2/3 = 0.6667
    assert res["overall"]["number_of_s1_entities"] == 3
    assert res["overall"]["macro_f0_5"] == pytest.approx(2.0 / 3.0)
    
    # US breakdown: S1-1 (1.0) and S1-2 (1.0) -> macro F0.5 = 1.0
    assert res["by_country"]["US"]["number_of_s1_entities"] == 2
    assert res["by_country"]["US"]["macro_f0_5"] == pytest.approx(1.0)
    
    # France breakdown: S1-3 (0.0) -> macro F0.5 = 0.0
    assert res["by_country"]["France"]["number_of_s1_entities"] == 1
    assert res["by_country"]["France"]["macro_f0_5"] == pytest.approx(0.0)
