import pytest
import pandas as pd
from src.evaluation.measure_blocking_recall import (
    evaluate_blocking_recall,
    evaluate_blocking_channels,
    synthetic_dummy_candidate_generator
)

def test_blocking_recall_perfect():
    s1_df = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "country": ["US", "India"]})
    s2_df = pd.DataFrame({"entity_id": ["S2-10", "S2-20"], "country": ["US", "India"]})
    s3_df = pd.DataFrame({"entity_id": ["S3-100"], "country": ["US"]})
    
    gt = {
        "S1-1": {"S2-10", "S3-100"},
        "S1-2": set()  # Zero-match singleton
    }
    
    cands = {
        "S1-1": {"S2-10", "S3-100", "S2-20"}, # Perfect recall + 1 extra candidate
        "S1-2": set()
    }
    
    res = evaluate_blocking_recall(s1_df, s2_df, s3_df, gt, cands, channel_name="test")
    
    assert res["total_s1_entities"] == 2
    assert res["s1_entities_with_gt_matches"] == 1
    assert res["s1_entities_with_zero_gt_matches"] == 1
    assert res["macro_blocking_recall"] == 1.0
    assert res["micro_blocking_recall"] == 1.0
    assert res["s1_entities_with_any_missed_match"] == 0
    assert res["s1_entities_with_all_missed_matches"] == 0
    assert res["total_true_matches"] == 2
    assert res["total_recovered_true_matches"] == 2
    assert res["total_missed_true_matches"] == 0

def test_blocking_recall_partial_and_missed():
    s1_df = pd.DataFrame({"entity_id": ["S1-1", "S1-2"], "country": ["US", "India"]})
    s2_df = pd.DataFrame({"entity_id": ["S2-10", "S2-20"], "country": ["US", "India"]})
    s3_df = pd.DataFrame({"entity_id": ["S3-100"], "country": ["US"]})
    
    gt = {
        "S1-1": {"S2-10", "S3-100"},  # 2 true matches
        "S1-2": {"S2-20"}            # 1 true match
    }
    
    # Candidate generator misses S3-100 for S1-1 (recall=0.5), and misses S2-20 for S1-2 (recall=0.0)
    cands = {
        "S1-1": {"S2-10"},
        "S1-2": set()
    }
    
    res = evaluate_blocking_recall(s1_df, s2_df, s3_df, gt, cands, channel_name="partial_test")
    
    # S1-1 recall = 0.5, S1-2 recall = 0.0 -> Macro recall = 0.25
    assert res["macro_blocking_recall"] == 0.25
    # Micro recall = 1 recovered / 3 total = 1/3
    assert res["micro_blocking_recall"] == pytest.approx(1.0 / 3.0)
    assert res["s1_entities_with_any_missed_match"] == 2
    assert res["s1_entities_with_all_missed_matches"] == 1

def test_blocking_channels_and_union():
    s1_df = pd.DataFrame({"entity_id": ["S1-1"], "country": ["US"]})
    s2_df = pd.DataFrame({"entity_id": ["S2-10", "S2-20"], "country": ["US", "US"]})
    s3_df = pd.DataFrame({"entity_id": [], "country": []})
    
    gt = {"S1-1": {"S2-10", "S2-20"}}
    
    ch1 = {"S1-1": {"S2-10"}}
    ch2 = {"S1-1": {"S2-20"}}
    
    res = evaluate_blocking_channels(s1_df, s2_df, s3_df, gt, {"channel1": ch1, "channel2": ch2})
    
    assert res["individual_channels"]["channel1"]["macro_blocking_recall"] == 0.5
    assert res["individual_channels"]["channel2"]["macro_blocking_recall"] == 0.5
    # UNION channel combines candidates {S2-10, S2-20} -> recall = 1.0
    assert res["union_channel"]["macro_blocking_recall"] == 1.0

def test_synthetic_candidate_generator():
    s1_df = pd.DataFrame({"entity_id": ["S1-1", "S1-2"]})
    s2_df = pd.DataFrame({"entity_id": ["S2-1", "S2-2"]})
    s3_df = pd.DataFrame({"entity_id": ["S3-1"]})
    
    cands = synthetic_dummy_candidate_generator(s1_df, s2_df, s3_df, k_dummy=1)
    assert "S1-1" in cands
    assert cands["S1-1"] == {"S2-1", "S3-1"}
