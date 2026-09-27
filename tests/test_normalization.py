import pytest
from src.normalization.text_normalizer import normalize_name, normalize_name_core, tokenize_name
from src.normalization.address_normalizer import normalize_address, tokenize_address

def test_normalize_name_basic():
    assert normalize_name(None) == ""
    assert normalize_name("") == ""
    assert normalize_name("   ") == ""
    assert normalize_name("Acme Corp") == "acme corp"
    assert normalize_name("ACME   CORPORATION! ") == "acme corporation"

def test_normalize_name_legal_suffixes():
    text = "Baba Technologies Private Limited"
    # Preserves legal suffixes by default
    assert normalize_name(text) == "baba technologies private limited"
    # Strips legal suffixes in normalize_name_core
    assert normalize_name_core(text) == "baba technologies"

def test_normalize_name_suffix_variations():
    assert normalize_name_core("Shlok Electronics Pvt Ltd") == "shlok electronics"
    assert normalize_name_core("Kisan Avas Ltd.") == "kisan avas"
    assert normalize_name_core("Bay Creative Stone LLC") == "bay creative stone"
    assert normalize_name_core("LLC Eye Clinic") == "eye clinic"
    assert normalize_name_core("Straight Edge Book Store P.C.") == "straight edge book store"

def test_normalize_name_unicode_and_scripts():
    # Accent marks
    assert normalize_name("Nieto Álgonquin LLC") == "nieto algonquin llc"
    # Regional scripts (Telugu, Hindi, Kannada)
    telugu_text = "Baba టెక్నాలజీస్ ప్రైవేట్ లిమిటెڈ"
    norm_telugu = normalize_name(telugu_text)
    assert "baba" in norm_telugu
    assert "టెక్నాలజీస్" in norm_telugu  # Telugu characters preserved!

def test_normalize_address_basic():
    assert normalize_address(None) == ""
    assert normalize_address("   ") == ""
    assert normalize_address("4018- Annapolis Rd, HALETHORPE, MD") == "4018 annapolis rd halethorpe md"
    assert normalize_address("#4120 Highway 77, Marion, Arkansas") == "4120 highway 77 marion arkansas"
    assert normalize_address("##2337 Pendragon Rd") == "2337 pendragon rd"

def test_tokenize_helpers():
    assert tokenize_name("Baba Technologies Pvt Ltd", remove_legal_suffixes=True) == ["baba", "technologies"]
    assert tokenize_address("123 Main St, Unit #4") == ["123", "main", "st", "unit", "4"]
