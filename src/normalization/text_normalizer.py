import re
import unicodedata
from typing import List, Optional

# Default list of common business legal suffixes observed in ER datasets
DEFAULT_LEGAL_SUFFIXES = [
    "private limited",
    "pvt ltd",
    "pvt limited",
    "pvtltd",
    "private ltd",
    "incorporated",
    "corporation",
    "company",
    "limited",
    "gmbh",
    "sarl",
    "sarlu",
    "s a",
    "ltd",
    "inc",
    "llc",
    "llp",
    "corp",
    "pc",
    "p c",
    "lp",
    "co",
    "sa",
]

def _clean_unicode(text: str) -> str:
    """
    1. Fast-path ASCII strings.
    2. Decompose Latin accented characters and strip Latin combining diacritical marks.
    3. Replace Unicode Punctuation (P category) and Symbols (S category) with spaces.
    4. Preserves Letters (L), Numbers (N), Whitespace (Z), and Indic Combining Marks (M).
    """
    if text.isascii():
        return re.sub(r'[^\w\s]', ' ', text)
        
    nfkd = unicodedata.normalize('NFKD', text)
    cleaned = []
    for c in nfkd:
        cat = unicodedata.category(c)
        if cat == 'Mn' and ord(c) <= 0x036F:
            continue
        elif cat.startswith('P') or cat.startswith('S'):
            cleaned.append(' ')
        else:
            cleaned.append(c)
            
    return unicodedata.normalize('NFC', "".join(cleaned))

def normalize_name(
    text: Optional[str],
    remove_legal_suffixes: bool = False,
    legal_suffixes: Optional[List[str]] = None
) -> str:
    """
    Conservative business name normalization.
    """
    if text is None or not isinstance(text, str):
        return ""
        
    s = text.strip()
    if not s:
        return ""
        
    # Unicode cleaning
    s = _clean_unicode(s)
    
    # Case normalization
    s = s.lower()
    
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    
    if remove_legal_suffixes:
        suffixes = legal_suffixes if legal_suffixes is not None else DEFAULT_LEGAL_SUFFIXES
        sorted_suffixes = sorted(suffixes, key=len, reverse=True)
        
        changed = True
        while changed and s:
            changed = False
            for suf in sorted_suffixes:
                pattern = r'(?:\b|_)' + re.escape(suf) + r'(?:\b|_)'
                if re.search(pattern + r'\s*$', s):
                    s = re.sub(pattern + r'\s*$', '', s).strip()
                    changed = True
                    break
                elif re.search(r'^\s*' + pattern, s):
                    s = re.sub(r'^\s*' + pattern, '', s).strip()
                    changed = True
                    break
                    
    # Final whitespace cleanup
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_name_core(
    text: Optional[str],
    legal_suffixes: Optional[List[str]] = None
) -> str:
    return normalize_name(text, remove_legal_suffixes=True, legal_suffixes=legal_suffixes)

def tokenize_name(text: Optional[str], remove_legal_suffixes: bool = False) -> List[str]:
    norm = normalize_name(text, remove_legal_suffixes=remove_legal_suffixes)
    return norm.split() if norm else []
