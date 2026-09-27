import re
import unicodedata
from typing import Optional, List

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

def normalize_address(text: Optional[str]) -> str:
    """
    Conservative business address normalization.
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
    return s

def tokenize_address(text: Optional[str]) -> List[str]:
    norm = normalize_address(text)
    return norm.split() if norm else []
