"""
MeloTTS frontend processing module
"""
from typing import List, Tuple
from .config import LexiconItem


def text_to_ids(text: str, lexicon: dict, token_map: dict) -> Tuple[List[int], List[int]]:
    """
    Convert normalized text to Token ID and Tone ID sequences
    
    Args:
        text: Normalized text
        lexicon: Lexicon dictionary
        token_map: Token mapping dictionary
    
    Returns:
        Tuple of (ids, tones)
    """
    ids = []
    tones = []
    
    # Use smart segmentation to split text into words/characters/punctuation
    segments = smart_segment(text)
    
    for word in segments:
        if not word.strip():
            continue
        lower_word = word.lower()
        
        # Look up in lexicon (prefer full match)
        if append_ids_from_lexicon(lower_word, lexicon, token_map, ids, tones):
            continue
        
        # Look up in Token
        if word in token_map:
            append_token(token_map[word], 0, ids, tones)
            continue
        
        # OOV character-by-character fallback processing
        if len(word) > 1:
            sub_ids = []
            sub_tones = []
            
            for char in word:
                char_str = char
                lower_char = char_str.lower()
                
                # Try single character lookup in lexicon (e.g., 'a' -> [phone...])
                if append_ids_from_lexicon(lower_char, lexicon, token_map, sub_ids, sub_tones):
                    continue
                
                # Try single character direct lookup in Token
                if char_str in token_map:
                    append_token(token_map[char_str], 0, sub_ids, sub_tones)
                    continue
                
                # Single character also cannot be processed
                print(f"[WARN] OOV character lost: {char_str} (in {word})")
            
            if len(sub_ids) > 0:
                ids.extend(sub_ids)
                tones.extend(sub_tones)
                continue
        
        # Completely unprocessable
        print(f"[WARN] Skipping completely unrecognized character/word: {word}")
    
    # End padding
    append_token(0, 0, ids, tones)
    
    if len(ids) <= 1:
        raise ValueError("Generated Token sequence is empty")
    
    return ids, tones


def append_ids_from_lexicon(key: str, lexicon: dict, token_map: dict,
                           ids: List[int], tones: List[int]) -> bool:
    """
    Try to find and append IDs from lexicon, return whether successful
    
    Args:
        key: Lookup key
        lexicon: Lexicon dictionary
        token_map: Token mapping dictionary
        ids: ID list (will be modified)
        tones: Tone list (will be modified)
    
    Returns:
        Whether successful
    """
    if key not in lexicon:
        return False
    
    item = lexicon[key]
    for i, phone in enumerate(item.phones):
        if phone not in token_map:
            print(f"[ERROR] Lexicon contains unknown phone: {phone} (key: {key})")
            continue
        t_val = item.tones[i] if i < len(item.tones) else 0
        append_token(token_map[phone], t_val, ids, tones)
    
    return True


def append_token(id_val: int, tone: int, ids: List[int], tones: List[int]) -> None:
    """
    Helper function: uniformly append format [0, id]
    
    Args:
        id_val: Token ID
        tone: Tone value
        ids: ID list (will be modified)
        tones: Tone list (will be modified)
    """
    ids.append(0)
    ids.append(id_val)
    tones.append(0)
    tones.append(tone)


def smart_segment(text: str) -> List[str]:
    """
    Simple segmentation logic: distinguish Chinese characters/symbols from English/number words
    
    Args:
        text: Input text
    
    Returns:
        Segmentation result list
    """
    segments = []
    buffer = []
    
    for char in text:
        # English, numbers, single quotes are processed continuously as part of word
        if ('a' <= char <= 'z') or ('A' <= char <= 'Z') or ('0' <= char <= '9') or char == '\'':
            buffer.append(char)
        else:
            # Encounter non-English/number, first settle previous buffer
            if buffer:
                segments.append(''.join(buffer))
                buffer = []
            # Current character as independent segment (Chinese characters, punctuation)
            segments.append(char)
    
    # Settle remaining buffer
    if buffer:
        segments.append(''.join(buffer))
    
    return segments

