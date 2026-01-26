"""
MeloTTS frontend processing module
"""
from typing import List, Tuple
import jieba
from .config import LexiconItem

# Silence jieba loading messages
jieba.setLogLevel(jieba.logging.INFO)


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
    Smart segmentation using jieba for Chinese and rule-based for English/numbers.
    
    Benefits of jieba:
    - Better word boundary detection (e.g., "今天天气" -> ["今天", "天气"] instead of char-by-char)
    - Improved pronunciation for multi-character words in lexicon
    
    Args:
        text: Input text
    
    Returns:
        Segmentation result list
    """
    segments = []
    buffer = []
    
    def flush_buffer():
        """Flush English/number buffer to segments"""
        if buffer:
            segments.append(''.join(buffer))
            buffer.clear()
    
    def flush_chinese_buffer(chinese_buffer: List[str]):
        """Use jieba to segment Chinese text buffer"""
        if chinese_buffer:
            chinese_text = ''.join(chinese_buffer)
            # jieba.cut returns generator, convert to list
            words = list(jieba.cut(chinese_text))
            segments.extend(words)
            chinese_buffer.clear()
    
    chinese_buffer = []
    
    for char in text:
        # English, numbers, single quotes -> buffer as word
        if ('a' <= char <= 'z') or ('A' <= char <= 'Z') or ('0' <= char <= '9') or char == '\'':
            flush_chinese_buffer(chinese_buffer)
            buffer.append(char)
        # Chinese characters -> collect for jieba
        elif '\u4e00' <= char <= '\u9fff':
            flush_buffer()
            chinese_buffer.append(char)
        # Punctuation and other characters -> flush and add directly
        else:
            flush_buffer()
            flush_chinese_buffer(chinese_buffer)
            if char.strip():  # Skip whitespace
                segments.append(char)
    
    # Flush remaining buffers
    flush_buffer()
    flush_chinese_buffer(chinese_buffer)
    
    return segments

