"""
MeloTTS frontend processing module
"""
from typing import List, Tuple, Optional
import jieba
from .config import LexiconItem
from .polyphone import resolve_polyphone, is_polyphone

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
    
    # Track position in original text for polyphone disambiguation
    current_pos = 0
    
    for word in segments:
        if not word.strip():
            continue
        lower_word = word.lower()
        
        # Find position in original text
        word_pos = text.find(word, current_pos)
        if word_pos >= 0:
            current_pos = word_pos
        
        # Look up in lexicon (prefer full match)
        if append_ids_from_lexicon(lower_word, lexicon, token_map, ids, tones):
            current_pos += len(word)
            continue
        
        # Look up in Token
        if word in token_map:
            append_token(token_map[word], 0, ids, tones)
            current_pos += len(word)
            continue
        
        # OOV character-by-character fallback processing with polyphone support
        if len(word) > 1:
            sub_ids = []
            sub_tones = []
            
            for i, char in enumerate(word):
                char_str = char
                lower_char = char_str.lower()
                char_position = current_pos + i
                
                # Check if this is a polyphone character
                if is_polyphone(char_str):
                    pinyin = resolve_polyphone(text, char_str, char_position)
                    if pinyin and append_ids_from_lexicon_by_pinyin(pinyin, lexicon, token_map, sub_ids, sub_tones):
                        continue
                
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
                current_pos += len(word)
                continue
        
        # Single character with polyphone check
        if len(word) == 1 and is_polyphone(word):
            pinyin = resolve_polyphone(text, word, current_pos)
            if pinyin and append_ids_from_lexicon_by_pinyin(pinyin, lexicon, token_map, ids, tones):
                current_pos += 1
                continue
        
        # Completely unprocessable
        print(f"[WARN] Skipping completely unrecognized character/word: {word}")
        current_pos += len(word)
    
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


def append_ids_from_lexicon_by_pinyin(pinyin: str, lexicon: dict, token_map: dict,
                                      ids: List[int], tones: List[int]) -> bool:
    """
    根据拼音查找词典并追加 IDs（用于多音字消歧）
    
    拼音格式: 声母+韵母+声调数字 (如 "zhang3", "chang2", "hang2")
    
    Args:
        pinyin: 拼音字符串 (如 "zhang3")
        lexicon: Lexicon dictionary
        token_map: Token mapping dictionary
        ids: ID list (will be modified)
        tones: Tone list (will be modified)
    
    Returns:
        Whether successful
    """
    # 从拼音中提取声调
    tone = 0
    pinyin_base = pinyin
    if pinyin and pinyin[-1].isdigit():
        tone = int(pinyin[-1])
        pinyin_base = pinyin[:-1]
    
    # 尝试在 lexicon 中查找拼音
    # lexicon 的 key 可能是汉字或拼音，这里直接用拼音查找
    if pinyin_base in lexicon:
        item = lexicon[pinyin_base]
        for i, phone in enumerate(item.phones):
            if phone not in token_map:
                continue
            t_val = item.tones[i] if i < len(item.tones) else tone
            append_token(token_map[phone], t_val, ids, tones)
        return True
    
    # 如果 lexicon 中没有拼音，尝试将拼音拆分为声母+韵母
    # 常见声母
    initials = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 
                'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'r', 'y', 'w']
    
    initial = ''
    final = pinyin_base
    
    for ini in initials:
        if pinyin_base.startswith(ini):
            initial = ini
            final = pinyin_base[len(ini):]
            break
    
    # 尝试查找声母和韵母的组合
    phones_to_add = []
    
    if initial:
        if initial in token_map:
            phones_to_add.append((token_map[initial], 0))
        elif initial in lexicon:
            item = lexicon[initial]
            for i, phone in enumerate(item.phones):
                if phone in token_map:
                    phones_to_add.append((token_map[phone], 0))
    
    if final:
        if final in token_map:
            phones_to_add.append((token_map[final], tone))
        elif final in lexicon:
            item = lexicon[final]
            for i, phone in enumerate(item.phones):
                if phone in token_map:
                    t = item.tones[i] if i < len(item.tones) else tone
                    phones_to_add.append((token_map[phone], t))
    
    if phones_to_add:
        for phone_id, t in phones_to_add:
            append_token(phone_id, t, ids, tones)
        return True
    
    return False


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

