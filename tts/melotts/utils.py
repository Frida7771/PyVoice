"""
MeloTTS utility functions
"""
from typing import Dict
from .config import LexiconItem


def load_lexicon(path: str) -> Dict[str, LexiconItem]:
    """
    Load pronunciation lexicon
    
    Data format: word phone1 phone2 ... tone1 tone2 ...
    
    Args:
        path: Path to lexicon.txt file
    
    Returns:
        Lexicon dictionary
    """
    lex = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            
            word = parts[0]
            rest = parts[1:]
            
            # Number of phones must equal number of tones, so remaining part must be even
            if len(rest) % 2 != 0:
                print(f"Skipping invalid lexicon line: {word}")
                continue
            
            mid = len(rest) // 2
            phones = rest[:mid]
            tone_strings = rest[mid:]
            
            tones = []
            for t_str in tone_strings:
                try:
                    tones.append(int(t_str))
                except ValueError:
                    continue
            
            lex[word] = LexiconItem(phones=phones, tones=tones)
    
    return lex


def load_tokens(path: str) -> Dict[str, int]:
    """
    Load Token ID mapping table
    
    Data format: token id
    
    Args:
        path: Path to tokens.txt file
    
    Returns:
        Dictionary mapping Token string to ID
    """
    token_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                try:
                    token_id = int(parts[1])
                    token_map[token] = token_id
                except ValueError:
                    continue
    
    return token_map

