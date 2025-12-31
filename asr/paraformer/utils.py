"""
Paraformer ASR utility functions
"""
from typing import Dict, Tuple
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from .config import SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE
from internal.mediautil import reformat_wav_bytes, pcm_bytes_to_float32


def load_tokens(path: str) -> Dict[int, str]:
    """
    Load Token ID mapping table
    
    Data format: token id
    
    Args:
        path: Path to tokens.txt file
    
    Returns:
        Dictionary mapping Token ID to string
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
                    token_map[token_id] = token
                except ValueError:
                    continue
    return token_map


def load_cmvn(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse am.mvn file
    
    Returns neg_mean (negative mean) and inv_std (inverse standard deviation)
    
    Args:
        path: Path to am.mvn file
    
    Returns:
        Tuple of (neg_mean, inv_std)
    """
    neg_mean = None
    inv_std = None
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('<LearnRateCoef>'):
                continue
            
            # Read data and convert to float32
            parts = line.split()
            values = []
            data_parts = parts[3:len(parts)-1]
            for v in data_parts:
                try:
                    values.append(float(v))
                except ValueError:
                    continue
            
            if neg_mean is None:
                neg_mean = np.array(values, dtype=np.float32)
            else:
                inv_std = np.array(values, dtype=np.float32)
                break
    
    if neg_mean is None or inv_std is None or len(neg_mean) == 0 or len(inv_std) == 0:
        raise ValueError("No valid CMVN data found")
    
    return neg_mean, inv_std


def parse_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """
    Convert WAV byte stream to float32 array
    
    Args:
        wav_bytes: WAV file byte stream
    
    Returns:
        float32 audio sample array
    """
    # Audio format conversion
    target_bytes = reformat_wav_bytes(wav_bytes, SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE)
    return pcm_bytes_to_float32(target_bytes[44:], BITS_PER_SAMPLE)

