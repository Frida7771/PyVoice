"""
Paraformer ASR utility functions
"""
from typing import Dict, Tuple
import numpy as np
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from .config import SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE
from internal.mediautil import reformat_wav_bytes, pcm_bytes_to_float32

# Supported audio formats (requires ffmpeg for non-wav formats)
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.webm'}


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


def load_audio_file(file_path: str) -> bytes:
    """
    Load audio file and convert to WAV format bytes.
    Supports: WAV, MP3, FLAC, OGG, M4A, AAC, WMA, WEBM
    
    Note: Non-WAV formats require ffmpeg installed on the system.
    
    Args:
        file_path: Path to audio file
    
    Returns:
        WAV format byte stream
    
    Raises:
        ValueError: If file format is not supported
        RuntimeError: If ffmpeg is not installed (for non-WAV formats)
    """
    import subprocess
    import tempfile
    import os
    
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    # Direct read for WAV files
    if suffix == '.wav':
        with open(file_path, 'rb') as f:
            return f.read()
    
    # Check if format is supported
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    
    # Use ffmpeg directly for format conversion
    # Output: 16kHz, mono, 16-bit PCM WAV
    try:
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        # ffmpeg command: convert to 16kHz mono 16-bit WAV
        cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-ar', str(SAMPLE_RATE),  # Sample rate: 16000
            '-ac', str(CHANNELS),      # Channels: 1 (mono)
            '-sample_fmt', 's16',      # 16-bit
            '-f', 'wav',
            tmp_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Check if ffmpeg is not installed
            if 'not found' in result.stderr.lower() or 'no such file' in result.stderr.lower():
                raise RuntimeError(
                    f"ffmpeg is required for {suffix} format. "
                    "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
                )
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        
        # Read converted WAV
        with open(tmp_path, 'rb') as f:
            wav_bytes = f.read()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return wav_bytes
        
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg is required for {suffix} format. "
            "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

