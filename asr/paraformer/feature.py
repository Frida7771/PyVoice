"""
Paraformer ASR feature extraction module
"""
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from .config import SAMPLE_RATE
from internal.mediautil import (
    pre_emphasis, hamming_window, mel_filters, fft, apply_cmvn
)

# Global cache
_window = None
_mel_filters = None


def extract_features(samples: np.ndarray, neg_mean: np.ndarray, 
                    inv_std: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Feature processing
    
    Pipeline: Wave -> FilterBank -> LFR -> CMVN
    
    Args:
        samples: Audio sample data
        neg_mean: CMVN negative mean
        inv_std: CMVN inverse standard deviation
    
    Returns:
        Tuple of (feature array, frame count)
    """
    global _window, _mel_filters
    
    mel_bins = 80
    lfr_m = 7  # Window size
    lfr_n = 6  # Window shift
    
    # Extract FilterBank
    fbank_data, num_frames = compute_filter_bank(samples, SAMPLE_RATE, mel_bins)
    if num_frames == 0:
        raise ValueError("FBank feature extraction failed: frame count less than 1")
    
    # Apply LFR (Low Frame Rate)
    lfr_data, lfr_frames = apply_lfr(fbank_data, num_frames, mel_bins, lfr_m, lfr_n)
    if lfr_frames == 0:
        raise ValueError("LFR feature extraction failed: frame count less than 1")
    
    # CMVN
    if len(neg_mean) > 0 and len(inv_std) > 0:
        apply_cmvn(lfr_data, neg_mean.tolist(), inv_std.tolist())
    
    # Flatten to 1D array
    total_len = lfr_frames * (mel_bins * lfr_m)
    flattened = np.zeros(total_len, dtype=np.float32)
    row_size = mel_bins * lfr_m
    for i, frame in enumerate(lfr_data):
        flattened[i*row_size:(i+1)*row_size] = frame
    
    return flattened, lfr_frames


def compute_filter_bank(samples: np.ndarray, sample_rate: int, 
                        mel_bins: int) -> Tuple[List[List[float]], int]:
    """
    Compute FilterBank features
    
    Args:
        samples: Audio samples
        sample_rate: Sample rate
        mel_bins: Number of Mel bins
    
    Returns:
        Tuple of (feature matrix, frame count)
    """
    global _window, _mel_filters
    
    frame_len = 400  # 25ms @ 16kHz
    frame_shift = 160  # 10ms @ 16kHz
    fft_size = 512  # Next power of 2
    
    # Initialize window and filters (only once)
    if _window is None:
        _window = hamming_window(frame_len)
        _mel_filters = mel_filters(sample_rate, fft_size, mel_bins, 0, 0)
    
    # Pre-emphasis
    emphasized = pre_emphasis(samples, 0.97)
    
    # Prepare base data
    num_samples = len(emphasized)
    if num_samples < frame_len:
        return [], 0
    
    # Calculate frame count
    num_frames = (num_samples - frame_len) // frame_shift + 1
    
    # Allocate result matrix
    features = []
    
    for i in range(num_frames):
        start = i * frame_shift
        
        # Windowing & fill FFT buffer
        fft_buffer = np.zeros(fft_size, dtype=np.complex128)
        for j in range(fft_size):
            if j < frame_len:
                val = emphasized[start + j] * _window[j]
                fft_buffer[j] = complex(val, 0)
            else:
                fft_buffer[j] = 0  # Zero padding
        
        # FFT transform
        spectrum = fft(fft_buffer)
        
        # Calculate Mel energy
        feature_frame = np.zeros(mel_bins, dtype=np.float32)
        for k in range(mel_bins):
            sum_val = 0.0
            # Traverse first half of FFT result (Nyquist)
            for j in range(fft_size // 2 + 1):
                w = _mel_filters[k][j]
                if w > 0:
                    # Power = |X|^2
                    power = abs(spectrum[j]) ** 2
                    sum_val += power * w
            
            if sum_val < 1e-7:
                sum_val = 1e-7
            feature_frame[k] = np.log(sum_val)
        
        features.append(feature_frame.tolist())
    
    return features, num_frames


def apply_lfr(inputs: List[List[float]], num_frames: int, input_dim: int, 
              lfr_m: int, lfr_n: int) -> Tuple[List[List[float]], int]:
    """
    LFR (Low Frame Rate)
    
    Args:
        inputs: Input feature matrix
        num_frames: Input frame count
        input_dim: Input dimension
        lfr_m: LFR window size
        lfr_n: LFR window shift step
    
    Returns:
        Tuple of (output feature matrix, output frame count)
    """
    if num_frames < lfr_m:
        return [], 0
    
    # Calculate LFR output frame count
    out_frames = (num_frames - lfr_m) // lfr_n + 1
    out_dim = input_dim * lfr_m
    
    output = []
    
    for i in range(out_frames):
        output_frame = np.zeros(out_dim, dtype=np.float32)
        start_frame = i * lfr_n
        
        # Concatenate M frames
        for j in range(lfr_m):
            src_idx = start_frame + j
            # Target position: j * input_dim to (j+1) * input_dim
            dest_pos = j * input_dim
            output_frame[dest_pos:dest_pos+input_dim] = inputs[src_idx]
        
        output.append(output_frame.tolist())
    
    return output, out_frames

