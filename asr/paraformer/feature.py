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
    
    # Flatten to 1D array (vectorized)
    flattened = np.array(lfr_data, dtype=np.float32).flatten()
    
    return flattened, lfr_frames


def compute_filter_bank(samples: np.ndarray, sample_rate: int, 
                        mel_bins: int) -> Tuple[List[List[float]], int]:
    """
    Compute FilterBank features (vectorized implementation)
    
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
        _mel_filters = np.array(mel_filters(sample_rate, fft_size, mel_bins, 0, 0), dtype=np.float32)
    
    # Pre-emphasis
    emphasized = pre_emphasis(samples, 0.97)
    
    # Prepare base data
    num_samples = len(emphasized)
    if num_samples < frame_len:
        return [], 0
    
    # Calculate frame count
    num_frames = (num_samples - frame_len) // frame_shift + 1
    
    # Vectorized frame extraction using stride tricks
    # Create frame indices
    frame_starts = np.arange(num_frames) * frame_shift
    frame_indices = frame_starts[:, np.newaxis] + np.arange(frame_len)
    
    # Extract all frames at once: (num_frames, frame_len)
    frames = emphasized[frame_indices]
    
    # Apply window to all frames: (num_frames, frame_len)
    windowed = frames * _window
    
    # Zero-pad to fft_size and FFT all frames at once
    padded = np.zeros((num_frames, fft_size), dtype=np.float64)
    padded[:, :frame_len] = windowed
    
    # FFT all frames: (num_frames, fft_size)
    spectra = np.fft.fft(padded, axis=1)
    
    # Power spectrum (first half only): (num_frames, fft_size//2+1)
    power_spectrum = np.abs(spectra[:, :fft_size // 2 + 1]) ** 2
    
    # Apply mel filters: (num_frames, mel_bins)
    # mel_filters is (mel_bins, fft_size//2+1), power_spectrum is (num_frames, fft_size//2+1)
    mel_energy = np.dot(power_spectrum, _mel_filters.T)
    
    # Floor and log
    mel_energy = np.maximum(mel_energy, 1e-7)
    features = np.log(mel_energy).astype(np.float32)
    
    return features.tolist(), num_frames


def apply_lfr(inputs: List[List[float]], num_frames: int, input_dim: int, 
              lfr_m: int, lfr_n: int) -> Tuple[List[List[float]], int]:
    """
    LFR (Low Frame Rate) - vectorized implementation
    
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
    
    # Convert to numpy array for vectorized operations
    inputs_arr = np.array(inputs, dtype=np.float32)
    
    # Create output array
    out_dim = input_dim * lfr_m
    output = np.zeros((out_frames, out_dim), dtype=np.float32)
    
    # Vectorized frame concatenation
    for i in range(out_frames):
        start_frame = i * lfr_n
        # Concatenate lfr_m consecutive frames
        output[i] = inputs_arr[start_frame:start_frame + lfr_m].flatten()
    
    return output.tolist(), out_frames

