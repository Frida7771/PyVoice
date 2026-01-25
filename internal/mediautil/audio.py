"""
Audio signal processing module
"""
import numpy as np
from typing import List


def pre_emphasis(samples: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Pre-emphasis filter, enhances high frequencies, balances spectral energy
    
    Difference formula: y[t] = x[t] - alpha * x[t-1]
    
    Args:
        samples: Input audio data
        alpha: Pre-emphasis coefficient, recommended 0.97
    
    Returns:
        Processed audio data
    """
    if len(samples) == 0:
        return np.array([], dtype=np.float32)
    
    # Vectorized: y[0] = x[0], y[1:] = x[1:] - alpha * x[:-1]
    return np.append(samples[0], samples[1:] - alpha * samples[:-1]).astype(np.float32)


def hamming_window(size: int) -> np.ndarray:
    """
    Generate Hamming window, reduces spectral leakage
    
    Hamming formula: 0.54 - 0.46 * cos(2πn / (N-1))
    
    Args:
        size: Window size
    
    Returns:
        Hamming window array
    """
    if size <= 1:
        return np.ones(size, dtype=np.float32)
    
    # Vectorized window generation
    n = np.arange(size)
    return (0.54 - 0.46 * np.cos(2 * np.pi * n / (size - 1))).astype(np.float32)


def hann_window(size: int) -> np.ndarray:
    """
    Generate Hann window
    
    Hann formula: 0.5 * (1 - cos(2πn / N))
    
    Args:
        size: Window size
    
    Returns:
        Hann window array
    """
    if size <= 0:
        return np.array([], dtype=np.float32)
    if size == 1:
        return np.array([1.0], dtype=np.float32)
    
    # Vectorized window generation
    n = np.arange(size)
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * n / size))).astype(np.float32)


def fft(x: np.ndarray) -> np.ndarray:
    """
    Fast Fourier Transform, time domain to frequency domain
    
    Args:
        x: Time domain data (waveform), complex array
    
    Returns:
        Frequency domain data, complex array
    """
    return np.fft.fft(x)


def mel_filters(sample_rate: int, fft_size: int, mel_bin_count: int, 
                f_min: float = 0.0, f_max: float = 0.0) -> List[List[float]]:
    """
    Generate Mel filter bank weight matrix, maps linear frequency to Mel scale
    
    Args:
        sample_rate: Sample rate
        fft_size: FFT window size
        mel_bin_count: Number of Mel bins
        f_min: Minimum frequency
        f_max: Maximum frequency
    
    Returns:
        Mel filter matrix
    """
    # Default parameter values
    if f_max == 0:
        f_max = sample_rate / 2.0
    if f_min < 0:
        f_min = 0
    
    # Vectorized Hz <-> Mel conversion (HTK formula)
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    
    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10, mel / 2595.0) - 1.0)
    
    mel_min = hz_to_mel(np.array([f_min]))[0]
    mel_max = hz_to_mel(np.array([f_max]))[0]
    
    # Vectorized: calculate all mel/hz/bin points at once
    mel_points = np.linspace(mel_min, mel_max, mel_bin_count + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.minimum(bin_points, fft_size // 2)
    
    # Build filter matrix using vectorized operations
    num_bins = fft_size // 2 + 1
    filters = np.zeros((mel_bin_count, num_bins), dtype=np.float32)
    fft_freqs = np.arange(num_bins)
    
    for i in range(mel_bin_count):
        start_bin = bin_points[i]
        center_bin = bin_points[i + 1]
        end_bin = bin_points[i + 2]
        
        # Slaney normalization
        enorm = 2.0 / (hz_points[i + 2] - hz_points[i]) if hz_points[i + 2] != hz_points[i] else 0.0
        
        # Left slope (ascending): vectorized
        if center_bin > start_bin:
            left_mask = (fft_freqs >= start_bin) & (fft_freqs < center_bin)
            filters[i, left_mask] = (fft_freqs[left_mask] - start_bin) / (center_bin - start_bin) * enorm
        
        # Right slope (descending): vectorized
        if end_bin > center_bin:
            right_mask = (fft_freqs >= center_bin) & (fft_freqs < end_bin)
            filters[i, right_mask] = (end_bin - fft_freqs[right_mask]) / (end_bin - center_bin) * enorm
        
        # Center peak
        if center_bin < num_bins:
            filters[i, center_bin] = enorm
    
    return filters.tolist()


def apply_cmvn(features: List[List[float]], neg_mean: List[float], inv_std: List[float]) -> None:
    """
    Cepstral Mean and Variance Normalization (CMVN)
    
    Formula: result = (x + negMean) * invStd
    
    Args:
        features: Feature matrix (will be modified in-place)
        neg_mean: Negative mean vector
        inv_std: Inverse standard deviation vector
    """
    if not features:
        return
    
    # Convert to numpy for vectorized operation
    neg_mean_arr = np.array(neg_mean, dtype=np.float32)
    inv_std_arr = np.array(inv_std, dtype=np.float32)
    
    for i in range(len(features)):
        dim = len(features[i])
        check_len = min(len(neg_mean), dim, len(inv_std))
        
        # Vectorized: apply CMVN to each row
        row = np.array(features[i][:check_len], dtype=np.float32)
        row = (row + neg_mean_arr[:check_len]) * inv_std_arr[:check_len]
        features[i][:check_len] = row.tolist()

