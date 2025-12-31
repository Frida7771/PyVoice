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
    
    output = np.zeros_like(samples, dtype=np.float32)
    output[0] = samples[0]
    for i in range(1, len(samples)):
        output[i] = samples[i] - alpha * samples[i-1]
    return output


def hamming_window(size: int) -> np.ndarray:
    """
    Generate Hamming window, reduces spectral leakage
    
    Hamming formula: 0.54 - 0.46 * cos(2πn / (N-1))
    
    Args:
        size: Window size
    
    Returns:
        Hamming window array
    """
    window = np.zeros(size, dtype=np.float32)
    for i in range(size):
        window[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / (size - 1))
    return window


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
    
    window = np.zeros(size, dtype=np.float32)
    factor = 2.0 * np.pi / size
    for i in range(size):
        window[i] = 0.5 * (1.0 - np.cos(factor * i))
    return window


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
    
    # Conversion functions (using standard HTK formula: 2595 * log10)
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    
    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)
    
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    
    # Calculate center frequency points (Hz)
    # Use count + 2 points to define count triangles (left, center, right)
    mel_points = np.zeros(mel_bin_count + 2)
    hz_points = np.zeros(mel_bin_count + 2)
    bin_points = np.zeros(mel_bin_count + 2, dtype=int)
    
    step = (mel_max - mel_min) / (mel_bin_count + 1)
    
    for i in range(mel_bin_count + 2):
        mel = mel_min + i * step
        mel_points[i] = mel
        hz = mel_to_hz(mel)
        hz_points[i] = hz
        
        # Calculate corresponding FFT Bin index
        # Bin = (Hz / SampleRate) * FFTSize
        bin_val = int(np.floor((fft_size + 1) * hz / sample_rate))
        
        if bin_val > fft_size // 2:
            bin_val = fft_size // 2
        bin_points[i] = bin_val
    
    # Build filter matrix
    filters = []
    
    for i in range(mel_bin_count):
        filter_row = np.zeros(fft_size // 2 + 1, dtype=np.float32)
        
        start_bin = bin_points[i]
        center_bin = bin_points[i + 1]
        end_bin = bin_points[i + 2]
        
        # Slaney normalization (Area Normalization)
        # Ensure filters of different widths have the same energy
        # Width = right frequency - left frequency
        enorm = 2.0 / (hz_points[i + 2] - hz_points[i])
        
        # Left slope (ascending)
        for j in range(start_bin, center_bin):
            filter_row[j] = float((j - start_bin) / (center_bin - start_bin) * enorm)
        
        # Right slope (descending)
        for j in range(center_bin, end_bin):
            filter_row[j] = float((end_bin - j) / (end_bin - center_bin) * enorm)
        
        if center_bin < len(filter_row):
            filter_row[center_bin] = float(enorm)
        
        filters.append(filter_row.tolist())
    
    return filters


def apply_cmvn(features: List[List[float]], neg_mean: List[float], inv_std: List[float]) -> None:
    """
    Cepstral Mean and Variance Normalization (CMVN)
    
    Formula: result = (x + negMean) * invStd
    
    Args:
        features: Feature matrix (will be modified)
        neg_mean: Negative mean vector
        inv_std: Inverse standard deviation vector
    """
    for i in range(len(features)):
        dim = len(features[i])
        # Safety check to prevent dimension mismatch out of bounds
        check_len = min(len(neg_mean), dim, len(inv_std))
        
        for j in range(check_len):
            features[i][j] = (features[i][j] + neg_mean[j]) * inv_std[j]

