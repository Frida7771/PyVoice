"""
Benchmark: Compare vectorized vs loop-based audio feature extraction

Usage:
    python benchmark.py
"""
import time
import numpy as np

# ============ Vectorized Version (Current) ============

def compute_filter_bank_vectorized(samples: np.ndarray, sample_rate: int = 16000, 
                                    mel_bins: int = 80):
    """Vectorized FilterBank extraction"""
    frame_len = 400
    frame_shift = 160
    fft_size = 512
    
    # Pre-emphasis
    emphasized = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    
    num_samples = len(emphasized)
    if num_samples < frame_len:
        return np.array([]), 0
    
    num_frames = (num_samples - frame_len) // frame_shift + 1
    
    # Hamming window
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))
    
    # Mel filters (simplified)
    mel_filters = np.random.rand(mel_bins, fft_size // 2 + 1).astype(np.float32)
    
    # === VECTORIZED: Extract all frames at once ===
    frame_starts = np.arange(num_frames) * frame_shift
    frame_indices = frame_starts[:, np.newaxis] + np.arange(frame_len)
    frames = emphasized[frame_indices]
    
    # Apply window to all frames at once
    windowed = frames * window
    
    # Zero-pad and FFT all frames at once
    padded = np.zeros((num_frames, fft_size), dtype=np.float64)
    padded[:, :frame_len] = windowed
    spectra = np.fft.fft(padded, axis=1)
    
    # Power spectrum
    power_spectrum = np.abs(spectra[:, :fft_size // 2 + 1]) ** 2
    
    # Mel filterbank (matrix multiply)
    mel_energy = np.dot(power_spectrum, mel_filters.T)
    mel_energy = np.maximum(mel_energy, 1e-7)
    features = np.log(mel_energy).astype(np.float32)
    
    return features, num_frames


# ============ Loop-based Version (Old) ============

def compute_filter_bank_loop(samples: np.ndarray, sample_rate: int = 16000, 
                              mel_bins: int = 80):
    """Loop-based FilterBank extraction (original slow version)"""
    frame_len = 400
    frame_shift = 160
    fft_size = 512
    
    # Pre-emphasis (loop)
    emphasized = np.zeros(len(samples))
    emphasized[0] = samples[0]
    for i in range(1, len(samples)):
        emphasized[i] = samples[i] - 0.97 * samples[i-1]
    
    num_samples = len(emphasized)
    if num_samples < frame_len:
        return [], 0
    
    num_frames = (num_samples - frame_len) // frame_shift + 1
    
    # Hamming window
    window = []
    for i in range(frame_len):
        window.append(0.54 - 0.46 * np.cos(2 * np.pi * i / (frame_len - 1)))
    
    # Mel filters (simplified)
    mel_filters = np.random.rand(mel_bins, fft_size // 2 + 1).astype(np.float32)
    
    features = []
    
    # === LOOP: Process each frame one by one ===
    for frame_idx in range(num_frames):
        start = frame_idx * frame_shift
        
        # Extract frame
        frame = []
        for i in range(frame_len):
            frame.append(emphasized[start + i])
        
        # Apply window
        windowed = []
        for i in range(frame_len):
            windowed.append(frame[i] * window[i])
        
        # Zero-pad
        padded = windowed + [0.0] * (fft_size - frame_len)
        
        # FFT (using numpy, but called per-frame)
        spectrum = np.fft.fft(padded)
        
        # Power spectrum
        power = []
        for i in range(fft_size // 2 + 1):
            power.append(abs(spectrum[i]) ** 2)
        
        # Mel filterbank (loop)
        mel_energy = []
        for m in range(mel_bins):
            energy = 0.0
            for i in range(fft_size // 2 + 1):
                energy += power[i] * mel_filters[m, i]
            mel_energy.append(max(energy, 1e-7))
        
        # Log
        log_mel = [np.log(e) for e in mel_energy]
        features.append(log_mel)
    
    return features, num_frames


# ============ Benchmark ============

def run_benchmark():
    print("=" * 60)
    print("PyVoice Audio Feature Extraction Benchmark")
    print("=" * 60)
    
    # Generate test audio (3 seconds @ 16kHz)
    duration_sec = 3
    sample_rate = 16000
    samples = np.random.randn(sample_rate * duration_sec).astype(np.float32)
    
    print(f"\nTest audio: {duration_sec}s @ {sample_rate}Hz = {len(samples)} samples")
    print("-" * 60)
    
    # Warm up
    compute_filter_bank_vectorized(samples[:1000])
    compute_filter_bank_loop(samples[:1000])
    
    # Benchmark vectorized version
    iterations = 10
    
    print(f"\nRunning {iterations} iterations...")
    
    start = time.perf_counter()
    for _ in range(iterations):
        compute_filter_bank_vectorized(samples)
    vectorized_time = (time.perf_counter() - start) / iterations
    
    # Benchmark loop version
    start = time.perf_counter()
    for _ in range(iterations):
        compute_filter_bank_loop(samples)
    loop_time = (time.perf_counter() - start) / iterations
    
    # Results
    speedup = loop_time / vectorized_time
    reduction = (1 - vectorized_time / loop_time) * 100
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Loop-based version:       {loop_time*1000:.2f} ms")
    print(f"Vectorized version:       {vectorized_time*1000:.2f} ms")
    print(f"Speedup:                  {speedup:.1f}x")
    print(f"Latency reduction:        {reduction:.0f}%")
    print("=" * 60)
    
    # Generate resume-friendly statement
    print("\n📝 Resume statement:")
    print(f"   'Engineered audio feature extraction pipeline using NumPy")
    print(f"    vectorization, reducing inference latency by {reduction:.0f}%.'")
    

if __name__ == "__main__":
    run_benchmark()

