"""
Audio processing utility module
"""
from .wav import (
    WavHeader, parse_wav_header, read_wav_header, write_wav, save_wav,
    float32_to_pcm_bytes, float32_to_wav_bytes, pcm_bytes_to_float32,
    reformat_wav_bytes, change_channels, resample_safe, resample_linear
)
from .audio import (
    pre_emphasis, hamming_window, hann_window, fft, mel_filters, apply_cmvn
)

__all__ = [
    'WavHeader', 'parse_wav_header', 'read_wav_header', 'write_wav', 'save_wav',
    'float32_to_pcm_bytes', 'float32_to_wav_bytes', 'pcm_bytes_to_float32',
    'reformat_wav_bytes', 'change_channels', 'resample_safe', 'resample_linear',
    'pre_emphasis', 'hamming_window', 'hann_window', 'fft', 'mel_filters', 'apply_cmvn'
]

