"""
WAV file processing module
"""
import struct
import wave
import io
from typing import Tuple, Optional
import numpy as np


class WavHeader:
    """WAV file header structure"""
    
    def __init__(self, audio_format: int = 1, num_channels: int = 1, 
                 sample_rate: int = 16000, bits_per_sample: int = 16, 
                 data_size: int = 0):
        self.audio_format = audio_format
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.bits_per_sample = bits_per_sample
        self.data_size = data_size
    
    def get_duration(self) -> float:
        """Calculate audio duration (seconds)"""
        if self.sample_rate == 0:
            return 0.0
        byte_rate = self.sample_rate * self.num_channels * self.bits_per_sample // 8
        if byte_rate == 0:
            return 0.0
        return self.data_size / byte_rate
    
    def __str__(self) -> str:
        return (f"WAV Header: [Format: {self.audio_format}, "
                f"Channels: {self.num_channels}, Rate: {self.sample_rate}Hz, "
                f"Bits: {self.bits_per_sample}, DataSize: {self.data_size} bytes, "
                f"Duration: {self.get_duration():.2f}s]")


def parse_wav_header(data: bytes) -> WavHeader:
    """
    Parse WAV header from byte stream
    
    Args:
        data: Byte stream containing WAV header information (at least 44 bytes)
    
    Returns:
        WavHeader object
    """
    if len(data) < 44:
        raise ValueError("WAV header data length must be at least 44 bytes")
    
    # Validate RIFF and WAVE identifiers
    if data[0:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError("Not a valid RIFF/WAVE file")
    
    # Parse header fields
    audio_format = struct.unpack('<H', data[20:22])[0]
    num_channels = struct.unpack('<H', data[22:24])[0]
    sample_rate = struct.unpack('<I', data[24:28])[0]
    bits_per_sample = struct.unpack('<H', data[34:36])[0]
    data_size = struct.unpack('<I', data[40:44])[0]
    
    return WavHeader(audio_format, num_channels, sample_rate, bits_per_sample, data_size)


def read_wav_header(file_path: str) -> WavHeader:
    """
    Read WAV header from file
    
    Args:
        file_path: Path to WAV file
    
    Returns:
        WavHeader object
    """
    with open(file_path, 'rb') as f:
        header_bytes = f.read(44)
    return parse_wav_header(header_bytes)


def write_wav(w, pcm_data: bytes, sample_rate: int, channels: int, bits_per_sample: int) -> None:
    """
    Write PCM data as WAV format
    
    Args:
        w: Write target (file object or BytesIO)
        pcm_data: Raw PCM data
        sample_rate: Sample rate
        channels: Number of channels
        bits_per_sample: Bits per sample
    """
    if sample_rate <= 0 or channels <= 0 or bits_per_sample <= 0:
        raise ValueError(f"Invalid parameters: rate={sample_rate}, chan={channels}, bit={bits_per_sample}")
    
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    # Build WAV header
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    header += struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, channels, sample_rate, 
                         byte_rate, block_align, bits_per_sample)
    header += struct.pack('<4sI', b'data', data_size)
    
    w.write(header)
    w.write(pcm_data)


def save_wav(file_path: str, pcm_data: bytes, sample_rate: int, 
             channels: int, bits_per_sample: int) -> None:
    """
    Save PCM data as local WAV file
    
    Args:
        file_path: File path
        pcm_data: Raw PCM data
        sample_rate: Sample rate
        channels: Number of channels
        bits_per_sample: Bits per sample
    """
    with open(file_path, 'wb') as f:
        write_wav(f, pcm_data, sample_rate, channels, bits_per_sample)


def float32_to_pcm_bytes(data: np.ndarray, bits_per_sample: int) -> bytes:
    """
    Convert standard float audio data to PCM byte stream with specified bit depth
    
    Args:
        data: Audio sample array, value range should be [-1.0, 1.0]
        bits_per_sample: Bit depth, supports 16, 24, 32
    
    Returns:
        PCM byte stream
    """
    if len(data) == 0:
        return b''
    
    # Handle NaN and clipping
    data = np.clip(data, -1.0, 1.0)
    data = np.nan_to_num(data, nan=0.0)
    
    if bits_per_sample == 16:
        scale = 32767.0
        samples = (data * scale).astype(np.int16)
        return samples.tobytes()
    elif bits_per_sample == 24:
        scale = 8388607.0
        samples = (data * scale).astype(np.int32)
        # 24-bit Little Endian: [Low, Mid, High]
        output = bytearray(len(data) * 3)
        for i, val in enumerate(samples):
            output[i*3] = val & 0xFF
            output[i*3+1] = (val >> 8) & 0xFF
            output[i*3+2] = (val >> 16) & 0xFF
        return bytes(output)
    elif bits_per_sample == 32:
        scale = 2147483647.0
        samples = (data * scale).astype(np.int32)
        return samples.tobytes()
    else:
        raise ValueError(f"Unsupported bit depth: {bits_per_sample}")


def float32_to_wav_bytes(data: np.ndarray, sample_rate: int, 
                        channels: int, bits_per_sample: int) -> bytes:
    """
    Convert standard float audio data to complete WAV file byte stream
    
    Args:
        data: Raw audio data (mono or stereo interleaved)
        sample_rate: Sample rate
        channels: Number of channels
        bits_per_sample: Bits per sample
    
    Returns:
        Complete WAV file byte stream
    """
    if sample_rate <= 0 or channels <= 0 or bits_per_sample <= 0:
        raise ValueError(f"Invalid parameters: rate={sample_rate}, chan={channels}, bit={bits_per_sample}")
    
    # Convert Float32 data to raw PCM byte stream
    pcm_body = float32_to_pcm_bytes(data, bits_per_sample)
    
    # Write to buffer
    buf = io.BytesIO()
    write_wav(buf, pcm_body, sample_rate, channels, bits_per_sample)
    return buf.getvalue()


def pcm_bytes_to_float32(data: bytes, bits_per_sample: int) -> np.ndarray:
    """
    Convert PCM byte stream to float32 array
    
    Args:
        data: Raw PCM data
        bits_per_sample: Bit depth, supports 16, 24, 32
    
    Returns:
        float32 array
    """
    if bits_per_sample not in [16, 24, 32]:
        raise ValueError(f"Unsupported bit depth: {bits_per_sample}")
    
    bytes_per_sample = bits_per_sample // 8
    if len(data) % bytes_per_sample != 0:
        raise ValueError("PCM data length is not aligned with bit per sample")
    
    num_samples = len(data) // bytes_per_sample
    
    if bits_per_sample == 16:
        samples = np.frombuffer(data, dtype=np.int16)
        return samples.astype(np.float32) / 32767.0
    elif bits_per_sample == 24:
        # 24-bit requires special handling
        samples = []
        for i in range(num_samples):
            offset = i * 3
            b0 = int(data[offset])
            b1 = int(data[offset + 1])
            b2 = int(data[offset + 2])
            val_int = (b2 << 16) | (b1 << 8) | b0
            # Handle sign bit
            if val_int & 0x800000:
                val_int |= 0xFF000000  # Extend sign bit to 32 bits
            samples.append(val_int)
        samples = np.array(samples, dtype=np.int32)
        return samples.astype(np.float32) / 8388607.0
    elif bits_per_sample == 32:
        samples = np.frombuffer(data, dtype=np.int32)
        return samples.astype(np.float32) / 2147483647.0


def reformat_wav_bytes(wav_data: bytes, target_rate: int, 
                       target_channels: int, target_bits_per_sample: int) -> bytes:
    """
    WAV byte stream format conversion
    
    Supports: bit depth conversion, sample rate conversion, channel conversion
    
    Args:
        wav_data: Raw WAV file data
        target_rate: Target sample rate
        target_channels: Target number of channels
        target_bits_per_sample: Target bit depth
    
    Returns:
        Converted WAV byte stream
    """
    # Parse original header
    header = parse_wav_header(wav_data)
    
    current_rate = header.sample_rate
    current_channels = header.num_channels
    current_bits_per_sample = header.bits_per_sample
    
    # No conversion needed, return directly
    if (target_rate == current_rate and 
        target_channels == current_channels and 
        target_bits_per_sample == current_bits_per_sample):
        return wav_data
    
    # Extract PCM data and convert to float32
    header_size = 44
    pcm_raw = wav_data[header_size:]
    samples = pcm_bytes_to_float32(pcm_raw, current_bits_per_sample)
    
    # Channel conversion
    if target_channels > 0 and target_channels != current_channels:
        samples = change_channels(samples, current_channels, target_channels)
        current_channels = target_channels
    
    # Resampling
    if target_rate > 0 and target_rate != current_rate:
        samples = resample_safe(samples, current_rate, target_rate, current_channels)
        current_rate = target_rate
    
    # Target bit depth
    if target_bits_per_sample <= 0:
        target_bits_per_sample = current_bits_per_sample
    
    # Encode back to WAV
    return float32_to_wav_bytes(samples, current_rate, current_channels, target_bits_per_sample)


def change_channels(data: np.ndarray, src_channel: int, dst_channel: int) -> np.ndarray:
    """
    Audio data channel conversion
    
    Supports: Stereo(2) -> Mono(1), Mono(1) -> Stereo(2)
    
    Args:
        data: Raw audio data
        src_channel: Source number of channels
        dst_channel: Target number of channels
    
    Returns:
        Converted audio data
    """
    if src_channel == dst_channel:
        return data
    
    # Stereo to Mono
    if src_channel == 2 and dst_channel == 1:
        length = len(data) // 2
        output = np.zeros(length, dtype=np.float32)
        for i in range(length):
            l = data[i * 2]
            r = data[i * 2 + 1]
            output[i] = (l + r) / 2.0
        return output
    
    # Mono to Stereo
    if src_channel == 1 and dst_channel == 2:
        output = np.zeros(len(data) * 2, dtype=np.float32)
        for i in range(len(data)):
            output[i * 2] = data[i]  # L
            output[i * 2 + 1] = data[i]  # R
        return output
    
    raise ValueError(f"Unsupported channel conversion: {src_channel} -> {dst_channel}")


def resample_safe(data: np.ndarray, old_rate: int, new_rate: int, channels: int) -> np.ndarray:
    """
    Safe resampling function, supports mono/stereo
    
    Args:
        data: Raw data
        old_rate: Source sample rate
        new_rate: Target sample rate
        channels: Number of channels
    
    Returns:
        Resampled data
    """
    if old_rate == new_rate:
        return data
    
    try:
        from scipy import signal
    except ImportError:
        # If scipy is not available, use simple linear interpolation
        return resample_linear(data, old_rate, new_rate)
    
    # Mono: direct resampling
    if channels == 1:
        num_samples = int(len(data) * new_rate / old_rate)
        return signal.resample(data, num_samples).astype(np.float32)
    
    # Stereo: split -> resample -> merge
    if channels == 2:
        length = len(data) // 2
        left = data[::2]
        right = data[1::2]
        
        num_samples = int(length * new_rate / old_rate)
        left_resampled = signal.resample(left, num_samples).astype(np.float32)
        right_resampled = signal.resample(right, num_samples).astype(np.float32)
        
        # Merge
        output = np.zeros(num_samples * 2, dtype=np.float32)
        output[::2] = left_resampled
        output[1::2] = right_resampled
        return output
    
    return data


def resample_linear(data: np.ndarray, old_rate: int, new_rate: int) -> np.ndarray:
    """
    Linear interpolation resampling, suitable for single track continuous data
    
    Args:
        data: Raw data
        old_rate: Source sample rate
        new_rate: Target sample rate
    
    Returns:
        Resampled data
    """
    if old_rate == new_rate:
        return data
    
    ratio = old_rate / new_rate
    new_len = int(len(data) / ratio)
    output = np.zeros(new_len, dtype=np.float32)
    
    for i in range(new_len):
        src_idx_float = i * ratio
        src_idx_int = int(src_idx_float)
        
        if src_idx_int >= len(data) - 1:
            output[i] = data[-1]
            continue
        
        fraction = src_idx_float - src_idx_int
        sample1 = data[src_idx_int]
        sample2 = data[src_idx_int + 1]
        
        output[i] = sample1 + fraction * (sample2 - sample1)
    
    return output
