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


def apply_vad(samples: np.ndarray, 
              sample_rate: int = SAMPLE_RATE,
              frame_ms: int = 25,
              hop_ms: int = 10,
              threshold_db: float = -40,
              margin_ms: int = 100) -> np.ndarray:
    """
    Voice Activity Detection - 去除音频首尾静音段
    
    使用能量检测方法，移除音频开头和结尾的静音/低能量部分，
    保留有效语音区域，提高 ASR 识别准确率。
    
    Args:
        samples: 音频样本数据 (float32, 范围 [-1, 1])
        sample_rate: 采样率 (默认 16000)
        frame_ms: 帧长度（毫秒）
        hop_ms: 帧移动步长（毫秒）
        threshold_db: 能量阈值（相对于最大能量，dB），低于此阈值视为静音
        margin_ms: 边界保护边距（毫秒），避免切掉语音起始/结束
    
    Returns:
        去除首尾静音后的音频样本
    """
    if len(samples) == 0:
        return samples
    
    # 计算帧参数
    frame_size = int(sample_rate * frame_ms / 1000)
    hop_size = int(sample_rate * hop_ms / 1000)
    margin_samples = int(sample_rate * margin_ms / 1000)
    
    # 确保音频足够长
    if len(samples) < frame_size:
        return samples
    
    # 计算每帧能量 (dB)
    num_frames = (len(samples) - frame_size) // hop_size + 1
    energies = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = samples[start:end]
        
        # RMS 能量
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        # 转换为 dB
        energies[i] = 20 * np.log10(rms + 1e-10)
    
    # 计算动态阈值（相对于最大能量）
    max_energy = np.max(energies)
    threshold = max(max_energy + threshold_db, -60)  # 绝对下限 -60 dB
    
    # 找到有效语音区间
    active_frames = energies > threshold
    
    if not active_frames.any():
        # 没有检测到语音，返回原始音频
        return samples
    
    # 找到第一个和最后一个活跃帧
    first_active = np.argmax(active_frames)
    last_active = len(active_frames) - np.argmax(active_frames[::-1]) - 1
    
    # 转换为样本索引，并添加边距
    start_sample = max(0, first_active * hop_size - margin_samples)
    end_sample = min(len(samples), (last_active + 1) * hop_size + margin_samples)
    
    # 如果裁剪后太短，返回原始音频
    min_duration_samples = int(sample_rate * 0.1)  # 最少保留 100ms
    if end_sample - start_sample < min_duration_samples:
        return samples
    
    return samples[start_sample:end_sample]


def apply_vad_advanced(samples: np.ndarray,
                       sample_rate: int = SAMPLE_RATE,
                       frame_ms: int = 25,
                       hop_ms: int = 10,
                       energy_threshold_db: float = -40,
                       zcr_threshold: float = 0.3,
                       margin_ms: int = 100) -> np.ndarray:
    """
    高级 VAD - 结合能量和过零率检测
    
    过零率 (Zero-Crossing Rate) 可以帮助区分语音和噪声：
    - 语音通常有较高的过零率
    - 背景噪声的过零率相对稳定
    
    Args:
        samples: 音频样本数据
        sample_rate: 采样率
        frame_ms: 帧长度（毫秒）
        hop_ms: 帧移动步长（毫秒）
        energy_threshold_db: 能量阈值（dB）
        zcr_threshold: 过零率阈值（相对于平均值的倍数）
        margin_ms: 边界保护边距（毫秒）
    
    Returns:
        去除首尾静音后的音频样本
    """
    if len(samples) == 0:
        return samples
    
    frame_size = int(sample_rate * frame_ms / 1000)
    hop_size = int(sample_rate * hop_ms / 1000)
    margin_samples = int(sample_rate * margin_ms / 1000)
    
    if len(samples) < frame_size:
        return samples
    
    num_frames = (len(samples) - frame_size) // hop_size + 1
    energies = np.zeros(num_frames)
    zcrs = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = samples[start:end]
        
        # 能量
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        energies[i] = 20 * np.log10(rms + 1e-10)
        
        # 过零率
        signs = np.sign(frame)
        sign_changes = np.abs(np.diff(signs))
        zcrs[i] = np.sum(sign_changes > 0) / (frame_size - 1)
    
    # 能量检测
    max_energy = np.max(energies)
    energy_threshold = max(max_energy + energy_threshold_db, -60)
    energy_active = energies > energy_threshold
    
    # 过零率检测（高于平均值一定倍数）
    mean_zcr = np.mean(zcrs)
    zcr_active = zcrs > mean_zcr * zcr_threshold
    
    # 结合两种检测：能量必须满足，过零率作为辅助
    active_frames = energy_active & (zcr_active | (energies > max_energy - 20))
    
    if not active_frames.any():
        return samples
    
    first_active = np.argmax(active_frames)
    last_active = len(active_frames) - np.argmax(active_frames[::-1]) - 1
    
    start_sample = max(0, first_active * hop_size - margin_samples)
    end_sample = min(len(samples), (last_active + 1) * hop_size + margin_samples)
    
    min_duration_samples = int(sample_rate * 0.1)
    if end_sample - start_sample < min_duration_samples:
        return samples
    
    return samples[start_sample:end_sample]


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

