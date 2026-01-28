"""
Paraformer ASR engine module
"""
import numpy as np
import onnxruntime as ort
from typing import Dict, Optional
import os

from .config import Config
from .utils import load_tokens, load_cmvn, parse_wav_bytes, load_audio_file, apply_vad
from .feature import extract_features
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from onnx_config import OnnxConfig


class Engine:
    """Encapsulates Paraformer ASR ONNX runtime and related resources"""
    
    def __init__(self, cfg: Config, enable_vad: bool = True):
        """
        Initialize Paraformer ASR engine
        
        Args:
            cfg: Configuration object
            enable_vad: Whether to enable VAD preprocessing (default True)
        """
        # Initialize ONNX
        onnx_config = OnnxConfig(
            onnx_runtime_lib_path=cfg.onnx_runtime_lib_path,
            num_threads=cfg.num_threads
        )
        onnx_config.new()
        
        # Load resources (Tokens and CMVN)
        self.token_map = load_tokens(cfg.tokens_path)
        self.neg_mean, self.inv_std = load_cmvn(cfg.cmvn_path)
        
        # VAD settings
        self.enable_vad = enable_vad
        
        # Create ONNX session
        input_names = ["speech", "speech_lengths"]
        output_names = ["logits"]
        
        sess_options = onnx_config.session_options
        self.session = ort.InferenceSession(
            cfg.model_path,
            sess_options=sess_options
        )
    
    def recognize_file(self, audio_path: str) -> str:
        """
        Read audio file and perform speech recognition.
        Supports: WAV, MP3, FLAC, OGG, M4A, AAC, WMA, WEBM
        
        Note: Non-WAV formats require ffmpeg installed on the system.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Recognized text
        """
        wav_bytes = load_audio_file(audio_path)
        return self.recognize_bytes(wav_bytes)
    
    def recognize_bytes(self, wav_bytes: bytes) -> str:
        """
        Read WAV byte stream and perform speech recognition
        
        Args:
            wav_bytes: Audio file byte stream
        
        Returns:
            Recognized text
        """
        samples = parse_wav_bytes(wav_bytes)
        return self.recognize(samples)
    
    def recognize(self, samples: np.ndarray, use_vad: bool = None) -> str:
        """
        Recognize float32 audio sample data
        
        Args:
            samples: 16KHz mono audio data, range [-1, 1]
            use_vad: Override VAD setting for this call (None = use default)
        
        Returns:
            Recognized text
        """
        if len(samples) == 0:
            raise ValueError("Input audio data is empty")
        
        # Apply VAD preprocessing to remove silence
        should_use_vad = use_vad if use_vad is not None else self.enable_vad
        if should_use_vad:
            samples = apply_vad(samples)
            if len(samples) == 0:
                return ""
        
        # Feature extraction
        features, feat_len = extract_features(samples, self.neg_mean, self.inv_std)
        
        # Inference
        token_ids = self._run_inference(features, feat_len)
        
        # Decode
        text = self._decode(token_ids)
        return text
    
    def _run_inference(self, features: np.ndarray, feat_len: int) -> list:
        """
        Run inference
        
        Args:
            features: Feature array
            feat_len: Feature length
        
        Returns:
            Token ID list
        """
        # Build input tensors
        # speech: [1, feat_len, 560]
        speech_shape = (1, feat_len, 560)
        speech_data = features.reshape(speech_shape).astype(np.float32)
        
        # speech_lengths: [1]
        speech_lengths = np.array([feat_len], dtype=np.int32)
        
        # Execute inference
        inputs = {
            "speech": speech_data,
            "speech_lengths": speech_lengths
        }
        outputs = self.session.run(["logits"], inputs)
        
        # Get results
        logits = outputs[0]  # [1, T_out, TokenSize]
        if len(logits.shape) < 3:
            raise ValueError(f"Output result dimension error: {logits.shape}")
        
        return self._get_token_ids(logits[0], logits.shape[1], logits.shape[2])
    
    def _get_token_ids(self, token_scores: np.ndarray, steps: int, token_size: int) -> list:
        """
        Get token ids
        
        Args:
            token_scores: Token score array, shape (steps, token_size)
            steps: Number of time steps
            token_size: Token size
        
        Returns:
            Token ID list
        """
        ids = []
        # token_scores is a 2D array (steps, token_size)
        for t in range(steps):
            # Directly get all scores for row t
            cur_step_scores = token_scores[t]
            max_idx = int(np.argmax(cur_step_scores))
            ids.append(max_idx)
        
        return ids
    
    def _decode(self, ids: list) -> str:
        """
        Decode token ids to text
        
        Args:
            ids: Token ID list
        
        Returns:
            Decoded text
        """
        result = []
        for idx in ids:
            if idx in self.token_map:
                word = self.token_map[idx]
                if word in ["<blank>", "<s>", "</s>", "<unk>"]:
                    result.append(' ')
                elif word.endswith("@@"):
                    word = word.replace("@@", "")
                    result.append(word)
                else:
                    result.append(word)
                    result.append(' ')
        
        return ''.join(result).strip()

