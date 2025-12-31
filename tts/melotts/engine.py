"""
MeloTTS engine module
"""
import numpy as np
import onnxruntime as ort
from typing import List

from .config import Config, SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE, SPEAKER_ID
from .utils import load_tokens, load_lexicon
from .frontend import text_to_ids
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from internal.convertutil import text_to_chinese
from internal.mediautil import float32_to_wav_bytes
from onnx_config import OnnxConfig


class Engine:
    """Encapsulates MeloTTS ONNX runtime and related resources"""
    
    def __init__(self, cfg: Config):
        """
        Initialize MeloTTS engine
        
        Args:
            cfg: Configuration object
        """
        # Initialize ONNX
        onnx_config = OnnxConfig(
            onnx_runtime_lib_path=cfg.onnx_runtime_lib_path,
            num_threads=cfg.num_threads
        )
        onnx_config.new()
        
        if not cfg.model_path or not cfg.token_path or not cfg.lexicon_path:
            raise ValueError("Model, Tokens, and Lexicon file paths cannot be empty")
        
        # Load resources (Tokens and Lexicon)
        self.token_map = load_tokens(cfg.token_path)
        self.lexicon = load_lexicon(cfg.lexicon_path)
        self.config = cfg
        
        # Create ONNX session
        sess_options = onnx_config.session_options
        self.session = ort.InferenceSession(
            cfg.model_path,
            sess_options=sess_options
        )
    
    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        """
        Convert text to speech data (float32 PCM)
        
        Args:
            text: Text to convert
            speed: Speed adjustment, larger value means faster, 1.0 is normal speed
        
        Returns:
            float32 PCM audio data
        """
        # Text normalization
        normalized_text = text_to_chinese(text)
        
        # Text to ID (G2P)
        input_ids, tone_ids = text_to_ids(normalized_text, self.lexicon, self.token_map)
        
        # Execute ONNX inference
        return self._run_inference(input_ids, tone_ids, speed)
    
    def synthesize_to_wav(self, text: str, speed: float = 1.0) -> bytes:
        """
        Convert text to WAV format byte stream
        
        Args:
            text: Text to convert
            speed: Speed adjustment, larger value means faster, 1.0 is normal speed
        
        Returns:
            WAV format byte stream
        """
        pcm_data = self.synthesize(text, speed)
        return float32_to_wav_bytes(pcm_data, SAMPLE_RATE, CHANNELS, BITS_PER_SAMPLE)
    
    def _run_inference(self, input_ids: List[int], tone_ids: List[int], 
                      speed: float) -> np.ndarray:
        """
        Run inference
        
        Args:
            input_ids: Input ID list
            tone_ids: Tone ID list
            speed: Speech speed
        
        Returns:
            Audio data (float32 array)
        """
        seq_length = len(input_ids)
        
        # Build input tensors
        x = np.array([input_ids], dtype=np.int64)
        x_lengths = np.array([seq_length], dtype=np.int64)
        tones = np.array([tone_ids], dtype=np.int64)
        sid = np.array([SPEAKER_ID], dtype=np.int64)
        
        # Parameter control
        # noise_scale (0.667), length_scale (1.0 / speed), noise_scale_w (0.8)
        # Note: length_scale controls speed, larger value means slower, so use 1.0/speed
        noise_scale = 0.667
        length_scale = 1.0 / speed if speed > 0 else 1.0
        noise_scale_w = 0.8
        
        noise = np.array([noise_scale], dtype=np.float32)
        length = np.array([length_scale], dtype=np.float32)
        noise_w = np.array([noise_scale_w], dtype=np.float32)
        
        inputs = {
            "x": x,
            "x_lengths": x_lengths,
            "tones": tones,
            "sid": sid,
            "noise_scale": noise,
            "length_scale": length,
            "noise_scale_w": noise_w
        }
        
        # Execute
        outputs = self.session.run(["y"], inputs)
        
        # Get result
        result = outputs[0][0]  # Remove batch dimension
        return result.astype(np.float32)

