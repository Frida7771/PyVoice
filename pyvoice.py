"""
PyVoice SDK - Simple API for Speech Recognition and Text-to-Speech

Usage:
    from pyvoice import ASR, TTS
    
    # Speech Recognition
    asr = ASR()
    text = asr.recognize("audio.wav")
    
    # Text-to-Speech
    tts = TTS()
    audio_bytes = tts.synthesize("你好世界")
    tts.synthesize_to_file("你好世界", "output.wav")
"""
import os
from pathlib import Path

# Get project root directory
_PROJECT_ROOT = Path(__file__).parent

# Default model paths
_DEFAULT_ASR_CONFIG = {
    "model_path": str(_PROJECT_ROOT / "paraformer_weights" / "model.int8.onnx"),
    "tokens_path": str(_PROJECT_ROOT / "paraformer_weights" / "tokens.txt"),
    "cmvn_path": str(_PROJECT_ROOT / "paraformer_weights" / "am.mvn"),
}

_DEFAULT_TTS_CONFIG = {
    "model_path": str(_PROJECT_ROOT / "melo_weights" / "model.onnx"),
    "token_path": str(_PROJECT_ROOT / "melo_weights" / "tokens.txt"),
    "lexicon_path": str(_PROJECT_ROOT / "melo_weights" / "lexicon.txt"),
}


class ASR:
    """
    Automatic Speech Recognition (语音识别)
    
    Usage:
        asr = ASR()
        text = asr.recognize("audio.mp3")  # Supports WAV, MP3, FLAC, etc.
    """
    
    def __init__(self, model_path: str = None, tokens_path: str = None, cmvn_path: str = None):
        """
        Initialize ASR engine.
        
        Args:
            model_path: Path to ONNX model (optional, uses default if not provided)
            tokens_path: Path to tokens.txt (optional)
            cmvn_path: Path to am.mvn (optional)
        """
        from asr.paraformer.config import Config
        from asr.paraformer.engine import Engine
        
        config = Config(
            model_path=model_path or _DEFAULT_ASR_CONFIG["model_path"],
            tokens_path=tokens_path or _DEFAULT_ASR_CONFIG["tokens_path"],
            cmvn_path=cmvn_path or _DEFAULT_ASR_CONFIG["cmvn_path"],
        )
        self._engine = Engine(config)
    
    def recognize(self, audio_path: str) -> str:
        """
        Recognize speech from audio file.
        Supports: WAV, MP3, FLAC, OGG, M4A, AAC, WMA, WEBM
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Recognized text
        """
        return self._engine.recognize_file(audio_path)
    
    def recognize_bytes(self, audio_bytes: bytes) -> str:
        """
        Recognize speech from WAV bytes.
        
        Args:
            audio_bytes: WAV format audio bytes
        
        Returns:
            Recognized text
        """
        return self._engine.recognize_bytes(audio_bytes)


class TTS:
    """
    Text-to-Speech (语音合成)
    
    Usage:
        tts = TTS()
        audio_bytes = tts.synthesize("你好世界")
        tts.synthesize_to_file("你好世界", "output.wav")
    """
    
    def __init__(self, model_path: str = None, token_path: str = None, lexicon_path: str = None):
        """
        Initialize TTS engine.
        
        Args:
            model_path: Path to ONNX model (optional, uses default if not provided)
            token_path: Path to tokens.txt (optional)
            lexicon_path: Path to lexicon.txt (optional)
        """
        from tts.melotts.config import Config
        from tts.melotts.engine import Engine
        
        config = Config(
            model_path=model_path or _DEFAULT_TTS_CONFIG["model_path"],
            token_path=token_path or _DEFAULT_TTS_CONFIG["token_path"],
            lexicon_path=lexicon_path or _DEFAULT_TTS_CONFIG["lexicon_path"],
        )
        self._engine = Engine(config)
    
    def synthesize(self, text: str, speed: float = 1.0) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speed: Speech speed (1.0 = normal, >1 = faster, <1 = slower)
        
        Returns:
            WAV format audio bytes
        """
        return self._engine.synthesize_to_wav(text, speed)
    
    def synthesize_to_file(self, text: str, output_path: str, speed: float = 1.0) -> str:
        """
        Synthesize speech and save to file.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            speed: Speech speed (1.0 = normal)
        
        Returns:
            Output file path
        """
        audio_bytes = self.synthesize(text, speed)
        
        # Ensure directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(audio_bytes)
        
        return output_path

