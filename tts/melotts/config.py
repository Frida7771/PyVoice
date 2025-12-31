"""
MeloTTS configuration module
"""

# Sample rate, default is 44100
SAMPLE_RATE = 44100
# Speaker ID
SPEAKER_ID = 1
# Number of channels
CHANNELS = 1
# Bits per sample
BITS_PER_SAMPLE = 16


class Config:
    """Configuration parameters for MeloTTS engine"""
    
    def __init__(self, onnx_runtime_lib_path: str = "", model_path: str = "",
                 token_path: str = "", lexicon_path: str = "",
                 use_cuda: bool = False, num_threads: int = 0):
        """
        Initialize configuration
        
        Args:
            onnx_runtime_lib_path: onnxruntime library path (usually not needed for Python version)
            model_path: ONNX model path
            token_path: tokens.txt path
            lexicon_path: lexicon.txt path
            use_cuda: Whether to enable CUDA (optional)
            num_threads: ONNX thread count, default determined by CPU cores
        """
        self.onnx_runtime_lib_path = onnx_runtime_lib_path
        self.model_path = model_path
        self.token_path = token_path
        self.lexicon_path = lexicon_path
        self.use_cuda = use_cuda
        self.num_threads = num_threads
    
    @staticmethod
    def default_config() -> 'Config':
        """
        Return a default configuration (based on common directory structure)
        
        Returns:
            Config object
        """
        return Config(
            model_path="./melo_weights/model.onnx",
            token_path="./melo_weights/tokens.txt",
            lexicon_path="./melo_weights/lexicon.txt",
        )


class LexiconItem:
    """Stores phone and corresponding tone information"""
    
    def __init__(self, phones: list, tones: list):
        """
        Initialize
        
        Args:
            phones: Phone list
            tones: Tone list
        """
        self.phones = phones
        self.tones = tones

