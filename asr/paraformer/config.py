"""
Paraformer ASR configuration module
"""

# Sample rate
SAMPLE_RATE = 16000
# Number of channels
CHANNELS = 1
# Bits per sample
BITS_PER_SAMPLE = 16


class Config:
    """Configuration parameters for Paraformer model"""
    
    def __init__(self, onnx_runtime_lib_path: str = "", model_path: str = "",
                 tokens_path: str = "", cmvn_path: str = "", 
                 use_cuda: bool = False, num_threads: int = 0):
        """
        Initialize configuration
        
        Args:
            onnx_runtime_lib_path: onnxruntime library path (usually not needed for Python version)
            model_path: ONNX model path
            tokens_path: tokens.txt path
            cmvn_path: am.mvn file path
            use_cuda: Whether to enable CUDA (optional)
            num_threads: ONNX thread count, default determined by CPU cores
        """
        self.onnx_runtime_lib_path = onnx_runtime_lib_path
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.cmvn_path = cmvn_path
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
            model_path="./paraformer_weights/model.int8.onnx",
            tokens_path="./paraformer_weights/tokens.txt",
            cmvn_path="./paraformer_weights/am.mvn",
        )

