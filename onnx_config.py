"""
ONNX Runtime configuration and initialization module
"""
import onnxruntime as ort
from typing import Optional


class OnnxConfig:
    """ONNX Runtime configuration class, responsible for initializing ONNX Runtime environment"""
    
    _initialized = False
    _init_error = None
    
    def __init__(self, onnx_runtime_lib_path: str = "", num_threads: int = 0):
        """
        Initialize ONNX configuration
        
        Args:
            onnx_runtime_lib_path: ONNX Runtime dynamic library path (optional, usually not needed for Python version)
            num_threads: CPU inference thread count, <=0 means use default value
        """
        self.onnx_runtime_lib_path = onnx_runtime_lib_path
        self.num_threads = num_threads
        self.session_options: Optional[ort.SessionOptions] = None
    
    def new(self) -> None:
        """
        Initialize ONNX Runtime (executed only once globally)
        """
        if not OnnxConfig._initialized:
            # Python version of onnxruntime doesn't need manual library path initialization
            # Just create SessionOptions
            OnnxConfig._initialized = True
        
        # Create SessionOptions
        opts = ort.SessionOptions()
        
        # Set CPU inference thread count (if specified)
        if self.num_threads > 0:
            opts.intra_op_num_threads = self.num_threads
            opts.inter_op_num_threads = self.num_threads
        
        self.session_options = opts

