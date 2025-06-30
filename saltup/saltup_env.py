import os

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python <3.8
    raise ImportError(
        "The importlib.metadata module is not available in this Python version. "
        "Please upgrade to Python 3.8 or later."
    )
    
def _get_version_from_metadata():
    try:
        return version("saltup")
    except PackageNotFoundError:
        return "unknown"

 
class _SaltupEnv:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_SaltupEnv, cls).__new__(cls)
        return cls._instance

    @property
    def VERSION(self):
        return _get_version_from_metadata()
    
    @property
    def SALTUP_KERAS_TRAIN_SHUFFLE(self):
        return os.getenv("SALTUP_KERAS_TRAIN_SHUFFLE", "True").lower() in ("true", "1", "yes")
    
    @property
    def SALTUP_KERAS_TRAIN_VERBOSE(self):
        return int(os.getenv("SALTUP_KERAS_TRAIN_VERBOSE", 1))
    
    @property
    def SALTUP_NN_MNG_USE_GPU(self):
        return os.getenv("SALTUP_NN_MNG_USE_GPU", "1").lower() in ("true", "1", "yes")
    
    @property
    def SALTUP_BBOX_INNER_FORMAT(self):
        return int(os.getenv("SALTUP_BBOX_INNER_FORMAT", 1)) # CORNERS_NORMALIZED = 1

    @property
    def SALTUP_BBOX_FLOAT_PRECISION(self):
        return int(os.getenv("SALTUP_BBOX_FLOAT_PRECISION", "4"))
    
    @property
    def SALTUP_BBOX_NORMALIZATION_TOLERANCE(self):
        return float(os.getenv("SALTUP_BBOX_NORMALIZATION_TOLERANCE", 1e-2))

    @property
    def SALTUP_ONNX_OPSET(self):
            return int(os.getenv("SALTUP_ONNX_OPSET", 16))
    
# Create a singleton instance for easy access
SaltupEnv = _SaltupEnv()