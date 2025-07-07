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
    
    @property
    def SALTUP_TRAINING_KERAS_COMPILE_ARGS(self):
        """
        Additional arguments for Keras model compilation.
        Set via environment variable SALTUP_TRAINING_KERAS_COMPILE_ARGS as:
        1. JSON string: SALTUP_TRAINING_KERAS_COMPILE_ARGS='{"metrics": ["accuracy"], "run_eagerly": true}'
        2. Path to JSON file: SALTUP_TRAINING_KERAS_COMPILE_ARGS='/path/to/compile_args.json'
        """
        import json
        args_value = os.getenv("SALTUP_TRAINING_KERAS_COMPILE_ARGS", "{}")
        
        # Check if it's a file path
        if os.path.isfile(args_value):
            try:
                with open(args_value, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        else:
            # Treat as JSON string
            try:
                return json.loads(args_value)
            except json.JSONDecodeError:
                return {}
    
    @property
    def SALTUP_TRAINING_KERAS_FIT_ARGS(self):
        """
        Additional arguments for Keras model fit method.
        Set via environment variable SALTUP_TRAINING_KERAS_FIT_ARGS as:
        1. JSON string: SALTUP_TRAINING_KERAS_FIT_ARGS='{"workers": 4, "use_multiprocessing": true}'
        2. Path to JSON file: SALTUP_TRAINING_KERAS_FIT_ARGS='/path/to/fit_args.json'
        """
        import json
        args_value = os.getenv("SALTUP_TRAINING_KERAS_FIT_ARGS", "{}")
        
        # Check if it's a file path
        if os.path.isfile(args_value):
            try:
                with open(args_value, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        else:
            # Treat as JSON string
            try:
                return json.loads(args_value)
            except json.JSONDecodeError:
                return {}
    
# Create a singleton instance for easy access
SaltupEnv = _SaltupEnv()