from abc import ABC, abstractmethod
import logging

class BaseYoloHandler(ABC):
    """
    Abstract base class for YOLO handlers. Defines the required methods 
    and common structure for YOLO-based model handling.
    """
    
    def __init__(self, args):
        """
        Initialize the base handler with configuration arguments.
        """
        pass
    
    @abstractmethod
    def train(self):
        """
        Abstract method for training the model. Must be implemented in subclasses.
        """
        logging.error("Attempted to invoke the abstract 'train' method. This must be implemented in a subclass.")
        raise NotImplementedError("The 'train' method must be implemented in a subclass.")
    
    @abstractmethod
    def model_info(self):
        """
        Abstract method to generate and export model metadata.
        """
        logging.error("Attempted to invoke the abstract 'model_info' method. This must be implemented in a subclass.")
        raise NotImplementedError("The 'model_info' method must be implemented in a subclass.")

    @abstractmethod
    def preprocess(self, input):
        """
        Abstract method for model preprocess. Must be implemented in subclasses.
        """
        logging.error("Attempted to invoke the abstract 'preprocess' method. This must be implemented in a subclass.")
        raise NotImplementedError("The 'preprocess' method must be implemented in a subclass.")

    @abstractmethod
    def postprocess(self, input):
        """
        Abstract method for model postprocess. Must be implemented in subclasses.
        """
        logging.error("Attempted to invoke the abstract 'postprocess' method. This must be implemented in a subclass.")
        raise NotImplementedError("The 'postprocess' method must be implemented in a subclass.")
    
    @abstractmethod
    def inference(self, input):
        """
        Abstract method for model inference. Must be implemented in subclasses.
        """
        logging.error("Attempted to invoke the abstract 'inference' method. This must be implemented in a subclass.")
        raise NotImplementedError("The 'inference' method must be implemented in a subclass.")

    @abstractmethod
    def qat(self):
        """
        Abstract method for model quantization. Must be implemented in subclasses.
        """
        logging.error("Attempted to invoke the abstract 'qat' method. This must be implemented in a subclass.")
        raise NotImplementedError("The 'qat' method must be implemented in a subclass.")
    
    @abstractmethod
    def export_onnx(self):
        """
        Abstract method for export the model in ONNX. Must be implemented in subclasses.
        """
        logging.error("Attempted to invoke the abstract 'export_onnx' method. This must be implemented in a subclass.")
        raise NotImplementedError("The 'export_onnx' method must be implemented in a subclass.")
