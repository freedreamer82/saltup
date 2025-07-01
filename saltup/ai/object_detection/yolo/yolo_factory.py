from enum import IntEnum

from saltup.ai.object_detection.yolo.yolo import BaseYolo
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.impl.yolo_anchors_based import YoloAnchorsBased
from saltup.ai.object_detection.yolo.impl.yolo_damo import YoloDamo
from saltup.ai.object_detection.yolo.impl.yolo_nas import YoloNas
from saltup.ai.object_detection.yolo.impl.yolo_ultralytics import YoloUltralytics
from saltup.ai.nn_model import NeuralNetworkModel


class YoloFactory:
    """
    Factory class for creating YOLO object detection model instances based on the specified YOLO type.

    Methods
    -------
    create(yolo_type: YoloType, model_or_path, number_class: int, **kwargs) -> BaseYolo
        Static method to instantiate and return a YOLO model of the specified type.

        Parameters
        ----------
        yolo_type : YoloType
            The type of YOLO model to create (e.g., ANCHORS_BASED, ULTRALYTICS, SUPERGRAD, DAMO).
        model_or_path : str or object
            Either a string path to the model file or a pre-loaded model object (e.g., Keras or Torch model).
        number_class : int
            The number of classes for object detection.
        **kwargs
            Additional keyword arguments required for specific YOLO types (e.g., 'anchors' for ANCHORS_BASED).

        Returns
        -------
        BaseYolo
            An instance of a subclass of BaseYolo corresponding to the specified YOLO type.

        Raises
        ------
        ValueError
            If an unknown YOLO type is provided.
    """
    @staticmethod
    def create(yolo_type: YoloType, model_or_path, number_class: int, **kwargs) -> BaseYolo:
        # model_or_path can be either a path (str) or an already loaded keras/torch model
        if isinstance(model_or_path, str):
            model = NeuralNetworkModel(model_or_path)
        else:
            model = model_or_path   

        if yolo_type == YoloType.ANCHORS_BASED:
            anchors = kwargs.get('anchors')
            if isinstance(anchors, str):
                anchors = BaseYolo.load_anchors(anchors)
            elif not isinstance(anchors, (list, tuple)) and not hasattr(anchors, 'shape'):
                raise ValueError("anchors must be a file path, a list/tuple, or a numpy.ndarray")
            return YoloAnchorsBased(model, number_class, anchors)
        elif yolo_type == YoloType.ULTRALYTICS:
            return YoloUltralytics(model, number_class, **kwargs)
        elif yolo_type == YoloType.SUPERGRAD:
            return YoloNas(model, number_class, **kwargs)
        elif yolo_type == YoloType.DAMO:
            return YoloDamo(model, number_class, **kwargs)
        else:
            raise ValueError(f"Unknown processor type: {yolo_type}")