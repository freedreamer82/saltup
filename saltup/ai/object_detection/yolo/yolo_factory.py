from enum import IntEnum
from saltup.ai.object_detection.yolo.yolo import BaseYolo
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.impl.yolo_anchors_based import YoloAnchorsBased

class YoloFactory:
    @staticmethod
    def create(yolo_type: YoloType, modelpath: str, number_class:int ,**kwargs) -> BaseYolo:
        if yolo_type == YoloType.ANCHORS_BASED:
            return YoloAnchorsBased(yolo_type, modelpath, number_class, BaseYolo.load_anchors(kwargs['anchors']))
        # elif yolo_type == YoloType.ULTRALYTICS:
        #     from saltup.ai.object_detection.yolo.preprocessing.ultralytics_preprocess import UltralyticsPreprocess
        #     return YoloUltralytics(modelpath, number_class, **kwargs)
        # elif yolo_type == YoloType.SUPERGRAD:
        #     from saltup.ai.object_detection.yolo.preprocessing.supergradients_preprocess import SupergradPreprocess
        #     return YoloNas(modelpath, number_class, **kwargs)
        # elif yolo_type == YoloType.DAMO:
        #     from saltup.ai.object_detection.yolo.preprocessing.damo_preprocess import DamoPreprocessing
        #     return YoloDamo(modelpath, number_class, **kwargs)
        else:
            raise ValueError(f"Unknown processor type: {yolo_type}")