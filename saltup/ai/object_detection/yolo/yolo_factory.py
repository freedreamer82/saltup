from enum import IntEnum
from saltup.ai.object_detection.yolo.yolo import BaseYolo
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.impl.yolo_anchors_based import YoloAnchorsBased
from saltup.ai.object_detection.yolo.impl.yolo_damo import YoloDamo
from saltup.ai.object_detection.yolo.impl.yolo_nas import YoloNas
from saltup.ai.object_detection.yolo.impl.yolo_ultralytics import YoloUltralytics

class YoloFactory:
    @staticmethod
    def create(yolo_type: YoloType, modelpath: str, number_class:int ,**kwargs) -> BaseYolo:
        if yolo_type == YoloType.ANCHORS_BASED:
            return YoloAnchorsBased(yolo_type, modelpath, number_class, BaseYolo.load_anchors(kwargs['anchors']))
        elif yolo_type == YoloType.ULTRALYTICS:
             return YoloUltralytics(yolo_type, modelpath, number_class, **kwargs)
        elif yolo_type == YoloType.SUPERGRAD:
            return YoloNas(yolo_type, modelpath, number_class, **kwargs)
        elif yolo_type == YoloType.DAMO:
            return YoloDamo(yolo_type, modelpath, number_class, **kwargs)
        else:
            raise ValueError(f"Unknown processor type: {yolo_type}")