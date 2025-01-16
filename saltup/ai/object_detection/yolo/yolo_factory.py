from enum import IntEnum
from saltup.ai.object_detection.yolo import BaseYolo, YoloType
from saltup.ai.object_detection.yolo.impl import YoloAnchorsBased ,YoloUltralitics, YoloNas, YoloDamo

class YoloFactory:
    @staticmethod
    def create(yolo_type: YoloType, modelpath: str, **kwargs) -> BaseYolo:
        if yolo_type == YoloType.ANCHORS_BASED:
            from saltup.ai.object_detection.preprocessing.impl import AnchorsBasedPreprocess
            return YoloAnchorsBased(yolo_type, modelpath, kwargs['anchors'])
        elif yolo_type == YoloType.ULTRALYTICS:
            from saltup.ai.object_detection.preprocessing.impl import UltralyticsPreprocess
            return YoloUltralitics(modelpath, **kwargs)
        elif yolo_type == YoloType.SUPERGRAD:
            from saltup.ai.object_detection.preprocessing.impl import SupergradPreprocess
            return YoloNas(modelpath, **kwargs)
        elif yolo_type == YoloType.DAMO:
            from saltup.ai.object_detection.preprocessing.impl import DamoPreprocessing
            return YoloDamo(modelpath, **kwargs)
        else:
            raise ValueError(f"Unknown processor type: {yolo_type}")