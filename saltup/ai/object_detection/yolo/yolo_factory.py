from enum import IntEnum
from saltup.ai.object_detection.yolo import BaseYolo , YoloType 

class YoloFactory:
    @staticmethod
    def create(yolo_type: YoloType) -> BaseYolo:
        if yolo_type == YoloType.ANCHORS_BASED:
           # from saltup.ai.object_detection.preprocessing.impl import AnchorsBasedPreprocess
           # return AnchorsBasedPreprocess()
        elif yolo_type == YoloType.ULTRALITICS:
           # from saltup.ai.object_detection.preprocessing.impl import UltralyticsPreprocess
           # return UltralyticsPreprocess()
        elif yolo_type == YoloType.SUPERGRAD:
          #  from saltup.ai.object_detection.preprocessing.impl import SupergradPreprocess
          #  return SupergradPreprocess()
        elif yolo_type == YoloType.DAMO:
          #  from saltup.ai.object_detection.preprocessing.impl import DamoPreprocessing
          #  return DamoPreprocessing()
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")