from enum import IntEnum
from saltup.ai.object_detection.preprocessing import Preprocessing

class PreprocessingType(IntEnum):
    ANCHORS_BASED = 0
    ULTRALITICS = 1
    SUPERGRAD = 2
    DAMO = 3

class PreprocessingFactory:
    @staticmethod
    def create(processor_type: PreprocessingType) -> Preprocessing:
        if processor_type == PreprocessingType.ANCHORS_BASED:
            from saltup.ai.object_detection.preprocessing.impl import AnchorsBasedPreprocess
            return AnchorsBasedPreprocess()
        elif processor_type == PreprocessingType.ULTRALITICS:
            from saltup.ai.object_detection.preprocessing.impl import UltraliticsPreprocess
            return UltraliticsPreprocess()
        elif processor_type == PreprocessingType.SUPERGRAD:
            from saltup.ai.object_detection.preprocessing.impl import SupergradPreprocess
            return SupergradPreprocess()
        elif processor_type == PreprocessingType.DAMO:
            from saltup.ai.object_detection.preprocessing.impl import DamoPreprocessing
            return DamoPreprocessing()
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")