from enum import IntEnum
from saltup.ai.object_detection.preprocessing import Preprocessing
from saltup.ai.object_detection.preprocessing.impl import (
    AnchorsBasedPreprocess,
    DamoPreprocessing,
    SupergradPreprocess,
    UltraliticsPreprocess
)

class PreprocessingType(IntEnum):
    ANCHORS_BASED = 0
    ULTRALITICS = 1
    SUPERGRAD = 2
    DAMO = 3

class PreprocessingFactory:
    @staticmethod
    def create(processor_type: PreprocessingType) -> Preprocessing:
        processors = {
            PreprocessingType.ANCHORS_BASED: AnchorsBasedPreprocess,
            PreprocessingType.ULTRALITICS: UltraliticsPreprocess,
            PreprocessingType.SUPERGRAD: SupergradPreprocess,
            PreprocessingType.DAMO: DamoPreprocessing
        }
        processor_class = processors.get(processor_type)
        if processor_class is None:
            raise ValueError(f"Unknown processor type: {processor_type}")
        return processor_class()