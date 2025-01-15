from enum import IntEnum
from saltup.ai.object_detection.postprocessing import Postprocessing
from saltup.ai.object_detection.postprocessing.impl import (
    AnchorsBasedPostprocess,
    DamoPostprocess,
    SupergradPostprocess,
    UltralyticsPostprocess
)

class PostprocessingType(IntEnum):
    ANCHORS_BASED = 0
    ULTRALYTICS = 1
    SUPERGRAD = 2
    DAMO = 3

class PostprocessingFactory:
    @staticmethod
    def create(processor_type: PostprocessingType) -> Postprocessing:
        processors = {
            PostprocessingType.ANCHORS_BASED: AnchorsBasedPostprocess,
            PostprocessingType.ULTRALYTICS: UltralyticsPostprocess,
            PostprocessingType.SUPERGRAD: SupergradPostprocess,
            PostprocessingType.DAMO: DamoPostprocess
        }
        processor_class = processors.get(processor_type)
        if processor_class is None:
            raise ValueError(f"Unknown processor type: {processor_type}")
        return processor_class()