from enum import IntEnum
from saltup.ai.object_detection.yolo import BaseYolo, YoloType


class YoloFactory:
	@staticmethod
	def create(yolo_type: YoloType) -> BaseYolo:
		if yolo_type == YoloType.ANCHORS_BASED:
			# from saltup.ai.object_detection.preprocessing.impl import AnchorsBasedPreprocess
			# return AnchorsBasedPreprocess()
			pass
		elif yolo_type == YoloType.ULTRALYTICS:
			# from saltup.ai.object_detection.preprocessing.impl import UltralyticsPreprocess
			# return UltralyticsPreprocess()
			pass
		elif yolo_type == YoloType.SUPERGRAD:
			#  from saltup.ai.object_detection.preprocessing.impl import SupergradPreprocess
			#  return SupergradPreprocess()
			pass
		elif yolo_type == YoloType.DAMO:
			#  from saltup.ai.object_detection.preprocessing.impl import DamoPreprocessing
			#  return DamoPreprocessing()
			pass
		else:
			raise ValueError(f"Unknown processor type: {yolo_type}")
