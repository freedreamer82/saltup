from enum import IntEnum
from saltup.ai.quantization import Quantize
from saltup.ai.quantization.impl import (
    KerasQuantize,
    OnnxStaticQuantize,
    OnnxDynamicQuantize,
)

class QuantizationType(IntEnum):
    KERAS = 0
    STATIC_ONNX = 1
    DYNAMIC_ONNX = 2

class QuantizationFactory:
    @staticmethod
    def create(quantizer_type: QuantizationType) -> Quantize:
        quantizers = {
            QuantizationType.KERAS: KerasQuantize,
            QuantizationType.STATIC_ONNX: OnnxStaticQuantize,
            QuantizationType.DYNAMIC_ONNX: OnnxDynamicQuantize,
        }
        quantizer_class = quantizers.get(quantizer_type)
        if quantizer_class is None:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")
        return quantizer_class()