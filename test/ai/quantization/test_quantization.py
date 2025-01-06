import pytest
import numpy as np
import os

from saltup.ai.quantization import (
    QuantizationFactory,
    QuantizationType,
    Quantize
)
from saltup.ai.quantization.impl import (
    OnnxDynamicQuantize, 
    OnnxStaticQuantize,
    KerasQuantize
)


class TestBaseQuantization:
    """Test the abstract base Quantization class."""

    class ConcreteQuantization(Quantize):
        """Concrete class for testing abstract base class."""

        def __call__(self):
            return super().__call__()

    def test_validate_input_none(self):
        """Test validation of None input."""
        quantizer = self.ConcreteQuantization()
        with pytest.raises(ValueError, match="model path cannot be None"):
            quantizer._validate_input_model(None)

    def test_abstract_process_method(self):
        """Test that abstract quantize method raises NotImplementedError."""
        quantizer = self.ConcreteQuantization()
        with pytest.raises(NotImplementedError):
            quantizer.__call__()


class TestQuantizationFactory:
    """Test the Quantization factory class."""

    def test_valid_quantizer_creation(self):
        """Test creation of all valid quantizer types."""
        valid_types = [
            (QuantizationType.STATIC_ONNX, OnnxStaticQuantize),
            (QuantizationType.DYNAMIC_ONNX, OnnxDynamicQuantize),
            (QuantizationType.KERAS, KerasQuantize)
        ]

        for proc_type, expected_class in valid_types:
            quantizer = QuantizationFactory.create(proc_type)
            assert isinstance(quantizer, expected_class)

    def test_invalid_quantizer_type(self):
        """Test factory behavior with invalid quantizer type."""
        with pytest.raises(ValueError, match="Unknown quantizer type"):
            QuantizationFactory.create(999)

    