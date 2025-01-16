import pytest
import numpy as np
import os

from saltup.ai.keras_utils.keras_to_onnx import (
    convert_keras_to_onnx, verify_onnx_model
)

class TestOnnxKeras:
    @pytest.fixture
    def keras_model(self, root_dir):
        keras_model_path = os.path.join(str(root_dir),  'results', 'models', 'model_1.keras')
        return keras_model_path
    
    def test_keras_to_onnx(self, keras_model, root_dir):
        onnx_path = os.path.join(str(root_dir),  'results', 'models', 'model_1.onnx')
        onnx, keras = convert_keras_to_onnx(keras_model, onnx_path)
        
        keras_pred, onnx_pred = verify_onnx_model(onnx_path, keras)
        
if __name__ == '__main__':
    pytest.main(['-v', __file__])