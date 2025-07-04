import os
import tempfile
import torch
from saltup.ai.utils.torch.to_onnx import convert_torch_to_onnx

import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

def test_convert_pytorch_to_onnx_success():
    model = DummyModel()
    input_shape = (1, 4)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "dummy.onnx")
        result = convert_torch_to_onnx(model, input_shape, output_path)
        assert result is True
        assert os.path.exists(output_path)

def test_convert_pytorch_to_onnx_invalid_model():
    # Passing a non-model object should fail
    input_shape = (1, 4)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "dummy.onnx")
        result = convert_torch_to_onnx(None, input_shape, output_path)
        assert result is False

def test_convert_pytorch_to_onnx_invalid_path():
    model = DummyModel()
    input_shape = (1, 4)
    # Use an invalid path to trigger exception
    output_path = "/invalid_path/dummy.onnx"
    result = convert_torch_to_onnx(model, input_shape, output_path)
    assert result is False