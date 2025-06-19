import torch
import onnx
from torch import nn

def convert_pytorch_to_onnx(
    pytorch_model_path: str,
    onnx_path: str,
    input_shape: tuple,
    opset: int = 16,
    device: str = "cpu"
) -> tuple[onnx.ModelProto, nn.Module]:
    """
    Converts a PyTorch model (.pt or .pth) to ONNX format

    Args:
        pytorch_model_path: Path to the .pt or .pth model to convert
        onnx_path: Path where to save the ONNX model
        input_shape: Shape of the model input (excluding batch size), e.g. (3, 224, 224)
        opset: ONNX opset version to use (default: 16)
        device: Device to load the model on ("cpu" or "cuda")
    Returns:
        Tuple of (ONNX model proto, loaded PyTorch model)
    """
    # 1. Load PyTorch model
    model: nn.Module = torch.load(pytorch_model_path, map_location=device)
    model.eval()

    # 2. Create dummy input
    dummy_input = torch.randn(1, *input_shape, device=device)

    # 3. Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    # 4. Load ONNX model
    model_proto = onnx.load(onnx_path)
    return model_proto, model
