import torch
import torch.onnx
import numpy as np

def convert_pytorch_to_onnx(model, input_shape, output_path, model_name="model"):
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to convert
        input_shape: Tuple representing input shape (batch_size, channels, height, width)
        output_path: Path where the ONNX model will be saved
        model_name: Name for the ONNX model (default: "model")
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input tensor
        dummy_input = torch.randn(input_shape)
        
        # Export the model to ONNX
        torch.onnx.export(
            model,                          # PyTorch model
            dummy_input,                    # Model input (or tuple for multiple inputs)
            output_path,                    # Output file path
            export_params=True,             # Store trained parameter weights
            opset_version=11,               # ONNX version to export to
            do_constant_folding=True,       # Execute constant folding optimization
            input_names=['input'],          # Model input names
            output_names=['output'],        # Model output names
            dynamic_axes={                  # Variable length axes
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model successfully converted to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting model to ONNX: {str(e)}")
        return False

def verify_onnx_model(onnx_path, pytorch_model, input_shape):
    """
    Verify that the ONNX model produces the same output as PyTorch model.
    
    Args:
        onnx_path: Path to the ONNX model
        pytorch_model: Original PyTorch model
        input_shape: Input shape for testing
    
    Returns:
        bool: True if outputs match, False otherwise
    """
    try:
        import onnxruntime as ort
        
        # Create test input
        test_input = torch.randn(input_shape)
        
        # Get PyTorch output
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).numpy()
        
        # Get ONNX output
        ort_session = ort.InferenceSession(onnx_path)
        onnx_output = ort_session.run(None, {'input': test_input.numpy()})[0]
        
        # Compare outputs
        if np.allclose(pytorch_output, onnx_output, rtol=1e-03, atol=1e-05):
            print("✓ ONNX model verification successful - outputs match!")
            return True
        else:
            print("✗ ONNX model verification failed - outputs don't match")
            return False
            
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        return False