<div align="center">
<img src="docs/saltup-banner.png">
</div>

# Saltup 🧂

The universal AI toolkit that works with your existing models and data, regardless of format.

Stop wrestling with format conversions. Stop rewriting data loaders. Start building.

## Installation

```bash
pip install git+https://github.com/freedreamer82/saltup.git
```

For development:
```bash
git clone https://github.com/freedreamer82/saltup.git
cd saltup && ./makePackage.sh -d
```

## The Magic

Load any model format through one interface:
```python
from saltup.ai.nn_model import NeuralNetworkModel

model = NeuralNetworkModel("model.pt")       # PyTorch
model = NeuralNetworkModel("model.keras")    # TensorFlow/Keras  
model = NeuralNetworkModel("model.onnx")     # ONNX
model = NeuralNetworkModel("model.tflite")   # TensorFlow Lite

# Same interface, any format
predictions = model.predict(your_data)
```

Work with any dataset format:
```python
from saltup.ai.object_detection.dataset.loader_factory import DataLoaderFactory

# Auto-detects COCO, Pascal VOC, YOLO formats
train_dl, val_dl, test_dl = DataLoaderFactory.get_dataloaders("./your_dataset")
```

Train across frameworks seamlessly:
```python
from saltup.ai.training.train import training

# Mix TensorFlow data with PyTorch models, or vice versa
training(model, train_dataloader, validation_dataloader)
```

## Core Capabilities

**🔄 Format Freedom**  
One interface for .pt, .keras, .onnx, .tflite models. Load once, use everywhere.

**📊 Dataset Flexibility**  
Auto-detect and work with COCO, Pascal VOC, YOLO formats without manual conversion.

**🔀 Cross-Framework**  
Mix TensorFlow and PyTorch components in the same pipeline. Best of both worlds.

**🚀 Production Ready**  
Built-in quantization, model conversion, and deployment tools for real-world use.

## Ready-to-Use Tools

Get productive immediately with battle-tested command-line utilities:

```bash
# Model operations
saltup_keras2onnx model.keras              # Convert between formats
saltup_onnx_quantization model.onnx        # Optimize for deployment

# Quick inference  
saltup_yolo_image_inference model.pt image.jpg
saltup_yolo_video_inference model.onnx video.mp4
saltup_yolo_s3_inference model.tflite s3://bucket/images/

# Dataset utilities
saltup_yolo_count_classes ./dataset        # Analyze your data
saltup_info                                # Package information
```

## Real-World Workflows

**Model Format Pipeline:**
Train in PyTorch → Convert to ONNX → Quantize → Deploy to embedded device

**Cross-Framework Training:**
Load COCO dataset → Train with TensorFlow → Export as PyTorch → Optimize with ONNX

**Dataset Format Freedom:**
Pascal VOC annotations → Auto-load as YOLO → Train any model → Deploy anywhere

## Why Saltup?

Because your time is valuable. Because formats shouldn't dictate your architecture. Because production deployment should be simple, not a research project.

Work with what you have. Build what you need. Deploy where you want.
