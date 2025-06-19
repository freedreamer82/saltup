import os
import gc
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader as pytorch_DataGenerator

from saltup.ai.classification.datagenerator import (
    keras_ClassificationDataGenerator,
    pytorch_ClassificationDataGenerator,
)
from saltup.ai.object_detection.utils.metrics import Metric


def evaluate_model(
    model: Union[str, tf.keras.Model, torch.nn.Module, "tf.lite.Interpreter"],
    test_gen: Union[keras_ClassificationDataGenerator, pytorch_DataGenerator],
    output_dir: str = None,
    loss_function: callable = None,
    confusion_matrix: bool = False
) -> float:
    """
    Evaluate a classification model on the test set.

    Args:
        model (Union[str, tf.keras.Model, torch.nn.Module, tf.lite.Interpreter]):
            Model instance or path to the model file.
        test_gen (Union[keras_ClassificationDataGenerator, pytorch_DataGenerator]):
            Test data generator.
        output_dir (str, optional):
            Directory to save output files such as confusion matrix images.
        loss_function (callable, optional):
            Loss function for evaluation (used only for PyTorch models).
        confusion_matrix (bool, optional):
            Whether to compute and save the confusion matrix plot.

    Raises:
        ValueError: If the model type or test generator is unsupported.

    Returns:
        Tuple[Metric, Dict[int, Metric]]:
            Global metric and per-class metrics.
    """
    global_metric = Metric()
    if isinstance(test_gen, keras_ClassificationDataGenerator):
        class_names = test_gen.dataloader.get_classes().keys()
    elif isinstance(test_gen, pytorch_DataGenerator):
        class_names = test_gen.dataset.dataloader.get_classes().keys()
    else:
        raise ValueError("Unsupported test generator type.")

    metric_per_class = {i: Metric() for i in range(len(class_names))}

    # Determine model type and load if necessary
    loaded_model = None
    tflite_interpreter = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_extension = None

    # If a path is provided, load the model
    if isinstance(model, str):
        model_extension = os.path.splitext(model)[1].replace('.', '')
        if model.endswith('.keras'):
            print("\n--- Evaluate Keras model ---")
            loaded_model = tf.keras.models.load_model(model)
            model_extension = "keras"
        elif model.endswith('.pt'):
            print("\n--- Evaluate PyTorch model ---")
            loaded_model = torch.jit.load(model) if model.endswith(".pt") else torch.load(model)
            loaded_model.eval()
            loaded_model.to(device)
            model_extension = "pt"
        elif model.endswith('.tflite'):
            print("\n--- Evaluate TFLite model ---")
            tflite_interpreter = tf.lite.Interpreter(model_path=model)
            model_extension = "tflite"
        else:
            raise ValueError("Unsupported model type. Please use a Keras, PyTorch or TFLite model.")
    # If a model instance is provided
    elif isinstance(model, tf.keras.Model):
        print("\n--- Evaluate Keras model (instance) ---")
        loaded_model = model
        model_extension = "keras"
    elif isinstance(model, torch.nn.Module):
        print("\n--- Evaluate PyTorch model (instance) ---")
        loaded_model = model
        loaded_model.eval()
        loaded_model.to(device)
        model_extension = "pt"
    elif hasattr(model, "get_input_details") and hasattr(model, "get_output_details"):
        print("\n--- Evaluate TFLite model (instance) ---")
        tflite_interpreter = model
        model_extension = "tflite"
    else:
        raise ValueError("Unsupported model type or path.")

    pbar = tqdm(test_gen, desc="Processing data", dynamic_ncols=True)

    all_true_labels = []
    all_pred_labels = []

    for images, labels in pbar:
        if model_extension == "keras":
            predictions = loaded_model.predict(images, verbose=0)
        elif model_extension == "pt":
            with torch.no_grad():
                X_batch = images.to(device)
                y_batch = labels.to(device)
                if y_batch.ndim == 1:
                    true_labels = y_batch
                elif y_batch.ndim == 2:
                    true_labels = torch.argmax(y_batch, dim=1)
                else:
                    raise ValueError(f"Unexpected label shape: {y_batch.shape}")

                outputs = loaded_model(X_batch)
                predictions = outputs.cpu().numpy()
                labels = true_labels.cpu().numpy()
        elif model_extension == "tflite":
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            tflite_interpreter.allocate_tensors()

            input_index = input_details[0]['index']
            output_index = output_details[0]['index']

            if input_details[0]['dtype'] == np.float32:
                x_test_lite = images.astype(np.float32)
            else:
                scale, zero_point = input_details[0]["quantization"]
                x_test_lite = np.uint8(images / scale + zero_point)

            if x_test_lite.shape != tuple(input_details[0]['shape']):
                print('Resizing input tensor to:', x_test_lite.shape)
                tflite_interpreter.resize_tensor_input(input_index, x_test_lite.shape)
                tflite_interpreter.allocate_tensors()

            tflite_interpreter.set_tensor(input_index, x_test_lite)
            tflite_interpreter.invoke()

            predictions = tflite_interpreter.get_tensor(output_index)
            if len(predictions) != len(labels):
                raise ValueError(f"Mismatch between predictions ({len(predictions)}) and labels ({len(labels)})")
        else:
            raise ValueError("Unknown model type during evaluation.")

        # === Evaluation loop ===
        for i, pred in enumerate(predictions):
            pred_class = int(np.argmax(pred))
            true_class = int(np.argmax(labels[i]) if isinstance(labels[i], (np.ndarray, list)) else labels[i])

            all_true_labels.append(true_class)
            all_pred_labels.append(pred_class)

            if pred_class == true_class:
                global_metric.addTP(1)
                metric_per_class[pred_class].addTP(1)
            else:
                metric_per_class[pred_class].addFP(1)
                metric_per_class[true_class].addFN(1)
                
                # Add both to global (because per-class metrics sum to global)
                global_metric.addFP(1)
                global_metric.addFN(1)
                
        dict_tqdm = {
            "tp": global_metric.getTP(),
            "fp": global_metric.getFP(),
            "fn": global_metric.getFN(),
            "accuracy": global_metric.getAccuracy()
        }

        pbar.set_postfix(**dict_tqdm)
        
    
    # ==== Confusion Matrix ====
    if confusion_matrix:
        labels_range = [i for i in range(len(class_names))]
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels_range)
        plt.figure(figsize=(10, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues')

        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cm_path = os.path.join(output_dir, f"_{model_extension}_confusion_matrix.png")
            plt.savefig(cm_path, bbox_inches="tight")
            print(f"Confusion matrix saved at {cm_path}")
            plt.show()

    # Free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return global_metric, metric_per_class

