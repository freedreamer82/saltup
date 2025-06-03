from saltup.ai.classification.datagenerator import keras_ClassificationDataGenerator, ClassificationDataloader, pytorch_ClassificationDataGenerator
from saltup.utils.data.image.image_utils import Image, ColorMode
from saltup.ai.keras_utils.keras_to_tflite_quantization import *
from saltup.ai.keras_utils.keras_to_onnx import *
from saltup.ai.training.training_callbacks import *
from saltup.ai.base_dataformat.base_datagen import BaseDatagenerator, kfoldGenerator
from typing import Iterator, Tuple, Any, List, Tuple, Union
import os
import shutil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import albumentations as A
from saltup.ai.object_detection.utils.metrics import Metric
import copy
import gc
import torch
from torch.utils.data import DataLoader as pytorch_DataGenerator
import tensorflow as tf


def evaluate_model(model_path:str, 
                   test_gen:Union[keras_ClassificationDataGenerator, pytorch_DataGenerator],
                   output_dir:str=None,
                   loss_function:callable=None) -> float:
    """function to evaluate the model on the test set.

    Args:
        model_path (str): path to the model
        test_gen (Union[keras_ClassificationDataGenerator, pytorch_ClassificationDataGenerator]): test data generator
        test_data_dir (str): folder containing the test data
        loss_function (callable, optional): loss_function for evaluation. Only used when evaluating a PyTorch model. Defaults to None.

    Raises:
        ValueError: if the model is not a keras or pytorch model
        ValueError: if the model is not a keras model and loss_function is None

    Returns:
        float: Accuracy of the model on the test set
    """
    # Evaluate the model on the test set

    global_metric = Metric()
    if isinstance(test_gen, keras_ClassificationDataGenerator):
        class_names = test_gen.dataloader.get_classes().keys()
    elif isinstance(test_gen, pytorch_DataGenerator):
        class_names = test_gen.dataset.dataloader.get_classes().keys()
    
    print("The class names are:", class_names)
    metric_per_class = {i: Metric() for i in range(len(class_names))}

    # Load model once before loop
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path.endswith('.keras'):
        print("\n--- Evaluate Keras model ---")
        model = tf.keras.models.load_model(model_path)

    elif model_path.endswith('.pt'):
        print("\n--- Evaluate PyTorch model ---")
        if loss_function is None:
            raise ValueError("The model is not a keras model, please provide a loss_function.")

        model = torch.jit.load(model_path) if model_path.endswith(".pt") else torch.load(model_path)
        model.eval()
        model.to(device)

    elif model_path.endswith('.tflite'):
        print("\n--- Evaluate TFLite model ---")
        tflite_interpreter = tf.lite.Interpreter(model_path=model_path)


    else:
        raise ValueError("Unsupported model type. Please use a Keras, PyTorch or TFLite model.")

    pbar = tqdm(test_gen, desc="Processing data", dynamic_ncols=True)

    all_true_labels = []
    all_pred_labels = []

    for images, labels in pbar:
        if model_path.endswith('.keras'):
            predictions = model.predict(images, verbose=0)

        elif model_path.endswith('.pt'):
            with torch.no_grad():
                X_batch = images.to(device)
                y_batch = labels.to(device)
                # Handle both one-hot encoded and non-one-hot encoded labels
                if y_batch.ndim == 1:  # If labels are not one-hot encoded
                    true_labels = y_batch
                elif y_batch.ndim == 2:  # If labels are one-hot encoded
                    true_labels = torch.argmax(y_batch, dim=1)
                else:
                    raise ValueError(f"Unexpected label shape: {y_batch.shape}")

                outputs = model(X_batch)
                loss = loss_function(outputs, true_labels)
                predictions = outputs.cpu().numpy()
                labels = true_labels.cpu().numpy()

        elif model_path.endswith('.tflite'):
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()  
            tflite_interpreter.allocate_tensors()

            input_index = input_details[0]['index']
            output_index = output_details[0]['index']
            
            # Ensure the input tensor is of type FLOAT32
            if input_details[0]['dtype'] == np.float32:
                x_test_lite = images.astype(np.float32)
            else:
                scale, zero_point = input_details[0]["quantization"]
                x_test_lite = np.uint8(images / scale + zero_point)
            
            # Resize the input tensor only if necessary
            if x_test_lite.shape != tuple(input_details[0]['shape']):
                print('Resizing input tensor to:', x_test_lite.shape)
                tflite_interpreter.resize_tensor_input(input_index, x_test_lite.shape)
                tflite_interpreter.allocate_tensors()
            
            tflite_interpreter.set_tensor(input_index, x_test_lite)
            tflite_interpreter.invoke()

            predictions = tflite_interpreter.get_tensor(output_index)
            # Ensure predictions and labels are aligned
            if len(predictions) != len(labels):
                raise ValueError(f"Mismatch between predictions ({len(predictions)}) and labels ({len(labels)})")

        # === Evaluation loop ===
        for i, pred in enumerate(predictions):
            pred_class = int(np.argmax(pred))
            true_class = int(np.argmax(labels[i]) if isinstance(labels[i], (np.ndarray, list)) else labels[i])

            all_true_labels.append(true_class)
            all_pred_labels.append(pred_class)

            if pred_class == true_class:
                # True Positive for the predicted class
                global_metric.addTP(1)
                metric_per_class[pred_class].addTP(1)
            else:
                # False Positive for the predicted class
                global_metric.addFP(1)
                metric_per_class[pred_class].addFP(1)

                # False Negative for the true class
                global_metric.addFN(1)
                metric_per_class[true_class].addFN(1)

        pbar.set_postfix(**global_metric.get_metrics())

    # ==== Confusion Matrix ====
    model_extension = os.path.splitext(model_path)[1].replace('.', '')
    labels = [i for i in range(len(class_names))]
    
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
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

    # ==== Final Report ====
    print("\n" + "="*80)
    print(f"{'METRICS SUMMARY':^80}")
    print("="*80 + "\n")
    print(f"{'Images processed:':<20} {len(test_gen.dataset) if hasattr(test_gen, 'dataset') else len(test_gen)}")

    print("\nPer class:")
    print("+" * 80)
    print(f"{'Label':<18} | {'Precision':<10} | {'Recall':<10} | {'Accuracy':<10}")
    print("-" * 80)
    for class_id, class_label in enumerate(class_names):
        metrics = metric_per_class[class_id]
        print(f"{class_label:<18} | {metrics.getPrecision():<10.4f} | {metrics.getRecall():<10.4f} | {metrics.getAccuracy():<10.4f}")

    print("\nOverall:")
    print(f"{'True Positives (TP):':<25} {global_metric.getTP()}")
    print(f"{'False Positives (FP):':<25} {global_metric.getFP()}")
    print(f"{'False Negatives (FN):':<25} {global_metric.getFN()}")
    print(f"{'Overall Precision:':<25} {global_metric.getPrecision():.4f}")
    print(f"{'Overall Recall:':<25} {global_metric.getRecall():.4f}")
    print(f"{'Overall Accuracy:':<25} {global_metric.getAccuracy():.4f}")
    print("=" * 80)

    # Free memory
    torch.cuda.empty_cache()
    gc.collect()

    return global_metric.getAccuracy()

