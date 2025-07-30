import os
import gc
from typing import Union, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader as pytorch_DataGenerator

from saltup.ai.classification.datagenerator import (
    keras_ClassificationDataGenerator,
    pytorch_ClassificationDataGenerator,
)
from saltup.ai.object_detection.utils.metrics import Metric
from saltup.ai.nn_model import NeuralNetworkModel, ModelType


def evaluate_model(
    model: Union[str, NeuralNetworkModel],
    test_gen: Union[keras_ClassificationDataGenerator, pytorch_DataGenerator],
    output_dir: str = None,
    conf_matrix: bool = False
) -> Tuple[Metric, Dict[int, Metric]]:
    """
    Evaluate a classification model on the test set.

    Args:
        model (Union[str, NeuralNetworkModel]):
            Model instance or path to the model file.
        test_gen (Union[keras_ClassificationDataGenerator, pytorch_DataGenerator]):
            Test data generator.
        output_dir (str, optional):
            Directory to save output files such as confusion matrix images.
        conf_matrix (bool, optional):
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
    if isinstance(model, str):
        current_model = NeuralNetworkModel(model_or_path=model)
    elif isinstance(model, NeuralNetworkModel):
        current_model = model
    elif not isinstance(model, NeuralNetworkModel):
        raise ValueError("Model must be a path to a model file or an instance of NeuralNetworkModel.")

    pbar = tqdm(test_gen, desc="Processing data", dynamic_ncols=True)

    all_true_labels = []
    all_pred_labels = []

    for images, labels in pbar:
        
        predictions = current_model.model_inference(images)
        # === Evaluation loop ===
        for i, pred in enumerate(predictions):
            pred_class = int(np.argmax(pred))
            label_item = labels[i]
            if isinstance(label_item, torch.Tensor):
                label_item = label_item.cpu().numpy()
            true_class = int(np.argmax(label_item) if isinstance(label_item, (np.ndarray, list)) else label_item)

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
    if conf_matrix:
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
            model_type = current_model.get_model_type()
            model_name = model_type.to_string()
            os.makedirs(output_dir, exist_ok=True)
            cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(cm_path, bbox_inches="tight")
            print(f"Confusion matrix saved at {cm_path}")
            plt.show()

    # Free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return global_metric, metric_per_class

