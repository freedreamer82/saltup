import os
import datetime
from dataclasses import dataclass, asdict
from typing import Union
import paho.mqtt.client as mqtt

import tensorflow as tf
import torch
from typing import List, Optional
 
@dataclass
class CallbackContext:
    """
    CallbackContext holds information about the current state of a training process for machine learning models.

    Note:
        The `epochs` and `best_epoch` attributes are 1-based (i.e., the first epoch is 1).

    Attributes:
        model (Union[tf.keras.Model, torch.nn.Module]): The model being trained.
        epochs (int, optional): The total number of epochs for training (1-based).
        batch_size (int, optional): The batch size used during training.
        loss (float, optional): The latest training loss value.
        val_loss (float, optional): The latest validation loss value.
        other_metrics (dict, optional): Additional metrics tracked during training.
        best_model (Union[tf.keras.Model, torch.nn.Module], optional): The best-performing model observed so far.
        best_epoch (int, optional): The epoch at which the best model was observed (1-based).
        best_loss (float, optional): The lowest training loss observed.
        best_val_loss (float, optional): The lowest validation loss observed.

    Methods:
        to_dict():
            Returns a dictionary representation of the context, including model types, training statistics, and any additional metrics.
    """
    model: Union[tf.keras.Model, torch.nn.Module]
    epochs: int = None
    batch_size: int = None
    loss: float = None
    val_loss: float = None
    other_metrics: dict = None
    best_model: Union[tf.keras.Model, torch.nn.Module] = None
    best_epoch: int = None
    best_loss: float = None
    best_val_loss: float = None

    def to_dict(self):
        tmp = {
            'model': type(self.model),
            'total_epochs': self.epochs,
            'batch_size': self.batch_size,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'best_model': type(self.best_model) if self.best_model else None,
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'best_val_loss': self.best_val_loss
        }
        return tmp | (self.other_metrics if self.other_metrics else {})


class BaseCallback:
    """
    BaseCallback is an abstract base class for creating custom training callbacks.

    This class provides a standard interface for managing callback data and handling
    training events such as the beginning and end of training, as well as the end of each epoch.
    Subclasses can override the event methods to implement custom behavior.

    Attributes:
        metrics (dict): A dictionary to store callback-specific metrics.
        metadata (dict): A dictionary to store callback-specific metadata.

    Methods:
        set_metrics(metrics: dict) -> None:
            Sets the internal metrics dictionary to the provided metrics.

        get_metrics() -> dict:
            Returns the current internal metrics dictionary.

        update_metrics(metrics: dict) -> None:
            Updates the internal metrics dictionary with the provided metrics.

        set_metadata(metadata: dict) -> None:
            Sets the internal metadata dictionary to the provided metadata.

        get_metadata() -> dict:
            Returns the current internal metadata dictionary.

        update_metadata(metadata: dict) -> None:
            Updates the internal metadata dictionary with the provided metadata.

        on_train_begin(context: CallbackContext) -> None:
            Called at the beginning of training. Can be overridden by subclasses.

        on_train_end(context: CallbackContext) -> None:
            Called at the end of training. Can be overridden by subclasses.

        on_epoch_end(epoch: int, context: CallbackContext) -> None:
            Called at the end of each epoch. Can be overridden by subclasses.
    """
    def __init__(self):
        self.metrics: dict = {}
        self.metadata: dict = {}

    def set_metrics(self, metrics: dict) -> None:
        self.metrics = metrics or {}

    def get_metrics(self) -> dict:
        return self.metrics
    
    def set_metadata(self, metadata: dict) -> None:
        self.metadata = metadata or {}

    def get_metadata(self) -> dict:
        return self.metadata

    def update_metadata(self, metadata: dict) -> None:
        if metadata:
            self.metadata.update(metadata)

    def update_metrics(self, metrics: dict) -> None:
        def _truncate_floats(d: dict, precision: int = 4) -> dict:
            """Truncate float values in a dictionary to a specified precision, recursively."""
            result = {}
            for k, v in d.items():
                if isinstance(v, float):
                    result[k] = round(v, precision)
                elif isinstance(v, dict):
                    result[k] = _truncate_floats(v, precision)
                else:
                    result[k] = v
            return result
        
        if metrics is not None or len(self.metrics) > 0:
            # Ensure metrics is actually a dictionary
            if not isinstance(metrics, dict):
                print(f"Warning: update_metrics called with non-dict argument: {type(metrics)}. Converting to dict.")
                if hasattr(metrics, 'to_dict'):
                    metrics = metrics.to_dict()
                else:
                    print(f"Error: Cannot convert {type(metrics)} to dict. Ignoring metrics update.")
                    return
            
            self.metrics.update(metrics)                        
            # Truncate float values in metrics
            self.metrics = _truncate_floats(self.metrics, precision=4)

    def on_train_begin(self, context: CallbackContext) -> None:
        pass

    def on_train_end(self, context: CallbackContext) -> None:
        pass

    def on_epoch_end(self, epoch: int, context: CallbackContext) -> None:
        pass


class _KerasCallbackAdapter(tf.keras.callbacks.Callback):
    """
    Adapter class to bridge a custom callback implementing the BaseCallback interface with the Keras Callback API.
    This class wraps a user-defined callback and ensures it is called at appropriate points during Keras model training.
    It tracks the best model according to a monitored metric and passes relevant training context to the custom callback.
    Args:
        custom_callback (BaseCallback): The user-defined callback to be invoked during training.
        monitor (str, optional): The metric name to monitor for best model tracking (default: "val_loss").
        mode (str, optional): Whether to minimize ("min") or maximize ("max") the monitored metric (default: "min").
    Attributes:
        cb (BaseCallback): The wrapped custom callback.
        monitor (str): The metric to monitor.
        mode (str): The mode for monitoring ("min" or "max").
        best_value (float): The best value observed for the monitored metric.
        best_model (tf.keras.Model or None): The best model observed so far.
        best_epoch (int or None): The epoch at which the best model was observed.
    Methods:
        on_train_begin(logs=None): Called at the beginning of training; passes context to the custom callback.
        on_train_end(logs=None): Called at the end of training; passes context to the custom callback.
        on_epoch_end(epoch, logs=None): Called at the end of each epoch; updates best model and passes context.
    """
    def __init__(self, custom_callback: BaseCallback, monitor="val_loss", mode="min"):
        super().__init__()
        self.cb = custom_callback
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else -float('inf')
        self.best_model = None
        self.best_epoch = None
        self.best_logs = None 

    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False
    
    def on_train_begin(self, logs=None):
        """Automatically retrieve the total number of epochs."""
        logs = logs or {}
        context = CallbackContext(
            model=self.model,
            epochs=self.params.get('epochs', None),
            batch_size=self.params.get('batch_size', None),
            loss=logs.get('loss', None),
            val_loss=logs.get('val_loss', None),
            other_metrics={k: v for k, v in logs.items() if k not in ['loss', 'val_loss']},
            best_model=self.best_model,
            best_epoch=self.best_epoch,
            best_loss=self.best_value if self.mode == "min" else None,
            best_val_loss=self.best_value if self.mode == "min" else None
        )
        if hasattr(self.cb, 'on_train_begin'):
            self.cb.on_train_begin(context)

    def _update_metrics_and_metadata(self, context):
        metrics = {}
        for k in ["loss", "val_loss", "best_loss", "best_val_loss","best_epoch"]:
            v = getattr(context, k, None)
            if v is not None:
                metrics[k] = v
        if metrics and hasattr(self.cb, 'update_metrics'):
            self.cb.update_metrics(metrics)

        meta = {
            "epochs": context.epochs,
            "batch_size": context.batch_size,
            "monitor": self.monitor,
            "mode": self.mode,
            "datetime": datetime.datetime.now().isoformat(),
        }
        filtered_meta = {k: v for k, v in meta.items() if v is not None}
        if filtered_meta and hasattr(self.cb, 'update_metadata'):
            self.cb.update_metadata(filtered_meta)


    def on_train_end(self, logs=None):
        logs = logs or {}
        context = CallbackContext(
            model=self.model,
            epochs=self.params.get('epochs', None),
            batch_size=self.params.get('batch_size', None),
            loss=logs.get('loss', None),
            val_loss=logs.get('val_loss', None),
            other_metrics={k: v for k, v in logs.items() if k not in ['loss', 'val_loss']},
            best_model=self.best_model,
            best_epoch=self.best_epoch,
            best_loss=self.best_value if self.mode == "min" else None,
            best_val_loss=self.best_value if self.mode == "min" else None
        )
        self._update_metrics_and_metadata(context)
        if hasattr(self.cb, 'on_train_end'):
            self.cb.on_train_end(context)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        is_better = False
        if current is not None:
            if self.mode == "min" and current < self.best_value:
                is_better = True
            elif self.mode == "max" and current > self.best_value:
                is_better = True
        if is_better:
            self.best_value = current
            self.best_model = tf.keras.models.clone_model(self.model)
            self.best_model.set_weights(self.model.get_weights())
            self.best_epoch = epoch + 1
            self.best_logs = logs.copy()  # <--- save the best

        # Use the best logs for best_loss/best_val_loss
        best_loss = self.best_logs.get('loss') if self.best_logs else None
        best_val_loss = self.best_logs.get('val_loss') if self.best_logs else None

        context = CallbackContext(
            model=self.model,
            epochs=self.params.get('epochs', None),
            batch_size=self.params.get('batch_size', None),
            loss=logs.get('loss', None),
            val_loss=logs.get('val_loss', None),
            other_metrics={k: v for k, v in logs.items() if k not in ['loss', 'val_loss']},
            best_model=self.best_model,
            best_epoch=self.best_epoch,
            best_loss=best_loss,
            best_val_loss=best_val_loss
        )

        self._update_metrics_and_metadata(context)
        if hasattr(self.cb, 'update_metrics'):
            self.cb.update_metrics({"epoch": epoch + 1})
        if hasattr(self.cb, 'on_epoch_end'):
            self.cb.on_epoch_end(epoch + 1, context)
        
class KFoldTrackingCallback(BaseCallback):
    """
    Callback to track best model, loss, and val_loss for each fold in k-fold cross-validation.
    Stores results in a dictionary: {fold_index: {"model": ..., "loss": ..., "val_loss": ...}}
    """
    def __init__(self):
        super().__init__()

        self.fold_results = {}
        self.current_fold = None

    def set_fold(self, fold_index: int):
        self.current_fold = fold_index

    def on_train_end(self, context: CallbackContext) -> None:
        if self.current_fold is None:
            raise ValueError("Current fold index not set. Call set_fold(fold_index) before training.")

        # Store results
        self.fold_results[self.current_fold] = {
            "model": context.best_model,
            "loss": context.best_loss,
            "val_loss": context.best_val_loss,
        }
    def get_fold_results(self):
            return self.fold_results     