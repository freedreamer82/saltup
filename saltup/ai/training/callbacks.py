import os
import datetime
from dataclasses import dataclass, asdict
from typing import Union
import paho.mqtt.client as mqtt

import tensorflow as tf
import torch


@dataclass
class CallbackContext:
    model: Union[tf.keras.Model, torch.nn.Module]
    epochs: int = None
    batch_size: int = None
    loss: float = None
    val_loss: float = None
    accuracy: float = None
    val_accuracy: float = None
    misc: dict = None
    best_model: Union[tf.keras.Model, torch.nn.Module] = None
    best_epoch: int = None

    def to_dict(self):
        tmp = {
            'model': type(self.model),
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'loss': self.loss,
            'val_loss': self.val_loss,
            'accuracy': self.accuracy,
            'val_accuracy': self.val_accuracy,
            'best_model': type(self.best_model) if self.best_model else None,  # opzionale
            'best_epoch': self.best_epoch
        }
        return tmp | (self.misc if self.misc else {})


class BaseCallback():
    def __init__(self, metric_keys: list = None):
        self.metric_keys = metric_keys  # If None, show all keys

    def filter_metrics(self, metrics):
        if metrics is None:
            return {}
        if self.metric_keys is None:
            return metrics
        return {k: v for k, v in metrics.items() if k in self.metric_keys}
    
    def on_train_begin(self, context: CallbackContext):
        pass

    def on_train_end(self, context: CallbackContext):
        pass
    
    def on_epoch_end(self, epoch, context: CallbackContext):
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
            accuracy=logs.get('accuracy', None),
            val_accuracy=logs.get('val_accuracy', None),
            misc={k: v for k, v in logs.items() if k not in ['loss', 'val_loss', 'accuracy', 'val_accuracy']}
        )
        self.cb.on_train_begin(context)
    
    def on_train_end(self, logs=None):
        logs = logs or {}
        context = CallbackContext(
            model=self.model,
            epochs=self.params.get('epochs', None),
            batch_size=self.params.get('batch_size', None),
            loss=logs.get('loss', None),
            val_loss=logs.get('val_loss', None),
            accuracy=logs.get('accuracy', None),
            val_accuracy=logs.get('val_accuracy', None),
            misc={k: v for k, v in logs.items() if k not in ['loss', 'val_loss', 'accuracy', 'val_accuracy']},
            best_model=self.best_model,
            best_epoch=self.best_epoch
        )
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
            # Save a copy of the best model
            self.best_model = tf.keras.models.clone_model(self.model)
            self.best_model.set_weights(self.model.get_weights())
            self.best_epoch = epoch + 1

        context = CallbackContext(
            model=self.model,
            epochs=self.params.get('epochs', None),
            batch_size=self.params.get('batch_size', None),
            loss=logs.get('loss', None),
            val_loss=logs.get('val_loss', None),
            accuracy=logs.get('accuracy', None),
            val_accuracy=logs.get('val_accuracy', None),
            misc={k: v for k, v in logs.items() if k not in ['loss', 'val_loss', 'accuracy', 'val_accuracy']},
            best_model=self.best_model,
            best_epoch=self.best_epoch
        )
        self.cb.on_epoch_end(epoch, context)
    

class MQTTCallback(BaseCallback):
    def __init__(self, broker, port, topic, client_id="keras-metrics", username=None, password=None, notebook_id=None):
        """
        Callback to send training metrics via MQTT.

        Args:
            broker (str): MQTT broker address.
            port (int): MQTT broker port.
            topic (str): MQTT topic to publish messages.
            client_id (str, optional): MQTT client ID. Default is "keras-metrics".
            username (str, optional): Username for authentication. Default is None.
            password (str, optional): Password for authentication. Default is None.
            notebook_id (str, optional): Identifier for the notebook. Default is None.
        """
        super().__init__()
        self.broker = broker
        self.port = port
        self.notebookid = notebook_id
        self.topic = topic
        self.client_id = client_id
        self.client = mqtt.Client(client_id=client_id)
        if username and password:
            self.client.username_pw_set(username, password)
        # Connect to the MQTT broker
        self.client.connect(self.broker, self.port)
        self.client.loop_start()  # Start the background loop to handle the connection

    def on_epoch_end(self, epoch, context: CallbackContext):
        """
        Method called at the end of each epoch.
        """        
        message = {
            "notebook": self.notebookid if self.notebookid is not None else "",
            "epoch": epoch + 1,
            "metrics": {k: round(v, 4) for k, v in context.to_dict().items() if isinstance(v, float)},
        }
        self.client.publish(self.topic, str(message))

    def on_train_end(self, context: CallbackContext):
        """
        Method called at the end of training.
        """
        self.client.loop_stop()  # Stop the MQTT loop
        self.client.disconnect()  # Disconnect from the broker


class FileLogger(BaseCallback):
    _instance = None  # Prevent accidental multiple instances

    def __init__(self, log_file, best_stats_file):
        if FileLogger._instance is not None:
            raise RuntimeError("FileLogger has already been initialized!")  # Prevent duplicates
        FileLogger._instance = self  # Save the instance

        super().__init__()
        self.log_file = log_file
        self.best_stats_file = best_stats_file
        self.best_val_loss = float('inf')
        self.best_metrics = None
        self.total_epochs = None  # Initialize total number of epochs

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # If the file does not exist, create the header
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write(f"# Training logs started at: {now}\n")
                f.write("epoch,total_epochs,loss,accuracy,val_loss,val_accuracy\n")

        if not os.path.exists(self.best_stats_file):
            with open(self.best_stats_file, "w") as f:
                f.write("# Best model statistics\n")
                f.write(f"# Last updated: {now}\n")
                f.write("epoch,loss,accuracy,val_loss,val_accuracy\n")

    def on_train_begin(self, context: CallbackContext):
        """Automatically retrieve the total number of epochs."""
        self.total_epochs = context.epochs
        print(f"🔹 Training will have {self.total_epochs} total epochs.")

    def on_epoch_end(self, epoch, context: CallbackContext):
        # Ensure values are not None
        loss = context.loss
        accuracy = context.accuracy
        val_loss = context.val_loss
        val_accuracy = context.val_accuracy

        # General logging with error handling
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{epoch+1},{self.total_epochs},{loss},{accuracy},{val_loss},{val_accuracy}\n")
        except Exception as e:
            print(f"❌ Error while writing log: {e}")

        # Check if this is the best model so far
        if isinstance(val_loss, (int, float)) and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_metrics = {
                'epoch': epoch + 1,
                'loss': loss,
                'accuracy': accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }

            # Safely write the new best statistics
            try:
                with open(self.best_stats_file, "w") as f:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"# Best model statistics - Updated: {now}\n")
                    f.write("epoch,loss,accuracy,val_loss,val_accuracy\n")
                    f.write(f"{self.best_metrics['epoch']},{self.best_metrics['loss']},"
                            f"{self.best_metrics['accuracy']},{self.best_metrics['val_loss']},"
                            f"{self.best_metrics['val_accuracy']}\n")
            except Exception as e:
                print(f"❌ Error while writing best statistics: {e}")