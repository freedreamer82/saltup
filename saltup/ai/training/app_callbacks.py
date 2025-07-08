import os
import sys
import io
import datetime
from typing import Optional, Set
from fnmatch import fnmatch

import paho.mqtt.client as mqtt

from saltup.ai.training.callbacks import BaseCallback, CallbackContext
from saltup.ai.nn_model import NeuralNetworkModel

import mlflow
from mlflow.tracking import MlflowClient

from saltup.ai.object_detection.yolo.yolo_factory import YoloFactory
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo import yolo
from saltup.ai.object_detection.yolo.impl.yolo_anchors_based import BaseYolo
from saltup.ai.object_detection.datagenerator.anchors_based_datagen import BaseDatagenerator
from saltup.utils.misc import PathDict
from saltup.ai.classification.evaluate import evaluate_model
 
class MQTTCallback(BaseCallback):
    def __init__(
        self,
        broker: str,
        port: int,
        topic: str,
        client_id: str = "keras-metrics",
        username: Optional[str] = None,
        password: Optional[str] = None,
        id: Optional[str] = None
    ):
        """
        Callback to send training metrics via MQTT.

        Args:
            broker (str): MQTT broker address.
            port (int): MQTT broker port.
            topic (str): MQTT topic to publish messages.
            client_id (str, optional): MQTT client ID. Default is "keras-metrics".
            username (str, optional): Username for authentication. Default is None.
            password (str, optional): Password for authentication. Default is None.
            id (str, optional): Identifier string. Default is None.
        """
        BaseCallback.__init__(self)
        self.broker = broker
        self.port = port
        self.id = id
        self.topic = topic
        self.client_id = client_id
        self.client = mqtt.Client(client_id=client_id)
        self.custom_data = {}
        if username and password:
            self.client.username_pw_set(username, password)
        # Connect to the MQTT broker
        self.client.connect(self.broker, self.port)
        self.client.loop_start()  # Start the background loop to handle the connection

    def on_epoch_end(self, epoch, context: CallbackContext):
        # Check if the MQTT client is connected before publishing
        if not self.client.is_connected():
            print("[MQTTCallback] Client is not connected. Attempting to reconnect...")
            try:
                self.client.reconnect()
            except Exception as e:
                print(f"[MQTTCallback] Reconnection failed: {e}")
        """
        Method called at the end of each epoch.
        """        
        metadata = {
            "id": self.id if self.id is not None else "",
        }
        self.update_metadata(metadata)

        metrics = self.get_metrics()
        metadata = self.get_metadata()
        message = {
            "metadata": metadata,
            "metrics": metrics
        }

        self.client.publish(self.topic, str(message))

    def on_train_end(self, context: CallbackContext):
        """
        Method called at the end of training.
        """
        self.client.loop_stop()  # Stop the MQTT loop
        self.client.disconnect()  # Disconnect from the broker


class MLflowCallback(BaseCallback):
    """Class dedicated to managing MLflow logic during training."""
    
    def __init__(self, mlflow_client: MlflowClient = None, mlflow_run_id: str = None, metrics_filter: Set = None, close_on_train_end: bool = False):
        """
        Initializes the MLflow callback.

        Args:
            mlflow_client (MlflowClient, optional): MLflow client for logging metrics.
            mlflow_run_id (str, optional): MLflow run ID.
            metrics_filter (Set, optional): Set of patterns (with wildcards) to select which metrics to log to MLflow. 
                    If None, all metrics are logged.
            close_on_train_end (bool, optional): Whether to close the MLflow run at the end of training. Default is False.
        """
        self.mlflow_client = mlflow_client
        self.mlflow_run_id = mlflow_run_id
        self.is_enabled = mlflow_client is not None and mlflow_run_id is not None
        # Ensure all patterns in metrics_filter start with '/'
        if metrics_filter is not None:
            self.metrics_filter = set(
            pattern if pattern.startswith('/') else '/' + pattern
            for pattern in metrics_filter
            )
        else:
            self.metrics_filter = None
        self.close_on_train_end = close_on_train_end
    
    def log_param(self, key, value):
        """Log a parameter to MLflow."""
        if not self.is_enabled:
            return
            
        try:
            self.mlflow_client.log_param(
                run_id=self.mlflow_run_id,
                key=key,
                value=value
            )
        except Exception as e:
            print(f"[MLflow] log_param error {key}: {e}")
    
    def log_metric(self, key, value, step=None):
        """Log a metric to MLflow."""
        if not self.is_enabled or value is None:
            return
            
        try:
            self.mlflow_client.log_metric(
                run_id=self.mlflow_run_id,
                key=key,
                value=float(value),
                step=step
            )
        except Exception as e:
            print(f"[MLflow] log_metric error {key}: {e}")
    
    def log_metrics_dict(self, metrics_dict, step=None):
        """Log a dictionary of metrics to MLflow."""
        if not self.is_enabled:
            return
            
        for key, value in metrics_dict.items():
            self.log_metric(key, value, step)
    
    def on_train_begin(self, context: CallbackContext):
        """Called at the beginning of training."""
        self.log_param("total_epochs", context.epochs)
    
    def on_epoch_end(self, epoch, context: CallbackContext):
        if not self.is_enabled:
            return
        
        metrics = PathDict(self.get_metrics())
        for key, value in metrics.items():
            should_log = value is not None
            if self.metrics_filter is not None:
                should_log = should_log and any(fnmatch(key, pattern) for pattern in self.metrics_filter)
                # Replace slashes with underscores for MLflow compatibility
                key = key.replace('/', '_')[1:]
            if should_log:
                # If value is a dict, log each subkey separately
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subvalue is not None:
                            try:
                                self.mlflow_client.log_metric(
                                    run_id=self.mlflow_run_id,
                                    key=f"{key}.{subkey}",
                                    value=float(subvalue),
                                    step=epoch
                                )
                            except Exception as e:
                                print(f"[MLflow] Errore log_metric {key}.{subkey}: {e}")
                else:
                    try:
                        self.mlflow_client.log_metric(
                            run_id=self.mlflow_run_id,
                            key=key,
                            value=float(value),
                            step=epoch
                        )
                    except Exception as e:
                        print(f"[MLflow] Errore log_metric {key}: {e}")
        
    def on_train_end(self, context: CallbackContext):
        """Called at the end of training."""
        # Log final metrics
        final_metrics = {
            "final_loss": getattr(context, 'final_loss', None),
        }
        
        self.log_metrics_dict(final_metrics)
        
        # Optionally close the MLflow run
        if self.is_enabled and self.close_on_train_end:
            self.mlflow_client.set_terminated(self.mlflow_run_id)


class FileLogger(BaseCallback):
    _instance = None  # Prevent accidental multiple instances

    def __init__(self, log_file: str, best_stats_file: str):
        """
        Initializes the FileLogger singleton instance.

        Args:
            log_file (str): Path to the file where training logs will be written.
            best_stats_file (str): Path to the file where the best model statistics will be saved.

        Raises:
            RuntimeError: If an instance of FileLogger has already been initialized.

        Side Effects:
            - Creates the log file and writes a header if it does not exist.
            - Creates the best stats file and writes a header if it does not exist.
            - Sets up initial values for tracking best validation loss and metrics.
        """
        if FileLogger._instance is not None:
            raise RuntimeError("FileLogger has already been initialized!")  # Prevent duplicates
        FileLogger._instance = self  # Save the instance

        BaseCallback.__init__(self)
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
                f.write("epoch,total_epochs,loss,val_loss\n")

        if not os.path.exists(self.best_stats_file):
            with open(self.best_stats_file, "w") as f:
                f.write("# Best model statistics\n")
                f.write(f"# Last updated: {now}\n")
                f.write("epoch,loss,val_loss\n")

    def on_train_begin(self, context: CallbackContext):
        """Automatically retrieve the total number of epochs."""
        self.total_epochs = context.epochs
        print(f"🔹 Training will have {self.total_epochs} total epochs.")

    def on_epoch_end(self, epoch, context: CallbackContext):
        # Ensure values are not None
        loss = context.loss
        val_loss = context.val_loss

        # General logging with error handling
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{epoch},{self.total_epochs},{loss},{val_loss}\n")
        except Exception as e:
            print(f"❌ Error while writing log: {e}")

        # Check if this is the best model so far
        if isinstance(val_loss, (int, float)) and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_metrics = {
                'epoch': epoch,
                'loss': loss,
                'val_loss': val_loss,
            }

            # Safely write the new best statistics
            try:
                with open(self.best_stats_file, "w") as f:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"# Best model statistics - Updated: {now}\n")
                    f.write("epoch,loss,val_loss\n")
                    f.write(f"{self.best_metrics['epoch']},{self.best_metrics['loss']},{self.best_metrics['val_loss']}\n")
            except Exception as e:
                print(f"❌ Error while writing best statistics: {e}")


class YoloEvaluationsCallback(BaseCallback):
    
    def __init__(
        self,
        yolo_type: YoloType, 
        datagen: BaseDatagenerator,
        end_of_train_datagen: BaseDatagenerator = None,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        every_epoch: int = 1,
        output_file: str = None,
        class_names: dict = None,
        yolo_factory_kwargs: dict = None
    ):
        """
        Initializes the callback for YOLO model training evaluation.

            yolo_type (YoloType): The type or version of YOLO model to use for evaluation.
            datagen (BaseDatagenerator): Data generator providing validation or test data during training.
            end_of_train_datagen (BaseDatagenerator, optional): Data generator used for evaluation at the end of training. Defaults to None.
            iou_threshold (float, optional): Intersection-over-Union threshold for considering a detection as true positive. Defaults to 0.5.
            confidence_threshold (float, optional): Minimum confidence score for a detection to be considered valid. Defaults to 0.5.
            every_epoch (int, optional): Frequency (in epochs) at which evaluation is performed. Defaults to 1 (every epoch).
            output_file (str, optional): Path to a file where evaluation results will be saved. If None, results are not saved to file. Defaults to None.
            class_names (dict, optional): Mapping from class indices to class names for reporting results. If None, default class names are used. Defaults to None.
            yolo_factory_kwargs (dict, optional): Additional keyword arguments to pass to the YOLO model factory. Defaults to empty dict.
        """
        if yolo_factory_kwargs is None:
            yolo_factory_kwargs = {}
        BaseCallback.__init__(self)
        self.datagen = datagen
        self.yolo_type = yolo_type
        self.end_of_train_datagen = end_of_train_datagen
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.every_epoch = every_epoch
        self.output_file = output_file
        self.class_names = class_names
        self.yolo_factory_kwargs = yolo_factory_kwargs

    def _print(self, msg):
        if self.output_file is not None:
            with open(self.output_file, "a") as f:
                print(msg, file=f)
        print(msg)

    def extract_per_class_metrics(self, metrics_per_class, number_classes):
        """
        Extracts per-class metrics (precision, recall, f1) into a dictionary.
        All metrics are rounded to 4 decimal places.
        """
        per_class_metrics = {}
        for class_id in range(number_classes):
            metric = metrics_per_class[class_id]
            per_class_metrics[class_id] = {
                "precision": round(metric.getPrecision(), 4),
                "recall": round(metric.getRecall(), 4),
                "f1_score": round(metric.getF1Score(), 4)
            }
        return per_class_metrics

    def on_train_end(self, context: CallbackContext):
        yolo_keras_best_model = YoloFactory.create(
            yolo_type=self.yolo_type,
            model_or_path=NeuralNetworkModel(context.best_model),
            number_class=self.datagen.num_classes,           
            **self.yolo_factory_kwargs
        )

        print("\n\n")
        self._print("=" * 80)
        self._print(f"{f'METRICS ON TRAIN END':^80}")
        self._print("=" * 80)
        if self.class_names is not None:
            self._print(f"class_names: {self.class_names}")

        metrics_dict = self._evaluate_metrics(yolo_keras_best_model, self.end_of_train_datagen)
        custom_data = {
            "best_model_per_class": metrics_dict["per_class"],
            "overall": metrics_dict["overall"],
            "map_50_95": metrics_dict["map_50_95"]
        }
        self.update_metrics(custom_data)

    def _evaluate_metrics(self, model: BaseYolo, datagen: BaseDatagenerator):
        output_stream = io.StringIO()
        streams = [sys.stdout, output_stream]
        if self.output_file is not None:
            f = open(self.output_file, "a")
            streams.append(f)
        results, metrics, mAP_50_95 = yolo.evaluate(
            yolo=model,
            dataloader=datagen.dataloader,
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold,
            output_streams=streams
        )
        if self.output_file is not None:
            f.close()
        number_classes = datagen.num_classes
        per_class_metrics = self.extract_per_class_metrics(results, number_classes)
        metrics_dict = {
            "per_class": per_class_metrics,
            "overall": metrics.get_metrics(),
            "map_50_95": mAP_50_95
        }
        return metrics_dict

    def on_epoch_end(self, epoch: int, context: CallbackContext):
        yolo_keras_best_model  = YoloFactory.create(
            yolo_type=self.yolo_type,
            model_or_path=NeuralNetworkModel(context.best_model),
            number_class=self.datagen.num_classes,           
            **self.yolo_factory_kwargs
        )

        if epoch % self.every_epoch == 0:
            print("\n\n")
            self._print("=" * 80)
            self._print(f"{f'METRICS SUMMARY FOR EPOCH {epoch}':^80}")
            self._print("=" * 80)

            self._print(f"Best model epoch: {context.best_epoch}")
            if self.class_names is not None:
                self._print(f"class_names: {self.class_names}")

            metrics_dict = self._evaluate_metrics(yolo_keras_best_model, self.datagen)

            custom_data = {
               "best_model_per_class": metrics_dict["per_class"],
               "overall": metrics_dict["overall"],
               "map_50_95": metrics_dict["map_50_95"]
            }
            self.update_metrics(custom_data)

class ClassificationEvaluationsCallback(BaseCallback):
    """
    Callback for evaluating classification metrics during and after model training.
    Args:
        datagen (BaseDatagenerator): Data generator used for evaluation during training (e.g., validation set).
        end_of_train_datagen (BaseDatagenerator, optional): Data generator used for evaluation at the end of training.
            If None, the main datagen is used.
        every_epoch (int, optional): Frequency (in epochs) at which to perform evaluation and print metrics.
            Defaults to 1 (every epoch).
        output_file (str, optional): Path to a file where metrics will be appended. If None, only prints to stdout.
        class_names (dict, optional): Mapping of class indices to class names for display in metrics output.
    Methods:
        on_train_end(context): Called at the end of training to evaluate and print/save metrics.
        on_epoch_end(epoch, context): Called at the end of each epoch (or every N epochs) to evaluate and print/save metrics.
    """
    
    def __init__(
        self,
        datagen: BaseDatagenerator, 
        end_of_train_datagen: BaseDatagenerator = None,
        every_epoch: int = 1,
        output_file: str = None,
        class_names: dict = None
    ):
        BaseCallback.__init__(self)
        self.datagen = datagen
        self.end_of_train_datagen = end_of_train_datagen
        self.every_epoch = every_epoch
        self.output_file = output_file
        self.class_names = class_names

    def _print(self, msg):
        if self.output_file is not None:
            with open(self.output_file, "a") as f:
                print(msg, file=f)
        print(msg)
        
    def extract_per_class_metrics(self, metrics_per_class, number_classes):
        """
        Extracts per-class metrics (precision, recall, f1) into a dictionary.
        All metrics are rounded to 4 decimal places.
        """
        per_class_metrics = {}
        for class_id in range(number_classes):
            metric = metrics_per_class[class_id]
            per_class_metrics[class_id] = {
                "accuracy": round(metric.getAccuracy(), 4)
            }
        return per_class_metrics

    def on_train_end(self, context: CallbackContext):
        model=context.best_model
        print("\n\n")
        self._print("=" * 80)
        self._print(f"{f'METRICS ON TRAIN END':^80}")
        self._print("=" * 80)
        if self.class_names is not None:
            self._print(f"class_names: {self.class_names}")

        global_metric, metric_per_class = evaluate_model(model, self.end_of_train_datagen)
        self._print(f"{'Images processed:':<20} {len(self.end_of_train_datagen.dataset) if hasattr(self.end_of_train_datagen, 'dataset') else len(self.end_of_train_datagen)}")

        self._print("\nPer class:")
        self._print("+" * 50)
        self._print(f"{'Label':<18} | {'Accuracy':<10}")
        self._print("-" * 50)
        for class_id, class_label in enumerate(self.class_names):
            metrics = metric_per_class[class_id]
            self._print(f"{class_label:<18} | {metrics.getAccuracy():<10.4f}")

        self._print("\nOverall:")
        self._print(f"{'True Positives (TP):':<25} {global_metric.getTP()}")
        self._print(f"{'False Positives (FP):':<25} {global_metric.getFP()}")
        self._print(f"{'Overall Accuracy:':<25} {global_metric.getAccuracy():.4f}")
        self._print("=" * 80)
        number_classes = self.end_of_train_datagen.num_classes
        per_class_metrics = self.extract_per_class_metrics(metric_per_class, number_classes)
        custom_data = {
            "per_class": per_class_metrics,
            "overall": global_metric.getAccuracy()
        }
        self.update_metrics(custom_data)

    def on_epoch_end(self, epoch: int, context: CallbackContext):
        model=context.best_model
        if epoch % self.every_epoch == 0:
            print("\n\n")
            self._print("=" * 80)
            self._print(f"{f'METRICS SUMMARY FOR EPOCH {epoch}':^80}")
            self._print("=" * 80)

            self._print(f"Best model epoch: {context.best_epoch} | Best loss: {context.best_loss:.4f} | Best val_loss: {context.best_val_loss:.4f}")
            if self.class_names is not None:
                self._print(f"class_names: {self.class_names}")
                
            global_metric, metric_per_class = evaluate_model(model, self.datagen)
            self._print(f"{'Images processed:':<20} {len(self.datagen.dataset.dataloader) if hasattr(self.datagen, 'dataset') else len(self.datagen.dataloader)}")

            self._print("\nPer class:")
            self._print("+" * 50)
            self._print(f"{'Label':<18} | {'Accuracy':<10}")
            self._print("-" * 50)
            for class_id, class_label in enumerate(self.class_names):
                metrics = metric_per_class[class_id]
                self._print(f"{class_label:<18} | {metrics.getAccuracy():<10.4f}")

            self._print("\nOverall:")
            self._print(f"{'True Positives (TP):':<25} {global_metric.getTP()}")
            self._print(f"{'False Positives (FP):':<25} {global_metric.getFP()}")
            self._print(f"{'Overall Accuracy:':<25} {global_metric.getAccuracy():.4f}")
            self._print("=" * 80)
            number_classes = self.datagen.num_classes
            per_class_metrics = self.extract_per_class_metrics(metric_per_class, number_classes)
            custom_data = {
                "per_class": per_class_metrics,
                "overall": global_metric.getAccuracy()
            }
            self.update_metrics(custom_data)
