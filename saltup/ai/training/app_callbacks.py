import os
import datetime
from dataclasses import dataclass, asdict
from typing import Union
import paho.mqtt.client as mqtt

import tensorflow as tf
import torch
from typing import List, Optional
import sys
import io
from saltup.ai.object_detection.yolo.yolo_factory import YoloFactory
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.ai.object_detection.utils.metrics import Metric
from saltup.ai.training.callbacks import BaseCallback, CallbackContext
from saltup.ai.nn_model import NeuralNetworkModel
from saltup.ai.object_detection.yolo import yolo 
from saltup.ai.object_detection.yolo.impl.yolo_anchors_based import BaseYolo
from saltup.ai.object_detection.datagenerator.anchors_based_datagen import BaseDatagenerator
from saltup.ai.classification.evaluate import evaluate_model

 
class MQTTCallback(BaseCallback):
    def __init__(self, broker, port, topic, client_id="keras-metrics", username=None, password=None, id=None):
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
        """
        Method called at the end of each epoch.
        """        
        message = {
            "id": self.id if self.id is not None else "",
            "epoch": epoch + 1,
            "datetime": datetime.datetime.now().isoformat(),
        }
        message.update({
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in context.to_dict().items()
            if isinstance(v, (float, int))
        })
        if self.data:
            message.update(self.data)
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
            "best_model_per_class": metrics_dict["per_class"]
        }
        self.update_data(custom_data)
        super().on_train_end(context)

    def _evaluate_metrics(self, model: BaseYolo, datagen: BaseDatagenerator):
        output_stream = io.StringIO()
        streams = [sys.stdout, output_stream]
        if self.output_file is not None:
            f = open(self.output_file, "a")
            streams.append(f)
        results, metrics = yolo.evaluate(
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
        }
        return metrics_dict

    def on_epoch_end(self, epoch: int, context: CallbackContext):
        yolo_keras_best_model  = YoloFactory.create(
            yolo_type=self.yolo_type,
            model_or_path=NeuralNetworkModel(context.best_model),
            number_class=self.datagen.num_classes,           
            **self.yolo_factory_kwargs
        )

        if (epoch + 1) % self.every_epoch == 0:
            print("\n\n")
            self._print("=" * 80)
            self._print(f"{f'METRICS SUMMARY FOR EPOCH {epoch + 1}':^80}")
            self._print("=" * 80)

            self._print(f"Best model epoch: {context.best_epoch}")
            if self.class_names is not None:
                self._print(f"class_names: {self.class_names}")

            metrics_dict = self._evaluate_metrics(yolo_keras_best_model, self.datagen)

            custom_data = {
               "best_model_per_class": metrics_dict["per_class"]
            }
            self.update_data(custom_data)
        super().on_epoch_end(epoch, context)
        
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
        super().__init__()
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

    def on_train_end(self, context: CallbackContext):
        model=context.best_model
        print("\n\n")
        self._print("=" * 80)
        self._print(f"{f'METRICS ON TRAIN END':^80}")
        self._print("=" * 80)
        if self.class_names is not None:
            self._print(f"class_names: {self.class_names}")

        global_metric, metric_per_class = evaluate_model(model, self.end_of_train_datagen)
        self._print(f"{'Images processed:':<20} {len(self.datagen.dataset) if hasattr(self.datagen, 'dataset') else len(self.datagen)}")

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
        super().on_train_end(context)

    def on_epoch_end(self, epoch: int, context: CallbackContext):
        model=context.best_model
        if (epoch + 1) % self.every_epoch == 0:
            print("\n\n")
            self._print("=" * 80)
            self._print(f"{f'METRICS SUMMARY FOR EPOCH {epoch + 1}':^80}")
            self._print("=" * 80)

            self._print(f"Best model epoch: {context.best_epoch + 1}")
            if self.class_names is not None:
                self._print(f"class_names: {self.class_names}")
                
            global_metric, metric_per_class = evaluate_model(model, self.datagen)
            self._print(f"{'Images processed:':<20} {len(self.datagen.dataset) if hasattr(self.datagen, 'dataset') else len(self.datagen)}")

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
        super().on_epoch_end(epoch, context)