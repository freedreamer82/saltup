import os
import datetime
import paho.mqtt.client as mqtt

import tensorflow as tf

class BaseCallback():
    def __init__(self, metric_keys: list = None):
        self.metric_keys = metric_keys  # If None, show all keys

    def filter_metrics(self, metrics):
        if metrics is None:
            return {}
        if self.metric_keys is None:
            return metrics
        return {k: v for k, v in metrics.items() if k in self.metric_keys}
    
    def on_train_begin(self, metrics=None):
        pass

    def on_train_end(self, metrics=None):
        pass
    
    def on_epoch_end(self, epoch, metrics=None):
        pass


class _KerasCallbackAdapter(tf.keras.callbacks.Callback):
    def __init__(self, custom_callback: BaseCallback):
        super().__init__()
        self.cb = custom_callback
    
    def _implements_train_batch_hooks(self):
        return False

    def _implements_test_batch_hooks(self):
        return False

    def _implements_predict_batch_hooks(self):
        return False
    
    def on_train_begin(self, logs=None):
        """Automatically retrieve the total number of epochs."""
        metrics = logs or {}
        metrics = metrics | self.params  # Merge logs with params
        self.cb.on_train_begin(metrics=self.cb.filter_metrics(metrics))
    
    def on_train_end(self, logs=None):
        metrics = logs or {}
        self.cb.on_train_end(metrics=self.cb.filter_metrics(metrics))

    def on_epoch_end(self, epoch, logs=None):
        metrics = logs or {}
        self.cb.on_epoch_end(epoch, metrics=self.cb.filter_metrics(metrics))
    

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

    def on_epoch_end(self, epoch, metrics=None):
        """
        Method called at the end of each epoch.
        """
        metrics = metrics or {}
        
        message = {
            "notebook": self.notebookid if self.notebookid is not None else "",
            "epoch": epoch + 1,
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
        }
        self.client.publish(self.topic, str(message))

    def on_train_end(self, metrics=None):
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

    def on_train_begin(self, metrics=None):
        """Automatically retrieve the total number of epochs."""
        metrics = metrics or {}
        self.total_epochs = metrics.get("epochs", None)
        print(f"🔹 Training will have {self.total_epochs} total epochs.")

    def on_epoch_end(self, epoch, metrics=None):
        metrics = metrics or {}

        # Ensure values are not None
        loss = metrics.get('loss', 'N/A')
        accuracy = metrics.get('accuracy', 'N/A')
        val_loss = metrics.get('val_loss', 'N/A')
        val_accuracy = metrics.get('val_accuracy', 'N/A')

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