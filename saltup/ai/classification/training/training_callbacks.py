import tensorflow as tf

class BaseCallback:
    def __init__(self, log_keys: list = None):
        self.log_keys = log_keys  # If None, show all keys

    def filter_logs(self, logs):
        if logs is None:
            return {}
        if self.log_keys is None:
            return logs
        return {k: v for k, v in logs.items() if k in self.log_keys}

    def on_train_end(self, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
class KerasCallbackAdapter(tf.keras.callbacks.Callback):
    def __init__(self, custom_callback: BaseCallback):
        super().__init__()
        self.cb = custom_callback

    def on_epoch_end(self, epoch, logs=None):
        self.cb.on_epoch_end(epoch, logs=self.cb.filter_logs(logs))
