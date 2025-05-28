class BaseCallback:
    def __init__(self, metric_keys: list = None):
        self.metric_keys = metric_keys  # If None, show all keys

    def filter_metrics(self, metrics):
        if metrics is None:
            return {}
        if self.metric_keys is None:
            return metrics
        return {k: v for k, v in metrics.items() if k in self.metric_keys}

    def on_train_end(self, metrics=None):
        pass
    
    def on_epoch_end(self, epoch, metrics=None):
        pass