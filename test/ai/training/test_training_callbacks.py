import pytest
from unittest import mock
from saltup.ai.training.callbacks import *
from saltup.ai.training.callbacks import _KerasCallbackAdapter


def test_base_callback_filter_logs_all_keys():
    metrics = {'loss': 0.5, 'accuracy': 0.8}
    cb = BaseCallback()
    filtered = cb.filter_metrics(metrics)
    assert filtered == metrics


def test_base_callback_filter_logs_selected_keys():
    metrics = {'loss': 0.5, 'accuracy': 0.8, 'val_loss': 0.6}
    cb = BaseCallback(metric_keys=['loss', 'val_loss'])
    filtered = cb.filter_metrics(metrics)
    assert filtered == {'loss': 0.5, 'val_loss': 0.6}


def test_base_callback_filter_logs_none_input():
    cb = BaseCallback(metric_keys=['loss'])
    assert cb.filter_metrics(None) == {}


def test_keras_callback_adapter_calls_custom_on_epoch_end():
    mock_cb = mock.Mock(spec=BaseCallback)
    mock_cb.filter_metrics.return_value = {'loss': 0.5}
    
    adapter = _KerasCallbackAdapter(mock_cb)
    adapter.on_epoch_end(1, metrics={'loss': 0.5, 'accuracy': 0.8})

    mock_cb.filter_metrics.assert_called_once_with({'loss': 0.5, 'accuracy': 0.8})
    mock_cb.on_epoch_end.assert_called_once_with(1, metrics={'loss': 0.5})
