import pytest
from unittest import mock
from saltup.ai.training.callbacks import _KerasCallbackAdapter, BaseCallback, CallbackContext


def test_base_callback_update_metrics_all_keys():
    """
    Test the update_metrics method to ensure it correctly updates all keys in the metrics dictionary.
    """
    metrics = {'loss': 0.5, 'accuracy': 0.8}
    cb = BaseCallback()
    cb.update_metrics(metrics)
    assert cb.get_metrics() == metrics


def test_base_callback_update_metrics_selected_keys():
    """
    Test the update_metrics method to ensure it updates only selected keys in the metrics dictionary.
    """
    metrics = {'loss': 0.5, 'accuracy': 0.8, 'val_loss': 0.6}
    cb = BaseCallback()
    cb.update_metrics({'loss': metrics['loss'], 'val_loss': metrics['val_loss']})
    assert cb.get_metrics() == {'loss': 0.5, 'val_loss': 0.6}


def test_base_callback_update_metrics_none_input():
    """
    Test the update_metrics method to ensure it handles None input gracefully.
    """
    cb = BaseCallback()
    cb.update_metrics(None)
    assert cb.get_metrics() == {}


def test_base_callback_update_metadata():
    """
    Test the update_metadata method to ensure it correctly updates metadata.
    """
    cb = BaseCallback()
    metadata = {'epochs': 10, 'batch_size': 32}
    cb.update_metadata(metadata)
    assert cb.get_metadata() == {'epochs': 10, 'batch_size': 32}


def test_keras_callback_adapter_calls_custom_on_epoch_end():
    """
    Test the _KerasCallbackAdapter to ensure it correctly calls the custom callback's methods during epoch end.
    """
    mock_cb = mock.Mock(spec=BaseCallback)
    mock_cb.update_metrics.return_value = None

    adapter = _KerasCallbackAdapter(mock_cb)
    adapter.params = {'epochs': 10, 'batch_size': 32}  # Mock the params attribute

    adapter.on_epoch_end(1, logs={'loss': 0.5, 'accuracy': 0.8})

    # Verify that update_metrics is called with the correct arguments.
    mock_cb.update_metrics.assert_any_call({'loss': 0.5, 'accuracy': 0.8})
    mock_cb.update_metrics.assert_any_call({'epoch': 2})
    # Verify that on_epoch_end is called once.
    mock_cb.on_epoch_end.assert_called_once()
