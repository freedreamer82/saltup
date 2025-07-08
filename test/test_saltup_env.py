import pytest
import os
import json
import tempfile
from unittest.mock import patch, mock_open
from pathlib import Path

from saltup.saltup_env import SaltupEnv, _SaltupEnv


class TestSaltupEnv:
    """Test suite for SaltupEnv singleton and its properties."""

    def test_singleton_pattern(self):
        """Test that SaltupEnv follows singleton pattern."""
        env1 = _SaltupEnv()
        env2 = _SaltupEnv()
        assert env1 is env2
        assert env1 is SaltupEnv

    def test_version_property(self):
        """Test VERSION property returns a string."""
        version = SaltupEnv.VERSION
        assert isinstance(version, str)
        assert len(version) > 0

    @patch.dict(os.environ, {"SALTUP_KERAS_TRAIN_SHUFFLE": "true"})
    def test_keras_train_shuffle_true(self):
        """Test SALTUP_KERAS_TRAIN_SHUFFLE returns True for 'true'."""
        assert SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE is True

    @patch.dict(os.environ, {"SALTUP_KERAS_TRAIN_SHUFFLE": "false"})
    def test_keras_train_shuffle_false(self):
        """Test SALTUP_KERAS_TRAIN_SHUFFLE returns False for 'false'."""
        assert SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE is False

    @patch.dict(os.environ, {"SALTUP_KERAS_TRAIN_SHUFFLE": "1"})
    def test_keras_train_shuffle_one(self):
        """Test SALTUP_KERAS_TRAIN_SHUFFLE returns True for '1'."""
        assert SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE is True

    @patch.dict(os.environ, {}, clear=True)
    def test_keras_train_shuffle_default(self):
        """Test SALTUP_KERAS_TRAIN_SHUFFLE default value."""
        assert SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE is True

    @patch.dict(os.environ, {"SALTUP_KERAS_TRAIN_VERBOSE": "2"})
    def test_keras_train_verbose_custom(self):
        """Test SALTUP_KERAS_TRAIN_VERBOSE with custom value."""
        assert SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE == 2

    @patch.dict(os.environ, {}, clear=True)
    def test_keras_train_verbose_default(self):
        """Test SALTUP_KERAS_TRAIN_VERBOSE default value."""
        assert SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE == 1

    @patch.dict(os.environ, {"SALTUP_NN_MNG_USE_GPU": "false"})
    def test_nn_mng_use_gpu_false(self):
        """Test SALTUP_NN_MNG_USE_GPU returns False."""
        assert SaltupEnv.SALTUP_NN_MNG_USE_GPU is False

    @patch.dict(os.environ, {"SALTUP_BBOX_INNER_FORMAT": "2"})
    def test_bbox_inner_format_custom(self):
        """Test SALTUP_BBOX_INNER_FORMAT with custom value."""
        assert SaltupEnv.SALTUP_BBOX_INNER_FORMAT == 2

    @patch.dict(os.environ, {"SALTUP_BBOX_FLOAT_PRECISION": "6"})
    def test_bbox_float_precision_custom(self):
        """Test SALTUP_BBOX_FLOAT_PRECISION with custom value."""
        assert SaltupEnv.SALTUP_BBOX_FLOAT_PRECISION == 6

    @patch.dict(os.environ, {"SALTUP_BBOX_NORMALIZATION_TOLERANCE": "0.05"})
    def test_bbox_normalization_tolerance_custom(self):
        """Test SALTUP_BBOX_NORMALIZATION_TOLERANCE with custom value."""
        assert SaltupEnv.SALTUP_BBOX_NORMALIZATION_TOLERANCE == 0.05

    @patch.dict(os.environ, {"SALTUP_ONNX_OPSET": "18"})
    def test_onnx_opset_custom(self):
        """Test SALTUP_ONNX_OPSET with custom value."""
        assert SaltupEnv.SALTUP_ONNX_OPSET == 18

    @patch.dict(os.environ, {"SALTUP_PYTORCH_DEVICE": "cuda:1"})
    def test_pytorch_device_custom(self):
        """Test SALTUP_PYTORCH_DEVICE with custom value."""
        assert SaltupEnv.SALTUP_PYTORCH_DEVICE == "cuda:1"

    @patch.dict(os.environ, {}, clear=True)
    def test_pytorch_device_default(self):
        """Test SALTUP_PYTORCH_DEVICE default value."""
        assert SaltupEnv.SALTUP_PYTORCH_DEVICE == "auto"


class TestKerasCompileArgs:
    """Test suite for SALTUP_TRAINING_KERAS_COMPILE_ARGS property."""

    @patch.dict(os.environ, {}, clear=True)
    def test_keras_compile_args_default(self):
        """Test default empty dictionary when no environment variable is set."""
        result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
        assert result == {}

    @patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": '{"metrics": ["accuracy"], "run_eagerly": true}'})
    def test_keras_compile_args_json_string(self):
        """Test parsing JSON string from environment variable."""
        result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
        expected = {"metrics": ["accuracy"], "run_eagerly": True}
        assert result == expected

    @patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": 'invalid json'})
    def test_keras_compile_args_invalid_json(self):
        """Test handling of invalid JSON string."""
        result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
        assert result == {}

    def test_keras_compile_args_json_file(self, tmp_path):
        """Test reading JSON from file."""
        json_file = tmp_path / "compile_args.json"
        test_data = {"metrics": ["accuracy", "precision"], "run_eagerly": False}
        json_file.write_text(json.dumps(test_data))
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": str(json_file)}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert result == test_data

    def test_keras_compile_args_nonexistent_file(self, tmp_path):
        """Test handling of non-existent file path."""
        nonexistent_file = tmp_path / "nonexistent.json"
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": str(nonexistent_file)}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert result == {}

    def test_keras_compile_args_invalid_json_file(self, tmp_path):
        """Test handling of file with invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("invalid json content")
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": str(json_file)}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert result == {}

    @patch("builtins.open", side_effect=IOError("File read error"))
    def test_keras_compile_args_file_io_error(self, mock_file):
        """Test handling of IO error when reading file."""
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": "/fake/path.json"}):
            with patch("os.path.isfile", return_value=True):
                result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
                assert result == {}


class TestKerasFitArgs:
    """Test suite for SALTUP_TRAINING_KERAS_FIT_ARGS property."""

    @patch.dict(os.environ, {}, clear=True)
    def test_keras_fit_args_default(self):
        """Test default empty dictionary when no environment variable is set."""
        result = SaltupEnv.SALTUP_TRAINING_KERAS_FIT_ARGS
        assert result == {}

    @patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_FIT_ARGS": '{"workers": 4, "use_multiprocessing": true}'})
    def test_keras_fit_args_json_string(self):
        """Test parsing JSON string from environment variable."""
        result = SaltupEnv.SALTUP_TRAINING_KERAS_FIT_ARGS
        expected = {"workers": 4, "use_multiprocessing": True}
        assert result == expected

    @patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_FIT_ARGS": 'invalid json'})
    def test_keras_fit_args_invalid_json(self):
        """Test handling of invalid JSON string."""
        result = SaltupEnv.SALTUP_TRAINING_KERAS_FIT_ARGS
        assert result == {}

    def test_keras_fit_args_json_file(self, tmp_path):
        """Test reading JSON from file."""
        json_file = tmp_path / "fit_args.json"
        test_data = {"workers": 8, "use_multiprocessing": False, "max_queue_size": 20}
        json_file.write_text(json.dumps(test_data))
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_FIT_ARGS": str(json_file)}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_FIT_ARGS
            assert result == test_data


class TestPytorchTrainingArgs:
    """Test suite for SALTUP_TRAINING_PYTORCH_ARGS property."""

    @patch.dict(os.environ, {}, clear=True)
    def test_pytorch_args_default(self):
        """Test default values when no environment variable is set."""
        result = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
        expected = {
            "use_scheduler_per_epoch": False,
            "early_stopping_patience": 0
        }
        assert result == expected

    @patch.dict(os.environ, {"SALTUP_TRAINING_PYTORCH_ARGS": '{"use_scheduler_per_epoch": true, "validation_frequency": 2}'})
    def test_pytorch_args_json_string_partial(self):
        """Test parsing JSON string with partial override."""
        result = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
        expected = {
            "use_scheduler_per_epoch": True,
            "early_stopping_patience": 0,  # Default
            "validation_frequency": 2  # Override
        }
        assert result == expected

    @patch.dict(os.environ, {"SALTUP_TRAINING_PYTORCH_ARGS": '{"gradient_clip_value": 1.5, "early_stopping_patience": 10}'})
    def test_pytorch_args_json_string_full_override(self):
        """Test parsing JSON string with multiple overrides."""
        result = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
        expected = {
            "use_scheduler_per_epoch": False,  # Default
            "early_stopping_patience": 10,  # Override
            "gradient_clip_value": 1.5  # Override
        }
        assert result == expected

    def test_pytorch_args_json_file(self, tmp_path):
        """Test reading JSON from file with defaults merge."""
        json_file = tmp_path / "pytorch_args.json"
        test_data = {
            "use_scheduler_per_epoch": True,
            "gradient_clip_value": 2.0,
            "custom_param": "ignored"  # This should be merged but not in defaults
        }
        json_file.write_text(json.dumps(test_data))
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_PYTORCH_ARGS": str(json_file)}):
            result = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
            expected = {
                "use_scheduler_per_epoch": True,  # Override
                "early_stopping_patience": 0,  # Default
                "gradient_clip_value": 2.0,  # Override
                "custom_param": "ignored"  # Additional param
            }
            assert result == expected

    @patch.dict(os.environ, {"SALTUP_TRAINING_PYTORCH_ARGS": 'invalid json'})
    def test_pytorch_args_invalid_json(self):
        """Test handling of invalid JSON string returns defaults."""
        result = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
        expected = {
            "use_scheduler_per_epoch": False,
            "early_stopping_patience": 0
        }
        assert result == expected

    def test_pytorch_args_invalid_json_file(self, tmp_path):
        """Test handling of file with invalid JSON returns defaults."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("invalid json content")
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_PYTORCH_ARGS": str(json_file)}):
            result = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
            expected = {
                "use_scheduler_per_epoch": False,
                "early_stopping_patience": 0
            }
            assert result == expected


class TestEnvironmentVariableTypes:
    """Test suite for proper type conversion and validation."""

    def test_boolean_environment_variables(self):
        """Test various boolean environment variable formats."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("YES", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("NO", False),
            ("invalid", False),
        ]
        
        for value, expected in test_cases:
            with patch.dict(os.environ, {"SALTUP_KERAS_TRAIN_SHUFFLE": value}):
                assert SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE is expected

    def test_integer_environment_variables(self):
        """Test integer environment variable conversion."""
        test_cases = [
            ("0", 0),
            ("1", 1),
            ("42", 42),
            ("-1", -1),
        ]
        
        for value, expected in test_cases:
            with patch.dict(os.environ, {"SALTUP_KERAS_TRAIN_VERBOSE": value}):
                assert SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE == expected

    def test_float_environment_variables(self):
        """Test float environment variable conversion."""
        test_cases = [
            ("0.0", 0.0),
            ("1.5", 1.5),
            ("0.01", 0.01),
            ("1e-2", 0.01),
        ]
        
        for value, expected in test_cases:
            with patch.dict(os.environ, {"SALTUP_BBOX_NORMALIZATION_TOLERANCE": value}):
                assert SaltupEnv.SALTUP_BBOX_NORMALIZATION_TOLERANCE == expected


class TestIntegration:
    """Integration tests for multiple environment variables."""

    def test_multiple_environment_variables(self):
        """Test setting multiple environment variables simultaneously."""
        env_vars = {
            "SALTUP_KERAS_TRAIN_SHUFFLE": "false",
            "SALTUP_KERAS_TRAIN_VERBOSE": "2",
            "SALTUP_PYTORCH_DEVICE": "cuda:0",
            "SALTUP_ONNX_OPSET": "17",
            "SALTUP_TRAINING_KERAS_COMPILE_ARGS": '{"metrics": ["accuracy"]}',
            "SALTUP_TRAINING_PYTORCH_ARGS": '{"gradient_clip_value": 1.0}'
        }
        
        with patch.dict(os.environ, env_vars):
            assert SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE is False
            assert SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE == 2
            assert SaltupEnv.SALTUP_PYTORCH_DEVICE == "cuda:0"
            assert SaltupEnv.SALTUP_ONNX_OPSET == 17
            assert SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS == {"metrics": ["accuracy"]}
            
            pytorch_args = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
            assert pytorch_args["gradient_clip_value"] == 1.0
            assert pytorch_args["use_scheduler_per_epoch"] is False  # Default preserved

    def test_with_real_json_files(self):
        """Test using real JSON files from test data."""
        test_data_path = Path(__file__).parent / "tests_data" / "saltup_env"
        
        # Test with existing JSON files
        keras_compile_file = test_data_path / "keras_compile_args.json"
        keras_fit_file = test_data_path / "keras_fit_args.json"
        pytorch_file = test_data_path / "pytorch_args.json"
        
        if keras_compile_file.exists():
            with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": str(keras_compile_file)}):
                result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
                assert "metrics" in result
                assert result["run_eagerly"] is True
        
        if keras_fit_file.exists():
            with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_FIT_ARGS": str(keras_fit_file)}):
                result = SaltupEnv.SALTUP_TRAINING_KERAS_FIT_ARGS
                assert result["workers"] == 8
                assert result["use_multiprocessing"] is True
        
        if pytorch_file.exists():
            with patch.dict(os.environ, {"SALTUP_TRAINING_PYTORCH_ARGS": str(pytorch_file)}):
                result = SaltupEnv.SALTUP_TRAINING_PYTORCH_ARGS
                assert result["use_scheduler_per_epoch"] is False
                assert result["early_stopping_patience"] == 15  # From test file


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_environment_variables(self):
        """Test handling of empty string environment variables."""
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": ""}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert result == {}

    def test_whitespace_only_environment_variables(self):
        """Test handling of whitespace-only environment variables."""
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": "   "}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert result == {}

    def test_directory_path_as_json_file(self, tmp_path):
        """Test handling when a directory path is provided instead of file."""
        directory = tmp_path / "config_dir"
        directory.mkdir()
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": str(directory)}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert result == {}

    def test_very_large_json_string(self):
        """Test handling of very large JSON string."""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        large_json = json.dumps(large_dict)
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": large_json}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert len(result) == 1000
            assert result["key_0"] == "value_0"
            assert result["key_999"] == "value_999"

    def test_nested_json_structure(self):
        """Test handling of nested JSON structures."""
        nested_json = json.dumps({
            "optimizer_config": {
                "learning_rate": 0.001,
                "parameters": {
                    "momentum": 0.9,
                    "decay": 1e-6
                }
            },
            "callbacks": [
                {"type": "early_stopping", "patience": 10},
                {"type": "reduce_lr", "factor": 0.5}
            ]
        })
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": nested_json}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert "optimizer_config" in result
            assert result["optimizer_config"]["learning_rate"] == 0.001
            assert len(result["callbacks"]) == 2

    def test_unicode_characters_in_json(self):
        """Test handling of unicode characters in JSON."""
        unicode_json = json.dumps({
            "description": "Тест с unicode символами",
            "emoji": "🚀🔥💯",
            "chinese": "测试中文字符"
        })
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": unicode_json}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert "description" in result
            assert result["emoji"] == "🚀🔥💯"
            assert result["chinese"] == "测试中文字符"
            assert result["optimizer_config"]["learning_rate"] == 0.001
            assert len(result["callbacks"]) == 2

    def test_unicode_characters_in_json(self):
        """Test handling of unicode characters in JSON."""
        unicode_json = json.dumps({
            "description": "Тест с unicode символами",
            "emoji": "🚀🔥💯",
            "chinese": "测试中文字符"
        })
        
        with patch.dict(os.environ, {"SALTUP_TRAINING_KERAS_COMPILE_ARGS": unicode_json}):
            result = SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS
            assert "description" in result
            assert result["emoji"] == "🚀🔥💯"
            assert result["chinese"] == "测试中文字符"
