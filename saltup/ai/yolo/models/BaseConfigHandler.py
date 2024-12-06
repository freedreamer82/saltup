import json
import yaml
import importlib
import logging
from abc import ABC, abstractmethod


class BaseConfigHandler(ABC):
    """
    An abstract base class to handle loading, validating, and processing configuration files 
    for the training process. Must be subclassed to implement domain-specific logic.
    """

    def __init__(self, file_path: str, postprocess: bool = True):
        """
        Initializes the Config class with a file path and loads the configuration.

        Parameters:
        file_path (str): The path to the configuration file.
        postprocess (bool): Whether to apply postprocessing functions on the loaded configuration.
        """
        self.file_path = file_path
        logging.info(f"Loading configuration from {file_path}")
        
        self._configs = self.open_file()
        self.validate_configs()

        # Define post-processing functions to apply if enabled
        self.postprocess_pipeline = [
            BaseConfigHandler.convert_none_strings,
            BaseConfigHandler.convert_str_to_float
        ]

        if postprocess:
            self.postprocess()

    def open_file(self):
        """
        Opens and reads the content of a JSON or YAML configuration file.

        Returns:
        dict: Parsed content of the file as a dictionary.
        
        Raises:
        ValueError: If the file format is not supported.
        """
        with open(self.file_path, "r") as file:
            if self.file_path.endswith(".json"):
                logging.debug("Loading JSON configuration file.")
                return json.load(file)
            elif self.file_path.endswith(".yaml") or self.file_path.endswith(".yml"):
                logging.debug("Loading YAML configuration file.")
                return yaml.safe_load(file)
            else:
                logging.error(f"Unsupported file format: {self.file_path}")
                raise ValueError(f"Unsupported file format: {self.file_path}")

    def postprocess(self):
        """
        Applies a series of postprocessing functions on the configuration dictionary.
        
        Raises:
        RuntimeError: If the postprocessing pipeline is empty or the configurations were not loaded.
        """
        if not self.postprocess_pipeline:
            logging.warning("Postprocess pipeline is empty.")
            raise RuntimeError("No methods in postprocess-pipeline")
        if not self._configs:
            logging.warning("Configurations are empty.")
            raise RuntimeError("No elements in configs. Maybe not loaded a configs file?")
        
        # Backup the current configuration before post-processing (Useful if you run in notebook)
        configs_bck = self._configs.copy()
        for callable_process in self.postprocess_pipeline:
            self._configs = callable_process(self._configs)

        # Validate the configuration after post-processing
        try:
            self.validate_configs()
        except Exception as e:
            logging.error("Validation failed after postprocessing. Reverting to the backup state.")
            self._configs = configs_bck
            raise e

    @abstractmethod
    def validate_configs(self):
        """
        Validates the configuration dictionary to ensure all required parameters are present and valid.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("The 'validate_configs' method must be implemented by a subclass.")

    @staticmethod
    def convert_none_strings(data):
        """
        Convert all values in a dictionary that are the string 'None' to the Python None type.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = BaseConfigHandler.convert_none_strings(value)
            elif value == "None":
                data[key] = None
        return data

    @staticmethod
    def convert_str_to_float(data):
        """
        Iterates over all elements in a dictionary, converting any string representations of floats, 
        including scientific notation, into actual float values.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = BaseConfigHandler.convert_str_to_float(value)
            elif isinstance(value, str):
                try:
                    if 'e' in value.lower() or '.' in value:
                        data[key] = float(value)
                except ValueError:
                    pass
        return data

    @staticmethod
    def instantiate_classes(data):
        """
        Recursively searches a nested dictionary or list for "_target_" keys to instantiate classes.
        """
        if isinstance(data, dict):
            if '_target_' in data:
                params = {k: BaseConfigHandler.instantiate_classes(v) for k, v in data.items() if k != '_target_'}
                module_path, class_name = data['_target_'].rsplit(".", 1)
                module = importlib.import_module(module_path)
                Class = getattr(module, class_name)
                return Class(**params)
            for key, value in data.items():
                data[key] = BaseConfigHandler.instantiate_classes(value)

        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = BaseConfigHandler.instantiate_classes(data[i])

        return data

    def get_configs(self):
        """
        Returns the configuration dictionary.
        """
        return self._configs
