from pathlib import Path
import pickle
from tqdm import tqdm
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Any

from saltup.utils.data.image.image_utils import Image, ColorMode, ImageFormat


class BaseDatasetLoader(ABC):
    """Base interface for dataset loaders.

    Abstract base class that defines the interface for dataset loaders.
    Provides basic functionality for loading and iterating over image-label pairs.
    """

    @abstractmethod
    def __iter__(self):
        """Returns iterator over image and label paths."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Returns total number of samples in dataset."""
        raise NotImplementedError

    @staticmethod
    def load_image(
        image_path: str, 
        color_mode: ColorMode = ColorMode.BGR
    ) -> Image:
        """Load and convert image to specified color mode.

        Args:
            image_path: Path to the image file
            color_mode: Target color mode ("RGB", "BGR", or "GRAY")

        Returns:
            np.ndarray: Image in specified color mode

        Raises:
            FileNotFoundError: If image file does not exist or cannot be loaded
            ValueError: If color conversion fails
        """
        return Image(image_path, color_mode)

    def save_dataset(
        self,
        filepath: str = '',
        process_fn: callable = lambda x: x,
        use_tqdm: bool = True
    ) -> Path:
        """Save dataset as a pickle file containing processed images and labels.

        Args:
            filepath: Path to save the pickle file. If empty, creates a timestamped file.
            process_fn: Function to process each image before saving.
            use_tqdm: Whether to display a progress bar.

        Returns:
            Path object pointing to the saved file.

        Raises:
            IOError: If unable to create directory or write file.
        """
        try:
            # Create filepath if not provided
            if not filepath:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"./dataset_{current_time}.pkl"

            # Ensure directory exists and add .pkl extension
            filepath = Path(filepath).with_suffix('.pkl')
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Process data with optional progress bar
            iterator = tqdm(self, disable=not use_tqdm, desc="Processing data")
            processed_data = [(process_fn(image.get_data()), label)
                              for image, label in iterator]

            # Save processed data
            with open(filepath, 'wb') as f:
                pickle.dump(processed_data, f)

            return filepath

        except Exception as e:
            print(f"Error saving data: {e}")
            raise
    
    @staticmethod
    def load_dataset(filepath: str) -> Iterator[Tuple[Any, Any]]:
        """Load a dataset from a pickle file and return an iterator.

        This method loads a dataset that was previously saved using the save() method.
        It returns an iterator that yields tuples of (image, label) pairs.

        Args:
            filepath: Path to the pickle file containing the saved dataset

        Returns:
            Iterator yielding (image, label) pairs

        Raises:
            FileNotFoundError: If the specified file does not exist
            pickle.UnpicklingError: If the file cannot be unpickled
            ValueError: If the file format is invalid

        Example:
            >>> loader = BaseDatasetLoader()
            >>> for image, label in loader.load("dataset.pkl"):
            ...     process_image(image)
            ...     process_label(label)
        """
        filepath = Path(filepath)

        # Verify file exists and has .pkl extension
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        if filepath.suffix != '.pkl':
            raise ValueError(
                f"Invalid file format. Expected .pkl file, got: {filepath.suffix}")

        try:
            # Load the pickle file
            with open(filepath, 'rb') as f:
                dataset = pickle.load(f)

            # Verify dataset format
            if not isinstance(dataset, list):
                raise ValueError(
                    "Invalid dataset format: expected a list of tuples")
            if len(dataset) > 0 and not isinstance(dataset[0], tuple):
                raise ValueError(
                    "Invalid dataset format: list elements must be tuples")

            # Create and return iterator
            return iter(dataset)

        except pickle.UnpicklingError as e:
            raise pickle.UnpicklingError(
                f"Failed to unpickle dataset file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
