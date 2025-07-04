from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, List, Tuple, Union

import base64
import io
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from saltup.ai.object_detection.utils.bbox import BBoxClassId
from saltup.utils.data.image.image_utils import Image, ColorMode


class StorageFormat(Enum):
    """Enum class for supported storage formats."""
    PICKLE = 'pkl'
    PARQUET = 'parquet'

class BaseDataloader(ABC):
    """Base interface for dataset loaders.

    Abstract base class that defines the interface for dataset loaders.
    Provides basic functionality for loading and iterating over image-label pairs.
    """

    _name: str = None

    def set_name(self, name: str):
        """Set the name of the dataloader."""
        self._name = name

    def get_name(self) -> str:
        """Get the name of the dataloader."""
        return self._name

    @abstractmethod
    def __iter__(self):
        """Returns iterator over image and label paths."""
        raise NotImplementedError
    
    @abstractmethod
    def __next__(self) -> Tuple[Image, List[BBoxClassId]]:
        """Get next item from dataset."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Returns total number of samples in dataset."""
        raise NotImplementedError
    
    @abstractmethod
    def split(self, ratio) -> List['BaseDataloader']:
        """Returns two dataloaders for training and validation sets."""
        raise NotImplementedError
    
    @abstractmethod
    def merge(dl1:'BaseDataloader', dg2:'BaseDataloader') -> 'BaseDataloader':
        """Merges another dataloader into this one."""
        raise NotImplementedError
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[
        Tuple[Image, Any],
        List[Tuple[Image, Any]]
    ]:
        """Get item(s) by index.
        
        Provides a default implementation that uses iteration to access elements.
        Subclasses can override this method with more efficient implementations
        that directly access their data structures.
        
        Args:
            idx: Integer index or slice object
            
        Returns:
            Single (image, annotations) tuple or list of tuples if slice
            
        Raises:
            IndexError: If index out of range
        """
        # Handle negative indices
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError("Index out of range")
            
            # Iterate until we reach the desired index
            iterator = iter(self)
            for _ in range(idx):
                next(iterator)
            return next(iterator)
            
        elif isinstance(idx, slice):
            # Get the actual indices from the slice
            indices = range(*idx.indices(len(self)))
            
            # Use the integer indexing we just defined
            return [self[i] for i in indices]
        
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

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
    
    @staticmethod
    def _serialize_array(arr: np.ndarray) -> str:
        """Convert numpy array to base64 string for Parquet storage."""
        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @staticmethod
    def _deserialize_array(b64_str: str) -> np.ndarray:
        """Convert base64 string back to numpy array."""
        buffer = io.BytesIO(base64.b64decode(b64_str))
        return np.load(buffer, allow_pickle=False)

    @staticmethod
    def _serialize_label(label) -> str:
        """Serialize label to JSON-compatible string."""
        return pickle.dumps(label).hex()

    @staticmethod
    def _deserialize_label(hex_str: str):
        """Deserialize label from hex string."""
        return pickle.loads(bytes.fromhex(hex_str))

    def save_dataset(
        self,
        filepath: str = '',
        process_fn: callable = lambda x: x,
        use_tqdm: bool = True,
        format: StorageFormat = StorageFormat.PICKLE
    ) -> Path:
        """Save dataset as a file containing processed images and labels.

        Args:
            filepath: Path to save the file. If empty, creates a timestamped file.
            process_fn: Function to process each image before saving.
            use_tqdm: Whether to display a progress bar.
            format: Format to save the dataset ('pickle' or 'parquet').

        Returns:
            Path object pointing to the saved file.

        Raises:
            IOError: If unable to create directory or write file.
        """
        try:
            # Create filepath if not provided
            if not filepath:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"./dataset_{current_time}.{format.value}"

            # Ensure directory exists and add appropriate extension
            filepath = Path(filepath).with_suffix(f'.{format.value}')
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Process data with optional progress bar
            iterator = tqdm(self, disable=not use_tqdm, desc="Processing data")
            processed_data = [(process_fn(image.get_data()), label)
                            for image, label in iterator]

            # Save processed data based on format
            if format == StorageFormat.PICKLE:
                with open(filepath, 'wb') as f:
                    pickle.dump(processed_data, f)
            elif format == StorageFormat.PARQUET:
                # Convert data for Parquet storage
                parquet_data = [
                    {
                        'image': self._serialize_array(img),
                        'label': self._serialize_label(lbl)
                    }
                    for img, lbl in processed_data
                ]
                df = pd.DataFrame(parquet_data)
                df.to_parquet(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return filepath

        except Exception as e:
            print(f"Error saving data: {e}")
            raise
    
    @staticmethod
    def load_dataset(
        filepath: str,
        format: StorageFormat = StorageFormat.PICKLE
    ) -> Iterator[Tuple[Any, Any]]:
        """Load a dataset from a file and return an iterator.

        This method loads a dataset that was previously saved using the save() method.
        It returns an iterator that yields tuples of (image, label) pairs.

        Args:
            filepath: Path to the file containing the saved dataset
            format: Format of the dataset file ('pickle' or 'parquet')

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

        # Verify file exists and has appropriate extension
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        if format == StorageFormat.PICKLE and filepath.suffix != '.pkl':
            raise ValueError(
                f"Invalid file format. Expected .pkl file, got: {filepath.suffix}")
        if format == StorageFormat.PARQUET and filepath.suffix != '.parquet':
            raise ValueError(
                f"Invalid file format. Expected .parquet file, got: {filepath.suffix}")

        try:
            # Load based on format
            if format == StorageFormat.PICKLE:
                with open(filepath, 'rb') as f:
                    dataset = pickle.load(f)
            elif format == StorageFormat.PARQUET:
                df = pd.read_parquet(filepath)
                # Convert back from Parquet storage format
                dataset = [
                    (
                        BaseDataloader._deserialize_array(row['image']),
                        BaseDataloader._deserialize_label(row['label'])
                    )
                    for row in df.to_dict('records')
                ]
            else:
                raise ValueError(f"Unsupported format: {format}")

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