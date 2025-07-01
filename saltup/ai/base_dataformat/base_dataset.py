from abc import ABC, abstractmethod
from typing import List, Any

from saltup.utils.data.image.image_utils import Image

class Dataset(ABC):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @abstractmethod
    def get_image(self, image_id: str) -> Image:
        """Returns the data item (e.g., image) corresponding to the specified ID."""
        raise NotImplementedError

    @abstractmethod
    def save_image(self, image: Image, image_id: str):
        """Saves the data item (e.g., image) with the specified ID."""
        raise NotImplementedError

    @abstractmethod
    def get_annotations(self, image_id: str) -> List[Any]:
        """Returns the annotation(s) associated with the specified data item ID."""
        raise NotImplementedError

    @abstractmethod
    def save_annotations(self, annotations: List[Any], image_id: str):
        """Saves the annotation(s) for the data item with the specified ID."""
        raise NotImplementedError

    @abstractmethod
    def save_image_annotations(self, image: Image, image_id: str, annotations: List[Any]):
        """Saves both the data item (e.g., image) and its annotation(s) with the specified ID."""
        raise NotImplementedError

    @abstractmethod
    def list_images(self, max_entries: int = -1) -> List[str]:
        """Returns a list of all data item IDs in the dataset."""
        raise NotImplementedError

    def list_annotations(self, max_entries: int = -1) -> List[str]:
        """Returns a list of all annotation IDs in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def check_integrity(self) -> bool:
        """Checks the integrity of the dataset."""
        raise NotImplementedError

    @abstractmethod
    def is_annotation_valid(self, annotation: List[Any]) -> bool:
        """Checks if the given annotation is valid for the dataset's task."""
        raise NotImplementedError
