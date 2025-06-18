from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from saltup.utils.data.image.image_utils import Image
from saltup.ai.object_detection.utils.bbox import BBox

class Dataset(ABC):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @abstractmethod
    def get_image(self, image_id: str) -> Image:
        """Returns the image corresponding to the specified ID."""
        pass

    @abstractmethod
    def save_image(self, image: Image, image_id: str):
        """Saves the image with the specified ID."""
        pass

    @abstractmethod
    def get_annotations(self, image_id: str) -> List[BBox]:
        """Returns the annotations corresponding to the specified image ID."""
        pass

    @abstractmethod
    def save_annotations(self, annotations: List[BBox], image_id: str):
        """Saves the annotations for the image with the specified ID."""
        pass

    @abstractmethod
    def save_image_annotations(self, image: Image, image_id: str, annotations: List[BBox]):
        """Saves both the image and its annotations with the specified ID."""
        pass

    @abstractmethod
    def list_images(self, max_entries: int = -1) -> List[str]:
        """Returns a list of all image names in the dataset."""
        pass

    def list_annotations(self, max_entries: int = -1) -> List[str]:
        """Returns a list of all image names in the dataset."""
        pass

    @abstractmethod
    def check_integrity(self) -> bool:
        """Checks the integrity of the dataset."""
        pass

    @abstractmethod
    def is_annotation_valid(self, annotation: List[BBox]) -> bool:
        """Checks if the given annotation is valid."""
        pass