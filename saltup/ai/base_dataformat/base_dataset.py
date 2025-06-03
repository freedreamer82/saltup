from abc import ABC, abstractmethod
from typing import List

from saltup.ai.object_detection.utils.bbox import BBoxClassId
from saltup.utils.data.image.image_utils import Image


class Dataset(ABC):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @abstractmethod
    def get_image(self, image_id: str) -> Image:
        """Returns the image corresponding to the specified ID."""
        raise NotImplementedError

    @abstractmethod
    def save_image(self, image: Image, image_id: str):
        """Saves the image with the specified ID."""
        raise NotImplementedError

    @abstractmethod
    def get_annotations(self, image_id: str) -> List[BBoxClassId]:
        """Returns the annotations corresponding to the specified image ID."""
        raise NotImplementedError

    @abstractmethod
    def save_annotations(self, annotations: List[BBoxClassId], image_id: str):
        """Saves the annotations for the image with the specified ID."""
        raise NotImplementedError

    @abstractmethod
    def save_image_annotations(self, image: Image, image_id: str, annotations: List[BBoxClassId]):
        """Saves both the image and its annotations with the specified ID."""
        raise NotImplementedError

    @abstractmethod
    def list_images_ids(self, max_entries: int = None) -> List[str]:
        """Returns a list of all image ids in the dataset."""
        raise NotImplementedError

    def list_annotations(self, max_entries: int = -1) -> List[str]:
        """Returns a list of all image ids in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def check_integrity(self) -> bool:
        """Checks the integrity of the dataset."""
        raise NotImplementedError

    @abstractmethod
    def is_annotation_valid(self, annotation: List[BBoxClassId]) -> bool:
        """Checks if the given annotation is valid."""
        raise NotImplementedError
