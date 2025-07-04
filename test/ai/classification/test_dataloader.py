import pytest
import numpy as np
from saltup.utils.data.image.image_utils import Image, ColorMode
from saltup.ai.classification.dataloader import ClassificationDataloader
import cv2
import os
import shutil
from PIL import Image as PILImage


IMG_SIZE = (64, 64)
ROOT_DIR = "test_dataset"


@pytest.fixture(scope="module")
def create_dummy_dataset():
    if os.path.exists(ROOT_DIR):
        shutil.rmtree(ROOT_DIR)
    os.makedirs(ROOT_DIR)

    class_dirs = ['cat', 'dog']
    for cls in class_dirs:
        class_path = os.path.join(ROOT_DIR, cls)
        os.makedirs(class_path)
        for i in range(2):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            PILImage.fromarray(img).save(os.path.join(class_path, f"{cls}_{i}.jpg"))
    
    yield ROOT_DIR
    shutil.rmtree(ROOT_DIR)


def test_get_num_samples_per_class(create_dummy_dataset):
    loader = ClassificationDataloader(source=create_dummy_dataset)
    num_samples = loader.get_num_samples_per_class()
    assert num_samples['cat'] == 2
    assert num_samples['dog'] == 2

def test_get_image_paths_and_labels(create_dummy_dataset):
    loader = ClassificationDataloader(source=create_dummy_dataset)
    image_paths = loader.get_image_paths()
    labels = loader.get_labels()
    assert len(image_paths) == 4
    assert len(labels) == 4
    assert all(isinstance(p, str) for p in image_paths)
    assert all(isinstance(l, int) for l in labels)

def test_split_returns_correct_ratios(create_dummy_dataset):
    loader = ClassificationDataloader(source=create_dummy_dataset)
    splits = loader.split([0.5, 0.5])
    assert len(splits) == 2
    total = sum(len(s) for s in splits)
    assert total == 4
    assert abs(len(splits[0]) - 2) <= 1  # Allow rounding
    assert abs(len(splits[1]) - 2) <= 1

def test_split_invalid_ratios(create_dummy_dataset):
    loader = ClassificationDataloader(source=create_dummy_dataset)
    with pytest.raises(ValueError):
        loader.split([0.3, 0.3])

def test_merge_success(create_dummy_dataset):
    loader1 = ClassificationDataloader(source=create_dummy_dataset)
    loader2 = ClassificationDataloader(source=create_dummy_dataset)
    merged = ClassificationDataloader.merge(loader1, loader2)
    assert len(merged) == 8
    assert merged.get_num_classes() == 2

def test_merge_fail_on_class_mismatch(create_dummy_dataset):
    loader1 = ClassificationDataloader(source=create_dummy_dataset, classes_dict={'cat': 0, 'dog': 1})
    loader2 = ClassificationDataloader(source=create_dummy_dataset, classes_dict={'cat': 0, 'dog': 2})
    with pytest.raises(ValueError):
        ClassificationDataloader.merge(loader1, loader2)

def test_get_extensions_and_img_size(create_dummy_dataset):
    loader = ClassificationDataloader(source=create_dummy_dataset, img_size=(64, 64, 3), extensions='jpg')
    assert loader.get_extensions() == 'jpg'
    assert loader.get_img_size() == (64, 64, 3)

def test_iter_and_next(create_dummy_dataset):
    loader = ClassificationDataloader(source=create_dummy_dataset)
    count = 0
    for img, label in loader:
        assert isinstance(img, np.ndarray)
        assert isinstance(label, int)
        count += 1
    assert count == 4

def test_get_idx_to_class(create_dummy_dataset):
    loader = ClassificationDataloader(source=create_dummy_dataset)
    idx_to_class = loader.get_idx_to_class()
    assert set(idx_to_class.values()) == {'cat', 'dog'}
    assert set(idx_to_class.keys()) == set([0, 1])

    img, label = loader[0]
    assert isinstance(img, np.ndarray)
    assert isinstance(label, int)

def test_dataloader_with_classes_dict(create_dummy_dataset):
    class_map = {'dog': 0, 'cat': 1}
    loader = ClassificationDataloader(source=create_dummy_dataset, classes_dict=class_map)
    
    assert loader.class_to_idx == class_map
    assert loader.idx_to_class == {0: 'dog', 1: 'cat'}
    assert loader.get_num_classes() == 2

    _, label = loader[0]
    assert label in [0, 1]
