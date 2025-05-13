import os
import shutil
import numpy as np
import pytest
import cv2
from PIL import Image as PILImage

from saltup.ai.classification.datagenerator.classification_datagen import ClassificationDataloader, keras_ClassificationDataGenerator, pytorch_ClassificationDataGenerator


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
    
def test_dataloader_auto_class_detection(create_dummy_dataset):
    loader = ClassificationDataloader(root_dir=create_dummy_dataset)
    
    assert len(loader) == 4
    assert set(loader.class_to_idx.keys()) == {'cat', 'dog'}
    assert loader.get_num_classes() == 2

    img, label = loader[0]
    assert isinstance(img, np.ndarray)
    assert isinstance(label, int)

def test_dataloader_with_classes_dict(create_dummy_dataset):
    class_map = {'dog': 0, 'cat': 1}
    loader = ClassificationDataloader(root_dir=create_dummy_dataset, classes_dict=class_map)
    
    assert loader.class_to_idx == class_map
    assert loader.idx_to_class == {0: 'dog', 1: 'cat'}
    assert loader.get_num_classes() == 2

    _, label = loader[0]
    assert label in [0, 1]

def test_keras_data_generator(create_dummy_dataset):
    loader = ClassificationDataloader(root_dir=create_dummy_dataset)
    datagen = keras_ClassificationDataGenerator(
        dataloader=loader,
        target_size=IMG_SIZE,
        num_classes=2,
        batch_size=2,
        preprocess=lambda x, target_size: cv2.resize(x, target_size)
    )
    X, y = datagen[0]
    assert X.shape == (2, *IMG_SIZE, 3)
    assert y.shape == (2, 2)


def test_pytorch_data_generator(create_dummy_dataset):
    loader = ClassificationDataloader(root_dir=create_dummy_dataset)
    datagen = pytorch_ClassificationDataGenerator(
        dataloader=loader,
        target_size=IMG_SIZE,
        num_classes=2,
        batch_size=2,
        preprocess=lambda x, target_size: cv2.resize(x, target_size)
    )

    img_tensor, label_tensor = datagen[0]
    assert img_tensor.shape == (3, *IMG_SIZE)
    assert label_tensor.shape == (2,)
