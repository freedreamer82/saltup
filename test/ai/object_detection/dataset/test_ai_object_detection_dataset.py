import pytest
import os
import numpy as np
from saltup.ai.object_detection.dataset.bbox_utils import (
    yolo_to_coco_bbox, coco_to_yolo_bbox
)
from saltup.ai.object_detection.dataset.yolo_darknet import (
    read_label, write_label, create_symlinks_by_class, replace_label_class, shift_class_ids
)

class TestDataProcessing:
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        imgs_dir = tmp_path / "images"
        lbls_dir = tmp_path / "labels"
        imgs_dir.mkdir()
        lbls_dir.mkdir()

        test_data = [
            ("img1.jpg", "0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.2"),
            ("img2.jpg", "1 0.4 0.6 0.3 0.2\n2 0.7 0.3 0.2 0.4")
        ]

        for img_name, label_content in test_data:
            (imgs_dir / img_name).touch()
            with open(lbls_dir / f"{img_name.replace('.jpg', '.txt')}", 'w') as f:
                f.write(label_content)

        return imgs_dir, lbls_dir

    def test_read_write_yolo_label(self, tmp_path):
        label_file = tmp_path / "test.txt"
        sample_labels = [[0, 0.5, 0.5, 0.2, 0.3], [1, 0.3, 0.4, 0.1, 0.2]]
        
        write_label(str(label_file), sample_labels)
        read_labels = read_label(str(label_file))
        
        assert len(read_labels) == len(sample_labels)
        for read, expected in zip(read_labels, sample_labels):
            np.testing.assert_array_almost_equal(read, expected)

    def test_create_symlinks_by_class(self, tmp_path):
        """Test creating symbolic links organized by class"""
        # Setup test directories and data
        imgs_dir = tmp_path / "images" 
        lbls_dir = tmp_path / "labels"
        dest_dir = tmp_path / "dest"
        imgs_dir.mkdir(), lbls_dir.mkdir()

        test_data = {
            "img1": ("0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.2", ["0", "1"]),
            "img2": ("1 0.4 0.6 0.3 0.2\n2 0.7 0.3 0.2 0.4", ["1", "2"]),
            "img3": ("0 0.2 0.3 0.4 0.3", ["0"])
        }

        # Create test files
        for name, (content, _) in test_data.items():
            (imgs_dir / f"{name}.jpg").touch()
            with open(lbls_dir / f"{name}.txt", 'w') as f:
                f.write(content)

        class_names = ["0", "1", "2"]
        create_symlinks_by_class(str(imgs_dir), str(lbls_dir), str(dest_dir), class_names)

        # Verify results
        assert (dest_dir / "classes.names").exists()
        with open(dest_dir / "classes.names") as f:
            assert f.read().strip().split("\n") == class_names

        # Check class directories and symlinks
        for name, (_, classes) in test_data.items():
            for class_id in classes:
                class_dir = dest_dir / class_id
                assert class_dir.is_dir()
                assert (class_dir / f"{name}.jpg").is_symlink()
                assert (class_dir / f"{name}.txt").is_symlink()
                
        # Verify symlink targets
        for name in test_data:
            for ext in [".jpg", ".txt"]:
                orig = (imgs_dir if ext == ".jpg" else lbls_dir) / f"{name}{ext}"
                for class_id in test_data[name][1]:
                    symlink = dest_dir / class_id / f"{name}{ext}"
                    assert os.path.realpath(symlink) == str(orig)
            
    def test_replace_yolo_label_class(self, sample_dataset, tmp_path):
        imgs_dir, lbls_dir = sample_dataset
        # Replace class 1 with class 3
        modified_count, modified_files = replace_label_class(1, 3, str(lbls_dir))
        
        # Verify results
        assert modified_count == 2  # Two files with class 1
        assert len(modified_files) == 2
        
        # Verify modified files
        for file_path in modified_files:
            with open(file_path, 'r') as f:
                content = f.read()
                assert "1" not in content.split()  # Class 1 shold not exists
                assert "3" in content.split()   # Shold be replaced with 3
    
    def test_shift_class_ids(self, tmp_path):
        # Setup test directory
        label_folder = tmp_path / "labels"
        output_folder = tmp_path / "output"
        label_folder.mkdir()
        
        # Create test files
        labels = {
            "img1.txt": "0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.2",
            "img2.txt": "2 0.4 0.6 0.3 0.2\n1 0.7 0.3 0.2 0.4"
        }
        
        for filename, content in labels.items():
            with open(label_folder / filename, 'w') as f:
                f.write(content)
        
        # Shift class IDs by 2
        shift_class_ids(str(label_folder), 2, str(output_folder))
        
        # Verify results
        assert output_folder.exists()
        
        # Check shifted class IDs in output files
        expected_shifts = {
            "img1.txt": ["2 ", "3 "],  # 0->2, 1->3
            "img2.txt": ["4 ", "3 "]   # 2->4, 1->3
        }
        
        for filename, expected_classes in expected_shifts.items():
            with open(output_folder / filename, 'r') as f:
                content = f.read()
                for class_id in expected_classes:
                    assert class_id in content
