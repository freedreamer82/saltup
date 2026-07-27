import pytest
import os
os.environ["SALTUP_BACKEND"] = "keras_tensorflow"
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from saltup.ai.base_dataformat.label_map import LabelMap
from saltup.ai.object_detection.utils.bbox import BBoxClassId, BBoxClassIdScore, BBoxFormat
from saltup.ai.object_detection.dataset.yolo_darknet import (
    YoloDarknetLoader, create_dataset_structure
)
from saltup.ai.object_detection.dataset.coco import COCOLoader
from saltup.ai.object_detection.dataset.pascal_voc import PascalVOCLoader
from saltup.ai.object_detection.dataset.loader_factory import DataLoaderFactory
from saltup.tools.collapse_labels import (
    _parse_mapping, _check_output_dir, collapse_dataset,
    collapse_yolo_labels, collapse_voc_annotations, collapse_coco_annotations
)


CLASS_NAMES = ['mouse', 'feeding', 'climbing', 'food']


def make_yolo_bbox(class_id, coords=(0.5, 0.5, 0.2, 0.2)):
    """Build a YOLO-flavoured annotation: id-carrying, empty class name."""
    return BBoxClassId(
        coordinates=list(coords), class_id=class_id, class_name="",
        fmt=BBoxFormat.YOLO, img_width=100, img_height=100
    )


def annotations_of(item) -> List[BBoxClassId]:
    """Pull the annotations out of a loader item.

    Loaders yield (path, image, annotations), but `__getitem__` is typed as
    returning either one such tuple or a list of them, so indexing it directly
    reads as a union to a type checker.
    """
    return item[2]


def items_of(sliced) -> List:
    """Normalize a loader slice result to a list of items.

    Same reason as `annotations_of`: the slice branch of `__getitem__` is not
    distinguished from the single-item branch by its type.
    """
    return sliced


def object_names(root) -> List[str]:
    """Read the <name> of every <object> in a parsed VOC annotation."""
    names = []
    for obj in root.findall('object'):
        name_elem = obj.find('name')
        assert name_elem is not None and name_elem.text is not None
        names.append(name_elem.text)
    return names


def object_tags(root, tag: str) -> List[str]:
    """Read an arbitrary child tag of every <object> in a parsed VOC annotation."""
    values = []
    for obj in root.findall('object'):
        elem = obj.find(tag)
        assert elem is not None and elem.text is not None
        values.append(elem.text)
    return values


def make_voc_bbox(class_name, coords=(10, 10, 30, 30)):
    """Build a VOC-flavoured annotation: name-carrying, class_id is None."""
    return BBoxClassId(
        coordinates=list(coords), class_id=None, class_name=class_name,
        fmt=BBoxFormat.CORNERS_ABSOLUTE, img_width=100, img_height=100
    )


class TestLabelMapCore:
    def test_collapse_by_name(self):
        label_map = LabelMap.collapse(
            ['feeding', 'climbing'], into='mouse', class_names=CLASS_NAMES
        )
        result = label_map.apply([make_yolo_bbox(i) for i in range(4)])
        assert [b.class_id for b in result] == [0, 0, 0, 3]

    def test_collapse_by_id(self):
        label_map = LabelMap.collapse([1, 2], into=0)
        result = label_map.apply([make_yolo_bbox(i) for i in range(4)])
        assert [b.class_id for b in result] == [0, 0, 0, 3]

    def test_collapse_from_dict(self):
        label_map = LabelMap.collapse({'feeding': 'mouse', 'climbing': 'mouse'},
                                      class_names=CLASS_NAMES)
        result = label_map.apply([make_yolo_bbox(i) for i in range(4)])
        assert [b.class_id for b in result] == [0, 0, 0, 3]

    def test_mixed_id_and_name_rules(self):
        label_map = LabelMap({1: 'mouse', 'climbing': 0}, class_names=CLASS_NAMES)
        result = label_map.apply([make_yolo_bbox(i) for i in range(4)])
        assert [b.class_id for b in result] == [0, 0, 0, 3]

    def test_unmapped_labels_pass_through(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES)
        result = label_map.apply([make_yolo_bbox(3)])
        assert [b.class_id for b in result] == [3]

    def test_drop_unmapped(self):
        """mouse (destination) and feeding (source) survive; climbing and food do not."""
        label_map = LabelMap.collapse(
            ['feeding'], into='mouse', class_names=CLASS_NAMES, drop_unmapped=True
        )
        result = label_map.apply([make_yolo_bbox(i) for i in range(4)])
        assert [b.class_id for b in result] == [0, 0]

    def test_original_annotations_are_not_mutated(self):
        """The loaders cache annotations (COCO), so apply() must not touch them."""
        annotations = [make_yolo_bbox(i) for i in range(4)]
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES)
        label_map.apply(annotations)
        assert [b.class_id for b in annotations] == [0, 1, 2, 3]

    def test_repeated_application_is_stable(self):
        """Repeated access to the same cached item must give the same result."""
        annotations = [make_yolo_bbox(i) for i in range(4)]
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES, reindex=True)
        first = [b.class_id for b in label_map.apply(annotations)]
        second = [b.class_id for b in label_map.apply(annotations)]
        assert first == second == [0, 0, 0, 1]

    def test_yolo_annotations_keep_empty_class_name(self):
        """BBoxClassId.get_data() returns the name when set, so it must stay empty."""
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES)
        result = label_map.apply([make_yolo_bbox(1)])
        assert result[0].class_name == ""
        assert result[0].class_id == 0

    def test_voc_annotations_keep_none_class_id(self):
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES)
        result = label_map.apply([make_voc_bbox(n) for n in CLASS_NAMES])
        assert [b.class_name for b in result] == ['mouse', 'mouse', 'mouse', 'food']
        assert all(b.class_id is None for b in result)

    def test_voc_mapping_without_class_names(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse')
        result = label_map.apply([make_voc_bbox('feeding'), make_voc_bbox('food')])
        assert [b.class_name for b in result] == ['mouse', 'food']

    def test_subclass_type_and_score_preserved(self):
        """A BBoxClassIdScore must not be downcast to BBoxClassId."""
        scored = BBoxClassIdScore(
            coordinates=[0.5, 0.5, 0.2, 0.2], class_id=1, class_name="",
            score=0.87, fmt=BBoxFormat.YOLO, img_width=100, img_height=100
        )
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES)
        result = label_map.apply([scored])[0]
        assert isinstance(result, BBoxClassIdScore)
        assert result.score == 0.87
        assert result.class_id == 0

    def test_coordinates_are_preserved(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES)
        original = make_yolo_bbox(1, coords=(0.25, 0.75, 0.1, 0.4))
        result = label_map.apply([original])[0]
        np.testing.assert_allclose(
            result.get_coordinates(fmt=BBoxFormat.YOLO),
            original.get_coordinates(fmt=BBoxFormat.YOLO)
        )

    def test_empty_annotations(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES)
        assert label_map.apply([]) == []


class TestSparseClassNames:
    """COCO category ids are arbitrary, so class_names also accepts {id: name}."""

    SPARSE = {1: 'mouse', 2: 'feeding', 5: 'food'}

    def test_name_rules_over_sparse_ids(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=self.SPARSE)
        result = label_map.apply([make_yolo_bbox(c) for c in (1, 2, 5)])
        assert [b.class_id for b in result] == [1, 1, 5]

    def test_class_mapping_over_sparse_ids(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=self.SPARSE)
        assert label_map.get_class_mapping() == {1: 1, 2: 1, 5: 5}
        assert label_map.get_num_classes() == 2

    def test_reindex_over_sparse_ids(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse',
                                      class_names=self.SPARSE, reindex=True)
        result = label_map.apply([make_yolo_bbox(c) for c in (1, 2, 5)])
        assert [b.class_id for b in result] == [0, 0, 1]
        assert label_map.get_class_names() == ['mouse', 'food']

    def test_unknown_sparse_id_rejected(self):
        with pytest.raises(ValueError, match="not a known class"):
            LabelMap({3: 1}, class_names=self.SPARSE)


class TestLabelMapReindex:
    def test_reindex_compacts_ids(self):
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES, reindex=True)
        result = label_map.apply([make_yolo_bbox(i) for i in range(4)])
        assert [b.class_id for b in result] == [0, 0, 0, 1]

    def test_reindex_exposes_mapping_and_names(self):
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES, reindex=True)
        assert label_map.get_class_mapping() == {0: 0, 1: 0, 2: 0, 3: 1}
        assert label_map.get_class_names() == ['mouse', 'food']
        assert label_map.get_num_classes() == 2

    def test_without_reindex_ids_keep_gaps(self):
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES)
        assert label_map.get_class_mapping() == {0: 0, 1: 0, 2: 0, 3: 3}
        assert label_map.get_num_classes() == 2

    def test_reindex_with_drop_unmapped(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES,
                                      drop_unmapped=True, reindex=True)
        result = label_map.apply([make_yolo_bbox(i) for i in range(4)])
        assert [b.class_id for b in result] == [0, 0]
        assert label_map.get_class_names() == ['mouse']

    def test_drop_unmapped_keeps_destination_class(self):
        """Collapsing feeding into mouse must never drop mouse itself."""
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES,
                                      drop_unmapped=True)
        result = label_map.apply([make_yolo_bbox(i) for i in range(4)])
        assert [b.class_id for b in result] == [0, 0]
        assert label_map.map_class_id(0) == 0
        assert label_map.map_class_id(3) is None

    def test_reindex_requires_class_names(self):
        with pytest.raises(ValueError, match="requires class_names"):
            LabelMap({1: 0}, reindex=True)


class TestLabelMapDedup:
    def test_duplicates_kept_by_default(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES)
        annotations = [make_yolo_bbox(0), make_yolo_bbox(1)]  # same box, both -> mouse
        assert len(label_map.apply(annotations)) == 2

    def test_dedup_removes_overlapping_same_class(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse',
                                      class_names=CLASS_NAMES, dedup_iou=0.9)
        annotations = [make_yolo_bbox(0), make_yolo_bbox(1)]
        assert len(label_map.apply(annotations)) == 1

    def test_dedup_keeps_distant_boxes(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse',
                                      class_names=CLASS_NAMES, dedup_iou=0.9)
        annotations = [make_yolo_bbox(0), make_yolo_bbox(1, coords=(0.9, 0.9, 0.05, 0.05))]
        assert len(label_map.apply(annotations)) == 2

    def test_dedup_keeps_different_classes(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse',
                                      class_names=CLASS_NAMES, dedup_iou=0.9)
        annotations = [make_yolo_bbox(0), make_yolo_bbox(3)]  # mouse + food, same box
        assert len(label_map.apply(annotations)) == 2



class TestLabelMapValidation:
    def test_empty_mapping(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LabelMap({})

    def test_unknown_source_name(self):
        with pytest.raises(ValueError, match="not in class_names"):
            LabelMap({'nope': 'mouse'}, class_names=CLASS_NAMES)

    def test_unknown_destination_name(self):
        with pytest.raises(ValueError, match="not in class_names"):
            LabelMap({'feeding': 'rodent'}, class_names=CLASS_NAMES)

    def test_unknown_id(self):
        with pytest.raises(ValueError, match="not a known class"):
            LabelMap({9: 0}, class_names=CLASS_NAMES)

    @pytest.mark.parametrize("iou", [0.0, 1.5, -0.5])
    def test_bad_dedup_iou(self, iou):
        with pytest.raises(ValueError, match="dedup_iou"):
            LabelMap({1: 0}, dedup_iou=iou)

    def test_invalid_label_type(self):
        with pytest.raises(ValueError, match="expected a class id"):
            LabelMap({1.5: 0})  # type: ignore[dict-item]  # wrong type is the point

    def test_collapse_requires_into_for_sequence(self):
        with pytest.raises(ValueError, match="`into` is required"):
            LabelMap.collapse(['feeding'])

    def test_collapse_rejects_into_with_dict(self):
        with pytest.raises(ValueError, match="must not be given"):
            LabelMap.collapse({'feeding': 'mouse'}, into='mouse')

    def test_id_annotation_with_nameless_destination_raises(self):
        """Mapping a YOLO id to a bare name is unresolvable without class_names."""
        label_map = LabelMap({1: 'mouse'})
        with pytest.raises(ValueError, match="no numeric class id"):
            label_map.apply([make_yolo_bbox(1)])


class TestLabelMapHelpers:
    def test_map_class_id(self):
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES)
        assert [label_map.map_class_id(i) for i in range(4)] == [0, 0, 0, 3]

    def test_map_class_id_dropped(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse',
                                      class_names=CLASS_NAMES, drop_unmapped=True)
        assert label_map.map_class_id(3) is None

    def test_map_class_name(self):
        label_map = LabelMap.collapse(['feeding'], into='mouse', class_names=CLASS_NAMES)
        assert label_map.map_class_name('feeding') == 'mouse'
        assert label_map.map_class_name('food') == 'food'


@pytest.fixture
def yolo_dataset(tmp_path):
    """A YOLO dataset with mouse/feeding/climbing/food annotations."""
    dataset_root = tmp_path / "mice"
    dirs = create_dataset_structure(str(dataset_root))

    samples = [
        ("img1.jpg", "0 0.5 0.5 0.2 0.3\n1 0.5 0.5 0.2 0.3"),   # mouse + feeding, same box
        ("img2.jpg", "2 0.4 0.6 0.3 0.2\n3 0.7 0.3 0.2 0.4"),   # climbing + food
    ]
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    for name, content in samples:
        cv2.imwrite(str(Path(dirs['images']['train']) / name), dummy_image)
        label_path = Path(dirs['labels']['train']) / name.replace('.jpg', '.txt')
        label_path.write_text(content)

    return dataset_root, dirs


@pytest.fixture
def voc_dataset(tmp_path):
    """A minimal Pascal VOC dataset."""
    root = tmp_path / "mice_voc"
    images_dir = root / "JPEGImages"
    annotations_dir = root / "Annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img1.jpg"), dummy_image)

    xml = """<annotation>
  <filename>img1.jpg</filename>
  <size><width>10</width><height>10</height><depth>3</depth></size>
  <object><name>mouse</name><difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
  <object><name>feeding</name><difficult>1</difficult>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
  <object><name>food</name><difficult>0</difficult>
    <bndbox><xmin>6</xmin><ymin>6</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
</annotation>"""
    (annotations_dir / "img1.xml").write_text(xml)
    return root, images_dir, annotations_dir


class TestLoaderIntegration:
    def test_yolo_loader_applies_label_map(self, yolo_dataset):
        _, dirs = yolo_dataset
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES)
        loader = YoloDarknetLoader(
            dirs['images']['train'], dirs['labels']['train'], label_map=label_map
        )
        all_ids = sorted(b.class_id for _, _, anns in loader for b in anns)
        assert all_ids == [0, 0, 0, 3]

    def test_yolo_loader_without_label_map_is_unchanged(self, yolo_dataset):
        _, dirs = yolo_dataset
        loader = YoloDarknetLoader(dirs['images']['train'], dirs['labels']['train'])
        all_ids = sorted(b.class_id for _, _, anns in loader for b in anns)
        assert all_ids == [0, 1, 2, 3]
        assert loader.get_label_map() is None

    def test_access_paths_agree(self, yolo_dataset):
        """__next__, __getitem__ and slicing must all apply the map identically."""
        _, dirs = yolo_dataset
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES)
        loader = YoloDarknetLoader(
            dirs['images']['train'], dirs['labels']['train'], label_map=label_map
        )
        by_iteration = [[b.class_id for b in anns] for _, _, anns in loader]
        by_index = [[b.class_id for b in annotations_of(loader[i])] for i in range(len(loader))]
        by_slice = [[b.class_id for b in annotations_of(item)] for item in items_of(loader[:])]
        assert by_iteration == by_index == by_slice

    def test_repeated_access_does_not_compound(self, yolo_dataset):
        _, dirs = yolo_dataset
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES, reindex=True)
        loader = YoloDarknetLoader(
            dirs['images']['train'], dirs['labels']['train'], label_map=label_map
        )
        first = [b.class_id for b in annotations_of(loader[0])]
        for _ in range(3):
            assert [b.class_id for b in annotations_of(loader[0])] == first

    def test_set_label_map_after_construction(self, yolo_dataset):
        _, dirs = yolo_dataset
        loader = YoloDarknetLoader(dirs['images']['train'], dirs['labels']['train'])
        loader.set_label_map(LabelMap.collapse([1, 2], into=0))
        all_ids = sorted(b.class_id for _, _, anns in loader for b in anns)
        assert all_ids == [0, 0, 0, 3]

    def test_voc_loader_applies_label_map(self, voc_dataset):
        _, images_dir, annotations_dir = voc_dataset
        label_map = LabelMap.collapse(['feeding'], into='mouse')
        loader = PascalVOCLoader(images_dir, annotations_dir, label_map=label_map)
        annotations = annotations_of(loader[0])
        assert [b.class_name for b in annotations] == ['mouse', 'mouse', 'food']

    def test_coco_loader_applies_label_map(self, tmp_path):
        """The documented COCO recipe: categories straight from the loader."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        cv2.imwrite(str(images_dir / "img1.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
        annotations_file = tmp_path / "ann.json"
        annotations_file.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 10, "height": 10}],
            # Sparse, non-zero-based category ids, as COCO allows.
            "categories": [{"id": 1, "name": "mouse"}, {"id": 2, "name": "feeding"},
                           {"id": 5, "name": "food"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5]},
                {"id": 3, "image_id": 1, "category_id": 5, "bbox": [6, 6, 3, 3]},
            ],
        }))

        base = COCOLoader(images_dir, annotations_file)
        assert base.get_category_names() == {1: 'mouse', 2: 'feeding', 5: 'food'}

        label_map = LabelMap.collapse(['feeding'], into='mouse',
                                      class_names=base.get_category_names())
        loader = COCOLoader(images_dir, annotations_file, label_map=label_map)
        assert [b.class_id for b in annotations_of(loader[0])] == [1, 1, 5]

    def test_factory_forwards_label_map(self, yolo_dataset):
        dataset_root, _ = yolo_dataset
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES)
        train, _, _ = DataLoaderFactory.create(dataset_root, label_map=label_map)
        assert train is not None
        assert train.get_label_map() is label_map
        all_ids = sorted(b.class_id for _, _, anns in train for b in anns)
        assert all_ids == [0, 0, 0, 3]


class TestCollapseLabelsCLI:
    def test_parse_mapping(self):
        assert _parse_mapping(['feeding=mouse', '2=0']) == {'feeding': 'mouse', 2: 0}

    @pytest.mark.parametrize("bad", ['feeding', 'feeding=', '=mouse'])
    def test_parse_mapping_rejects_malformed(self, bad):
        with pytest.raises(ValueError, match="invalid --map entry"):
            _parse_mapping([bad])

    def test_rejects_output_equal_to_input(self, tmp_path):
        with pytest.raises(ValueError, match="must differ from input_dir"):
            _check_output_dir(tmp_path, tmp_path, force=False)

    def test_rejects_output_inside_input(self, tmp_path):
        with pytest.raises(ValueError, match="is inside input_dir"):
            _check_output_dir(tmp_path, tmp_path / "out", force=False)

    def test_rejects_output_containing_input(self, tmp_path):
        inner = tmp_path / "inner"
        inner.mkdir()
        with pytest.raises(ValueError, match="contains input_dir"):
            _check_output_dir(inner, tmp_path, force=False)

    def test_rejects_non_empty_output(self, tmp_path):
        source = tmp_path / "src"
        destination = tmp_path / "dst"
        source.mkdir()
        destination.mkdir()
        (destination / "existing.txt").write_text("keep me")
        with pytest.raises(ValueError, match="--force"):
            _check_output_dir(source, destination, force=False)

    def test_force_clears_non_empty_output(self, tmp_path):
        source = tmp_path / "src"
        destination = tmp_path / "dst"
        source.mkdir()
        destination.mkdir()
        (destination / "existing.txt").write_text("clobber me")
        _check_output_dir(source, destination, force=True)
        assert not destination.exists()

    def test_yolo_end_to_end_leaves_input_untouched(self, yolo_dataset, tmp_path):
        dataset_root, dirs = yolo_dataset
        original = Path(dirs['labels']['train'] + "/img1.txt").read_text()

        output = tmp_path / "collapsed"
        label_map = LabelMap.collapse(['feeding', 'climbing'], into='mouse',
                                      class_names=CLASS_NAMES)
        fmt = collapse_dataset(dataset_root, output, label_map)

        assert fmt == 'yolo'
        assert Path(dirs['labels']['train'] + "/img1.txt").read_text() == original
        assert (output / "labels" / "train" / "img1.txt").read_text().split("\n")[0].startswith("0 ")

        loader = YoloDarknetLoader(output / "images" / "train", output / "labels" / "train")
        assert sorted(b.class_id for _, _, anns in loader for b in anns) == [0, 0, 0, 3]

    def test_yolo_rewrite_preserves_coordinate_text(self, yolo_dataset):
        _, dirs = yolo_dataset
        labels_dir = Path(dirs['labels']['train'])
        collapse_yolo_labels(labels_dir, LabelMap.collapse([1], into=0))
        line = (labels_dir / "img1.txt").read_text().splitlines()[1]
        assert line == "0 0.5 0.5 0.2 0.3"

    def test_yolo_rewrite_dedup(self, yolo_dataset):
        _, dirs = yolo_dataset
        labels_dir = Path(dirs['labels']['train'])
        label_map = LabelMap.collapse(['feeding'], into='mouse',
                                      class_names=CLASS_NAMES, dedup_iou=0.9)
        collapse_yolo_labels(labels_dir, label_map)
        assert len((labels_dir / "img1.txt").read_text().strip().splitlines()) == 1

    def test_voc_rewrite_preserves_unparsed_tags(self, voc_dataset):
        _, _, annotations_dir = voc_dataset
        collapse_voc_annotations(annotations_dir, LabelMap.collapse(['feeding'], into='mouse'))

        root = ET.parse(annotations_dir / "img1.xml").getroot()
        assert object_names(root) == ['mouse', 'mouse', 'food']
        # The <difficult> tag is not parsed by read_annotation and must survive.
        assert object_tags(root, 'difficult') == ['0', '1', '0']

    def test_voc_rewrite_drop_unmapped(self, voc_dataset):
        _, _, annotations_dir = voc_dataset
        label_map = LabelMap.collapse(['feeding'], into='mouse', drop_unmapped=True)
        collapse_voc_annotations(annotations_dir, label_map)
        root = ET.parse(annotations_dir / "img1.xml").getroot()
        # mouse (destination) and feeding (source) survive; food is dropped.
        assert object_names(root) == ['mouse', 'mouse']

    def test_coco_rewrite_preserves_unparsed_fields(self, tmp_path):
        annotations_file = tmp_path / "instances.json"
        annotations_file.write_text(json.dumps({
            "info": {"description": "mice"},
            "licenses": [{"id": 1, "name": "CC"}],
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 10, "height": 10}],
            "categories": [
                {"id": 1, "name": "mouse", "supercategory": "animal"},
                {"id": 2, "name": "feeding", "supercategory": "behaviour"},
                {"id": 3, "name": "food", "supercategory": "object"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5],
                 "iscrowd": 0, "area": 25, "segmentation": [[0, 0, 5, 0, 5, 5]]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5],
                 "iscrowd": 0, "area": 25, "segmentation": [[0, 0, 5, 0, 5, 5]]},
                {"id": 3, "image_id": 1, "category_id": 3, "bbox": [6, 6, 3, 3],
                 "iscrowd": 0, "area": 9, "segmentation": []},
            ],
        }))

        remapped, dropped = collapse_coco_annotations(
            annotations_file, LabelMap.collapse(['feeding'], into='mouse')
        )
        data = json.loads(annotations_file.read_text())

        assert (remapped, dropped) == (1, 0)
        assert [c['name'] for c in data['categories']] == ['mouse', 'food']
        assert [a['category_id'] for a in data['annotations']] == [1, 1, 3]
        # Fields the loaders never parse must survive the round trip.
        assert data['info'] == {"description": "mice"}
        assert data['licenses'] == [{"id": 1, "name": "CC"}]
        assert data['annotations'][0]['segmentation'] == [[0, 0, 5, 0, 5, 5]]
        assert data['categories'][0]['supercategory'] == 'animal'

    def test_coco_rewrite_drop_unmapped(self, tmp_path):
        annotations_file = tmp_path / "instances.json"
        annotations_file.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "img1.jpg", "width": 10, "height": 10}],
            "categories": [
                {"id": 1, "name": "mouse"},
                {"id": 2, "name": "feeding"},
                {"id": 3, "name": "food"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5]},
                {"id": 2, "image_id": 1, "category_id": 3, "bbox": [6, 6, 3, 3]},
            ],
        }))

        label_map = LabelMap.collapse(['feeding'], into='mouse', drop_unmapped=True)
        remapped, dropped = collapse_coco_annotations(annotations_file, label_map)
        data = json.loads(annotations_file.read_text())

        assert dropped == 1
        assert [c['name'] for c in data['categories']] == ['mouse']
        assert [a['category_id'] for a in data['annotations']] == [1]
