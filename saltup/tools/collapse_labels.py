#!/usr/bin/env python3
"""
CLI tool to collapse/rename labels of a dataset into a new copy on disk.

Unlike `saltup_yolo_replace_label_class`, this never edits the input dataset: it
copies the whole dataset tree to an output directory and rewrites only the
annotation files there. Annotations are rewritten in place in their original
container -- the COCO json dict, the VOC XML tree, the YOLO text lines -- so
fields the loaders do not parse (segmentation, iscrowd, licenses, pose,
truncated, difficult, ...) are preserved.

Example:
    saltup_collapse_labels ./mice ./mice_collapsed \\
        --map feeding=mouse climbing=mouse \\
        --class-names mouse feeding climbing food
"""
import argparse
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from saltup.utils import configure_logging
from saltup.ai.base_dataformat.label_map import LabelMap
from saltup.ai.object_detection.utils.bbox import BBoxClassId, BBoxFormat
from saltup.ai.object_detection.dataset.yolo_darknet import (
    is_yolo_darknet_dataset,
    get_dataset_paths as get_yolo_paths,
)
from saltup.ai.object_detection.dataset.coco import (
    is_coco_dataset,
    get_dataset_paths as get_coco_paths,
)
from saltup.ai.object_detection.dataset.pascal_voc import (
    is_pascal_voc_dataset,
    get_dataset_paths as get_voc_paths,
)


def _parse_label(token: str):
    """Interpret a CLI label token as an int class id when it looks like one."""
    try:
        return int(token)
    except ValueError:
        return token


def _parse_mapping(pairs: List[str]) -> Dict:
    """Parse `--map src=dst` tokens into a mapping dict."""
    mapping = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f"invalid --map entry {pair!r}, expected 'source=destination'")
        source, destination = pair.split('=', 1)
        if not source or not destination:
            raise ValueError(f"invalid --map entry {pair!r}, expected 'source=destination'")
        mapping[_parse_label(source)] = _parse_label(destination)
    return mapping


def _check_output_dir(input_dir: Path, output_dir: Path, force: bool) -> None:
    """Refuse output locations that would corrupt or silently clobber data.

    Raises:
        ValueError: If the output directory is the input directory, is nested
            inside it, or already exists with content and `force` is not set.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if output_dir == input_dir:
        raise ValueError(
            f"output_dir must differ from input_dir: {output_dir}. This tool never "
            f"rewrites a dataset in place."
        )
    if input_dir in output_dir.parents:
        raise ValueError(
            f"output_dir {output_dir} is inside input_dir {input_dir}, which would "
            f"corrupt the source dataset while copying"
        )
    if output_dir in input_dir.parents:
        raise ValueError(
            f"output_dir {output_dir} contains input_dir {input_dir}; refusing to write"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise ValueError(
                f"output_dir {output_dir} already exists and is not empty. "
                f"Pass --force to overwrite it."
            )
        shutil.rmtree(output_dir)


# Reference dimensions used to compare normalized YOLO boxes. IoU is scale
# invariant, so any consistent value works.
_IOU_REFERENCE_SIZE = 1000


def _parse_class_id(components: List[str]) -> Optional[int]:
    """Class id of a YOLO label line, or None if the line is not an annotation.

    A line qualifies only if it has at least five fields, an integer class id and
    four numeric coordinates -- anything else is passed through untouched.
    """
    if len(components) < 5:
        return None
    try:
        class_id = int(components[0])
        for coordinate in components[1:5]:
            float(coordinate)
    except ValueError:
        return None
    return class_id


class _YoloLine(NamedTuple):
    """One line of a YOLO label file, ready to be written back.

    `class_id` is None for a line that is not a valid annotation; such a line is
    preserved verbatim and never deduplicated.
    """
    class_id: Optional[int]
    text: str
    coordinates: Optional[List[str]] = None


def collapse_yolo_labels(labels_dir: Path, label_map: LabelMap) -> Tuple[int, int]:
    """Rewrite class ids in the YOLO label files of a directory.

    Only the class id field is rewritten; the coordinate text of each surviving
    line is preserved verbatim, so no precision is lost to a parse/format round
    trip. A line that is not a valid annotation is passed through untouched
    rather than discarded.

    Args:
        labels_dir: Directory of `.txt` label files to rewrite in place.
        label_map: The mapping to apply.

    Returns:
        Tuple of (files modified, annotations dropped).
    """
    logger = configure_logging.get_logger(__name__)
    modified = 0
    dropped = 0

    for label_file in sorted(labels_dir.rglob('*.txt')):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        kept: List[_YoloLine] = []
        changed = False
        for line in lines:
            components = line.split()
            old_id = _parse_class_id(components)
            if old_id is None:
                if components:      # not a blank line: keep it, don't silently drop
                    logger.warning(
                        f"{label_file}: keeping unparseable line {line.strip()!r}"
                    )
                    kept.append(_YoloLine(None, line.rstrip('\n')))
                continue
            new_id = label_map.map_class_id(old_id)
            if new_id is None:
                dropped += 1
                changed = True
                continue
            if new_id != old_id:
                changed = True
            components[0] = str(new_id)
            kept.append(_YoloLine(new_id, " ".join(components), components[1:5]))

        if label_map.dedup_iou is not None:
            before = len(kept)
            kept = _dedup_yolo_lines(kept, label_map, label_map.dedup_iou)
            if len(kept) != before:
                dropped += before - len(kept)
                changed = True

        if changed:
            with open(label_file, 'w') as f:
                f.writelines(entry.text + "\n" for entry in kept)
            modified += 1

    return modified, dropped


def _dedup_yolo_lines(
    lines: List['_YoloLine'],
    label_map: LabelMap,
    threshold: float
) -> List['_YoloLine']:
    """Drop same-class YOLO lines overlapping an already-kept line.

    Lines that are not annotations (`class_id is None`) are passed through and
    never compared.

    Args:
        lines: Surviving lines for one label file.
        label_map: Supplies the IoU variant to compare with.
        threshold: IoU at or above which a line is considered a duplicate.
    """
    def as_box(entry: '_YoloLine') -> Optional[BBoxClassId]:
        if entry.class_id is None or entry.coordinates is None:
            return None
        return BBoxClassId(
            coordinates=[float(c) for c in entry.coordinates],
            class_id=entry.class_id,
            class_name="",
            fmt=BBoxFormat.YOLO,
            img_width=_IOU_REFERENCE_SIZE,
            img_height=_IOU_REFERENCE_SIZE
        )

    kept: List[_YoloLine] = []
    kept_boxes: List[BBoxClassId] = []
    for entry in lines:
        box = as_box(entry)
        if box is not None and any(
            other.class_id == box.class_id
            and box.compute_iou(other, label_map.iou_type) >= threshold
            for other in kept_boxes
        ):
            continue
        kept.append(entry)
        if box is not None:
            kept_boxes.append(box)

    return kept


def collapse_voc_annotations(annotations_dir: Path, label_map: LabelMap) -> Tuple[int, int]:
    """Rewrite object names in the Pascal VOC XML files of a directory.

    The XML tree is mutated and rewritten, so tags this project does not parse
    (pose, truncated, difficult, ...) survive untouched.

    Args:
        annotations_dir: Directory of `.xml` files to rewrite in place.
        label_map: The mapping to apply.

    Returns:
        Tuple of (files modified, annotations dropped).
    """
    modified = 0
    dropped = 0

    for annotation_file in sorted(annotations_dir.rglob('*.xml')):
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        changed = False
        for obj in list(root.findall('object')):
            name_elem = obj.find('name')
            if name_elem is None or name_elem.text is None:
                continue
            new_name = label_map.map_class_name(name_elem.text)
            if new_name is None:
                root.remove(obj)
                dropped += 1
                changed = True
            elif new_name != name_elem.text:
                name_elem.text = new_name
                changed = True

        if changed:
            tree.write(annotation_file, encoding='utf-8', xml_declaration=False)
            modified += 1

    return modified, dropped


def collapse_coco_annotations(annotations_file: Path, label_map: LabelMap) -> Tuple[int, int]:
    """Rewrite category assignments in a COCO annotations JSON file.

    Categories are matched by name, since COCO category ids are arbitrary. The
    loaded json dict is edited and rewritten, so `segmentation`, `iscrowd`,
    `licenses`, `info` and any other fields are preserved.

    Args:
        annotations_file: COCO json file to rewrite in place.
        label_map: The mapping to apply.

    Returns:
        Tuple of (annotations remapped, annotations dropped).
    """
    raw = annotations_file.read_text()
    data = json.loads(raw)

    categories = data.get('categories', [])
    id_to_name = {cat['id']: cat['name'] for cat in categories}
    name_to_category = {cat['name']: cat for cat in categories}

    # Resolve every original category to the category it collapses into. Name
    # rules are used when the mapping has any, otherwise the rules are keyed by
    # category id -- which is what a bare `--map 4=1` means for COCO.
    resolved: Dict[int, Optional[str]] = {}
    for cat in categories:
        if label_map.matches_names:
            resolved[cat['id']] = label_map.map_class_name(cat['name'])
            continue
        destination_id = label_map.map_class_id(cat['id'])
        if destination_id is None:
            resolved[cat['id']] = None
        elif destination_id in id_to_name:
            resolved[cat['id']] = id_to_name[destination_id]
        else:
            raise ValueError(
                f"category id {cat['id']} maps to {destination_id}, which is not a "
                f"category of {annotations_file.name} (ids {sorted(id_to_name)})"
            )

    # Order surviving categories by the lowest original id that maps to them, so
    # that --reindex renumbers the same way LabelMap does.
    lowest_id: Dict[str, int] = {}
    for cat in categories:
        new_name = resolved[cat['id']]
        if new_name is not None:
            lowest_id.setdefault(new_name, cat['id'])
    surviving_names: List[str] = sorted(lowest_id, key=lambda name: lowest_id[name])

    # Keep the original category dicts (supercategory and friends) where possible.
    new_categories: List[Dict[str, Any]] = []
    name_to_new_id: Dict[str, int] = {}
    for position, name in enumerate(surviving_names):
        original = name_to_category.get(name)
        new_id = position if label_map.reindex else (original['id'] if original else position)
        category: Dict[str, Any] = dict(original) if original else {'name': name}
        category['id'] = new_id
        category['name'] = name
        new_categories.append(category)
        name_to_new_id[name] = new_id

    remapped = 0
    dropped = 0
    new_annotations = []
    for annotation in data.get('annotations', []):
        old_name = id_to_name.get(annotation['category_id'])
        new_name = resolved.get(annotation['category_id']) if old_name is not None else None
        if new_name is None:
            dropped += 1
            continue
        new_id = name_to_new_id[new_name]
        if new_id != annotation['category_id']:
            annotation['category_id'] = new_id
            remapped += 1
        new_annotations.append(annotation)

    data['categories'] = new_categories
    data['annotations'] = new_annotations

    # Keep the file roughly as we found it: rewriting a pretty-printed COCO file
    # as one long line makes it unreadable and undiffable.
    indent = 2 if re.search(r'\n\s+"', raw[:4096]) else None
    with open(annotations_file, 'w') as f:
        json.dump(data, f, indent=indent)

    return remapped, dropped


def collapse_dataset(
    input_dir: Path,
    output_dir: Path,
    label_map: LabelMap,
    force: bool = False,
    verbose: bool = False
) -> str:
    """Copy a dataset and collapse its labels in the copy.

    Args:
        input_dir: Root of the source dataset. Never modified.
        output_dir: Destination root, created by copying `input_dir`.
        label_map: The mapping to apply.
        force: Overwrite `output_dir` if it exists and is not empty.
        verbose: Print per-split progress.

    Returns:
        The detected dataset format ('yolo', 'coco' or 'voc').

    Raises:
        ValueError: If the dataset format is unsupported, or the output location
            is unsafe (see `_check_output_dir`).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise ValueError(f"input_dir is not a directory: {input_dir}")

    if is_yolo_darknet_dataset(input_dir):
        fmt = 'yolo'
    elif is_coco_dataset(input_dir):
        fmt = 'coco'
    elif is_pascal_voc_dataset(input_dir):
        fmt = 'voc'
    else:
        raise ValueError(f"Unsupported or unknown dataset type in directory: {input_dir}")

    # A rule can only bite if its flavour matches how the format names a class.
    # Without this the tool would copy the dataset and change nothing, reporting
    # success.
    if fmt == 'yolo' and not label_map.matches_ids:
        raise ValueError(
            "the mapping is name-based, but YOLO Darknet labels carry only class ids. "
            "Pass --class-names so the names can be resolved to ids."
        )
    if fmt == 'voc' and not label_map.matches_names:
        raise ValueError(
            "the mapping is id-based, but Pascal VOC annotations are identified by "
            "name. Pass --class-names so the ids can be resolved to names, or write "
            "the rules with class names."
        )

    if label_map.dedup_iou is not None and fmt != 'yolo':
        print(
            f"Warning: --dedup-iou is ignored for {fmt.upper()} datasets, which are "
            f"rewritten in their original container.",
            file=sys.stderr
        )

    if fmt != 'coco':
        print(f"Class mapping: {label_map.get_class_mapping()}")
        if label_map.reindex:
            print(f"Resulting classes: {label_map.get_class_names()}")

    _check_output_dir(input_dir, output_dir, force)

    print(f"Detected {fmt.upper()} dataset, copying {input_dir} -> {output_dir}")
    shutil.copytree(input_dir, output_dir, symlinks=True)

    total_modified = 0
    total_dropped = 0

    try:
        total_modified, total_dropped = _rewrite_annotations(
            fmt, output_dir, label_map, verbose
        )
    except Exception:
        # Never leave a half-collapsed dataset behind.
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    if fmt == 'coco':
        categories = _coco_categories(output_dir)
        if categories:
            print(f"Resulting categories: {categories}")

    print(f"Done: {total_modified} {'files' if fmt != 'coco' else 'annotations'} modified, "
          f"{total_dropped} annotations dropped")

    if total_modified == 0 and total_dropped == 0:
        print(
            "Warning: nothing changed -- no annotation matched any rule. Check that "
            "the labels in --map exist in this dataset.",
            file=sys.stderr
        )

    return fmt


def _coco_categories(output_dir: Path) -> List[Tuple[int, str]]:
    """Read back the categories actually written, so we report the truth."""
    categories: List[Tuple[int, str]] = []
    for annotations_file in [get_coco_paths(output_dir)[i] for i in (1, 3, 5)]:
        if not annotations_file:
            continue
        data = json.loads(Path(annotations_file).read_text())
        for cat in data.get('categories', []):
            if (cat['id'], cat['name']) not in categories:
                categories.append((cat['id'], cat['name']))
    return categories


def _rewrite_annotations(
    fmt: str,
    output_dir: Path,
    label_map: LabelMap,
    verbose: bool
) -> Tuple[int, int]:
    """Rewrite every split of the copied dataset. Returns (modified, dropped)."""
    total_modified = 0
    total_dropped = 0

    if fmt == 'yolo':
        paths = get_yolo_paths(output_dir)
        label_dirs = [paths[1], paths[3], paths[5]]
        for label_dir in label_dirs:
            if not label_dir:
                continue
            modified, dropped = collapse_yolo_labels(Path(label_dir), label_map)
            total_modified += modified
            total_dropped += dropped
            if verbose:
                print(f"  {label_dir}: {modified} files modified, {dropped} annotations dropped")

    elif fmt == 'voc':
        paths = get_voc_paths(output_dir)
        annotation_dirs = [paths[1], paths[3], paths[5]]
        for annotation_dir in annotation_dirs:
            if not annotation_dir:
                continue
            modified, dropped = collapse_voc_annotations(Path(annotation_dir), label_map)
            total_modified += modified
            total_dropped += dropped
            if verbose:
                print(f"  {annotation_dir}: {modified} files modified, {dropped} objects dropped")

    else:
        paths = get_coco_paths(output_dir)
        annotation_files = [paths[1], paths[3], paths[5]]
        for annotation_file in annotation_files:
            if not annotation_file:
                continue
            remapped, dropped = collapse_coco_annotations(Path(annotation_file), label_map)
            total_modified += remapped
            total_dropped += dropped
            if verbose:
                print(f"  {annotation_file}: {remapped} annotations remapped, {dropped} dropped")

    return total_modified, total_dropped


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collapse or rename dataset labels into a new copy of the dataset. "
            "The input dataset is never modified. Detects YOLO Darknet, COCO and "
            "Pascal VOC layouts automatically."
        ),
        epilog=(
            "Example: saltup_collapse_labels ./mice ./mice_collapsed "
            "--map feeding=mouse climbing=mouse "
            "--class-names mouse feeding climbing food"
        )
    )
    parser.add_argument("input_dir", type=str, help="Root directory of the source dataset")
    parser.add_argument("output_dir", type=str, help="Destination directory for the collapsed copy")
    parser.add_argument(
        "--map", dest="mapping", type=str, nargs='+', required=True, metavar="SRC=DST",
        help="Mapping rules, e.g. --map feeding=mouse climbing=mouse or --map 1=0 2=0. "
             "Tokens that parse as integers are treated as class ids, otherwise as class names."
    )
    parser.add_argument(
        "--class-names", type=str, nargs='+', default=None,
        help="Class names indexed by class id. Required to apply name-based rules to "
             "YOLO datasets, and required by --reindex."
    )
    parser.add_argument(
        "--drop-unmapped", action="store_true",
        help="Discard annotations whose label is not a source in --map (default: keep them). "
             "Destination classes are always kept."
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help="Renumber surviving classes contiguously from 0. Changes class ids relative "
             "to your existing class list -- the resulting mapping is printed."
    )
    parser.add_argument(
        "--dedup-iou", type=float, default=None, metavar="IOU",
        help="Drop boxes of the same resulting class overlapping an already-kept box "
             "above this IoU. Only applied to YOLO datasets."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite output_dir if it already exists and is not empty"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable per-split output")
    return parser.parse_args()


def main():
    args = get_args()

    try:
        label_map = LabelMap(
            _parse_mapping(args.mapping),
            class_names=args.class_names,
            drop_unmapped=args.drop_unmapped,
            reindex=args.reindex,
            dedup_iou=args.dedup_iou
        )

        # The resolved class mapping is printed by collapse_dataset, which knows
        # the format and so which id space the mapping is actually in.
        print(f"Rules: {label_map.mapping}")

        collapse_dataset(
            Path(args.input_dir),
            Path(args.output_dir),
            label_map,
            force=args.force,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
