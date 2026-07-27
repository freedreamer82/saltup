"""
Label mapping / collapsing for dataset loaders.

This module lets a dataset be *loaded* with some of its labels merged into another
label, without touching the annotation files on disk. The motivating case is a
dataset that annotates both an object and its behaviours -- e.g. a `mouse` class
alongside `feeding` and `climbing` -- where a detector may only be interested in seeing `mouse`:

    >>> label_map = LabelMap.collapse(
    ...     ["feeding", "climbing"], into="mouse",
    ...     class_names=["mouse", "feeding", "climbing", "food"]
    ... )
    >>> loader = YoloDarknetLoader(images_dir, labels_dir, label_map=label_map)

Labels that are not named in the mapping are left untouched (`food` stays `food`),
unless `drop_unmapped=True` is requested.

Annotations come from the loaders in two flavours: id-carrying (YOLO, COCO) and
name-carrying (Pascal VOC, whose `class_id` is `None`). A `LabelMap` matches on
whichever the annotation provides, and writes back in the same flavour it found --
a YOLO annotation keeps `class_name == ""` so that `BBoxClassId.get_data()` keeps
returning an integer, and a VOC annotation keeps being identified by name. Pass
`class_names` to bridge the two, so that name-based rules can be applied to
id-carrying annotations and vice versa.

A `class_names` sequence is indexed by class id, which matches YOLO's contiguous
`0..n-1` id space. COCO `category_id`s are arbitrary and need not start at 0, so
for COCO pass the dataset's own categories as a sparse mapping instead:

    >>> loader = COCOLoader(images_dir, annotations_file)
    >>> label_map = LabelMap.collapse(["feeding"], into="mouse",
    ...                               class_names=loader.get_category_names())
"""

import copy
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from saltup.ai.object_detection.utils.bbox import BBoxClassId, IoUType

Label = Union[int, str]


class _Target:
    """Resolved destination of a mapping rule, as an (id, name) pair.

    Either component may be None when it cannot be resolved: a name target
    without `class_names` has no id, and an id target without `class_names`
    has no name.
    """

    __slots__ = ("class_id", "class_name", "label")

    def __init__(self, class_id: Optional[int], class_name: Optional[str], label: Label):
        self.class_id = class_id
        self.class_name = class_name
        self.label = label

    def __repr__(self):
        return f"_Target(class_id={self.class_id}, class_name={self.class_name!r})"


class LabelMap:
    """Collapse or rename labels of a dataset at load time.

    Attributes:
        mapping: The original mapping, as given by the caller.
        class_names: Optional list of class names indexed by class id, used to
            translate between name-based rules and id-based annotations.
        drop_unmapped: Whether annotations not named in the mapping are discarded.
        reindex: Whether surviving classes are renumbered contiguously from 0.
        dedup_iou: Optional IoU threshold above which two boxes that ended up in
            the same class are considered duplicates and only the first is kept.
    """

    def __init__(
        self,
        mapping: Dict[Label, Label],
        class_names: Optional[Union[Sequence[str], Mapping[int, str]]] = None,
        *,
        drop_unmapped: bool = False,
        reindex: bool = False,
        dedup_iou: Optional[float] = None,
        iou_type: IoUType = IoUType.IOU
    ):
        """Build a label map.

        Args:
            mapping: Rules as `{source_label: destination_label}`. Keys and values
                may each be a class id (int) or a class name (str), and the two may
                be mixed.
            class_names: Class names, either as a sequence indexed by class id
                (YOLO's contiguous ids) or as a sparse `{class_id: name}` mapping
                (COCO category ids, which need not start at 0 -- see
                `COCOLoader.get_category_names()`). Required to apply name-based
                rules to id-carrying annotations, and required by `reindex`.
            drop_unmapped: If True, annotations whose label is not a key of
                `mapping` are discarded. Destination classes are always kept, so
                collapsing `feeding` into `mouse` never drops `mouse` itself.
                Defaults to False, which keeps every label.
            reindex: If True, renumber the surviving classes contiguously from 0,
                so the result can be fed to a datagenerator. Requires `class_names`.
                Defaults to False, which preserves the original ids -- and therefore
                leaves gaps where classes were collapsed away.
            dedup_iou: If set, drop boxes of the same resulting class that overlap
                an already-kept box with an IoU greater than or equal to this
                threshold. Defaults to None (keep every box).
            iou_type: IoU variant used for deduplication. Defaults to IoUType.IOU.

        Raises:
            ValueError: If the mapping is empty, refers to a name absent from
                `class_names`, refers to an id outside `class_names`, if `reindex`
                is requested without `class_names`, or if `dedup_iou` is outside
                (0, 1].
        """
        if not mapping:
            raise ValueError("mapping must not be empty")
        if dedup_iou is not None and not 0.0 < dedup_iou <= 1.0:
            raise ValueError(f"dedup_iou must be in (0, 1], got {dedup_iou}")
        if reindex and class_names is None:
            raise ValueError(
                "reindex=True requires class_names: the full set of classes cannot "
                "be inferred from the mapping alone"
            )

        self.mapping = dict(mapping)
        self.class_names = class_names
        self.drop_unmapped = drop_unmapped
        self.reindex = reindex
        self.dedup_iou = dedup_iou
        self.iou_type = iou_type

        # Accept both a list indexed by class id (YOLO) and a sparse
        # {class_id: name} mapping (COCO category ids need not start at 0).
        if class_names is None:
            self._id_to_name: Dict[int, str] = {}
        elif isinstance(class_names, Mapping):
            self._id_to_name = dict(class_names)
        else:
            self._id_to_name = dict(enumerate(class_names))

        self._name_to_id: Dict[str, int] = {
            name: class_id for class_id, name in self._id_to_name.items()
        }

        # Rules split by the flavour of the key, so an annotation can be matched
        # on whichever of id / name it carries.
        self._by_id: Dict[int, _Target] = {}
        self._by_name: Dict[str, _Target] = {}
        for source, destination in self.mapping.items():
            target = self._resolve(destination, role="destination")
            source_id, source_name = self._resolve_source(source)
            if source_id is not None:
                self._by_id[source_id] = target
            if source_name is not None:
                self._by_name[source_name] = target

        # A destination class is implicitly kept: collapsing "feeding" into "mouse"
        # must never drop "mouse" itself when drop_unmapped is set. Rules given
        # explicitly take precedence, so chained rules still work.
        for destination in set(self.mapping.values()):
            target = self._resolve(destination, role="destination")
            if target.class_id is not None and target.class_id not in self._by_id:
                self._by_id[target.class_id] = target
            if target.class_name is not None and target.class_name not in self._by_name:
                self._by_name[target.class_name] = target

        self._reindex_map: Dict[int, int] = {}
        self._final_class_names: Optional[List[str]] = None
        if self.class_names is not None:
            self._build_class_mapping()


    def _resolve(self, label: Label, role: str) -> _Target:
        """Resolve a label to an (id, name) pair using `class_names` when available."""
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise ValueError(
                f"invalid {role} label {label!r}: expected a class id (int) or a class name (str)"
            )

        if isinstance(label, int):
            if self.class_names is not None and label not in self._id_to_name:
                raise ValueError(
                    f"{role} class id {label} is not a known class; class_names covers "
                    f"ids {sorted(self._id_to_name)}"
                )
            name = self._id_to_name.get(label) if self.class_names is not None else None
            return _Target(class_id=label, class_name=name, label=label)

        if self.class_names is not None:
            if label not in self._name_to_id:
                raise ValueError(
                    f"{role} class name {label!r} is not in class_names "
                    f"{self.class_names!r}; add it if you are collapsing into a new class"
                )
            return _Target(class_id=self._name_to_id[label], class_name=label, label=label)

        return _Target(class_id=None, class_name=label, label=label)

    def _resolve_source(self, label: Label) -> Tuple[Optional[int], Optional[str]]:
        """Resolve a source label to the id and/or name it should match on."""
        target = self._resolve(label, role="source")
        return target.class_id, target.class_name

    def _build_class_mapping(self) -> None:
        """Precompute, for every original class id, the class it ends up as."""
        assert self.class_names is not None

        # Original id -> post-collapse target (None marks a dropped class).
        collapsed: Dict[int, Optional[_Target]] = {}
        for class_id, name in self._id_to_name.items():
            target = self._by_id.get(class_id) or self._by_name.get(name)
            if target is not None:
                collapsed[class_id] = target
            elif self.drop_unmapped:
                collapsed[class_id] = None
            else:
                collapsed[class_id] = _Target(class_id, name, class_id)

        self._collapsed = collapsed

        if not self.reindex:
            self._final_class_names = None
            return

        # Renumber surviving classes, ordered by their lowest original id so the
        # result is stable and predictable.
        survivors: List[int] = []
        for class_id in sorted(collapsed):
            target = collapsed[class_id]
            if target is None or target.class_id is None:
                continue
            if target.class_id not in survivors:
                survivors.append(target.class_id)

        self._reindex_map = {old: new for new, old in enumerate(survivors)}
        self._final_class_names = [self._id_to_name[old] for old in survivors]

    def get_class_mapping(self) -> Dict[Label, Optional[Label]]:
        """Return the overall original-label -> final-label mapping.

        Returns:
            When `class_names` is available, a dict keyed by original class id
            whose values are the final class ids (after collapsing and, if
            enabled, reindexing), or None for classes that are dropped. Without
            `class_names`, a dict keyed by the source labels of the mapping,
            since the full set of classes is unknown.
        """
        if self.class_names is None:
            return {
                source: target.label
                for source, target in
                [(s, self._resolve(d, role="destination")) for s, d in self.mapping.items()]
            }

        result: Dict[Label, Optional[Label]] = {}
        for class_id, target in self._collapsed.items():
            if target is None or target.class_id is None:
                result[class_id] = None
            elif self.reindex:
                result[class_id] = self._reindex_map.get(target.class_id)
            else:
                result[class_id] = target.class_id
        return result

    def get_class_names(self) -> Optional[List[str]]:
        """Return the class names after collapsing, or None if not determinable.

        With `reindex=True` the returned list is indexed by the *new* class ids and
        is exactly the set of surviving classes. Without reindexing the original
        class names are returned ordered by class id -- the ids themselves are
        unchanged, so the list keeps entries for classes that were collapsed away;
        use `get_class_mapping()` to see what happened to each.
        """
        if self._final_class_names is not None:
            return list(self._final_class_names)
        if self.class_names is None:
            return None
        return [self._id_to_name[class_id] for class_id in sorted(self._id_to_name)]

    def get_num_classes(self) -> Optional[int]:
        """Return the number of distinct classes remaining after collapsing.

        Returns None when `class_names` was not provided, since the full set of
        classes is then unknown.
        """
        if self.class_names is None:
            return None
        if self.reindex:
            return len(self._final_class_names or [])
        return len({v for v in self.get_class_mapping().values() if v is not None})

    def map_class_id(self, class_id: int) -> Optional[int]:
        """Map a single class id, without going through an annotation object.

        Args:
            class_id: The original class id.

        Returns:
            The resulting class id, or None if the class is dropped.

        Raises:
            ValueError: If the destination has no numeric class id.
        """
        target = self._by_id.get(class_id)
        if target is None:
            if self.drop_unmapped:
                return None
            new_id = class_id
        else:
            if target.class_id is None:
                raise ValueError(
                    f"cannot map class id {class_id} to {target.label!r}: the destination "
                    f"has no numeric class id. Pass class_names to LabelMap."
                )
            new_id = target.class_id

        if self.reindex:
            return self._reindex_map.get(new_id)
        return new_id

    def map_class_name(self, class_name: str) -> Optional[str]:
        """Map a single class name, without going through an annotation object.

        Args:
            class_name: The original class name.

        Returns:
            The resulting class name, or None if the class is dropped.
        """
        target = self._by_name.get(class_name)
        if target is None:
            return None if self.drop_unmapped else class_name
        return target.class_name if target.class_name is not None else class_name

    def apply(self, annotations: List[BBoxClassId]) -> List[BBoxClassId]:
        """Apply the mapping to a list of annotations.

        The input annotations are never modified: each kept box is shallow-copied,
        preserving its concrete type (so a `BBoxClassIdScore` keeps its score), and
        only its class fields are rewritten. This makes `apply` safe to call
        repeatedly on annotations a loader holds in memory, as COCO does.

        Args:
            annotations: Annotations as produced by a dataloader.

        Returns:
            A new list of annotations with mapped class ids / names, unmapped
            entries dropped if `drop_unmapped` is set, and near-duplicate boxes
            removed if `dedup_iou` is set.

        Raises:
            ValueError: If an id-carrying annotation must be mapped to a
                destination that has no numeric class id.
        """
        mapped: List[BBoxClassId] = []

        for annotation in annotations:
            class_id = annotation.class_id
            class_name = annotation.class_name

            target = None
            if class_id is not None and class_id in self._by_id:
                target = self._by_id[class_id]
            elif class_name and class_name in self._by_name:
                target = self._by_name[class_name]

            if target is None:
                if self.drop_unmapped:
                    continue
                new_id, new_name = class_id, class_name
            else:
                if class_id is not None and target.class_id is None:
                    raise ValueError(
                        f"cannot map class id {class_id} to {target.label!r}: the destination "
                        f"has no numeric class id. Pass class_names to LabelMap so names can "
                        f"be resolved to ids."
                    )
                # Write back in the flavour the annotation already used, so that
                # BBoxClassId.get_data() keeps returning the same kind of value.
                new_id = target.class_id if class_id is not None else class_id
                new_name = target.class_name if class_name else class_name

            if self.reindex and new_id is not None:
                if new_id not in self._reindex_map:
                    # Only reachable when the class is dropped by drop_unmapped.
                    continue
                new_id = self._reindex_map[new_id]

            new_annotation = copy.copy(annotation)
            new_annotation.class_id = new_id
            new_annotation.class_name = new_name
            mapped.append(new_annotation)

        if self.dedup_iou is not None:
            mapped = self._deduplicate(mapped, self.dedup_iou)

        return mapped

    def _deduplicate(self, annotations: List[BBoxClassId], threshold: float) -> List[BBoxClassId]:
        """Drop boxes of the same resulting class overlapping an earlier kept box.

        BBox requires image dimensions at construction, so every annotation is
        guaranteed to be comparable here.

        Args:
            annotations: Annotations to filter, already mapped.
            threshold: IoU at or above which a box is considered a duplicate.
        """
        kept: List[BBoxClassId] = []
        for annotation in annotations:
            duplicate = any(
                (annotation.class_id, annotation.class_name) == (other.class_id, other.class_name)
                and annotation.compute_iou(other, self.iou_type) >= threshold
                for other in kept
            )
            if not duplicate:
                kept.append(annotation)
        return kept

    @classmethod
    def collapse(
        cls,
        labels: Union[Dict[Label, Label], Sequence[Label]],
        into: Optional[Label] = None,
        class_names: Optional[Union[Sequence[str], Mapping[int, str]]] = None,
        **kwargs
    ) -> 'LabelMap':
        """Build a LabelMap that collapses labels into another label.

        Args:
            labels: Either a `{source: destination}` dict, or a sequence of source
                labels to collapse into `into`.
            into: Destination label, required when `labels` is a sequence.
            class_names: Class names indexed by class id, see `__init__`.
            **kwargs: Forwarded to `__init__` (`drop_unmapped`, `reindex`,
                `dedup_iou`, `iou_type`).

        Returns:
            The configured LabelMap.

        Raises:
            ValueError: If `into` is given together with a dict, or omitted with a
                sequence.

        Example:
            >>> LabelMap.collapse(["feeding", "climbing"], into="mouse",
            ...                   class_names=["mouse", "feeding", "climbing", "food"])
        """
        if isinstance(labels, dict):
            if into is not None:
                raise ValueError("`into` must not be given when `labels` is a dict")
            mapping = dict(labels)
        else:
            if into is None:
                raise ValueError("`into` is required when `labels` is a sequence of labels")
            mapping = {label: into for label in labels}

        return cls(mapping, class_names=class_names, **kwargs)

    def __repr__(self):
        return (
            f"LabelMap(mapping={self.mapping!r}, class_names={self.class_names!r}, "
            f"drop_unmapped={self.drop_unmapped}, reindex={self.reindex}, "
            f"dedup_iou={self.dedup_iou})"
        )
