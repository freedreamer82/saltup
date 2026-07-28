# Label Mapping

Collapse or rename dataset labels **at load time**, without touching the annotation files on disk.

## The problem

You have a dataset of mice annotated with both an object class and behaviour classes:

| id | name |
|----|----------|
| 0  | mouse    |
| 1  | feeding  |
| 2  | climbing |
| 3  | food     |

You want to train a detector that only knows about `mouse`. A `feeding` box *is* a mouse, and so is
a `climbing` box — but `food` is something else entirely and must stay as it is. Rather than
duplicating the dataset or rewriting label files, load it with a `LabelMap`:

```python
from saltup.ai.base_dataformat.label_map import LabelMap
from saltup.ai.object_detection.dataset.loader_factory import DataLoaderFactory

label_map = LabelMap.collapse(
    ["feeding", "climbing"], into="mouse",
    class_names=["mouse", "feeding", "climbing", "food"]
)

train_dl, val_dl, test_dl = DataLoaderFactory.create("./mice", label_map=label_map)
```

An image annotated with one `mouse` box and one `feeding` box now yields two `mouse` boxes. A `food`
box is still `food`. The files under `./mice` are never modified.

## Writing the rules

Rules are `{source: destination}` pairs. Sources and destinations may each be a **class id** or a
**class name**, and the two can be mixed.

```python
# a sequence collapsed into one destination
LabelMap.collapse(["feeding", "climbing"], into="mouse", class_names=CLASS_NAMES)

# an explicit dict, for renaming several things at once
LabelMap.collapse({"feeding": "mouse", "climbing": "mouse"}, class_names=CLASS_NAMES)

# by id -- no class_names needed
LabelMap.collapse([1, 2], into=0)

# mixed
LabelMap({1: "mouse", "climbing": 0}, class_names=CLASS_NAMES)
```

`LabelMap(mapping, ...)` and `LabelMap.collapse(...)` are the same thing; `collapse` just saves you
writing `{x: y, z: y}` by hand.

## `class_names`

Annotations reach a `LabelMap` in one of two flavours:

| Format | `class_id` | `class_name` |
|--------|-----------|--------------|
| YOLO Darknet | the integer from the label file | `""` |
| COCO | the raw `category_id` | `""` |
| Pascal VOC | `None` | the `<name>` from the XML |

A `LabelMap` matches on whichever the annotation carries. So id-based rules work on YOLO and COCO
out of the box, and name-based rules work on VOC out of the box. **`class_names` is what lets you
cross over** — name-based rules on a YOLO dataset, or id-based rules on a VOC one.

It takes either form:

```python
# a sequence indexed by class id -- YOLO's contiguous 0..n-1 ids
LabelMap.collapse(["feeding"], into="mouse", class_names=["mouse", "feeding", "climbing", "food"])

# a sparse {class_id: name} mapping -- COCO category ids need not start at 0
LabelMap.collapse(["feeding"], into="mouse", class_names={1: "mouse", 2: "feeding", 5: "food"})
```

For COCO, take the categories straight from the loader:

```python
from saltup.ai.object_detection.dataset.coco import COCOLoader

base = COCOLoader(images_dir, annotations_file)
label_map = LabelMap.collapse(["feeding"], into="mouse",
                              class_names=base.get_category_names())
loader = COCOLoader(images_dir, annotations_file, label_map=label_map)
```

`class_names` is also validated: a rule naming a class that does not exist raises `ValueError` when
the `LabelMap` is built, not silently mid-epoch.

## Options

### `drop_unmapped` — discard everything else

By default, labels you did not name are passed through unchanged. Set `drop_unmapped=True` to keep
only the classes involved in the mapping:

```python
label_map = LabelMap.collapse(["feeding"], into="mouse",
                              class_names=CLASS_NAMES, drop_unmapped=True)
# mouse -> mouse, feeding -> mouse, climbing dropped, food dropped
```

Destination classes are always kept: collapsing `feeding` into `mouse` never discards `mouse`
itself, even though `mouse` is not a *source* of any rule.

### `reindex` — renumber the surviving classes

Collapsing leaves gaps. `{0,1,2,3}` collapsed to `{mouse, food}` gives ids `{0, 3}` — fine for
inspection, but a datagenerator allocates `5 + num_classes` and indexes it by `class_id`, so the
gap breaks training. `reindex=True` compacts the ids:

```python
label_map = LabelMap.collapse(["feeding", "climbing"], into="mouse",
                              class_names=CLASS_NAMES, reindex=True)

label_map.get_class_mapping()   # {0: 0, 1: 0, 2: 0, 3: 1}
label_map.get_class_names()     # ['mouse', 'food']
label_map.get_num_classes()     # 2
```

> ⚠️ Reindexing changes class ids relative to any external class list, trained checkpoint, or
> previously exported dataset. It is off by default. Print `get_class_mapping()` and keep it with
> the run whenever you enable it.

It requires `class_names` — the full set of classes cannot be inferred from the rules alone, so
`reindex=True` without it raises `ValueError`.

### `dedup_iou` — merge duplicate boxes

Collapsing `mouse` and `feeding` on the *same* animal gives you two heavily overlapping `mouse`
boxes. That is often exactly what the annotation meant, so nothing is dropped by default. When you
do want them merged, set an IoU threshold:

```python
label_map = LabelMap.collapse(["feeding", "climbing"], into="mouse",
                              class_names=CLASS_NAMES, dedup_iou=0.9)
```

Boxes are compared only against boxes of the **same resulting class**; the first one in the file
wins. `iou_type` selects the variant (`IoUType.IOU` by default, also `DIOU`, `CIOU`, `GIOU`).

## Using it with the loaders

Every loader takes `label_map` as its last keyword argument:

```python
from saltup.ai.object_detection.dataset.yolo_darknet import YoloDarknetLoader

loader = YoloDarknetLoader(images_dir, labels_dir, label_map=label_map)
```

This works for `YoloDarknetLoader`, `COCOLoader`, `PascalVOCLoader` and their S3 counterparts
`YoloDarknetS3Loader`, `COCOS3Loader`, `PascalVOCS3Loader`. `DataLoaderFactory.create()` forwards it
to whichever format it detects.

You can also attach or swap one after construction:

```python
loader.set_label_map(label_map)
loader.get_label_map()
```

The map is applied on every read, through every access path — iteration, `loader[i]`, and slicing
all agree.

### Feeding a datagenerator

`num_classes` must match the post-collapse class count, and ids must be contiguous:

```python
label_map = LabelMap.collapse(["feeding", "climbing"], into="mouse",
                              class_names=CLASS_NAMES, reindex=True)
loader = YoloDarknetLoader(images_dir, labels_dir, label_map=label_map)

datagen = AnchorsBasedDatagen(
    dataloader=loader,
    anchors=anchors,
    target_size=(416, 416),
    grid_size=(13, 13),
    num_classes=label_map.get_num_classes(),   # 2
)
```

## Inspecting a map

```python
label_map.get_class_mapping()   # {original id: final id}, None where dropped
label_map.get_class_names()     # surviving names, indexed by new id when reindexing
label_map.get_num_classes()     # distinct classes remaining
label_map.map_class_id(2)       # map one id;   None if dropped
label_map.map_class_name("feeding")  # map one name; None if dropped
```

Without `class_names` the full class set is unknown, so `get_class_mapping()` returns just the
rules you gave and `get_num_classes()` returns `None`.

## Writing a collapsed dataset to disk

For a permanent copy, use the CLI:

```bash
saltup_collapse_labels ./mice ./mice_collapsed \
    --map feeding=mouse climbing=mouse \
    --class-names mouse feeding climbing food
```

It auto-detects YOLO / COCO / Pascal VOC, copies the dataset, and rewrites the annotations in the
copy.

| Flag | Meaning |
|------|---------|
| `--map SRC=DST ...` | Rules. Tokens that parse as integers are class ids, otherwise class names. |
| `--class-names ...` | Class names indexed by class id. |
| `--drop-unmapped` | Discard labels not named in `--map`. |
| `--reindex` | Renumber surviving classes from 0. |
| `--dedup-iou IOU` | Merge overlapping same-class boxes (YOLO datasets only). |
| `--force` | Overwrite a non-empty output directory. |
| `-v` | Per-split progress. |

The tool is deliberately conservative:

- **It never writes in place.** It refuses an output directory that is the input, is nested inside
  the input, or contains the input.
- **It refuses to clobber.** A non-empty output directory is an error unless you pass `--force`.
- **It preserves what it does not understand.** Annotations are rewritten in their original
  container — the COCO json dict, the VOC XML tree, and only field 0 of a YOLO line — so
  `segmentation`, `iscrowd`, `area`, `licenses`, `info`, `pose`, `truncated` and `difficult` all
  survive, and YOLO coordinate text is preserved character-for-character.

## Guarantees

- **Input annotations are never mutated.** `apply()` shallow-copies each box and rewrites only its
  class fields. COCO caches its annotation objects, so mutation would compound across repeated
  reads; copying makes repeated access stable.
- **Concrete types survive.** A `BBoxClassIdScore` stays a `BBoxClassIdScore` with its `score`
  intact, so a map can be applied to detector output as well as ground truth.
- **The annotation flavour is preserved.** YOLO annotations keep `class_name == ""` (so
  `BBoxClassId.get_data()` keeps returning an integer), and VOC annotations keep `class_id is None`.
  A map never changes the *shape* of what a loader gives you, only the class it names.
- **Loaders without a `label_map` are completely unaffected** — the default is `None` and the whole
  path is skipped.

## Gotchas

- Mapping a **YOLO id to a bare name** with no `class_names` cannot be resolved to a number and
  raises `ValueError`. Pass `class_names` so names can be turned into ids.
- **COCO category ids are not indices.** A `class_names` *list* is indexed by id, which suits YOLO;
  for COCO pass `loader.get_category_names()` as a `{id: name}` mapping instead.
- **`reindex` and `dedup_iou` are lossy in different ways** — one changes ids, the other drops
  boxes. Both are off by default; turn them on deliberately.
- **`reindex` requires `class_names` to cover every class actually present.** An annotation whose
  class is outside `class_names` cannot be renumbered, so it raises rather than disappearing. Extend
  `class_names`, or pass `drop_unmapped=True` to discard such classes deliberately.
- **A rule only bites if its flavour matches the format.** Name rules need `class_names` to reach
  YOLO's integer ids; id rules need it to reach VOC's names. The CLI refuses the mismatch outright
  instead of copying your dataset and changing nothing.
