import os
os.environ["SALTUP_BACKEND"] = "keras_tensorflow"

from typing import Any, Callable, Iterator, List, Tuple

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from saltup.ai.base_dataformat.base_dataloader import BaseDataloader
from saltup.ai.nn_model import NeuralNetworkModel
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat
from saltup.ai.object_detection.yolo.yolo import BaseYolo, YoloHookEvent, evaluate
from saltup.ai.object_detection.yolo.yolo_type import YoloType
from saltup.utils.data.image import ColorMode, Image, ImageFormat

H, W = 32, 32
BOX = BBox(img_height=H, img_width=W, coordinates=[2, 2, 12, 12], fmt=BBoxFormat.CORNERS_ABSOLUTE)


class StubYolo(BaseYolo):
    """Identity model; the native post-process returns one fixed box."""

    def __init__(self, model_path: str) -> None:
        super().__init__(YoloType.ANCHORS_BASED, NeuralNetworkModel(model_path), number_class=2)

    def get_input_info(self) -> Tuple[tuple, ColorMode, ImageFormat]:
        return (H, W, 3), ColorMode.RGB, ImageFormat.HWC

    @staticmethod
    def preprocess(
        image: Image,
        target_height: int,
        target_width: int,
        normalize_method: Callable = lambda x: x,
        apply_padding: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.zeros((1, target_height, target_width, 3), dtype=np.float32)

    def postprocess(
        self,
        raw_output: np.ndarray,
        image_height: int,
        image_width: int,
        confidence_thr: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> List[Any]:
        return [(BOX, 0, 0.9)]


@pytest.fixture
def yolo(tmp_path) -> Iterator[StubYolo]:
    tensor_in = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, H, W, 3])
    tensor_out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, H, W, 3])
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "g", [tensor_in], [tensor_out])
    path = str(tmp_path / "identity.onnx")
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)]), path)
    yield StubYolo(path)


def make_image() -> Image:
    return Image(np.zeros((H, W, 3), dtype=np.uint8), ColorMode.RGB)


def postprocess_with_extras(raw_output, image_height, image_width, confidence_thr, iou_threshold):
    return [(BOX, 1, 0.4)], {"objectness": [0.8]}


def test_without_hooks_output_is_unchanged_and_metadata_empty(yolo: StubYolo) -> None:
    out = yolo.run(make_image())
    assert out.get_boxes() == [(BOX, 0, 0.9)]
    assert out.get_metadata() == {}


def test_extras_reach_the_hook_but_not_the_boxes(yolo: StubYolo) -> None:
    seen = {}

    def hook(model, inputs, outputs, extras):
        seen.update(model=model, inputs=inputs, outputs=outputs)
        return {"objectness": extras["objectness"]}

    yolo.register_hook(YoloHookEvent.POSTPROCESS_END, hook)
    out = yolo.run(make_image(), postprocess=postprocess_with_extras)

    assert out.get_boxes() == [(BOX, 1, 0.4)]
    assert out.get_metadata() == {"objectness": [0.8]}
    assert seen["model"] is yolo
    assert seen["outputs"] == [(BOX, 1, 0.4)]
    assert isinstance(seen["inputs"], (np.ndarray, list))  # raw model output


def test_extras_are_read_only_and_empty_for_a_plain_postprocess(yolo: StubYolo) -> None:
    def hook(model, inputs, outputs, extras):
        assert dict(extras) == {}
        with pytest.raises(TypeError):
            extras["x"] = 1  # type: ignore[index]

    yolo.register_hook("postprocess_end", hook)
    yolo.run(make_image())


def test_hooks_merge_in_order_and_handle_detaches(yolo: StubYolo) -> None:
    order: List[str] = []

    def hook_a(*_):
        order.append("a")
        return {"a": 1}

    def hook_b(*_):
        order.append("b")
        return None

    handle_a = yolo.register_hook(YoloHookEvent.POSTPROCESS_END, hook_a)
    yolo.register_hook(YoloHookEvent.POSTPROCESS_END, hook_b, prepend=True)
    assert yolo.run(make_image()).get_metadata() == {"a": 1}
    assert order == ["b", "a"]

    handle_a.remove()
    assert yolo.run(make_image()).get_metadata() == {}


def test_skipped_postprocess_runs_no_hook(yolo: StubYolo) -> None:
    yolo.register_hook(YoloHookEvent.POSTPROCESS_END, lambda *_: pytest.fail("hook must not run"))
    assert yolo.run(make_image(), postprocess=False).get_metadata() == {}


def test_unknown_event_fails_at_registration(yolo: StubYolo) -> None:
    with pytest.raises(ValueError):
        yolo.register_hook("postprocess_ending", lambda *_: None)


def test_duplicate_metadata_keys_raise(yolo: StubYolo) -> None:
    yolo.register_hook(YoloHookEvent.POSTPROCESS_END, lambda *_: {"k": 1})
    yolo.register_hook(YoloHookEvent.POSTPROCESS_END, lambda *_: {"k": 2})
    with pytest.raises(ValueError):
        yolo.run(make_image())


class ListDataloader(BaseDataloader):
    def __init__(self, data: List[Any]) -> None:
        self.data = data
        self.index = 0

    def __iter__(self) -> "ListDataloader":
        self.index = 0
        return self

    def __next__(self) -> Any:
        if self.index >= len(self.data):
            raise StopIteration
        self.index += 1
        return self.data[self.index - 1]

    def __len__(self) -> int:
        return len(self.data)

    def split(self, ratio: Any) -> List[BaseDataloader]:
        return [self, self]

    def merge(dl1: BaseDataloader, dg2: BaseDataloader) -> BaseDataloader:  # type: ignore[override]
        return dl1


def test_evaluate_forwards_the_custom_postprocess(yolo: StubYolo) -> None:
    gt = BBox(img_height=H, img_width=W, coordinates=[2, 2, 12, 12], fmt=BBoxFormat.CORNERS_ABSOLUTE)
    dataloader = ListDataloader([("img.png", make_image(), [(gt, 1)])])

    native, _, _ = evaluate(yolo, dataloader, confidence_threshold=0.1)
    custom, _, _ = evaluate(yolo, dataloader, confidence_threshold=0.1, postprocess=postprocess_with_extras)

    # native predicts class 0 (wrong), the custom post-process predicts class 1 (right)
    assert native[1].getTP() == 0
    assert custom[1].getTP() == 1
