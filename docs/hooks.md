# Hooks

Expose values computed **inside** a model's processing stages, without changing what the model
returns.

## Hooks or callbacks?

Saltup has both, and they answer different questions.

| | Callbacks | Hooks |
|---|---|---|
| Attached to | a **loop**: the training run | a **processing stage** of a model object, e.g. the post-process |
| Fire on | loop events: train begin, epoch end, train end | every time the stage runs, e.g. once per image |
| Receive | aggregated values: losses, metrics (`CallbackContext`) | the stage's actual input, output and internal values |
| Lifetime | passed to `training(...)`, live for that run | registered on the model, live until `handle.remove()` |
| Where | `saltup.ai.training.callbacks` | `saltup.ai.hooks`, `BaseYolo.register_hook` |
| Typical use | tracking (MLflow, MQTT), checkpoints, evaluation every N epochs | debugging, analysing internal values (e.g. objectness), application metrics |

Callbacks watch the model from the outside; hooks watch it from the inside. The split follows
common practice: Keras callbacks follow the `fit`/`evaluate`/`predict` loops and receive `logs`,
while PyTorch module hooks (`register_forward_hook`) are attached to a module and receive its
inputs and outputs on every forward pass.

## Using a hook

A hook is a function registered on an event of a model. `BaseYolo.run()` calls it when the event
is reached:

```python
hook(model, inputs, outputs, extras) -> dict | None
```

- `model`: the `BaseYolo` that is running.
- `inputs`, `outputs`: input and output of the stage that just ended. For
  `YoloHookEvent.POSTPROCESS_END` they are the raw model output and the final
  `[(bbox, class_id, score), ...]`.
- `extras`: internal values the stage chose to expose (read only, possibly empty).
- return: a dict of named values to add to `YoloOutput.get_metadata()`, or `None`.

```python
from saltup.ai.object_detection.yolo.yolo import YoloHookEvent

def box_count(model, raw_output, boxes, extras):
    return {"box_count": len(boxes)}

handle = yolo.register_hook(YoloHookEvent.POSTPROCESS_END, box_count)
out = yolo.run(image)
out.get_boxes()                    # unchanged
out.get_metadata()["box_count"]
handle.remove()                    # detach

with yolo.register_hook("postprocess_end", box_count):   # detached on exit
    out = yolo.run(image)
```

Without hooks, `get_metadata()` is an empty dict and nothing else changes.

## Exposing internal values from a post-process

A post-process may return `(boxes, extras)` instead of `boxes`. `run()` keeps `boxes` as the
output and passes `extras` to the hooks, which decide what to expose:

```python
def my_postprocess(raw_output, image_height, image_width, confidence_thr, iou_threshold):
    ...
    return boxes, {"objectness": kept_objectness}     # aligned with boxes

def objectness_hook(model, raw_output, boxes, extras):
    return {"objectness": extras["objectness"]}

yolo.register_hook(YoloHookEvent.POSTPROCESS_END, objectness_hook)
out = yolo.run(image, postprocess=my_postprocess)
out.get_metadata()["objectness"]

# the same custom post-process can be used when evaluating
evaluate(yolo, dataloader, postprocess=my_postprocess)
```

The native post-processes of saltup return only `boxes`, so their output is unchanged.

## Rules

- Hooks of an event run in registration order; `register_hook(..., prepend=True)` puts one first.
- An unknown event raises `ValueError` at registration, so a typo never goes unnoticed.
- Hooks observe: they cannot change the boxes, and `extras` is read only.
- Two hooks returning the same key raise `ValueError`: nothing is overwritten silently.
- Exceptions raised inside a hook propagate to the caller of `run()`.
- Hooks run outside the measured post-processing time.
- `run(..., postprocess=False)` skips the post-process and its hooks.

## Events

| Event | `inputs` | `outputs` |
|---|---|---|
| `YoloHookEvent.POSTPROCESS_END` | raw model output | final `[(bbox, class_id, score), ...]` |

New events (e.g. after the pre-process or the inference) can be added with the same signature,
passing the input and output of their stage.
