"""Hooks: functions attached to a model that observe one processing stage from the inside.

Hooks and callbacks answer different questions:

- Callbacks (`saltup.ai.training.callbacks`) follow a *loop*. They are passed to a
  training run, fire on loop events (train begin, epoch end, train end) and receive
  aggregated values such as losses and metrics. They watch the model from the outside.
- Hooks follow a *processing stage*. They are registered on a model object, fire every
  time that stage runs (for example the post-process of one image) and receive the
  stage's actual inputs, outputs and the internal values the stage chose to expose.
  They watch the model from the inside, and stay attached until removed through the
  `HookHandle` returned at registration.

The design follows PyTorch module hooks (`register_forward_hook` returns a removable
handle, hooks run in registration order, `prepend` puts one first), with two
differences: events are validated by the model that owns them, so a misspelled event
fails at registration instead of never firing, and hooks cannot replace the stage
output, they can only return extra values.
"""
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple


class HookHandle:
    """Receipt of a hook registration, used only to remove that hook.

    It does not carry values: the model calls its hooks and passes them the values.
    It is also a context manager that removes the hook on exit:

        with model.register_hook(event, hook):
            model.run(image)
    """

    def __init__(self, entries: List[Tuple[object, Callable[..., Any]]], token: object) -> None:
        self._entries = entries
        self._token = token

    def remove(self) -> None:
        """Detach the hook. Calling it again does nothing."""
        # Match by token, not by function, so the same function registered twice
        # gets two handles and each one removes only its own registration
        self._entries[:] = [entry for entry in self._entries if entry[0] is not self._token]

    def __enter__(self) -> "HookHandle":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.remove()


class HookRegistry:
    """Hooks of one model, grouped by event, in call order.

    The owner (e.g. `BaseYolo`) validates event names before using the registry.
    """

    def __init__(self) -> None:
        self._entries: Dict[Hashable, List[Tuple[object, Callable[..., Any]]]] = {}

    def register(self, event: Hashable, hook: Callable[..., Any], prepend: bool = False) -> HookHandle:
        """Add `hook` to `event`, last by default or first with `prepend=True`."""
        if not callable(hook):
            raise TypeError(f"hook must be callable, got {type(hook).__name__}")
        entries = self._entries.setdefault(event, [])
        token = object()
        entries.insert(0 if prepend else len(entries), (token, hook))
        return HookHandle(entries, token)

    def get(self, event: Hashable) -> List[Callable[..., Any]]:
        """Hooks registered on `event`, in call order (empty list if none)."""
        return [hook for _, hook in self._entries.get(event, [])]


def merge_hook_results(results: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Merge the dictionaries returned by hooks; `None` means "nothing to add".

    Raises:
        TypeError: a hook returned something other than a dict or None.
        ValueError: two hooks returned the same key (nothing is overwritten silently).
    """
    merged: Dict[str, Any] = {}
    for result in results:
        if result is None:
            continue
        if not isinstance(result, dict):
            raise TypeError(f"a hook must return a dict or None, got {type(result).__name__}")
        duplicated = merged.keys() & result.keys()
        if duplicated:
            raise ValueError(f"hooks returned the same metadata keys: {sorted(duplicated)}")
        merged.update(result)
    return merged
