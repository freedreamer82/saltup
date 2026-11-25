import pytest

from saltup.ai.hooks import HookRegistry, merge_hook_results


def test_hooks_run_in_registration_order_and_prepend_goes_first() -> None:
    registry = HookRegistry()
    first, second, early = (lambda: 1), (lambda: 2), (lambda: 0)
    registry.register("event", first)
    registry.register("event", second)
    registry.register("event", early, prepend=True)

    assert registry.get("event") == [early, first, second]
    assert registry.get("other") == []


def test_handle_removes_only_its_own_registration() -> None:
    registry = HookRegistry()
    hook = lambda: None
    handle_a = registry.register("event", hook)
    registry.register("event", hook)

    handle_a.remove()
    handle_a.remove()  # idempotent

    assert registry.get("event") == [hook]


def test_handle_as_context_manager_removes_on_exit() -> None:
    registry = HookRegistry()
    hook = lambda: None
    with registry.register("event", hook):
        assert registry.get("event") == [hook]
    assert registry.get("event") == []


def test_register_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        HookRegistry().register("event", "not a function")  # type: ignore[arg-type]


def test_merge_hook_results() -> None:
    assert merge_hook_results([]) == {}
    assert merge_hook_results([None, {"a": 1}, None, {"b": 2}]) == {"a": 1, "b": 2}

    with pytest.raises(ValueError, match="'a'"):
        merge_hook_results([{"a": 1}, {"a": 2}])
    with pytest.raises(TypeError):
        merge_hook_results([[("a", 1)]])  # type: ignore[list-item]
