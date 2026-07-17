"""
Unit tests for :class:`MotionDetectionOptions.from_gui` — specifically the
relationship between ``move_threshold`` and the ``grid_quadrants`` parameter.

Why these tests exist
---------------------
The docstring of :meth:`MotionDetectionOptions.from_gui` promises that *"Absolute
detection sensitivity is preserved across grid ... changes: ``move_threshold`` is
rescaled ... by the cell size ... so the same real motion still triggers."*

The movement signal is built from the **per-cell normalised** centroid
(``cnorm = (ex/qw, ey/qh)`` in ``motion_detection``, video_utils.py:907-963), not
from absolute pixel coordinates.  Because of that normalisation:

  * a horizontal image-space movement of fraction ``F`` of the *image width*
    produces a per-cell normalised x-shift of ``F / (cell_width_fraction)`` =
    ``F * cols``  (a cell spans ``1/cols`` of the image width);
  * the peak signal for that pure-horizontal move is therefore ``100 * F * cols``;
  * detection triggers when ``signal > move_threshold``
    -> at image-fraction ``F_trigger = move_threshold / (100 * cols)``.

So for the *same* absolute image-space movement to trigger regardless of grid,
``move_threshold`` must scale **in proportion to the number of grid columns**
(``cols`` returned by :func:`_compute_grid`).  That is exactly the scenario the
user described: with grid=4 a threshold of 50 triggers once an object reaches the
first quarter of the image, so with grid=8 (twice as many columns) the threshold
must be 100 to trigger at the same absolute position.

These tests encode that invariant.  They are pure unit tests — no video decoding,
no network.
"""

import pytest

from saltup.utils.data.video.video_utils import (
    MotionDetectionOptions,
    _compute_grid,
)


# All grids are compared against grid=4, whose column count is fixed by
# _compute_grid(4) -> (rows=2, cols=2).
_BASE_COLS = _compute_grid(4)[1]


def _threshold_ratio_for(n_quadrants: int) -> float:
    """move_threshold(grid=n) / move_threshold(grid=4) from from_gui."""
    base = MotionDetectionOptions.from_gui(grid_quadrants=4)
    other = MotionDetectionOptions.from_gui(grid_quadrants=n_quadrants)
    return other.move_threshold / base.move_threshold


def _trigger_image_fraction(n_quadrants: int) -> float:
    """Image-width fraction an object must traverse to trigger, derived from the
    actual signal geometry (centroid normalised per cell, pure horizontal move):
    ``F_trigger = move_threshold / (100 * cols)``."""
    opts = MotionDetectionOptions.from_gui(grid_quadrants=n_quadrants)
    _, cols = _compute_grid(n_quadrants)
    return opts.move_threshold / (100.0 * cols)


@pytest.mark.parametrize("n_quadrants", [2, 4, 6, 8, 9, 12, 16])
def test_threshold_scales_with_grid_columns(n_quadrants: int):
    """``move_threshold`` must scale with the number of grid columns so the same
    absolute image-space horizontal movement triggers regardless of the grid.

    Expected ratio = cols(grid=n) / cols(grid=4).
    """
    _, cols = _compute_grid(n_quadrants)
    expected_ratio = cols / _BASE_COLS
    actual_ratio = _threshold_ratio_for(n_quadrants)
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6), (
        f"grid={n_quadrants}: expected move_threshold ratio {expected_ratio:.4f} "
        f"(cols {cols}/{_BASE_COLS}) to preserve absolute sensitivity, "
        f"got {actual_ratio:.4f}"
    )


def test_grid8_threshold_is_twice_grid4():
    """Mirrors the user's concrete scenario: grid=4 with threshold 50 triggers at
    the first quarter of the image; grid=8 (double the columns) must require
    threshold 100 — i.e. a 2x ratio — to trigger at the same absolute position.
    """
    ratio = _threshold_ratio_for(8)
    assert ratio == pytest.approx(2.0, rel=1e-6), (
        f"grid=8 should need 2x the grid=4 threshold to preserve absolute "
        f"sensitivity, got ratio {ratio:.4f}"
    )


def test_absolute_image_movement_trigger_is_grid_invariant():
    """The image-width fraction an object must cross to trigger must be identical
    across grids (the whole point of 'absolute sensitivity preserved')."""
    grids = [4, 6, 8, 16]
    trigger_fractions = [_trigger_image_fraction(n) for n in grids]
    lo, hi = min(trigger_fractions), max(trigger_fractions)
    assert hi == pytest.approx(lo, rel=1e-6), (
        f"absolute trigger image-fraction is not grid-invariant: {trigger_fractions}"
    )
