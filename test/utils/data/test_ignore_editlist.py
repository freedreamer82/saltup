"""
Equivalence test: decoding with ``ignore_edit_list=True`` must yield the same
frames as decoding a copy whose ``moov`` edit list was byte-neutralized by
``Mp4EditListPatcher``.

This guarantees the two edit-list neutralization mechanisms agree:
  * serving time (browser)  -> Mp4EditListPatcher rewrites the moov bytes
  * export/decode time      -> FFmpeg's ``ignore_editlist`` demuxer flag

so a frame index annotated in the browser maps to the same picture the
extraction task pulls.

Requires a local ``ffmpeg`` binary to synthesize an MP4 carrying an edit list.
The test skips (never fails) when ffmpeg is unavailable or the available
encoder does not emit an edit list we can neutralize.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from saltup.utils.data.video.video_utils import process_video, get_video_properties, VideoReadOptions
from saltup.utils.data.video.mp4_edit_list import Mp4EditListPatcher

_FFMPEG = shutil.which("ffmpeg")


def _make_editlist_mp4(path: Path) -> None:
    """Encode a short clip with B-frames so the muxer writes an edit list."""
    cmd = [
        _FFMPEG, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=3",
        "-c:v", "libx264", "-bf", "2", "-g", "15", "-pix_fmt", "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True)


def _collect_frames(video_input, frame_numbers, ignore_edit_list):
    """Decode the given frame numbers, returning {frame_number: ndarray}."""
    collected = {}

    def _cb(image, frame_number, _total, _metadata=None):
        collected[frame_number] = image.get_data().copy()
        return image

    process_video(
        video_input=str(video_input),
        callback=_cb,
        frame_numbers=list(frame_numbers),
        options=VideoReadOptions(ignore_edit_list=ignore_edit_list),
    )
    return collected


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg binary not available")
def test_ignore_editlist_matches_byte_patched_decode(tmp_path):
    original = tmp_path / "editlist.mp4"
    try:
        _make_editlist_mp4(original)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("could not synthesize an MP4 (encoder unavailable)")

    raw = original.read_bytes()

    # The fixture must actually carry a neutralizable edit list, else there is
    # nothing to compare and the test would be vacuous.
    patched = Mp4EditListPatcher.patch_bytes(raw)
    if patched is None:
        pytest.skip("this ffmpeg build did not emit a neutralizable edit list")

    # Same byte size and offsets (header-only rewrite) — sanity check.
    assert len(patched) == len(raw)

    patched_file = tmp_path / "patched.mp4"
    patched_file.write_bytes(patched)

    # (a) reported frame count must agree on both timelines.
    props_flag = get_video_properties(str(original), options=VideoReadOptions(ignore_edit_list=True))
    props_patch = get_video_properties(str(patched_file))
    assert props_flag.total_frames == props_patch.total_frames, (
        f"total_frames differ: flag={props_flag.total_frames} patched={props_patch.total_frames}"
    )
    total = props_flag.total_frames

    # (b) selected indices must decode to identical pictures.
    frame_numbers = [n for n in (0, 1, 5, 10, 20) if n < total]
    assert frame_numbers, "fixture has too few frames"
    via_flag = _collect_frames(original, frame_numbers, ignore_edit_list=True)
    via_patch = _collect_frames(patched_file, frame_numbers, ignore_edit_list=False)
    assert set(via_flag) == set(via_patch) == set(frame_numbers)
    for n in frame_numbers:
        assert np.array_equal(via_flag[n], via_patch[n]), (
            f"frame {n} differs between ignore_editlist and byte-patched decode"
        )

    # (c) a full sequential decode must yield the same number of frames AND the
    # same pixels for every one — guards against divergence at the tail.
    seq_flag = _collect_frames(original, range(total), ignore_edit_list=True)
    seq_patch = _collect_frames(patched_file, range(total), ignore_edit_list=False)
    assert len(seq_flag) == len(seq_patch), (
        f"decoded frame count differs: flag={len(seq_flag)} patched={len(seq_patch)}"
    )
    for n in seq_flag:
        assert np.array_equal(seq_flag[n], seq_patch[n]), f"frame {n} differs in full decode"
