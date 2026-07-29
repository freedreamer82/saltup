import saltup.utils.data.video.video_utils as video_utils
from saltup.utils.data.video.video_utils import (
    parse_avi_header,
    parse_mov_header,
    parse_mp4_header,
    parse_video_header,
)


def test_parse_avi_header_reads_basic_stream_metadata():
    avih_data = bytearray(40)
    avih_data[0:4] = (40_000).to_bytes(4, "little")
    avih_data[16:20] = (250).to_bytes(4, "little")
    avih_data[32:36] = (1280).to_bytes(4, "little")
    avih_data[36:40] = (720).to_bytes(4, "little")

    header = b"RIFF" + (100).to_bytes(4, "little") + b"AVI " + b"JUNK" + b"avih" + (40).to_bytes(4, "little") + bytes(avih_data)
    result = parse_avi_header(header)

    assert result["format"] == "AVI"
    assert result["width"] == 1280
    assert result["height"] == 720
    assert result["total_frames"] == 250
    assert result["fps"] == 25.0


def test_parse_mp4_header_extracts_duration_and_dimensions():
    head = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"

    mvhd = bytearray(b"mvhd" + (b"\x00" * 40))
    mvhd[4] = 0
    mvhd[16:20] = (1000).to_bytes(4, "big")
    mvhd[20:24] = (5000).to_bytes(4, "big")

    tkhd = bytearray(b"tkhd" + (b"\x00" * 100))
    tkhd[4] = 0
    tkhd[80:84] = (1920 << 16).to_bytes(4, "big")
    tkhd[84:88] = (1080 << 16).to_bytes(4, "big")

    header = head + b"moov" + bytes(mvhd) + b"trak" + bytes(tkhd)
    result = parse_mp4_header(header)

    assert result["format"] == "MP4"
    assert result["width"] == 1920
    assert result["height"] == 1080
    assert result["duration"] == 5.0


def test_parse_mov_header_reuses_mp4_parsing_logic():
    header = b"\x00\x00\x00\x18ftypqt  " + (b"\x00" * 32)
    result = parse_mov_header(header)
    assert result["format"] == "MOV"


def test_parse_video_header_mp4_reads_moov_from_tail(tmp_path):
    mp4_path = tmp_path / "internet_style.mp4"
    head = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + (b"\x00" * 64)

    mvhd = bytearray(b"mvhd" + (b"\x00" * 40))
    mvhd[4] = 0
    mvhd[16:20] = (1000).to_bytes(4, "big")
    mvhd[20:24] = (5000).to_bytes(4, "big")

    tkhd = bytearray(b"tkhd" + (b"\x00" * 100))
    tkhd[4] = 0
    tkhd[80:84] = (1920 << 16).to_bytes(4, "big")
    tkhd[84:88] = (1080 << 16).to_bytes(4, "big")

    tail = b"moov" + bytes(mvhd) + b"trak" + bytes(tkhd)
    mp4_path.write_bytes(head + (b"\x00" * 256) + tail)

    result = parse_video_header(mp4_path)
    assert result
    assert result.format == "MP4"
    assert result.width == 1920
    assert result.height == 1080
    assert result.duration == 5.0


def test_parse_video_header_dispatches_avi(tmp_path):
    avih_data = bytearray(40)
    avih_data[0:4] = (40_000).to_bytes(4, "little")   # 25 fps
    avih_data[16:20] = (250).to_bytes(4, "little")
    avih_data[32:36] = (640).to_bytes(4, "little")
    avih_data[36:40] = (480).to_bytes(4, "little")
    payload = (
        b"RIFF" + (100).to_bytes(4, "little") + b"AVI "
        + b"JUNK" + b"avih" + (40).to_bytes(4, "little") + bytes(avih_data)
    )
    avi_path = tmp_path / "sample.avi"
    avi_path.write_bytes(payload)

    result = parse_video_header(avi_path)
    assert result
    assert result.format == "AVI"
    assert result.width == 640
    assert result.height == 480
    assert result.fps == 25.0


def test_parse_video_header_unsupported_bytes(tmp_path):
    file_path = tmp_path / "video.unknown"
    file_path.write_bytes(b"\x00\x01\x02\x03")
    result = parse_video_header(file_path)
    assert not result
    assert result.error


def test_parse_video_header_unsupported_format(tmp_path):
    # Matroska has a known extension but no parser: must fail, not raise.
    file_path = tmp_path / "clip.mkv"
    file_path.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 60)
    result = parse_video_header(file_path)
    assert not result          # clean failure, no exception
    assert result.error
    assert "mkv" in result.error.lower()


def _fast_start_mp4(width=1280, height=720, duration_s=3.0):
    head = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"
    timescale, dur = 1000, int(duration_s * 1000)
    mvhd = bytearray(b"mvhd" + b"\x00" * 40)
    mvhd[4] = 0
    mvhd[16:20] = timescale.to_bytes(4, "big")
    mvhd[20:24] = dur.to_bytes(4, "big")
    tkhd = bytearray(b"tkhd" + b"\x00" * 100)
    tkhd[4] = 0
    tkhd[80:84] = (width << 16).to_bytes(4, "big")
    tkhd[84:88] = (height << 16).to_bytes(4, "big")
    return head + b"moov" + bytes(mvhd) + b"trak" + bytes(tkhd)


def test_parse_video_header_over_http(tmp_path):
    import functools
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    (tmp_path / "v.mp4").write_bytes(_fast_start_mp4(1280, 720, 3.0))

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = HTTPServer(("127.0.0.1", 0), handler)
    server.RequestHandlerClass.log_message = lambda *a, **k: None
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = parse_video_header(f"http://127.0.0.1:{port}/v.mp4?sig=x")
        assert result
        assert result.format == "MP4"
        assert result.width == 1280
        assert result.height == 720
        assert result.duration == 3.0
    finally:
        server.shutdown()
        thread.join(timeout=5)
