import struct

from saltup.utils.data.image.image_utils import (
	parse_gif_header,
	parse_heic_header,
	parse_image_header,
	parse_jpeg_header,
	parse_png_header,
	parse_tiff_header,
	parse_webp_header,
)


def _jpeg_bytes(width: int = 48, height: int = 32, channels: int = 3, precision: int = 8) -> bytes:
	app0 = b"\xff\xe0\x00\x10" + (b"\x00" * 14)
	sof0 = b"\xff\xc0\x00\x11" + bytes([precision])
	sof0 += struct.pack(">H", height)
	sof0 += struct.pack(">H", width)
	sof0 += bytes([channels]) + (b"\x01\x11\x00" * channels)
	return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def _png_bytes(width: int = 640, height: int = 480, bit_depth: int = 8, color_type: int = 2) -> bytes:
	signature = b"\x89PNG\r\n\x1a\n"
	ihdr_data = struct.pack(">I", width) + struct.pack(">I", height) + bytes([bit_depth, color_type, 0, 0, 0])
	return signature + struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + b"\x00\x00\x00\x00"


def _webp_vp8x_bytes(width: int = 300, height: int = 200) -> bytes:
	width_minus_one = (width - 1).to_bytes(3, "little")
	height_minus_one = (height - 1).to_bytes(3, "little")
	payload = b"\x00\x00\x00\x00" + width_minus_one + height_minus_one
	riff_size = (4 + 4 + len(payload)).to_bytes(4, "little")
	return b"RIFF" + riff_size + b"WEBP" + b"VP8X" + len(payload).to_bytes(4, "little") + payload


def _gif_bytes(width: int = 120, height: int = 90) -> bytes:
	return b"GIF89a" + struct.pack("<H", width) + struct.pack("<H", height) + b"\x80\x00\x00"


def _tiff_bytes(width: int = 256, height: int = 128, bit_depth: int = 8, channels: int = 3) -> bytes:
	header = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
	entries = [
		struct.pack("<HHI4s", 256, 4, 1, struct.pack("<I", width)),
		struct.pack("<HHI4s", 257, 4, 1, struct.pack("<I", height)),
		struct.pack("<HHI4s", 258, 3, 1, struct.pack("<H", bit_depth) + b"\x00\x00"),
		struct.pack("<HHI4s", 277, 3, 1, struct.pack("<H", channels) + b"\x00\x00"),
	]
	ifd = struct.pack("<H", len(entries)) + b"".join(entries) + struct.pack("<I", 0)
	return header + ifd


def _heic_bytes(width: int = 1920, height: int = 1080) -> bytes:
	ftyp = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heic"
	ispe = b"\x00\x00\x00\x14ispe\x00\x00\x00\x00" + struct.pack(">I", width) + struct.pack(">I", height)
	return ftyp + ispe


def test_parse_jpeg_header_success():
	data = _jpeg_bytes(width=300, height=200, channels=3, precision=8)
	result = parse_jpeg_header(data)
	assert result["format"] == "JPEG"
	assert result["width"] == 300
	assert result["height"] == 200
	assert result["channels"] == 3
	assert result["bit_depth"] == 8


def test_parse_png_header_success():
	data = _png_bytes(width=640, height=360, bit_depth=8, color_type=6)
	result = parse_png_header(data)
	assert result["format"] == "PNG"
	assert result["width"] == 640
	assert result["height"] == 360
	assert result["channels"] == 4
	assert result["color_mode"] == "RGBA"


def test_parse_webp_header_success():
	data = _webp_vp8x_bytes(width=512, height=256)
	result = parse_webp_header(data)
	assert result["format"] == "WEBP"
	assert result["width"] == 512
	assert result["height"] == 256


def test_parse_gif_header_success():
	data = _gif_bytes(width=99, height=55)
	result = parse_gif_header(data)
	assert result["format"] == "GIF"
	assert result["width"] == 99
	assert result["height"] == 55


def test_parse_tiff_header_success():
	data = _tiff_bytes(width=800, height=600, bit_depth=8, channels=3)
	result = parse_tiff_header(data)
	assert result["format"] == "TIFF"
	assert result["width"] == 800
	assert result["height"] == 600
	assert result["channels"] == 3


def test_parse_heic_header_success():
	data = _heic_bytes(width=1024, height=768)
	result = parse_heic_header(data)
	assert result["format"] == "HEIC"
	assert result["width"] == 1024
	assert result["height"] == 768


def test_parse_image_header_dispatch_for_supported_formats(tmp_path):
	samples = [
		("sample.jpg", _jpeg_bytes(400, 300), "JPEG", 400, 300),
		("sample.png", _png_bytes(320, 240), "PNG", 320, 240),
		("sample.webp", _webp_vp8x_bytes(160, 90), "WEBP", 160, 90),
		("sample.gif", _gif_bytes(88, 66), "GIF", 88, 66),
		("sample.tiff", _tiff_bytes(128, 64), "TIFF", 128, 64),
		("sample.heic", _heic_bytes(1920, 1080), "HEIC", 1920, 1080),
	]

	for filename, payload, expected_format, expected_width, expected_height in samples:
		file_path = tmp_path / filename
		file_path.write_bytes(payload)
		result = parse_image_header(file_path)
		assert result  # metadata decoded
		assert result.format == expected_format
		assert result.width == expected_width
		assert result.height == expected_height


def test_parse_image_header_dispatch_without_extension(tmp_path):
	# Magic-byte detection works even when the file has no/incorrect extension.
	file_path = tmp_path / "mystery.bin"
	file_path.write_bytes(_png_bytes(320, 240))
	result = parse_image_header(file_path)
	assert result
	assert result.format == "PNG"
	assert (result.width, result.height) == (320, 240)


def test_parse_image_header_invalid_signature(tmp_path):
	file_path = tmp_path / "sample.bmp"
	file_path.write_bytes(b"BM\x00\x00\x00\x00")
	result = parse_image_header(file_path)
	assert not result
	assert "Invalid BMP signature" in result.error


def test_parse_image_header_unsupported_format(tmp_path):
	# ICO is not handled: unknown extension + unrecognized magic -> clean failure.
	file_path = tmp_path / "favicon.ico"
	file_path.write_bytes(b"\x00\x00\x01\x00" + b"\x00" * 60)
	result = parse_image_header(file_path)
	assert not result
	assert result.error


def test_parse_image_header_over_http(tmp_path):
	import functools
	import threading
	from http.server import HTTPServer, SimpleHTTPRequestHandler

	(tmp_path / "pic.png").write_bytes(_png_bytes(512, 384))

	handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
	server = HTTPServer(("127.0.0.1", 0), handler)
	server.RequestHandlerClass.log_message = lambda *a, **k: None  # silence logs
	port = server.server_address[1]
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	try:
		# Presigned-style URL: query string, format recovered from magic bytes.
		url = f"http://127.0.0.1:{port}/pic.png?X-Amz-Signature=abc"
		result = parse_image_header(url)
		assert result
		assert result.format == "PNG"
		assert (result.width, result.height) == (512, 384)
	finally:
		server.shutdown()
		thread.join(timeout=5)
