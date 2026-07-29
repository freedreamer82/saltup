from saltup.utils.data.audio.audio_utils import (
	parse_audio_header,
	parse_flac_header,
	parse_mp3_header,
	parse_wav_header,
)


def _wav_bytes(sample_rate: int = 44100, channels: int = 2, bit_depth: int = 16) -> bytes:
	byte_rate = sample_rate * channels * (bit_depth // 8)
	block_align = channels * (bit_depth // 8)
	fmt_chunk = (
		b"fmt "
		+ (16).to_bytes(4, "little")
		+ (1).to_bytes(2, "little")
		+ channels.to_bytes(2, "little")
		+ sample_rate.to_bytes(4, "little")
		+ byte_rate.to_bytes(4, "little")
		+ block_align.to_bytes(2, "little")
		+ bit_depth.to_bytes(2, "little")
	)
	data_chunk = b"data" + (0).to_bytes(4, "little")
	riff_size = (4 + len(fmt_chunk) + len(data_chunk)).to_bytes(4, "little")
	return b"RIFF" + riff_size + b"WAVE" + fmt_chunk + data_chunk


def _flac_bytes(sample_rate: int = 48000, channels: int = 2, bit_depth: int = 16, total_samples: int = 12345) -> bytes:
	block = bytearray(34)
	bits64 = (
		(sample_rate & ((1 << 20) - 1)) << 44
		| ((channels - 1) & 0x7) << 41
		| (bit_depth & 0x1F) << 36
		| (total_samples & ((1 << 36) - 1))
	)
	block[10:18] = bits64.to_bytes(8, "big")
	streaminfo_header = bytes([0x00]) + (34).to_bytes(3, "big")
	return b"fLaC" + streaminfo_header + bytes(block)


def _mp3_frame_bytes() -> bytes:
	return b"\xff\xfb\x90\x64" + (b"\x00" * 16)


def test_parse_wav_header_success():
	data = _wav_bytes(sample_rate=22050, channels=1, bit_depth=16)
	result = parse_wav_header(data)
	assert result["format"] == "WAV"
	assert result["sample_rate"] == 22050
	assert result["channels"] == 1
	assert result["bit_depth"] == 16
	assert result["audio_format"] == 1


def test_parse_flac_header_success():
	data = _flac_bytes(sample_rate=48000, channels=2, bit_depth=16, total_samples=9999)
	result = parse_flac_header(data)
	assert result["format"] == "FLAC"
	assert result["sample_rate"] == 48000
	assert result["channels"] == 2
	assert result["bit_depth"] == 16
	assert result["audio_format"] == "PCM"
	assert result["total_samples"] == 9999


def test_parse_mp3_header_success():
	data = _mp3_frame_bytes()
	result = parse_mp3_header(data)
	assert result["format"] == "MP3"
	assert result["sample_rate"] == 44100
	assert result["channels"] == 2
	assert result["audio_format"] == "PCM"
	assert result["bit_depth"] is None


def test_parse_audio_header_dispatch_supported_formats(tmp_path):
	samples = [
		("sample.wav", _wav_bytes(16000, 1, 16), "WAV", 16000, 1),
		("sample.flac", _flac_bytes(32000, 2, 16, 42), "FLAC", 32000, 2),
		("sample.mp3", _mp3_frame_bytes(), "MP3", 44100, 2),
	]

	for filename, payload, expected_format, expected_sr, expected_channels in samples:
		file_path = tmp_path / filename
		file_path.write_bytes(payload)
		result = parse_audio_header(file_path)
		assert result
		assert result.format == expected_format
		assert result.sample_rate == expected_sr
		assert result.channels == expected_channels


def test_parse_audio_header_unrecognized_bytes(tmp_path):
	file_path = tmp_path / "sample.xyz"
	file_path.write_bytes(b"not-audio")
	result = parse_audio_header(file_path)
	assert not result
	assert result.error


def test_parse_audio_header_unsupported_format(tmp_path):
	# OGG has a known extension but no parser: must fail, not raise.
	file_path = tmp_path / "song.ogg"
	file_path.write_bytes(b"OggS" + b"\x00" * 60)
	result = parse_audio_header(file_path)
	assert not result
	assert result.error
	assert "ogg" in result.error.lower()


def test_parse_audio_header_over_http(tmp_path):
	import functools
	import threading
	from http.server import HTTPServer, SimpleHTTPRequestHandler

	(tmp_path / "a.wav").write_bytes(_wav_bytes(16000, 1, 16))

	handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
	server = HTTPServer(("127.0.0.1", 0), handler)
	server.RequestHandlerClass.log_message = lambda *a, **k: None
	port = server.server_address[1]
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	try:
		result = parse_audio_header(f"http://127.0.0.1:{port}/a.wav?sig=x")
		assert result
		assert result.format == "WAV"
		assert result.sample_rate == 16000
		assert result.channels == 1
	finally:
		server.shutdown()
		thread.join(timeout=5)
