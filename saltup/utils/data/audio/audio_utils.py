"""
Audio loading utilities for the saltup library.

This module provides generic functions for loading audio data from files
and in-memory byte streams.  It supports both encoded formats (wav, mp3,
flac, …) via ``librosa`` and headerless raw PCM.

**Dependency note** — ``librosa`` is an *optional* dependency
(part of the ``saltup[full]`` extra).  Functions in this module raise
a clear ``ImportError`` at call time if ``librosa`` is not installed,
keeping the base ``saltup`` package lightweight.

Typical usage::

    from saltup.utils.data.audio.audio_utils import load_audio, load_audio_from_stream

    audio, sr = load_audio("recording.wav", sr=44100)
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy librosa import — fail gracefully at call time, not at module import.
# ---------------------------------------------------------------------------

_librosa = None


def _ensure_librosa():
    """Import librosa on first use and cache the reference. Reduce overhead for users who don't need it."""
    global _librosa
    if _librosa is None:
        try:
            import librosa
            _librosa = librosa
        except ImportError as exc:
            raise ImportError(
                "librosa is required for audio processing utilities. "
                "Install it with:  pip install saltup[full]  or  pip install librosa"
            ) from exc
    return _librosa


# =============================================================================
# Audio Loading
# =============================================================================

def load_audio(
    audio_path: str,
    sr: int = 44100,
    is_raw: bool = False,
    raw_format: str = "int16",
    n_channels: int = 1,
    downsample: bool = False,
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and return the waveform as a 1-D float array.

    Supports two modes:

    * **Encoded files** (wav, mp3, flac, …) — decoded via ``librosa.load``.
    * **Raw headerless files** — read directly from disk as a flat
      binary blob with the specified ``raw_format``.

    In both cases the returned array is mono, float32-normalised to the
    range ``[-1, 1]``.

    Args:
        audio_path: Path to the audio file.
        sr: Target sample rate in Hz.  ``librosa.load`` will resample
            automatically.  For raw files this is the *assumed* original
            sample rate.  Defaults to 44 100.
        is_raw: If ``True``, treat the file as headerless raw PCM.
            Defaults to ``False``.
        raw_format: NumPy dtype string for raw files (e.g. ``"int16"``,
            ``"int32"``, ``"float32"``).  Ignored when ``is_raw`` is
            ``False``.  Defaults to ``"int16"``.
        n_channels: Number of interleaved channels in the raw file.
            Multi-channel data is averaged to mono.  Defaults to 1.
        downsample: If ``True``, resample the audio to *target_sr*
            after loading.  Defaults to ``False``.
        target_sr: Target sample rate when *downsample* is ``True``.
            Defaults to 16 000.

    Returns:
        A tuple ``(audio_data, sample_rate)`` where *audio_data* is a
        1-D ``float32`` NumPy array and *sample_rate* is the effective
        sample rate after any downsampling.

    Raises:
        IOError: If the file cannot be read or decoded.

    Examples:
        >>> y, sr = load_audio("speech.wav", sr=16000)
        >>> y.dtype
        dtype('float32')
    """
    librosa = _ensure_librosa()

    try:
        if is_raw:
            audio_data = np.fromfile(audio_path, dtype=raw_format)

            # De-interleave multi-channel data
            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels)

            # Normalise integer formats to [-1, 1]
            audio_data = _normalise_pcm(audio_data, raw_format)

            # Mix to mono
            if n_channels > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Optional downsampling
            if downsample and target_sr < sr:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                logger.info("Downsampled raw audio to %d Hz", target_sr)

            logger.info(
                "Loaded raw audio: %d samples, %d channel(s) at %d Hz",
                len(audio_data), n_channels, sr,
            )
            return audio_data, sr

        else:
            # librosa handles mp3, wav, flac, ogg, …
            y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)

            if downsample and target_sr < sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
                logger.info("Downsampled audio to %d Hz", target_sr)

            logger.info("Loaded audio: %d samples at %d Hz", len(y), sr)
            return y, sr

    except Exception as exc:
        raise IOError(f"Cannot load audio file {audio_path}: {exc}") from exc


def load_audio_from_stream(
    stream: io.RawIOBase,
    sr: int = 44100,
    raw_format: str = "int16",
    n_channels: int = 1,
) -> Tuple[np.ndarray, int]:
    """Load raw audio from an in-memory byte stream (e.g. from MinIO / S3).

    The stream is consumed in a single ``read()`` call and interpreted as
    headerless PCM with the specified format and channel layout.

    Args:
        stream: A file-like or ``BytesIO`` object containing raw audio
            bytes.
        sr: Assumed sample rate of the raw data.  Defaults to 44 100.
        raw_format: NumPy dtype for the raw samples.  Defaults to
            ``"int16"``.
        n_channels: Number of interleaved channels.  Defaults to 1.

    Returns:
        ``(audio_data, sample_rate)`` — same convention as
        :func:`load_audio`.

    Raises:
        IOError: If the stream cannot be decoded.

    Examples:
        >>> import io
        >>> buf = io.BytesIO(b"\\x00" * 1000)
        >>> y, sr = load_audio_from_stream(buf, sr=16000)
    """
    try:
        audio_bytes = stream.read()
        audio_data = np.frombuffer(audio_bytes, dtype=raw_format)

        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)

        audio_data = _normalise_pcm(audio_data, raw_format)

        if n_channels > 1:
            audio_data = np.mean(audio_data, axis=1)

        logger.info(
            "Loaded audio from stream: %d samples at %d Hz",
            len(audio_data), sr,
        )
        return audio_data, sr

    except Exception as exc:
        raise IOError(f"Cannot load audio from stream: {exc}") from exc


# =============================================================================
# Private Helpers
# =============================================================================

def _normalise_pcm(audio_data: np.ndarray, raw_format: str) -> np.ndarray:
    """Normalise integer PCM data to float32 in [-1, 1]."""
    if raw_format == "int16":
        return audio_data.astype(np.float32) / 32768.0
    elif raw_format == "int32":
        return audio_data.astype(np.float32) / 2147483648.0
    # float32 or other formats: return as-is
    return audio_data.astype(np.float32)


# =============================================================================
# Header parsing
# =============================================================================
 
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from saltup.utils.data.image.image_utils import FileExtensionType
from saltup.utils.data.header_info import (
    DEFAULT_CHUNK,
    MAX_HEADER_SIZE,
    HeaderInfo,
    PathOrUrl,
    extension_from_source,
    parse_header_from_path,
)

_WAV_AUDIO_FORMATS = {
    1: "PCM",
    3: "IEEE_FLOAT",
    6: "A_LAW",
    7: "MU_LAW",
    65534: "EXTENSIBLE",
}


def _get_wav_audio_format_name(audio_format_code: int | None) -> str | None:
    """Return a readable WAV audio format label for a numeric code."""
    if audio_format_code is None:
        return None
    return _WAV_AUDIO_FORMATS.get(audio_format_code, f"UNKNOWN_{audio_format_code}")

def parse_wav_header(data: bytes) -> dict:
    """Parse a RIFF/WAVE header from bytes and return basic fields.

    Returns: dict with keys: format, sample_rate, channels, bit_depth, audio_format,
    audio_format_name
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {"format": "WAV", "error": "Invalid WAV/RIFF header"}

    offset = 12
    sample_rate = None
    channels = None
    bit_depth = None
    audio_format = None

    while offset + 8 <= len(data):
        chunk_id = data[offset:offset+4]
        chunk_size = int.from_bytes(data[offset+4:offset+8], "little")
        offset += 8
        if offset + chunk_size > len(data):
            break

        if chunk_id == b"fmt ":
            # fmt chunk minimum 16 bytes
            fmt = data[offset:offset+chunk_size]
            if len(fmt) >= 16:
                audio_format = int.from_bytes(fmt[0:2], "little")
                channels = int.from_bytes(fmt[2:4], "little")
                sample_rate = int.from_bytes(fmt[4:8], "little")
                bit_depth = int.from_bytes(fmt[14:16], "little")
                return {
                    "format": "WAV",
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "bit_depth": bit_depth,
                    "audio_format": audio_format,
                    "audio_format_name": _get_wav_audio_format_name(audio_format),
                    "total_samples": None,  # not in fmt chunk
                }

        offset += ((chunk_size + 1) // 2) * 2  # chunks are word-aligned

    return {"format": "WAV", "error": "fmt chunk not found"}


def parse_flac_header(data: bytes) -> dict:
    """Parse a FLAC header (fLaC + STREAMINFO) and extract sample rate/channels/bitdepth.
    """
    if len(data) < 8 or data[:4] != b"fLaC":
        return {"format": "FLAC", "error": "Invalid FLAC signature"}

    offset = 4
    # iterate metadata blocks until STREAMINFO (type 0) found
    while offset + 4 <= len(data):
        header = data[offset]
        is_last = (header >> 7) & 1
        block_type = header & 0x7F
        block_length = int.from_bytes(data[offset+1:offset+4], "big")
        offset += 4
        if offset + block_length > len(data):
            break

        if block_type == 0:  # STREAMINFO
            block = data[offset:offset+block_length]
            if len(block) < 34:
                return {"format": "FLAC", "error": "STREAMINFO too short"}
            # bytes 10..17 contain sample rate (20), channels (3), bits-per-sample (5), total_samples (36)
            bits64 = int.from_bytes(block[10:18], "big")
            sample_rate = bits64 >> (64 - 20)
            channels = (bits64 >> (64 - 20 - 3)) & 0x7
            bits_per_sample = (bits64 >> (64 - 20 - 3 - 5)) & 0x1F
            total_samples = bits64 & ((1 << 36) - 1)
            return {
                "format": "FLAC",
                "sample_rate": int(sample_rate),
                "channels": int(channels) + 1 if channels is not None else None,
                "bit_depth": int(bits_per_sample),
                "audio_format": "PCM",  # PCM
                "total_samples": int(total_samples),
            }

        offset += block_length
        if is_last:
            break

    return {"format": "FLAC", "error": "STREAMINFO not found"}


def parse_mp3_header(data: bytes) -> dict:
    """Attempt to find an MPEG frame header and extract sample rate and channels.

    This is a best-effort parser: it looks for a syncword (0xFFF) and decodes
    fields from the first valid frame header it finds.
    """
    # skip ID3 tag if present
    pos = 0
    if data[:3] == b"ID3" and len(data) >= 10:
        # synchsafe size in bytes 6..9
        size_bytes = data[6:10]
        size = ((size_bytes[0] & 0x7F) << 21) | ((size_bytes[1] & 0x7F) << 14) | ((size_bytes[2] & 0x7F) << 7) | (size_bytes[3] & 0x7F)
        pos = 10 + size

    def _parse_frame_at(i: int):
        if i + 4 > len(data):
            return None
        header = int.from_bytes(data[i:i+4], "big")
        if (header >> 21) & 0x7FF != 0x7FF:
            return None
        version_id = (header >> 19) & 0x3
        sample_rate_idx = (header >> 10) & 0x3
        channel_mode = (header >> 6) & 0x3

        versions = {0: "2.5", 1: "reserved", 2: "2", 3: "1"}
        sr_map = {
            "1": [44100, 48000, 32000],
            "2": [22050, 24000, 16000],
            "2.5": [11025, 12000, 8000],
        }
        version = versions.get(version_id, None)
        sample_rate = None
        if version in sr_map and sample_rate_idx in (0,1,2):
            sample_rate = sr_map[version][sample_rate_idx]
        channels = 1 if channel_mode == 3 else 2
        return {"format": "MP3", 
                "sample_rate": sample_rate, 
                "channels": channels, 
                "audio_format": "PCM", #PCM
                "bit_depth": None,
                "total_samples": None}

    # search for a valid frame header
    while pos < len(data) - 4:
        if data[pos] == 0xFF and (data[pos+1] & 0xE0) == 0xE0:
            parsed = _parse_frame_at(pos)
            if parsed:
                return parsed
        pos += 1

    return {"format": "MP3", "error": "No frame header found"}

def _extension_hint(source: PathOrUrl) -> Optional[FileExtensionType]:
    """Map the extension of a path/URI to a FileExtensionType, or None."""
    try:
        return FileExtensionType(extension_from_source(source))
    except ValueError:
        return None


def parse_audio_header(
    source: PathOrUrl,
    *,
    chunk: int = DEFAULT_CHUNK,
    max_header_size: int = MAX_HEADER_SIZE,
) -> "AudioHeaderInfo":
    """Read an audio header incrementally (from a path or URL) and decode it.

    A first slice of *chunk* bytes is read and grown only as far as needed to
    decode the header (see
    :func:`~saltup.utils.data.header_info.parse_header_from_path`); no file size
    is required. The format is auto-detected from the magic bytes
    (WAV / FLAC / MP3), falling back to the extension recovered from the
    path/URL only when detection fails.

    Args:
        source: Audio file path or ``http(s)://`` URL (e.g. a presigned URL).
        chunk: Bytes to read per step while probing the header.
        max_header_size: Hard ceiling on bytes read, so a corrupt file whose
            header never resolves cannot be read in full. A smaller file stops
            earlier at end-of-stream.

    Returns:
        An :class:`AudioHeaderInfo` (truthy when metadata was decoded).
    """
    fallback = _extension_hint(source)

    def _parse(data: bytes, _unused=None) -> "AudioHeaderInfo":
        info = parse_header(data)
        if not info and fallback is not None:
            info = parse_header(data, hint=fallback)
        return info

    return parse_header_from_path(
        source, _parse, initial_read=chunk, max_read=max_header_size
    )


# ---- Byte-buffer header parsing (format auto-detected from magic bytes) ----

_AUDIO_PARSERS = {
    FileExtensionType.WAV: parse_wav_header,
    FileExtensionType.FLAC: parse_flac_header,
    FileExtensionType.MP3: parse_mp3_header,
}


@dataclass
class AudioHeaderInfo(HeaderInfo):
    """Metadata decoded from an audio header. See :class:`HeaderInfo`."""

    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    audio_format: Optional[object] = None
    audio_format_name: Optional[str] = None
    total_samples: Optional[int] = None

    @classmethod
    def supported_formats(cls) -> Tuple[FileExtensionType, ...]:
        """Audio formats this header parser can decode."""
        return tuple(_AUDIO_PARSERS)

    @classmethod
    def _from_parser_dict(cls, parsed: dict) -> "AudioHeaderInfo":
        error = parsed.get("error")
        return cls(
            format=parsed.get("format"),
            has_metadata=error is None,
            error=error,
            sample_rate=parsed.get("sample_rate"),
            channels=parsed.get("channels"),
            bit_depth=parsed.get("bit_depth"),
            audio_format=parsed.get("audio_format"),
            audio_format_name=parsed.get("audio_format_name"),
            total_samples=parsed.get("total_samples"),
        )


def _detect_audio_format(data: bytes) -> Optional[FileExtensionType]:
    """Detect an audio format from the leading *magic bytes* of a buffer.

    Returns the matching :class:`FileExtensionType`, or ``None`` when the buffer
    does not start with any recognized audio signature.
    """
    if len(data) < 4:
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return FileExtensionType.WAV
    if data[:4] == b"fLaC":
        return FileExtensionType.FLAC
    if data[:3] == b"ID3":
        return FileExtensionType.MP3
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:  # MPEG audio syncword
        return FileExtensionType.MP3
    return None


def parse_header(data: bytes, hint: Optional[FileExtensionType] = None) -> AudioHeaderInfo:
    """Parse an audio header directly from a byte buffer.

    The format is auto-detected from the leading magic bytes, dispatched to the
    matching per-format parser, and the result reports whether metadata was
    successfully decoded.

    Args:
        data: Raw bytes containing at least the file header.
        hint: Optional expected format. When given it takes precedence over
            magic-byte detection; when omitted the format is detected.

    Returns:
        An :class:`AudioHeaderInfo`. It is truthy (``bool(info) is True``) when
        metadata was read; otherwise ``info.error`` explains why not.
    """
    if not data:
        return AudioHeaderInfo(error="Empty buffer")

    fmt = hint if hint is not None else _detect_audio_format(data)
    if fmt is None:
        return AudioHeaderInfo(error="Unrecognized audio format")

    parser = _AUDIO_PARSERS.get(fmt)
    if parser is None:
        return AudioHeaderInfo(error=f"No audio parser for format: {fmt.value}")

    return AudioHeaderInfo._from_parser_dict(parser(data))



