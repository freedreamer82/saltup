import copy
import random
import struct
from dataclasses import dataclass
from enum import IntEnum, auto, Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from saltup.utils.data.header_info import (
    DEFAULT_CHUNK,
    MAX_HEADER_SIZE,
    HeaderInfo,
    PathOrUrl,
    extension_from_source,
    parse_header_from_path,
)


class ColorsBGR(Enum):
    RED = (0, 0, 255)        # Rosso in formato BGR
    GREEN = (0, 255, 0)      # Verde in formato BGR
    BLUE = (255, 0, 0)       # Blu in formato BGR
    CYAN = (255, 255, 0)     # Ciano in formato BGR
    MAGENTA = (255, 0, 255)  # Magenta in formato BGR
    YELLOW = (0, 255, 255)   # Giallo in formato BGR
    ORANGE = (0, 165, 255)   # Arancione in formato BGR
    PURPLE = (128, 0, 128)   # Viola in formato BGR
    WHITE = (255, 255, 255)  # Bianco in formato BGR
    BLACK = (0, 0, 0)        # Nero in formato BGR

    def to_rgb(self):
        """
        Convert the BGR color to RGB format.

        Returns:
            tuple: The color in RGB format.
        """
        return self.value[::-1]  # Reverse the BGR tuple to get RGB


class ColorMode(IntEnum):
    RGB = auto()
    BGR = auto()
    GRAY = auto()

    def to_string(self):
        """
        Convert the ColorMode enum to its string representation.

        Returns:
            str: The string representation of the color mode.
        """
        if self == ColorMode.RGB:
            return "RGB"
        elif self == ColorMode.BGR:
            return "BGR"
        elif self == ColorMode.GRAY:
            return "GRAY"
        else:
            return "UNKNOWN"

class ImageFormat(IntEnum):
    HWC = auto()  # Height, Width, Channels (default)
    CHW = auto()  # Channels, Height, Width

class FileExtensionType(Enum):
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"
    SVG = "svg"
    WEBP = "webp"
    HEIC = "heic"
    HEIF = "heif"
    TIF = "tif"
    TIFF = "tiff"
    GIF = "gif"
    MOV = "mov"
    WEBM = "webm"
    FLV = "flv"
    WMV = "wmv"
    GP = "3gp"
    TS = "ts"
    M2TS = "m2ts"
    MTS = "mts"
    M3U8 = "m3u8"
    MP4 = "mp4"
    AVI = "avi"
    MKV = "mkv"
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    AAC = "aac"
    OGG = "ogg"
    M4A = "m4a"
    WMA = "wma"


# ---------------- Header parsing helpers ----------------
# Read and parse image headers (JPEG / PNG / WEBP / GIF / TIFF / HEIC)
def parse_jpeg_header(data: bytes) -> dict:
    if len(data) < 4 or data[:2] != b"\xFF\xD8":
        return {"format": "JPEG", "error": "Invalid JPEG signature"}

    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    i = 2
    while i + 1 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue

        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break

        marker = data[i]
        i += 1

        if marker == 0xD9:
            break
        if marker == 0xDA:
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            continue

        if i + 1 >= len(data):
            break

        segment_length = struct.unpack(">H", data[i:i + 2])[0]
        if segment_length < 2:
            break
        if i + segment_length > len(data):
            break

        if marker in sof_markers:
            if segment_length < 8:
                return {"format": "JPEG", "error": "Invalid SOF segment"}
            precision = data[i + 2]
            height = struct.unpack(">H", data[i + 3:i + 5])[0]
            width = struct.unpack(">H", data[i + 5:i + 7])[0]
            channels = data[i + 7]
            return {
                "format": "JPEG",
                "width": width,
                "height": height,
                "channels": channels,
                "bit_depth": precision,
                "color_mode": {1: "Grayscale", 3: "YCbCr/RGB", 4: "CMYK"}.get(channels, "Unknown"),
            }

        i += segment_length

    return {"format": "JPEG", "error": "SOF marker not found"}


def parse_png_header(data: bytes) -> dict:
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return {"format": "PNG", "error": "Invalid PNG signature"}

    if data[12:16] != b"IHDR":
        return {"format": "PNG", "error": "IHDR chunk not found"}

    width = struct.unpack(">I", data[16:20])[0]
    height = struct.unpack(">I", data[20:24])[0]
    bit_depth = data[24]
    color_type = data[25]

    channels_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    color_mode_map = {
        0: "Grayscale",
        2: "RGB",
        3: "Indexed",
        4: "Grayscale+Alpha",
        6: "RGBA",
    }

    return {
        "format": "PNG",
        "width": width,
        "height": height,
        "channels": channels_map.get(color_type, "Unknown"),
        "bit_depth": bit_depth,
        "color_mode": color_mode_map.get(color_type, "Unknown"),
    }


def parse_webp_header(data: bytes) -> dict:
    if len(data) < 21 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return {"format": "WEBP", "error": "Invalid WEBP signature"}

    chunk_type = data[12:16]
    payload = data[20:]

    if chunk_type == b"VP8 ":
        if len(payload) < 10 or payload[3:6] != b"\x9d\x01\x2a":
            return {"format": "WEBP", "error": "Invalid VP8 chunk"}
        width_raw = struct.unpack("<H", payload[6:8])[0]
        height_raw = struct.unpack("<H", payload[8:10])[0]
        width = width_raw & 0x3FFF
        height = height_raw & 0x3FFF
        return {
            "format": "WEBP",
            "width": width,
            "height": height,
            "channels": 3,
            "bit_depth": 8,
            "color_mode": "RGB",
        }

    if chunk_type == b"VP8L":
        if len(payload) < 5 or payload[0] != 0x2F:
            return {"format": "WEBP", "error": "Invalid VP8L chunk"}
        bits = int.from_bytes(payload[1:5], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return {
            "format": "WEBP",
            "width": width,
            "height": height,
            "channels": 4,
            "bit_depth": 8,
            "color_mode": "RGBA",
        }

    if chunk_type == b"VP8X":
        if len(payload) < 10:
            return {"format": "WEBP", "error": "Invalid VP8X chunk"}
        width = int.from_bytes(payload[4:7], "little") + 1
        height = int.from_bytes(payload[7:10], "little") + 1
        return {
            "format": "WEBP",
            "width": width,
            "height": height,
            "channels": 4,
            "bit_depth": 8,
            "color_mode": "RGB/RGBA",
        }

    return {"format": "WEBP", "error": "Unsupported WEBP chunk type"}


def parse_gif_header(data: bytes) -> dict:
    # Accept GIF signatures in a case-insensitive way and be a bit more lenient
    if len(data) < 10:
        return {"format": "GIF", "error": "Invalid GIF signature"}

    sig = data[:6].lower()
    if sig not in (b"gif87a", b"gif89a"):
        return {"format": "GIF", "error": "Invalid GIF signature"}

    width = struct.unpack("<H", data[6:8])[0]
    height = struct.unpack("<H", data[8:10])[0]
    packed = data[10] if len(data) > 10 else 0
    has_global_color_table = (packed & 0x80) != 0
    channels = 3 if has_global_color_table else 1

    return {
        "format": "GIF",
        "width": width,
        "height": height,
        "channels": channels,
        "bit_depth": 8,
        "color_mode": "Indexed",
    }


def parse_tiff_header(data: bytes) -> dict:
    if len(data) < 8:
        return {"format": "TIFF", "error": "Invalid TIFF header"}

    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        return {"format": "TIFF", "error": "Invalid TIFF byte order"}

    magic = struct.unpack(f"{endian}H", data[2:4])[0]
    if magic != 42:
        return {"format": "TIFF", "error": "Invalid TIFF magic number"}

    ifd_offset = struct.unpack(f"{endian}I", data[4:8])[0]
    if ifd_offset + 2 > len(data):
        return {"format": "TIFF", "error": "IFD offset out of range"}

    num_entries = struct.unpack(f"{endian}H", data[ifd_offset:ifd_offset + 2])[0]
    entry_start = ifd_offset + 2

    width = None
    height = None
    bit_depth = None
    channels = None

    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}

    def _read_ifd_value(value_type: int, value_count: int, value_or_offset: bytes):
        unit_size = type_sizes.get(value_type)
        if unit_size is None:
            return None
        total_size = unit_size * value_count

        if total_size <= 4:
            raw = value_or_offset[:total_size]
        else:
            offset = struct.unpack(f"{endian}I", value_or_offset)[0]
            if offset + total_size > len(data):
                return None
            raw = data[offset:offset + total_size]

        if value_type == 3:
            values = [struct.unpack(f"{endian}H", raw[i:i + 2])[0] for i in range(0, len(raw), 2)]
        elif value_type == 4:
            values = [struct.unpack(f"{endian}I", raw[i:i + 4])[0] for i in range(0, len(raw), 4)]
        elif value_type == 1:
            values = list(raw)
        else:
            return None
        return values

    for entry_idx in range(num_entries):
        pos = entry_start + entry_idx * 12
        if pos + 12 > len(data):
            break

        tag = struct.unpack(f"{endian}H", data[pos:pos + 2])[0]
        value_type = struct.unpack(f"{endian}H", data[pos + 2:pos + 4])[0]
        value_count = struct.unpack(f"{endian}I", data[pos + 4:pos + 8])[0]
        value_or_offset = data[pos + 8:pos + 12]
        values = _read_ifd_value(value_type, value_count, value_or_offset)

        if not values:
            continue

        if tag == 256:
            width = values[0]
        elif tag == 257:
            height = values[0]
        elif tag == 258:
            bit_depth = values[0]
        elif tag == 277:
            channels = values[0]

    if width is None or height is None:
        return {"format": "TIFF", "error": "Width/Height tags not found"}

    if bit_depth is None:
        bit_depth = 8
    if channels is None:
        channels = 1

    color_mode = {
        1: "Grayscale",
        3: "RGB",
        4: "CMYK",
    }.get(channels, "Unknown")

    return {
        "format": "TIFF",
        "width": width,
        "height": height,
        "channels": channels,
        "bit_depth": bit_depth,
        "color_mode": color_mode,
    }


def parse_heic_header(data: bytes) -> dict:
    if len(data) < 16:
        return {"format": "HEIC", "error": "Invalid HEIC header"}

    if data[4:8] != b"ftyp":
        return {"format": "HEIC", "error": "ftyp box not found"}

    major_brand = data[8:12]
    valid_brands = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}
    if major_brand not in valid_brands:
        return {"format": "HEIC", "error": "Unsupported HEIC brand"}

    ispe_idx = data.find(b"ispe")
    if ispe_idx == -1 or ispe_idx + 16 > len(data):
        return {"format": "HEIC", "error": "ispe box not found"}

    width = struct.unpack(">I", data[ispe_idx + 8:ispe_idx + 12])[0]
    height = struct.unpack(">I", data[ispe_idx + 12:ispe_idx + 16])[0]

    return {
        "format": "HEIC",
        "width": width,
        "height": height,
        "channels": "Unknown",
        "bit_depth": "Unknown",
        "color_mode": "Unknown",
    }

def parse_bmp_header(data: bytes) -> dict:
    if len(data) < 54 or data[:2] != b"BM":
        return {"format": "BMP", "error": "Invalid BMP signature"}

    file_size = struct.unpack("<I", data[2:6])[0]
    width = struct.unpack("<I", data[18:22])[0]
    height = struct.unpack("<I", data[22:26])[0]
    planes = struct.unpack("<H", data[26:28])[0]
    bit_count = struct.unpack("<H", data[28:30])[0]

    if planes != 1:
        return {"format": "BMP", "error": "Unsupported number of planes"}

    color_mode = {
        1: "Monochrome",
        4: "16 colors",
        8: "256 colors",
        24: "RGB",
        32: "RGBA",
    }.get(bit_count, "Unknown")

    channels = 3 if bit_count in (24, 32) else 1

    return {
        "format": "BMP",
        "width": width,
        "height": height,
        "channels": channels,
        "bit_depth": bit_count,
        "color_mode": color_mode,
    }

def parse_svg_header(data: bytes) -> dict:
    if b"<svg" not in data.lower():
        return {"format": "SVG", "error": "SVG tag not found"}

    import re

    text = data.decode("utf-8", errors="ignore")
    m = re.search(r"<svg\b([^>]*)>", text, flags=re.IGNORECASE | re.DOTALL)
    attrs = m.group(1) if m else text

    def _parse_value(val: str):
        v = val.strip()
        m2 = re.match(r"^([0-9.+-eE]+)([a-z%]*)$", v)
        if not m2:
            return None
        num = float(m2.group(1))
        unit = m2.group(2).lower()
        if unit in ("", "px"):
            return int(round(num))
        if unit == "%":
            return None
        # convert common absolute units to px (assume 96dpi)
        if unit == "in":
            return int(round(num * 96))
        if unit == "cm":
            return int(round(num * 96.0 / 2.54))
        if unit == "mm":
            return int(round(num * 96.0 / 25.4))
        if unit == "pt":
            return int(round(num * 96.0 / 72.0))
        if unit == "pc":
            return int(round(num * 16.0))
        return None

    width = None
    height = None

    w_m = re.search(r"width\s*=\s*['\"]?([^'\">\s]+)", attrs, flags=re.IGNORECASE)
    h_m = re.search(r"height\s*=\s*['\"]?([^'\">\s]+)", attrs, flags=re.IGNORECASE)
    vb_m = re.search(r"viewBox\s*=\s*['\"]?([\d\.\-+eE\s,]+)['\"]?", attrs, flags=re.IGNORECASE)

    if w_m:
        width = _parse_value(w_m.group(1))
    if h_m:
        height = _parse_value(h_m.group(1))

    if (width is None or height is None) and vb_m:
        nums = re.split(r"[\s,]+", vb_m.group(1).strip())
        if len(nums) >= 4:
            try:
                vb_w = float(nums[2])
                vb_h = float(nums[3])
                if width is None:
                    width = int(round(vb_w))
                if height is None:
                    height = int(round(vb_h))
            except Exception:
                pass

    result = {
        "format": "SVG",
        "width": width if width is not None else "Unknown",
        "height": height if height is not None else "Unknown",
        "channels": "Unknown",
        "bit_depth": "Unknown",
        "color_mode": "Unknown",
    }

    return result

def _extension_hint(source: PathOrUrl) -> Optional[FileExtensionType]:
    """Map the extension of a path/URI to a FileExtensionType, or None."""
    try:
        return FileExtensionType(extension_from_source(source))
    except ValueError:
        return None


def parse_image_header(
    source: PathOrUrl,
    *,
    chunk: int = DEFAULT_CHUNK,
    max_header_size: int = MAX_HEADER_SIZE,
) -> "ImageHeaderInfo":
    """Read an image header incrementally (from a path or URL) and decode it.

    A first slice of *chunk* bytes is read and grown only as far as needed to
    decode the header (see
    :func:`~saltup.utils.data.header_info.parse_header_from_path`); no file size
    is required. The format is auto-detected from the magic bytes — so this
    works for files with a wrong or missing extension — and only falls back to
    the extension recovered from the path/URL when detection comes up empty.

    Args:
        source: Image file path or ``http(s)://`` URL (e.g. a presigned URL).
        chunk: Bytes to read per step while probing the header.
        max_header_size: Hard ceiling on bytes read, so a corrupt file whose
            header never resolves cannot be read in full. A smaller file stops
            earlier at end-of-stream.

    Returns:
        An :class:`ImageHeaderInfo` (truthy when metadata was decoded).
    """
    fallback = _extension_hint(source)

    def _parse(data: bytes, _unused=None) -> "ImageHeaderInfo":
        info = parse_header(data)
        if not info and fallback is not None:
            info = parse_header(data, hint=fallback)
        return info

    return parse_header_from_path(
        source, _parse, initial_read=chunk, max_read=max_header_size
    )


# ---- Byte-buffer header parsing (format auto-detected from magic bytes) ----

# HEIC/HEIF brands found right after the 'ftyp' box.
_HEIC_BRANDS = {
    b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx",
    b"heif", b"mif1", b"msf1",
}

_IMAGE_PARSERS = {
    FileExtensionType.JPEG: parse_jpeg_header,
    FileExtensionType.JPG: parse_jpeg_header,
    FileExtensionType.PNG: parse_png_header,
    FileExtensionType.WEBP: parse_webp_header,
    FileExtensionType.GIF: parse_gif_header,
    FileExtensionType.TIFF: parse_tiff_header,
    FileExtensionType.TIF: parse_tiff_header,
    FileExtensionType.HEIC: parse_heic_header,
    FileExtensionType.HEIF: parse_heic_header,
    FileExtensionType.BMP: parse_bmp_header,
    FileExtensionType.SVG: parse_svg_header,
}


@dataclass
class ImageHeaderInfo(HeaderInfo):
    """Metadata decoded from an image header. See :class:`HeaderInfo`."""

    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    color_mode: Optional[str] = None

    @classmethod
    def supported_formats(cls) -> Tuple[FileExtensionType, ...]:
        """Image formats this header parser can decode."""
        return tuple(_IMAGE_PARSERS)

    @classmethod
    def _from_parser_dict(cls, parsed: dict) -> "ImageHeaderInfo":
        error = parsed.get("error")
        return cls(
            format=parsed.get("format"),
            has_metadata=error is None,
            error=error,
            width=parsed.get("width"),
            height=parsed.get("height"),
            channels=parsed.get("channels"),
            bit_depth=parsed.get("bit_depth"),
            color_mode=parsed.get("color_mode"),
        )


def _detect_image_format(data: bytes) -> Optional[FileExtensionType]:
    """Detect an image format from the leading *magic bytes* of a buffer.

    Returns the matching :class:`FileExtensionType`, or ``None`` when the buffer
    does not start with any recognized image signature.
    """
    if len(data) < 4:
        return None
    if data[:2] == b"\xFF\xD8":
        return FileExtensionType.JPEG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return FileExtensionType.PNG
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return FileExtensionType.WEBP
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return FileExtensionType.GIF
    if data[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return FileExtensionType.TIFF
    if data[:2] == b"BM":
        return FileExtensionType.BMP
    if data[4:8] == b"ftyp" and data[8:12] in _HEIC_BRANDS:
        return FileExtensionType.HEIC
    head = data[:512].lstrip()
    if head[:5].lower() == b"<?xml" or b"<svg" in data[:512].lower():
        return FileExtensionType.SVG
    return None


def parse_header(data: bytes, hint: Optional[FileExtensionType] = None) -> ImageHeaderInfo:
    """Parse an image header directly from a byte buffer.

    The format is auto-detected from the leading magic bytes, dispatched to the
    matching per-format parser, and the result reports whether metadata was
    successfully decoded.

    Args:
        data: Raw bytes containing at least the file header.
        hint: Optional expected format. When given it takes precedence over
            magic-byte detection (useful when the signature is ambiguous or the
            buffer is a partial slice); when omitted the format is detected.

    Returns:
        An :class:`ImageHeaderInfo`. It is truthy (``bool(info) is True``) when
        metadata was read; otherwise ``info.error`` explains why not.
    """
    if not data:
        return ImageHeaderInfo(error="Empty buffer")

    fmt = hint if hint is not None else _detect_image_format(data)
    if fmt is None:
        return ImageHeaderInfo(error="Unrecognized image format")

    parser = _IMAGE_PARSERS.get(fmt)
    if parser is None:
        return ImageHeaderInfo(error=f"No image parser for format: {fmt.value}")

    return ImageHeaderInfo._from_parser_dict(parser(data))


# ---------------- End header parsing helpers ----------------

def generate_random_bgr_colors(num_colors):
    """
    Generates a list of distinct colors in BGR format. If the number of requested colors
    exceeds the number of predefined colors in the ColorsBGR enum, the colors are reused
    in a cyclic manner.

    Args:
        num_colors (int): Number of colors to generate.

    Returns:
        list: A list of colors in BGR format.
    """
    # Extract predefined colors from the ColorsBGR enum
    predefined_colors = [color.value for color in ColorsBGR]

    # If the number of requested colors is less than or equal to the predefined colors,
    # return a subset of the predefined colors.
    if num_colors <= len(predefined_colors):
        return predefined_colors[:num_colors]

    # If more colors are needed, reuse the predefined colors in a cyclic manner.
    colors = []
    for i in range(num_colors):
        # Cycle through the predefined colors
        color = predefined_colors[i % len(predefined_colors)]
        colors.append(color)

    return colors


class Image:
    def __init__(
        self,
        # Accepts either a file path or a NumPy array
        image_input: Union[str, Path, np.ndarray],
        color_mode: ColorMode = ColorMode.BGR
    ):
        """
        Initialize an Image instance.

        Args:
            image_input: Path to the image file (str or Path) or a NumPy array containing the image data.
            color_mode: Color mode of the image (BGR, RGB, or GRAY). Default is BGR.
            image_format: Format of the image (HWC or CHW). Default is HWC.
        """
        self.color_mode = color_mode
        self.image_format = ImageFormat.HWC  # Default format is HWC
        
        # Check if the input is a NumPy array
        if isinstance(image_input, np.ndarray):
            self.__image = self._process_image_data(image_input)
        else:
            # Otherwise, treat it as a file path
            self.image_path = Path(image_input) if isinstance(
                image_input, str) else image_input
            self.__image = self._load_image()

    def _process_image_data(self, image_data: np.ndarray) -> np.ndarray:
        """
        Process the provided NumPy array to ensure it matches the desired color mode and format.

        Args:
            image_data: NumPy array containing the image data.

        Returns:
            Processed image as a NumPy array with shape (h, w, 1) for grayscale or (h, w, 3) for RGB/BGR.
        """
        if not isinstance(image_data, np.ndarray):
            raise ValueError("image_data must be a NumPy array.")

        # Ensure the image is in HWC format because opencv works with HWC on that
        self.__image = self.convert_image_format(image_data, ImageFormat.HWC)

        # Convert the image to the desired color mode
        try:
            if self.color_mode == ColorMode.RGB:
                # If already RGB, do nothing
                if len(self.__image .shape) == 3 and self.__image .shape[-1] == 3:
                    pass
                else:
                    # Convert BGR to RGB if necessary
                    self.__image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2RGB)
            elif self.color_mode == ColorMode.GRAY:
                if len(self.__image .shape) == 2:  # If grayscale with shape (h, w), expand to (h, w, 1)
                    self.__image = np.expand_dims(self.__image, axis=-1)
                # If already (h, w, 1), do nothing
                elif len(self.__image .shape) == 3 and self.__image .shape[-1] == 1:
                    pass
                else:
                    # If it's a 3-channel image but not grayscale, convert to grayscale
                    self.__image = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
                    self.__image = np.expand_dims(self.__image, axis=-1)
        except cv2.error as e:
            raise ValueError(f"Error converting image color mode: {e}")

        return self.__image

    def _load_image(self) -> np.ndarray:
        """
        Load an image from the specified path and process it.

        Returns:
            Loaded and processed image as a NumPy array with shape (h, w, 1) for grayscale or (h, w, 3) for RGB/BGR.
        """
        # Check if the image file exists
        if not self.image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {self.image_path}")

        # Load the image using OpenCV
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_path}")

        # Process the loaded image to ensure it matches the desired color mode and format
        return self._process_image_data(image)

    def convert_color_mode(self, new_color_mode: ColorMode):
        """
        Convert the image to a new color mode in-place, ensuring the output has the correct shape
        based on the image format (HWC or CHW).

        Args:
            new_color_mode: The target color mode (BGR, RGB, or GRAY).
        """
        if self.color_mode == new_color_mode:
            return  # No conversion needed

        # Convert the image to the new color mode
        try:
            # Convert in HWC format
            if new_color_mode == ColorMode.RGB:
                if self.color_mode == ColorMode.BGR:
                    self.__image = cv2.cvtColor(
                        self.__image, cv2.COLOR_BGR2RGB)
                elif self.color_mode == ColorMode.GRAY:
                    self.__image = cv2.cvtColor(
                        self.__image, cv2.COLOR_GRAY2RGB)
            elif new_color_mode == ColorMode.BGR:
                if self.color_mode == ColorMode.RGB:
                    self.__image = cv2.cvtColor(
                        self.__image, cv2.COLOR_RGB2BGR)
                elif self.color_mode == ColorMode.GRAY:
                    self.__image = cv2.cvtColor(
                        self.__image, cv2.COLOR_GRAY2BGR)
            elif new_color_mode == ColorMode.GRAY:
                if self.color_mode == ColorMode.RGB:
                    self.__image = cv2.cvtColor(
                        self.__image, cv2.COLOR_RGB2GRAY)
                elif self.color_mode == ColorMode.BGR:
                    self.__image = cv2.cvtColor(
                        self.__image, cv2.COLOR_BGR2GRAY)
                # Ensure grayscale image has shape (h, w, 1) in HWC format
                self.__image = np.expand_dims(self.__image, axis=-1)
        except cv2.error as e:
            raise ValueError(f"Error converting image color mode: {e}")

        # Ensure the image has the correct number of dimensions
        if len(self.__image.shape) == 2:  # If the image is 2D (h, w), expand to (h, w, 1) or (1, h, w)
            if self.image_format == ImageFormat.HWC:
                self.__image = np.expand_dims(self.__image, axis=-1)
            elif self.image_format == ImageFormat.CHW:
                self.__image = np.expand_dims(self.__image, axis=0)

        # Update the color mode
        self.color_mode = new_color_mode

    def to_jpeg(self, quality: int = 95, save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Encode the image to JPEG format and return as numpy array.
        
        Args:
            quality: JPEG quality (0-100). Default is 95.
            save_path: Optional path to save the JPEG file. If None, only returns the buffer.
        
        Returns:
            np.ndarray: JPEG encoded image as a 1D numpy array (uint8).
        """
        # Convert to BGR if necessary (cv2.imencode requires BGR format)
        image_to_encode = copy.deepcopy(self.__image)
        
        # Handle grayscale images (remove channel dimension if present)
        if self.color_mode == ColorMode.GRAY and len(image_to_encode.shape) == 3:
            image_to_encode = np.squeeze(image_to_encode, axis=-1)
        elif self.color_mode == ColorMode.RGB:
            # Convert RGB to BGR for cv2.imencode
            image_to_encode = cv2.cvtColor(image_to_encode, cv2.COLOR_RGB2BGR)
        # If already BGR, use as is
        
        # Encode to JPEG format
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode('.jpg', image_to_encode, encode_params)
        
        if not success:
            raise RuntimeError("Failed to encode image to JPEG format")
        
        # Save to file if path is provided
        if save_path is not None:
            save_path = Path(save_path) if isinstance(save_path, str) else save_path
            with open(save_path, 'wb') as f:
                f.write(buffer.tobytes())
        
        # Return the JPEG buffer as numpy array
        return buffer.flatten()
    
    def copy(self) -> 'Image':
        """
        Create a deep copy of the current Image instance.

        Returns:
            A new Image instance with the same attributes as the current one.
        """
        # Create a new Image instance with the same image data, color mode, and format
        new_image = Image(
            image_input=copy.deepcopy(self.__image),
            color_mode=self.color_mode
        )
        return new_image

    # The remaining methods of the class remain unchanged
    def get_shape(self, format: ImageFormat = ImageFormat.HWC) -> tuple:
        """Get the shape of the image as a tuple (height, width, channels)."""
        if format == ImageFormat.HWC:
            return self.__image.shape
        elif format == ImageFormat.CHW:
            return self.__image.shape[::-1]

    def get_width(self) -> int:
        """Get the width of the image."""
        return self.__image.shape[1]

    def get_height(self) -> int:
        """Get the height of the image."""
        return self.__image.shape[0]

    def get_number_channel(self) -> int:
        """Get the number of channels in the image."""
        return self.__image.shape[2]

    def get_color_mode(self) -> ColorMode:
        """Get the color mode of the image."""
        return self.color_mode

    def get_data(self, format: ImageFormat = ImageFormat.HWC) -> np.ndarray:
        """Get the image as a NumPy array."""
        if format == ImageFormat.HWC:
            return self.__image
        elif format == ImageFormat.CHW:
            return self.convert_image_format(self.__image, ImageFormat.CHW)

    def resize(self, new_size: tuple) -> 'Image':
        """
        Resize the image to the specified dimensions while maintaining 3 dimensions.

        Args:
            new_size: A tuple (width, height) representing the new dimensions.

        Returns:
            self: The Image instance with the resized image.
        """
        # Resize the image using OpenCV
        self.__image = cv2.resize(self.__image, new_size)

        # Ensure the image has 3 dimensions
        if len(self.__image.shape) == 2:  # If the image is 2D (h, w), expand to (h, w, 1)
            self.__image = np.expand_dims(self.__image, axis=-1)
        # If already (h, w, 1), do nothing
        elif len(self.__image.shape) == 3 and self.__image.shape[2] == 1:
            pass
        # If already (h, w, 3), do nothing
        elif len(self.__image.shape) == 3 and self.__image.shape[2] == 3:
            pass
        else:
            # Handle unexpected shapes (e.g., 4D or invalid)
            raise ValueError(
                f"Unexpected image shape after resizing: {self.__image.shape}")

        return self

    def crop(self, crop_window: dict) -> 'Image':
        """Crop the image using the specified window."""
        self.__image = self.__image[crop_window['y_min']:crop_window['y_max'],
                                crop_window['x_min']:crop_window['x_max']]
        return self

    def invert_pixels(self) -> 'Image':
        """Invert the pixel values of the image."""
        self.__image = cv2.bitwise_not(self.__image)
        if len(self.__image.shape) == 3 and self.__image.shape[2] == 1:
            self.__image = np.expand_dims(self.__image, axis=-1)
        return self

    def save_raw(self, dest_path: str):
        """Save the image data as a raw binary file."""
        with open(dest_path, 'wb') as f:
            f.write(self.__image.tobytes())

    def save_jpg(self, dest_path: str):
        """Save the image as a JPG file."""
        if len(self.__image.shape) == 3 and self.__image.shape[2] == 3:
            image = cv2.cvtColor(self.__image, cv2.COLOR_RGB2BGR)
        else:
            image = self.__image
        cv2.imwrite(dest_path, image)

    def show(self, window_name: str = "Image", key: int = ord('q')):
        """Display the image in a window. Close the window when the specified key is pressed."""
        cv2.imshow(window_name, self.__image)
        while True:
            # Wait for 1 ms and check the key pressed
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == key or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    @classmethod
    def convert_image_format(cls, image: np.ndarray, target_format: ImageFormat) -> np.ndarray:
        """Convert an image between HWC and CHW formats."""
        if len(image.shape) not in {2, 3}:
            raise ValueError(
                f"Invalid image shape: {image.shape}. Expected 2D (H, W) or 3D (H, W, C).")

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if target_format == ImageFormat.CHW:
            if len(image.shape) == 3 and image.shape[2] <= 4:
                return np.transpose(image, (2, 0, 1))
            elif len(image.shape) == 3 and image.shape[0] <= 4:
                return image
        elif target_format == ImageFormat.HWC:
            if len(image.shape) == 3 and image.shape[0] <= 4:
                return np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 3 and image.shape[2] <= 4:
                return image
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

        raise ValueError(
            f"Cannot convert image with shape {image.shape} to {target_format} format.")

    @classmethod
    def jpg_to_raw(cls, input_file: str, grayscale: bool = False) -> Optional[np.ndarray]:
        """Convert a JPEG image to a raw array."""
        if grayscale:
            image_data = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
        else:
            image_data = cv2.imread(input_file)
            if image_data is not None:  # Check if image was loaded successfully
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    
        return image_data

    @classmethod
    def resize_image(cls, image: np.ndarray, new_size: tuple) -> np.ndarray:
        """Resize an image using OpenCV."""
        return cv2.resize(image, new_size)

    @classmethod
    def crop_image(cls, image: np.ndarray, crop_window: dict) -> np.ndarray:
        """Crop an image according to the specified window."""
        return image[crop_window['y_min']:crop_window['y_max'],
                     crop_window['x_min']:crop_window['x_max']]

    @classmethod
    def save_raw_image(cls, image: np.ndarray, dest_path: str):
        """Save raw image data to file."""
        with open(dest_path, 'wb') as f:
            f.write(image.tobytes())

    @classmethod
    def save_jpg_image(cls, image: np.ndarray, dest_path: str):
        """Save an image as JPG using OpenCV."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dest_path, image)

    @classmethod
    def invert_pixel(cls, img: np.ndarray) -> np.ndarray:
        """Invert the pixel values of a uint8 image."""
        inverted_img = cv2.bitwise_not(img)
        if len(img.shape) == 3 and img.shape[2] == 1:
            inverted_img = np.expand_dims(inverted_img, axis=-1)
        return inverted_img

    @classmethod
    def pad_image(cls, image: np.ndarray, target_h: int, target_w: int, image_format: ImageFormat = ImageFormat.HWC) -> np.ndarray:
        """Add padding if image dimensions are smaller than target size.

        Args:
            image: Input image in CHW or HWC format.
            target_h: Target height.
            target_w: Target width.
            image_format: Input image format (CHW or HWC).

        Returns:
            np.ndarray: Padded tensor matching target size, or original if no padding needed.
                        Output maintains the specified format and float32 precision.
        """
        # Convert to CHW format for consistent processing
        if image_format == ImageFormat.HWC:
            image = cls.convert_image_format(image, ImageFormat.CHW)

        # Extract dimensions
        c, h, w = image.shape

        # Return original image if no padding needed
        if h >= target_h and w >= target_w:
            if image_format == ImageFormat.HWC:
                image = cls.convert_image_format(image, ImageFormat.HWC)
            return image

        # Add padding only if necessary
        padded_img = 114 * np.ones((c, target_h, target_w), dtype=np.float32)
        padded_img[:, :h, :w] = image

        # Convert back to the original format
        if image_format == ImageFormat.HWC:
            padded_img = cls.convert_image_format(padded_img, ImageFormat.HWC)

        return padded_img
