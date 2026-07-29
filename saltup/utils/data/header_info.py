"""Shared result type and incremental reader for header/metadata parsing.

The per-media utility packages (:mod:`~saltup.utils.data.image`,
:mod:`~saltup.utils.data.audio`, :mod:`~saltup.utils.data.video`) each expose a
``parse_header(data, hint=None)`` function that decodes the metadata carried in
a raw byte buffer.  They all share the same *calling convention* and the same
*base result* defined here, but each returns its own domain-specific subclass
(``ImageHeaderInfo`` / ``AudioHeaderInfo`` / ``VideoHeaderInfo``) carrying the
fields that make sense for that media type.

This module also provides the machinery to feed those parsers *incrementally*
from either a local path or a remote URI (HTTP/HTTPS, e.g. presigned URLs):
:class:`ByteReader` abstracts range reads, and :func:`parse_header_from_path`
reads a small chunk and grows it only until the header is decoded — never
needing to know the total size up front.
"""

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Union

from saltup.utils.misc import is_url, extract_extension_from_url

# A header source: a local filesystem path, or a str holding an http(s) URL
# (e.g. a presigned URL). The standard library has no dedicated URL type — a
# Path models only the filesystem — so remote sources are plain ``str``.
PathOrUrl = Union[str, Path]

# Default number of bytes read per step while probing a header incrementally.
DEFAULT_CHUNK = 8 * 1024

# Safety ceiling on how many bytes a header probe may read. Guards against
# reading an entire (possibly huge or corrupt) file when the header is never
# found. A smaller file naturally stops earlier at end-of-stream. For a
# MP4/MOV with a large 'moov' box at the tail, raise this per call.
MAX_HEADER_SIZE = 5 * 1024 * 1024


@dataclass
class HeaderInfo:
    """Base result of a header parse over a raw byte buffer.

    Attributes:
        format: Detected container/codec name (e.g. ``"JPEG"``, ``"WAV"``,
            ``"MP4"``), or ``None`` when nothing recognizable was found.
        has_metadata: ``True`` when the parser decoded meaningful metadata from
            the supplied bytes; ``False`` when the buffer was empty,
            unrecognized, truncated, or malformed.
        error: Human-readable reason why parsing failed, set only when
            ``has_metadata`` is ``False``.

    The object is truthy exactly when metadata was read, so callers can write::

        info = parse_header(buffer)
        if info:
            use(info.width, info.height)
    """

    format: Optional[str] = None
    has_metadata: bool = False
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.has_metadata


# ---------------------------- byte sources ----------------------------------

class ByteReader:
    """Random-access byte source.

    Subclasses implement :meth:`read` (a forward range read) and
    :meth:`read_tail` (the last N bytes, for footer/trailer metadata). Neither
    requires the total size to be known in advance. The object is a context
    manager so it can be used with ``with``.
    """

    def read(self, offset: int, length: int) -> bytes:  # pragma: no cover - abstract
        """Return up to *length* bytes starting at *offset* (fewer means EOF)."""
        raise NotImplementedError

    def read_tail(self, length: int) -> bytes:  # pragma: no cover - abstract
        """Return up to the last *length* bytes of the source."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self) -> "ByteReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _FileByteReader(ByteReader):
    """Reads ranges from a local file via ``seek``/``read``."""

    def __init__(self, path: PathOrUrl):
        self._fh = open(Path(path), "rb")

    def read(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        self._fh.seek(offset)
        return self._fh.read(length)

    def read_tail(self, length: int) -> bytes:
        if length <= 0:
            return b""
        size = self._fh.seek(0, os.SEEK_END)
        self._fh.seek(max(0, size - length))
        return self._fh.read(length)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class _HttpByteReader(ByteReader):
    """Reads ranges from an HTTP/HTTPS URL using ``Range`` requests.

    Works with presigned URLs and never issues a preliminary ``HEAD``: it reads
    forward with ``bytes=start-end`` ranges and the tail with a suffix range
    ``bytes=-N``. If the server ignores ``Range`` (responding ``200`` with the
    full body), the body is cached and served from memory so subsequent reads
    don't re-download the file.
    """

    def __init__(self, url: str, timeout: float = 30.0):
        # requests is imported lazily so importing this module stays cheap and
        # does not hard-require the dependency for local-only usage.
        import requests

        self._requests = requests
        self._url = url
        self._timeout = timeout
        self._full: Optional[bytes] = None

    def _get(self, range_header: str) -> bytes:
        try:
            resp = self._requests.get(
                self._url, headers={"Range": range_header}, timeout=self._timeout
            )
        except Exception:
            return b""
        if resp.status_code == 206:
            return resp.content
        if resp.status_code == 200:
            # Range unsupported: cache the whole body and serve slices from it.
            self._full = resp.content
            return self._full
        return b""  # 416 (range not satisfiable) or any error

    def read(self, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if self._full is not None:
            return self._full[offset:offset + length]
        body = self._get(f"bytes={offset}-{offset + length - 1}")
        if self._full is not None:  # server returned the full body during _get
            return self._full[offset:offset + length]
        return body

    def read_tail(self, length: int) -> bytes:
        if length <= 0:
            return b""
        if self._full is not None:
            return self._full[-length:]
        body = self._get(f"bytes=-{length}")
        if self._full is not None:
            return self._full[-length:]
        return body


def open_byte_reader(source: PathOrUrl) -> ByteReader:
    """Open a :class:`ByteReader` for a local path or an HTTP(S) URI.

    Args:
        source: A filesystem path or an ``http(s)://`` URL (e.g. a presigned
            URL). ``s3://`` URIs are not read directly — pass a presigned HTTP
            URL, or download the object first via ``saltup.utils.data.s3``.
    """
    if is_url(source):
        return _HttpByteReader(str(source))
    return _FileByteReader(source)


def extension_from_source(source: PathOrUrl) -> str:
    """Best-effort file extension (without dot, lowercase) from a path or URI."""
    if is_url(source):
        return extract_extension_from_url(str(source)).lower().lstrip(".")
    return PurePosixPath(str(source)).suffix.lower().lstrip(".")


# ------------------------- incremental parsing -------------------------------

def parse_header_from_reader(
    reader: ByteReader,
    parser: Callable[[bytes, Optional[object]], "HeaderInfo"],
    *,
    hint: Optional[object] = None,
    initial_read: int = DEFAULT_CHUNK,
    growth: int = 4,
    max_read: Optional[int] = MAX_HEADER_SIZE,
    is_complete: Optional[Callable[["HeaderInfo"], bool]] = None,
) -> "HeaderInfo":
    """Feed *reader* to *parser* by accumulating bytes until the header decodes.

    Reads ``initial_read`` bytes, parses, and — while the header is neither
    complete nor definitively unrecognizable — reads more (growing the target
    geometrically) and retries. No total size is needed: reading stops on the
    first of these, whichever comes first:

    * the parser reports a complete header (``is_complete``),
    * the format could not even be *recognized* from the leading bytes (magic
      bytes live at the start, so more data cannot help),
    * a short read signals end-of-stream, or
    * the optional ``max_read`` ceiling is hit.

    See :func:`parse_header_from_path` for argument semantics; this lower-level
    entry point lets a caller reuse an already-open reader (e.g. to also read
    the file tail).
    """
    done = is_complete or (lambda info: info.has_metadata)

    data = b""
    target = initial_read if max_read is None else min(initial_read, max_read)
    eof = False
    while True:
        if not eof and len(data) < target:
            want = target - len(data)
            chunk = reader.read(len(data), want)
            data += chunk
            if len(chunk) < want:
                eof = True  # source exhausted

        info = parser(data, hint)
        if done(info):
            return info
        # Unrecognized after the first read: the signature is at the start, so
        # reading further bytes will not change the verdict.
        if info.format is None and len(data) >= min(initial_read, target):
            return info
        if eof or (max_read is not None and len(data) >= max_read):
            return info

        target *= growth
        if max_read is not None:
            target = min(target, max_read)


def parse_header_from_path(
    source: PathOrUrl,
    parser: Callable[[bytes, Optional[object]], "HeaderInfo"],
    *,
    hint: Optional[object] = None,
    initial_read: int = DEFAULT_CHUNK,
    growth: int = 4,
    max_read: Optional[int] = MAX_HEADER_SIZE,
    is_complete: Optional[Callable[["HeaderInfo"], bool]] = None,
) -> "HeaderInfo":
    """Read a header incrementally from a local path or HTTP(S) URI and parse it.

    A small chunk is read and handed to *parser*; if the header cannot yet be
    decoded, more bytes are read and it is retried, stopping as soon as the
    header is decoded, the format proves unrecognizable, or the stream ends.
    This needs no ``stat``/``HEAD`` and, for remote sources, only fetches the
    bytes it actually reads via range requests.

    Args:
        source: File path or ``http(s)://`` URL to read.
        parser: A ``parse_header``-style callable ``(data, hint) -> HeaderInfo``.
        hint: Optional format hint forwarded to *parser*.
        initial_read: Size of the first read.
        growth: Multiplicative factor applied to the read size on each retry.
        max_read: Upper bound on how many leading bytes to read; defaults to
            :data:`MAX_HEADER_SIZE`. ``None`` removes the ceiling (grow until the
            header decodes or the stream ends). Lower it for stricter caps, or
            raise it when metadata may sit far from the start.
        is_complete: Predicate deciding when the parsed header is good enough to
            stop. Defaults to ``bool(info)`` (i.e. ``info.has_metadata``).

    Returns:
        The :class:`HeaderInfo` produced by *parser* on the largest slice read.
    """
    with open_byte_reader(source) as reader:
        return parse_header_from_reader(
            reader, parser, hint=hint, initial_read=initial_read,
            growth=growth, max_read=max_read, is_complete=is_complete,
        )
