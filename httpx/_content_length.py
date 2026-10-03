"""
Strict validation of the response `Content-Length` header.

HTTPX is deliberately lenient about the length of response bodies by
default: a stream that ends early, or that yields more bytes than
advertised, is surfaced to the caller unchanged.  Some applications
need the stricter behaviour mandated by the HTTP specifications, so
this module provides an *opt-in* response hook,
[`enforce_content_length`][httpx.enforce_content_length], that:

* rejects responses carrying repeated, disagreeing `Content-Length`
  values (RFC 9112 section 6.2);
* raises a structured `ContentLengthTooShort` error if the raw body
  ends before the declared number of bytes has been received;
* raises a structured `ContentLengthTooLong` error as soon as the raw
  body yields more bytes than declared.

The check counts the *raw* (still compressed / encoded) response bytes,
so a gzip body whose decoded length differs from its declared length is
not a mismatch.  Responses to `HEAD` requests, together with the
status codes that define a bodyless response (`1xx`, `204`, `304`), are
left untouched, matching the HTTP message-framing rules.

Validation happens while the body is streamed, so the response is never
buffered into memory merely in order to measure it.
"""

from __future__ import annotations

import typing

from ._exceptions import (
    ContentLengthTooLong,
    ContentLengthTooShort,
    InvalidContentLength,
    request_context,
)
from ._models import Response
from ._types import AsyncByteStream, SyncByteStream

__all__ = [
    "enforce_content_length",
    "StrictAsyncContentLengthStream",
    "StrictSyncContentLengthStream",
]

# Status codes for which a response is defined as having no body.
# (RFC 9110 section 15.4.5, and the informational 1xx class.)
_BODYLESS_STATUS_CODES = frozenset({204, 304})


def _response_expects_body(response: Response) -> bool:
    """
    Return `False` for responses that are defined as carrying no body.

    `HEAD` responses may include a `Content-Length` header describing
    the representation that *would* have been sent for a `GET`, without
    actually sending a body, so it must not be validated.  Likewise for
    the 1xx, 204 and 304 status codes.
    """
    request = response._request
    if request is not None and request.method == "HEAD":
        return False
    if 100 <= response.status_code < 200:
        return False
    if response.status_code in _BODYLESS_STATUS_CODES:
        return False
    return True


def _parse_content_length(response: Response) -> int | None:
    """
    Resolve the declared response body length, or `None` if the
    response does not carry a usable `Content-Length` header.

    Repeated `Content-Length` header fields are split on commas (RFC
    allows a single field to be expressed as a comma-separated list) and
    whitespace-trimmed.  Multiple occurrences, or disagreeing values,
    are rejected outright instead of silently picking one value.
    """
    raw_values = response.headers.get_list("content-length")
    if not raw_values:
        return None

    distinct: list[str] = []
    for raw in raw_values:
        for part in raw.split(","):
            value = part.strip()
            if value not in distinct:
                distinct.append(value)

    if len(distinct) != 1:
        raise InvalidContentLength(
            "Received response with multiple, conflicting Content-Length "
            f"values: {', '.join(distinct)!r}.",
            values=distinct,
        )

    value = distinct[0]
    # `Content-Length` is a single non-negative decimal integer. Reject
    # things like "abc", "12.0", "+1", "0x5" or underscore-laden input.
    if not value.isdigit():
        raise InvalidContentLength(
            f"Received response with malformed Content-Length value: {value!r}.",
            values=[value],
        )

    return int(value)


def _enforce_content_length(response: Response) -> None:
    request = response._request
    with request_context(request=request):
        if not _response_expects_body(response):
            return

        # RFC 9112 section 6.2: when both Transfer-Encoding and
        # Content-Length are present, Content-Length does not frame the
        # body. Leave the framing entirely to the transfer coding.
        if "transfer-encoding" in response.headers:
            return

        expected_length = _parse_content_length(response)
        if expected_length is None:
            return

        if hasattr(response, "_content"):
            # The body is already fully materialised in memory (e.g. a
            # `Response(content=...)` instance). Check it directly rather
            # than wrapping a stream that will never be iterated.
            received_length = len(response._content)
            if received_length < expected_length:
                raise ContentLengthTooShort(
                    f"Expected response body with {expected_length} bytes, "
                    f"but received only {received_length} bytes "
                    "(connection closed prematurely).",
                    expected=expected_length,
                    received=received_length,
                )
            if received_length > expected_length:
                raise ContentLengthTooLong(
                    f"Expected response body with {expected_length} bytes, "
                    f"but received {received_length} bytes.",
                    expected=expected_length,
                    received=received_length,
                )
            return

        stream = response.stream
        if isinstance(stream, SyncByteStream):
            response.stream = StrictSyncContentLengthStream(
                stream, expected_length=expected_length
            )
        elif isinstance(stream, AsyncByteStream):
            response.stream = StrictAsyncContentLengthStream(
                stream, expected_length=expected_length
            )


class _ImmediateAwaitable:
    """
    An awaitable that resumes immediately, without scheduling a coroutine.

    Returning this (rather than a coroutine) means the validation can run
    synchronously when the hook is *called*, and still be `await`-ed when
    the same callable is installed as an `AsyncClient` event hook.
    """

    def __await__(self) -> typing.Iterator[None]:
        return
        yield  # pragma: no cover - marks this as a generator method


def enforce_content_length(response: Response) -> _ImmediateAwaitable:
    """
    Strictly enforce the response `Content-Length` header.

    Intended for use as a sync **or** async response event hook, e.g.:

    ```python
    client = httpx.Client(
        event_hooks={"response": [httpx.enforce_content_length]}
    )
    # ...or, equivalently:
    client = httpx.AsyncClient(
        event_hooks={"response": [httpx.enforce_content_length]}
    )
    ```

    The hook validates header values immediately (duplicate or malformed
    `Content-Length` headers raise `InvalidContentLength`), and wraps the
    response stream so that its raw byte count is checked as it is read:

    * a stream ending before the declared length raises
      `ContentLengthTooShort`;
    * a stream yielding more bytes than declared raises
      `ContentLengthTooLong` *immediately*, while streaming, rather than
      once the whole body has been consumed.

    Responses without a `Content-Length` header, `HEAD` responses and
    bodyless (`1xx` / `204` / `304`) responses are left unchanged, as is
    the default behaviour of `httpx.Response` when this hook is not in
    use.
    """
    _enforce_content_length(response)
    return _ImmediateAwaitable()


class StrictSyncContentLengthStream(SyncByteStream):
    """
    Wrap a sync response byte stream, enforcing a fixed `Content-Length`.

    The wrapper is an explicit iterator that keeps a running counter
    instead of buffering the body, so checking the length does not cause
    the whole response to be read into memory.  The wrapped stream is
    closed at most once - on completion, error, or explicit `close()`
    (e.g. the caller abandoning iteration) - regardless of how many times
    `close()` ends up being called.
    """

    def __init__(self, stream: SyncByteStream, *, expected_length: int) -> None:
        self._stream = stream
        self._expected_length = expected_length
        self._received_length = 0
        self._is_closed = False
        self._iterator: typing.Iterator[bytes] | None = None

    def __iter__(self) -> "StrictSyncContentLengthStream":
        self._iterator = iter(self._stream)
        return self

    def __next__(self) -> bytes:
        assert self._iterator is not None  # pragma: no cover
        try:
            chunk = next(self._iterator)
        except StopIteration:
            if self._received_length < self._expected_length:
                raise ContentLengthTooShort(
                    f"Expected response body with {self._expected_length} "
                    "bytes, but received only "
                    f"{self._received_length} bytes "
                    "(connection closed prematurely).",
                    expected=self._expected_length,
                    received=self._received_length,
                ) from None
            raise

        received_length = self._received_length + len(chunk)
        if received_length > self._expected_length:
            self._received_length = received_length
            # Fail straight away; do not hand the surplus bytes to the
            # caller and do not wait for the stream to end.
            raise ContentLengthTooLong(
                f"Expected response body with {self._expected_length} "
                "bytes, but received more than "
                f"{self._expected_length} bytes.",
                expected=self._expected_length,
                received=received_length,
            )
        self._received_length = received_length
        return chunk

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._stream.close()


class StrictAsyncContentLengthStream(AsyncByteStream):
    """
    Wrap an async response byte stream, enforcing a fixed `Content-Length`.

    See `StrictSyncContentLengthStream` for the semantics: streaming
    length checks with no full-body buffering, immediate failure when the
    declared length is exceeded, and a single underlying `aclose()`.
    """

    def __init__(self, stream: AsyncByteStream, *, expected_length: int) -> None:
        self._stream = stream
        self._expected_length = expected_length
        self._received_length = 0
        self._is_closed = False
        self._iterator: typing.AsyncIterator[bytes] | None = None

    def __aiter__(self) -> "StrictAsyncContentLengthStream":
        self._iterator = self._stream.__aiter__()
        return self

    async def __anext__(self) -> bytes:
        assert self._iterator is not None  # pragma: no cover
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            if self._received_length < self._expected_length:
                raise ContentLengthTooShort(
                    f"Expected response body with {self._expected_length} "
                    "bytes, but received only "
                    f"{self._received_length} bytes "
                    "(connection closed prematurely).",
                    expected=self._expected_length,
                    received=self._received_length,
                ) from None
            raise StopAsyncIteration

        received_length = self._received_length + len(chunk)
        if received_length > self._expected_length:
            self._received_length = received_length
            raise ContentLengthTooLong(
                f"Expected response body with {self._expected_length} "
                "bytes, but received more than "
                f"{self._expected_length} bytes.",
                expected=self._expected_length,
                received=received_length,
            )
        self._received_length = received_length
        return chunk

    async def aclose(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            await self._stream.aclose()
