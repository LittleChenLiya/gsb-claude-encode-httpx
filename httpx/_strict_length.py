"""
Strict `Content-Length` validation for response bodies.

This module is entirely *optional* and does not alter the default behaviour
of `httpx.Response` or `httpx.Client`.

It provides...

* `StrictLengthStream` / `AsyncStrictLengthStream` - byte stream wrappers
  that count the *raw* bytes delivered by a response stream and validate
  them against the declared `Content-Length`. The wrappers stream through
  incrementally and never buffer the response body into memory.
* `StrictLengthTransport` / `AsyncStrictLengthTransport` - transports that
  wrap another transport and attach strict length validation to every
  response.

Validation is performed against the raw, *pre-decoding* response body, so
gzip/brotli/zstd encoded responses are compared using their compressed
length, and decoded text length is never used.

Responses to `HEAD` requests, and `1xx`/`204`/`304` responses, never carry
a message body per HTTP semantics and are passed through untouched.
"""

from __future__ import annotations

import typing

from ._exceptions import ContentLengthError, StreamConsumed
from ._models import Headers, Request, Response
from ._status_codes import codes
from ._transports.base import AsyncBaseTransport, BaseTransport
from ._types import AsyncByteStream, SyncByteStream

__all__ = [
    "AsyncStrictLengthStream",
    "AsyncStrictLengthTransport",
    "StrictLengthStream",
    "StrictLengthTransport",
    "parse_content_length",
    "response_expects_body",
]


def parse_content_length(headers: Headers) -> int | None:
    """
    Parse the `Content-Length` header(s) of a response.

    Multiple `Content-Length` headers are accepted only when they all
    declare the same length (a comma separated list of identical values is
    also accepted). Conflicting values, or values that are not valid
    non-negative integers, raise `ContentLengthError`.

    Returns `None` when no `Content-Length` header is present (eg. a
    chunked transfer encoding).
    """
    values = headers.get_list("content-length")
    if not values:
        return None

    tokens: list[str] = []
    for value in values:
        tokens.extend(token.strip() for token in value.split(","))

    lengths: list[int] = []
    for token in tokens:
        if not token:
            raise ContentLengthError(
                "Malformed Content-Length header: empty value.",
                declared_length=None,
                received_length=0,
                reason="malformed_headers",
            )
        try:
            length = int(token)
        except ValueError:
            raise ContentLengthError(
                f"Malformed Content-Length header: {token!r} is not an integer.",
                declared_length=None,
                received_length=0,
                reason="malformed_headers",
            ) from None
        if length < 0:
            raise ContentLengthError(
                f"Malformed Content-Length header: {token!r} is negative.",
                declared_length=None,
                received_length=0,
                reason="malformed_headers",
            )
        lengths.append(length)

    if any(length != lengths[0] for length in lengths[1:]):
        raise ContentLengthError(
            "Conflicting Content-Length headers: "
            + ", ".join(f"{token!r}" for token in tokens)
            + ".",
            declared_length=None,
            received_length=0,
            reason="conflicting_headers",
        )

    return lengths[0]


def response_expects_body(request: Request, response: Response) -> bool:
    """
    Return `True` if the response is expected to carry a message body.

    Per HTTP semantics a response to a `HEAD` request never has a body, and
    neither do `1xx` (informational), `204` (No Content), or
    `304` (Not Modified) responses.
    """
    if request.method.upper() == "HEAD":
        return False
    if codes.is_informational(response.status_code):
        return False
    if response.status_code in (204, 304):
        return False
    return True


class StrictLengthStream(SyncByteStream):
    """
    A sync byte-stream wrapper that validates the raw number of bytes
    delivered against a declared `Content-Length`.

    Bytes are counted incrementally as they stream through - the response
    body is never read into memory. If the underlying stream delivers more
    bytes than declared, validation fails immediately (mid-stream, before
    the overshooting chunk is yielded); if it ends having delivered fewer
    bytes, validation fails once the stream is exhausted.

    The underlying stream is closed exactly once, whether the response is
    read to completion, closed early, or fails validation.

    **Parameters:**

    * `stream` - The underlying sync byte stream to wrap.
    * `content_length` - The declared `Content-Length`, or `None` if no
      length was declared (eg. chunked encoding).
    * `expects_body` - `False` for responses that never carry a body
      (`HEAD`, `1xx`, `204`, `304`), which are passed through untouched.
    """

    def __init__(
        self,
        stream: SyncByteStream,
        *,
        content_length: int | None,
        expects_body: bool = True,
    ) -> None:
        self._stream = stream
        self._content_length = content_length
        self._expects_body = expects_body
        self._num_bytes = 0
        self._closed = False
        self._started = False

    @property
    def num_bytes_downloaded(self) -> int:
        """
        The number of raw bytes delivered so far.
        """
        return self._num_bytes

    def __iter__(self) -> typing.Iterator[bytes]:
        if self._started:
            raise StreamConsumed()
        self._started = True

        iterator = self._stream.__iter__()
        try:
            if self._content_length is None or not self._expects_body:
                # No length was declared (eg. chunked transfer encoding),
                # or the response is one that never carries a message body.
                # Pass the stream through untouched.
                for chunk in iterator:
                    yield chunk
            else:
                for chunk in iterator:
                    self._num_bytes += len(chunk)
                    if self._num_bytes > self._content_length:
                        raise ContentLengthError(
                            "Response body exceeded the declared Content-Length: "
                            f"received {self._num_bytes} bytes but Content-Length "
                            f"is {self._content_length}.",
                            declared_length=self._content_length,
                            received_length=self._num_bytes,
                            reason="over_read",
                        )
                    yield chunk

                if self._num_bytes < self._content_length:
                    raise ContentLengthError(
                        "Response body ended before the declared Content-Length: "
                        f"received {self._num_bytes} bytes but Content-Length "
                        f"is {self._content_length}.",
                        declared_length=self._content_length,
                        received_length=self._num_bytes,
                        reason="short_read",
                    )
        finally:
            # Close the underlying iterator and stream exactly once, whether
            # the stream completed, failed validation, or was closed early.
            # Closing the iterator releases any suspended generator state.
            if hasattr(iterator, "close"):
                iterator.close()
            self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stream.close()


class AsyncStrictLengthStream(AsyncByteStream):
    """
    An async byte-stream wrapper that validates the raw number of bytes
    delivered against a declared `Content-Length`.

    See `StrictLengthStream` for details of the validation behaviour.
    """

    def __init__(
        self,
        stream: AsyncByteStream,
        *,
        content_length: int | None,
        expects_body: bool = True,
    ) -> None:
        self._stream = stream
        self._content_length = content_length
        self._expects_body = expects_body
        self._num_bytes = 0
        self._closed = False
        self._started = False

    @property
    def num_bytes_downloaded(self) -> int:
        """
        The number of raw bytes delivered so far.
        """
        return self._num_bytes

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        if self._started:
            raise StreamConsumed()
        self._started = True

        iterator = self._stream.__aiter__()
        try:
            if self._content_length is None or not self._expects_body:
                # No length was declared (eg. chunked transfer encoding),
                # or the response is one that never carries a message body.
                # Pass the stream through untouched.
                async for chunk in iterator:
                    yield chunk
            else:
                async for chunk in iterator:
                    self._num_bytes += len(chunk)
                    if self._num_bytes > self._content_length:
                        raise ContentLengthError(
                            "Response body exceeded the declared Content-Length: "
                            f"received {self._num_bytes} bytes but Content-Length "
                            f"is {self._content_length}.",
                            declared_length=self._content_length,
                            received_length=self._num_bytes,
                            reason="over_read",
                        )
                    yield chunk

                if self._num_bytes < self._content_length:
                    raise ContentLengthError(
                        "Response body ended before the declared Content-Length: "
                        f"received {self._num_bytes} bytes but Content-Length "
                        f"is {self._content_length}.",
                        declared_length=self._content_length,
                        received_length=self._num_bytes,
                        reason="short_read",
                    )
        finally:
            # Close the underlying iterator and stream exactly once, whether
            # the stream completed, failed validation, or was closed early.
            # Closing the iterator releases any suspended generator state.
            if hasattr(iterator, "aclose"):
                await iterator.aclose()
            await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._stream.aclose()


class StrictLengthTransport(BaseTransport):
    """
    A sync transport wrapper that attaches strict `Content-Length`
    validation to every response.

    Wrap any other transport with it to opt in...

        transport = httpx.StrictLengthTransport(httpx.HTTPTransport())
        client = httpx.Client(transport=transport)

    Conflicting or malformed `Content-Length` headers are rejected when the
    response is received; short or over-long bodies fail when the stream is
    consumed. The default behaviour of `httpx.Client` / `httpx.Response` is
    unchanged.
    """

    def __init__(self, transport: BaseTransport) -> None:
        self._transport = transport

    def handle_request(self, request: Request) -> Response:
        response = self._transport.handle_request(request)
        try:
            content_length = parse_content_length(response.headers)
            expects_body = response_expects_body(request, response)
            if isinstance(response.stream, SyncByteStream):
                response.stream = StrictLengthStream(
                    response.stream,
                    content_length=content_length,
                    expects_body=expects_body,
                )
        except ContentLengthError:
            # Release the underlying stream if validation could not be
            # attached.
            response.close()
            raise
        return response

    def close(self) -> None:
        self._transport.close()


class AsyncStrictLengthTransport(AsyncBaseTransport):
    """
    An async transport wrapper that attaches strict `Content-Length`
    validation to every response.

    Wrap any other transport with it to opt in...

        transport = httpx.AsyncStrictLengthTransport(httpx.AsyncHTTPTransport())
        client = httpx.AsyncClient(transport=transport)
    """

    def __init__(self, transport: AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: Request) -> Response:
        response = await self._transport.handle_async_request(request)
        try:
            content_length = parse_content_length(response.headers)
            expects_body = response_expects_body(request, response)
            if isinstance(response.stream, AsyncByteStream):
                response.stream = AsyncStrictLengthStream(
                    response.stream,
                    content_length=content_length,
                    expects_body=expects_body,
                )
        except ContentLengthError:
            # Release the underlying stream if validation could not be
            # attached.
            await response.aclose()
            raise
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()
