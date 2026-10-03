"""
Tests for the optional strict `Content-Length` response wrappers.

These tests use `httpx.MockTransport` together with counting streams to
verify that...

* short reads and over-long bodies produce structured `ContentLengthError`s
* validation uses the raw, pre-decoding body (gzip length, not decoded text)
* `HEAD` / `1xx` / `204` / `304` responses pass through per HTTP semantics
* conflicting / malformed `Content-Length` headers are rejected up front
* the underlying stream is closed exactly once on error or cancellation
* the response body is never buffered into memory for length counting
"""

from __future__ import annotations

import gzip
import typing

import pytest

import httpx

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class CountingStream(httpx.SyncByteStream):
    """
    A sync byte stream that records how many bytes it has produced and how
    many times it has been closed.
    """

    def __init__(self, chunks: typing.Iterable[bytes]) -> None:
        self._chunks = list(chunks)
        self.bytes_sent = 0
        self.close_count = 0

    def __iter__(self) -> typing.Iterator[bytes]:
        for chunk in self._chunks:
            self.bytes_sent += len(chunk)
            yield chunk

    def close(self) -> None:
        self.close_count += 1


class AsyncCountingStream(httpx.AsyncByteStream):
    """
    An async byte stream that records how many bytes it has produced and how
    many times it has been closed.
    """

    def __init__(self, chunks: typing.Iterable[bytes]) -> None:
        self._chunks = list(chunks)
        self.bytes_sent = 0
        self.close_count = 0

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        for chunk in self._chunks:
            self.bytes_sent += len(chunk)
            yield chunk

    async def aclose(self) -> None:
        self.close_count += 1


class ImmediateOverReadStream(httpx.SyncByteStream):
    """
    A sync stream whose first chunk overshoots the declared length, and
    which raises if iteration is resumed after the overshoot.
    """

    def __init__(self, chunk: bytes) -> None:
        self._chunk = chunk
        self.close_count = 0

    def __iter__(self) -> typing.Iterator[bytes]:
        yield self._chunk
        raise AssertionError("stream was resumed after the over-length chunk")

    def close(self) -> None:
        self.close_count += 1


class AsyncImmediateOverReadStream(httpx.AsyncByteStream):
    """
    An async stream whose first chunk overshoots the declared length, and
    which raises if iteration is resumed after the overshoot.
    """

    def __init__(self, chunk: bytes) -> None:
        self._chunk = chunk
        self.close_count = 0

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        yield self._chunk
        raise AssertionError("stream was resumed after the over-length chunk")

    async def aclose(self) -> None:
        self.close_count += 1


# ---------------------------------------------------------------------------
# Direct stream wrapper tests
# ---------------------------------------------------------------------------


def test_strict_length_stream_exact() -> None:
    stream = httpx.StrictLengthStream(
        httpx.ByteStream(b"Hello"), content_length=5
    )
    assert b"".join(stream) == b"Hello"
    assert stream.num_bytes_downloaded == 5


def test_strict_length_stream_short_read() -> None:
    stream = httpx.StrictLengthStream(
        httpx.ByteStream(b"abc"), content_length=5
    )
    with pytest.raises(httpx.ContentLengthError) as exc_info:
        b"".join(stream)

    error = exc_info.value
    assert error.reason == "short_read"
    assert error.declared_length == 5
    assert error.received_length == 3


def test_strict_length_stream_over_read_immediate() -> None:
    # The first chunk already exceeds the declared length, so validation
    # must fail immediately - the chunk must not be yielded and the stream
    # must not be read any further.
    stream = httpx.StrictLengthStream(
        ImmediateOverReadStream(b"abcdef"), content_length=5
    )
    iterator = iter(stream)
    with pytest.raises(httpx.ContentLengthError) as exc_info:
        next(iterator)

    error = exc_info.value
    assert error.reason == "over_read"
    assert error.declared_length == 5
    assert error.received_length == 6


def test_strict_length_stream_over_read_mid_stream() -> None:
    # The first chunk is fine, the second one overshoots.
    def gen() -> typing.Iterator[bytes]:
        yield b"ab"
        yield b"cdef"  # cumulative 6 > 5
        raise AssertionError("resumed after over-length chunk")

    stream = httpx.StrictLengthStream(
        _IterableStream(gen()), content_length=5
    )
    chunks = []
    with pytest.raises(httpx.ContentLengthError) as exc_info:
        for chunk in stream:
            chunks.append(chunk)

    assert chunks == [b"ab"]  # the overshooting chunk is never yielded
    error = exc_info.value
    assert error.reason == "over_read"
    assert error.received_length == 6


class _IterableStream(httpx.SyncByteStream):
    def __init__(self, iterator: typing.Iterator[bytes]) -> None:
        self._iterator = iterator

    def __iter__(self) -> typing.Iterator[bytes]:
        return self._iterator


def test_strict_length_stream_no_buffering() -> None:
    # Bytes must be counted incrementally - after pulling the first chunk
    # the underlying stream must have produced only that chunk, not the
    # whole body.
    counting = CountingStream([b"ab", b"cd", b"ef"])
    stream = httpx.StrictLengthStream(counting, content_length=6)

    iterator = iter(stream)
    assert next(iterator) == b"ab"
    assert counting.bytes_sent == 2

    assert next(iterator) == b"cd"
    assert counting.bytes_sent == 4

    assert next(iterator) == b"ef"
    assert counting.bytes_sent == 6


def test_strict_length_stream_passes_through_without_length() -> None:
    # No declared length (eg. chunked encoding) - pass through untouched.
    stream = httpx.StrictLengthStream(
        httpx.ByteStream(b"abc"), content_length=None
    )
    assert b"".join(stream) == b"abc"


def test_strict_length_stream_passes_through_without_body() -> None:
    # HEAD / no-body status - pass through untouched even with a length.
    stream = httpx.StrictLengthStream(
        httpx.ByteStream(b""), content_length=5, expects_body=False
    )
    assert b"".join(stream) == b""


def test_strict_length_stream_close_once_on_error() -> None:
    counting = CountingStream([b"abc"])
    stream = httpx.StrictLengthStream(counting, content_length=5)

    with pytest.raises(httpx.ContentLengthError):
        b"".join(stream)

    assert counting.close_count == 1
    # Closing again must not close the underlying stream a second time.
    stream.close()
    stream.close()
    assert counting.close_count == 1


def test_strict_length_stream_close_once_on_cancel() -> None:
    counting = CountingStream([b"ab", b"cd", b"ef"])
    stream = httpx.StrictLengthStream(counting, content_length=6)

    iterator = iter(stream)
    next(iterator)
    # Abandon the iterator early (cancellation).
    del iterator

    assert counting.close_count == 1


def test_strict_length_stream_cannot_be_consumed_twice() -> None:
    stream = httpx.StrictLengthStream(
        httpx.ByteStream(b"abc"), content_length=3
    )
    b"".join(stream)
    with pytest.raises(httpx.StreamConsumed):
        b"".join(stream)


# ---------------------------------------------------------------------------
# Header parsing tests
# ---------------------------------------------------------------------------


def test_parse_content_length_absent() -> None:
    assert httpx.parse_content_length(httpx.Headers()) is None


def test_parse_content_length_single() -> None:
    assert httpx.parse_content_length(
        httpx.Headers({"Content-Length": "5"})
    ) == 5


def test_parse_content_length_duplicate_identical() -> None:
    # Duplicate headers with the same value are accepted.
    headers = httpx.Headers(
        [("Content-Length", "5"), ("Content-Length", "5")]
    )
    assert httpx.parse_content_length(headers) == 5


def test_parse_content_length_comma_identical() -> None:
    headers = httpx.Headers({"Content-Length": "5, 5"})
    assert httpx.parse_content_length(headers) == 5


def test_parse_content_length_conflicting() -> None:
    headers = httpx.Headers(
        [("Content-Length", "5"), ("Content-Length", "7")]
    )
    with pytest.raises(httpx.ContentLengthError) as exc_info:
        httpx.parse_content_length(headers)
    assert exc_info.value.reason == "conflicting_headers"


def test_parse_content_length_malformed() -> None:
    with pytest.raises(httpx.ContentLengthError) as exc_info:
        httpx.parse_content_length(
            httpx.Headers({"Content-Length": "not-a-number"})
        )
    assert exc_info.value.reason == "malformed_headers"


def test_parse_content_length_negative() -> None:
    with pytest.raises(httpx.ContentLengthError) as exc_info:
        httpx.parse_content_length(
            httpx.Headers({"Content-Length": "-1"})
        )
    assert exc_info.value.reason == "malformed_headers"


def test_response_expects_body() -> None:
    request = httpx.Request("GET", "https://example.org")
    head_request = httpx.Request("HEAD", "https://example.org")

    assert httpx.response_expects_body(
        request, httpx.Response(200)
    ) is True
    assert httpx.response_expects_body(
        head_request, httpx.Response(200)
    ) is False
    assert httpx.response_expects_body(
        request, httpx.Response(204)
    ) is False
    assert httpx.response_expects_body(
        request, httpx.Response(304)
    ) is False
    assert httpx.response_expects_body(
        request, httpx.Response(100)
    ) is False
    assert httpx.response_expects_body(
        request, httpx.Response(199)
    ) is False


# ---------------------------------------------------------------------------
# Transport integration tests (MockTransport + counting stream)
# ---------------------------------------------------------------------------


def test_transport_short_read() -> None:
    counting = CountingStream([b"abc"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "5"}, stream=counting
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(httpx.ContentLengthError) as exc_info:
            client.get("https://example.org")

    error = exc_info.value
    assert error.reason == "short_read"
    assert error.declared_length == 5
    assert error.received_length == 3
    assert error.request is not None
    assert counting.close_count == 1


def test_transport_exact_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "5"}, content=b"Hello"
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        response = client.get("https://example.org")
    assert response.content == b"Hello"


def test_transport_over_read_immediate() -> None:
    counting = ImmediateOverReadStream(b"abcdef")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "5"}, stream=counting
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        with client.stream("GET", "https://example.org") as response:
            iterator = response.iter_raw()
            with pytest.raises(httpx.ContentLengthError) as exc_info:
                next(iterator)

    error = exc_info.value
    assert error.reason == "over_read"
    assert error.declared_length == 5
    assert error.received_length == 6
    assert counting.close_count == 1


def test_transport_gzip_uses_raw_length() -> None:
    # The declared length is the *compressed* length; the decoded text is
    # longer. Validation must compare the raw bytes, not the decoded text.
    raw = gzip.compress("你好，世界！这是一段用于测试的中文内容。".encode("utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(raw)),
                "Content-Encoding": "gzip",
            },
            content=raw,
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        response = client.get("https://example.org")

    decoded = "你好，世界！这是一段用于测试的中文内容。".encode("utf-8")
    assert response.content == decoded  # decoded (decompressed) content
    # decoded length differs from the declared raw length
    assert len(response.content) != len(raw)


def test_transport_head_with_length_no_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        return httpx.Response(
            200, headers={"Content-Length": "5"}, content=b""
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        response = client.head("https://example.org")
    assert response.content == b""


def test_transport_no_content_length_passes() -> None:
    # Chunked encoding carries no Content-Length - pass through.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Transfer-Encoding": "chunked"}, content=b"abc"
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        response = client.get("https://example.org")
    assert response.content == b"abc"


def test_transport_no_body_status_passes() -> None:
    for status_code in (204, 304, 100):
        def handler(request: httpx.Request, code: int = status_code) -> httpx.Response:
            return httpx.Response(
                code, headers={"Content-Length": "5"}, content=b""
            )

        transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
        with httpx.Client(transport=transport) as client:
            response = client.get("https://example.org")
        assert response.status_code == status_code


def test_transport_conflicting_headers_rejected_eagerly() -> None:
    counting = CountingStream([b"abcdefg"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Length", "5"), ("Content-Length", "7")],
            stream=counting,
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(httpx.ContentLengthError) as exc_info:
            client.get("https://example.org")

    assert exc_info.value.reason == "conflicting_headers"
    # The body must not have been consumed and the stream must be closed.
    assert counting.bytes_sent == 0
    assert counting.close_count == 1


def test_transport_malformed_header_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "not-a-number"},
            content=b"abc",
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(httpx.ContentLengthError) as exc_info:
            client.get("https://example.org")
    assert exc_info.value.reason == "malformed_headers"


def test_transport_close_once_on_cancel() -> None:
    counting = CountingStream([b"ab", b"cd", b"ef"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "6"}, stream=counting
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        with client.stream("GET", "https://example.org") as response:
            iterator = response.iter_raw()
            assert next(iterator) == b"ab"
            # Abandon the response without reading the rest.

    assert counting.close_count == 1
    response.close()
    assert counting.close_count == 1


def test_transport_does_not_buffer_body() -> None:
    counting = CountingStream([b"ab", b"cd", b"ef"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "6"}, stream=counting
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        with client.stream("GET", "https://example.org") as response:
            iterator = response.iter_raw()
            assert next(iterator) == b"ab"
            # Only the first chunk has been pulled through - the whole body
            # is never read into memory.
            assert counting.bytes_sent == 2
            assert next(iterator) == b"cd"
            assert counting.bytes_sent == 4
            assert next(iterator) == b"ef"
            assert counting.bytes_sent == 6


def test_default_client_unchanged() -> None:
    # Without the strict transport a short read must not raise.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "5"}, content=b"abc"
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = client.get("https://example.org")
    assert response.content == b"abc"


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_strict_length_stream_short_read() -> None:
    stream = httpx.AsyncStrictLengthStream(
        httpx.ByteStream(b"abc"), content_length=5
    )
    with pytest.raises(httpx.ContentLengthError) as exc_info:
        await _join_async(stream)

    error = exc_info.value
    assert error.reason == "short_read"
    assert error.declared_length == 5
    assert error.received_length == 3


@pytest.mark.anyio
async def test_async_strict_length_stream_over_read_immediate() -> None:
    stream = httpx.AsyncStrictLengthStream(
        AsyncImmediateOverReadStream(b"abcdef"), content_length=5
    )
    iterator = stream.__aiter__()
    with pytest.raises(httpx.ContentLengthError) as exc_info:
        await iterator.__anext__()

    error = exc_info.value
    assert error.reason == "over_read"
    assert error.received_length == 6


@pytest.mark.anyio
async def test_async_strict_length_stream_close_once_on_error() -> None:
    counting = AsyncCountingStream([b"abc"])
    stream = httpx.AsyncStrictLengthStream(counting, content_length=5)

    with pytest.raises(httpx.ContentLengthError):
        await _join_async(stream)

    assert counting.close_count == 1
    await stream.aclose()
    await stream.aclose()
    assert counting.close_count == 1


@pytest.mark.anyio
async def test_async_transport_short_read() -> None:
    counting = AsyncCountingStream([b"abc"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "5"}, stream=counting
        )

    transport = httpx.AsyncStrictLengthTransport(httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ContentLengthError) as exc_info:
            await client.get("https://example.org")

    error = exc_info.value
    assert error.reason == "short_read"
    assert error.declared_length == 5
    assert error.received_length == 3
    assert error.request is not None
    assert counting.close_count == 1


@pytest.mark.anyio
async def test_async_transport_over_read_immediate() -> None:
    counting = AsyncImmediateOverReadStream(b"abcdef")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "5"}, stream=counting
        )

    transport = httpx.AsyncStrictLengthTransport(httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://example.org") as response:
            iterator = response.aiter_raw()
            with pytest.raises(httpx.ContentLengthError) as exc_info:
                await iterator.__anext__()

    error = exc_info.value
    assert error.reason == "over_read"
    assert error.received_length == 6
    assert counting.close_count == 1


@pytest.mark.anyio
async def test_async_transport_gzip_uses_raw_length() -> None:
    raw = gzip.compress("你好，世界！这是一段用于测试的中文内容。".encode("utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(raw)),
                "Content-Encoding": "gzip",
            },
            content=raw,
        )

    transport = httpx.AsyncStrictLengthTransport(httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://example.org")

    decoded = "你好，世界！这是一段用于测试的中文内容。".encode("utf-8")
    assert response.content == decoded
    assert len(response.content) != len(raw)


@pytest.mark.anyio
async def test_async_transport_head_with_length_no_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        return httpx.Response(
            200, headers={"Content-Length": "5"}, content=b""
        )

    transport = httpx.AsyncStrictLengthTransport(httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.head("https://example.org")
    assert response.content == b""


@pytest.mark.anyio
async def test_async_transport_conflicting_headers_rejected() -> None:
    counting = AsyncCountingStream([b"abcdefg"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Length", "5"), ("Content-Length", "7")],
            stream=counting,
        )

    transport = httpx.AsyncStrictLengthTransport(httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ContentLengthError) as exc_info:
            await client.get("https://example.org")

    assert exc_info.value.reason == "conflicting_headers"
    assert counting.bytes_sent == 0
    assert counting.close_count == 1


@pytest.mark.anyio
async def test_async_strict_length_stream_close_once_on_cancel() -> None:
    # Abandoning the wrapper's async generator early must close the
    # underlying stream exactly once.
    counting = AsyncCountingStream([b"ab", b"cd", b"ef"])
    stream = httpx.AsyncStrictLengthStream(counting, content_length=6)

    iterator = stream.__aiter__()
    assert await iterator.__anext__() == b"ab"
    # Close the generator early (cancellation).
    await iterator.aclose()

    assert counting.close_count == 1
    await stream.aclose()
    assert counting.close_count == 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _join_async(stream: httpx.AsyncByteStream) -> bytes:
    return b"".join([chunk async for chunk in stream])
