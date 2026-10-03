import gzip

import pytest

import httpx
from httpx._content_length import enforce_content_length

# ---------------------------------------------------------------------------
# Fixtures: counting streams that record how many chunks they yielded and how
# many times they were closed, plus MockTransport handlers.
# ---------------------------------------------------------------------------


class CountingSyncStream(httpx.SyncByteStream):
    """
    A sync stream that yields `chunks` one at a time, counting the number
    of yielded chunks and the number of `close()` calls on the underlying
    stream.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.num_chunks_yielded = 0
        self.num_closed = 0

    def __iter__(self):
        for chunk in self.chunks:
            self.num_chunks_yielded += 1
            yield chunk

    def close(self) -> None:
        self.num_closed += 1


class CountingAsyncStream(httpx.AsyncByteStream):
    """
    An async counterpart of `CountingSyncStream`.

    Implemented as an explicit async iterator (rather than an async
    generator) so that, if a test ever leaves it unclosed, no "ignored
    exception in async generator" warning leaks out of the test run.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.num_chunks_yielded = 0
        self.num_closed = 0

    def __aiter__(self) -> "CountingAsyncStream":
        self._index = 0
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self._index]
        self._index += 1
        self.num_chunks_yielded += 1
        return chunk

    async def aclose(self) -> None:
        self.num_closed += 1


def streaming_response(
    method: str = "GET",
    chunks: list[bytes] | None = None,
    content_length: str | None = None,
    status_code: int = 200,
    headers: list[tuple[str, str]] | None = None,
    async_stream: bool = False,
) -> httpx.Response:
    raw_headers: list[tuple[str, str]] = list(headers or [])
    if content_length is not None:
        raw_headers.append(("Content-Length", content_length))
    chunks = chunks or []
    if async_stream:
        stream: httpx.AsyncByteStream | httpx.SyncByteStream = CountingAsyncStream(
            chunks
        )
    else:
        stream = CountingSyncStream(chunks)
    return httpx.Response(
        status_code,
        headers=raw_headers,
        stream=stream,
        request=httpx.Request(method, "https://example.org/"),
    )


def streaming_handler(chunks: list[bytes], content_length: str | int | None, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        return streaming_response(
            method=request.method,
            chunks=chunks,
            content_length=None if content_length is None else str(content_length),
            **kwargs,
        )

    return handler


# ---------------------------------------------------------------------------
# Sync client tests
# ---------------------------------------------------------------------------


def test_short_read_raises_content_length_too_short() -> None:
    # Declared length 5, body only gives "abc".
    transport = httpx.MockTransport(streaming_handler([b"abc"], 5))
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        with pytest.raises(httpx.ContentLengthTooShort) as exc_info:
            client.get("https://example.org/")

    error = exc_info.value
    assert error.expected == 5
    assert error.received == 3
    assert error.request.method == "GET"
    # Short reads are RemoteProtocolErrors, and therefore RequestErrors.
    assert isinstance(error, httpx.RemoteProtocolError)


def test_exact_length_passes_sync() -> None:
    transport = httpx.MockTransport(streaming_handler([b"ab", b"c", b"de"], 5))
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = client.get("https://example.org/")
    assert response.content == b"abcde"


def test_extra_bytes_raise_content_length_too_long_during_iteration() -> None:
    raw_stream = CountingSyncStream([b"abc", b"extra"])
    response = httpx.Response(
        200,
        headers={"Content-Length": "3"},
        stream=raw_stream,
        request=httpx.Request("GET", "https://example.org/"),
    )
    enforce_content_length(response)

    iterator = response.iter_raw()
    assert next(iterator) == b"abc"
    # The overshoot must fail at once, rather than at end-of-stream.
    with pytest.raises(httpx.ContentLengthTooLong) as exc_info:
        next(iterator)
    assert exc_info.value.expected == 3
    assert exc_info.value.received == 8

    # The surplus bytes were never handed to the caller.
    assert raw_stream.num_chunks_yielded == 2
    # Closing the failed response closes the underlying stream exactly once.
    response.close()
    assert raw_stream.num_closed == 1


def test_too_long_via_client_raises_and_closes_once() -> None:
    raw_stream = CountingSyncStream([b"abc", b"extra"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "3"},
            stream=raw_stream,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        with pytest.raises(httpx.ContentLengthTooLong):
            client.get("https://example.org/")

    assert raw_stream.num_closed == 1


def test_gzip_body_validated_against_raw_compressed_length() -> None:
    # Chinese text so the decoded size and the raw (gzip) size clearly differ.
    body = "你好，世界".encode("utf-8")
    compressed = gzip.compress(body)
    assert len(compressed) != len(body)

    raw_stream = CountingSyncStream([compressed[:10], compressed[10:]])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(compressed)),
                "Content-Encoding": "gzip",
            },
            stream=raw_stream,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = client.get("https://example.org/")
        # Decoded content differs in length from the declared, raw length.
        assert response.content == body
        assert response.text == "你好，世界"

    assert raw_stream.num_closed == 1


def test_gzip_short_read_uses_raw_compressed_length() -> None:
    body = "你好，世界".encode("utf-8")
    compressed = gzip.compress(body)
    truncated = compressed[:-5]

    raw_stream = CountingSyncStream([truncated])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(compressed)),
                "Content-Encoding": "gzip",
            },
            stream=raw_stream,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        with pytest.raises(httpx.ContentLengthTooShort) as exc_info:
            client.get("https://example.org/")

    assert exc_info.value.expected == len(compressed)
    assert exc_info.value.received == len(truncated)


def test_head_response_with_length_and_no_body_passes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "1234"},
            stream=httpx.ByteStream(b""),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = client.head("https://example.org/")
    assert response.status_code == 200
    assert response.headers["Content-Length"] == "1234"
    assert response.content == b""


@pytest.mark.parametrize("status_code", [100, 101, 204, 304])
def test_bodyless_status_codes_pass_with_length(status_code: int) -> None:
    transport = httpx.MockTransport(streaming_handler([], 10, status_code=status_code))
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = client.get("https://example.org/")
    assert response.status_code == status_code
    assert response.content == b""


def test_duplicate_disagreeing_content_length_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Length", "5"), ("Content-Length", "6")],
            stream=httpx.ByteStream(b"hello"),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        with pytest.raises(httpx.InvalidContentLength) as exc_info:
            client.get("https://example.org/")
    assert set(exc_info.value.values) == {"5", "6"}


def test_duplicate_agreeing_content_length_is_accepted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Length", "5"), ("Content-Length", "5")],
            stream=httpx.ByteStream(b"hello"),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = client.get("https://example.org/")
    assert response.content == b"hello"


def test_malformed_content_length_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "abc"},
            stream=httpx.ByteStream(b"hello"),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        with pytest.raises(httpx.InvalidContentLength):
            client.get("https://example.org/")


def test_transfer_encoding_takes_precedence_over_content_length() -> None:
    # Content-Length disagrees with the framed body, but Transfer-Encoding
    # is present, so Content-Length does not frame the message.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Length": "999",
                "Transfer-Encoding": "chunked",
            },
            stream=CountingSyncStream([b"hello"]),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = client.get("https://example.org/")
    assert response.content == b"hello"


def test_missing_content_length_is_not_validated() -> None:
    transport = httpx.MockTransport(streaming_handler([b"abc"], None))
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = client.get("https://example.org/")
    assert response.content == b"abc"


def test_default_client_without_hook_keeps_lenient_behaviour() -> None:
    # A short read without the hook must not raise - default behaviour
    # is unchanged.
    transport = httpx.MockTransport(streaming_handler([b"abc"], 5))
    with httpx.Client(transport=transport) as client:
        response = client.get("https://example.org/")
    assert response.content == b"abc"


def test_abandoned_sync_iteration_closes_stream_once() -> None:
    raw_stream = CountingSyncStream([b"ab", b"cde", b"f"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "6"},
            stream=raw_stream,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        with client.stream("GET", "https://example.org/") as response:
            chunks = []
            for chunk in response.iter_raw():
                chunks.append(chunk)
                if len(chunks) == 2:
                    # Give up midway; the context manager closes the
                    # response on the way out.
                    break
    assert raw_stream.num_closed == 1


def test_length_check_does_not_buffer_stream() -> None:
    # Many large chunks: merely installing the hook and iterating must not
    # materialise the whole body. We assert only one chunk at a time is ever
    # pulled from the underlying stream.
    chunks = [b"x" * 1024 for _ in range(5)]
    raw_stream = CountingSyncStream(chunks)
    response = httpx.Response(
        200,
        headers={"Content-Length": str(1024 * 5)},
        stream=raw_stream,
        request=httpx.Request("GET", "https://example.org/"),
    )
    enforce_content_length(response)

    iterator = response.iter_raw()
    next(iterator)
    assert raw_stream.num_chunks_yielded == 1
    # Drain the rest so cleanup runs.
    for _ in iterator:
        pass
    assert raw_stream.num_closed == 1


def test_materialised_response_short_body_is_rejected_by_hook() -> None:
    # A Response(content=...) already holds its body; the hook validates
    # it eagerly instead of wrapping an uniterated stream.
    response = httpx.Response(
        200,
        headers={"Content-Length": "5"},
        content=b"abc",
    )
    response.request = httpx.Request("GET", "https://example.org/")
    with pytest.raises(httpx.ContentLengthTooShort) as exc_info:
        enforce_content_length(response)
    assert exc_info.value.expected == 5
    assert exc_info.value.received == 3


def test_materialised_response_exact_length_passes() -> None:
    response = httpx.Response(200, headers={"Content-Length": "5"}, content=b"hello")
    response.request = httpx.Request("GET", "https://example.org/")
    enforce_content_length(response)  # must not raise
    assert response.content == b"hello"


# ---------------------------------------------------------------------------
# Async client tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_short_read_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return streaming_response(
            chunks=[b"abc"],
            content_length="5",
            async_stream=True,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        with pytest.raises(httpx.ContentLengthTooShort) as exc_info:
            await client.get("https://example.org/")
    assert exc_info.value.expected == 5
    assert exc_info.value.received == 3


@pytest.mark.anyio
async def test_async_exact_length_passes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return streaming_response(
            chunks=[b"ab", b"cde"],
            content_length="5",
            async_stream=True,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = await client.get("https://example.org/")
    assert response.content == b"abcde"


@pytest.mark.anyio
async def test_async_extra_bytes_fail_immediately_and_close_once() -> None:
    raw_stream = CountingAsyncStream([b"abc", b"extra"])
    response = httpx.Response(
        200,
        headers={"Content-Length": "3"},
        stream=raw_stream,
        request=httpx.Request("GET", "https://example.org/"),
    )
    enforce_content_length(response)

    iterator = response.aiter_raw()
    assert await iterator.__anext__() == b"abc"
    with pytest.raises(httpx.ContentLengthTooLong):
        await iterator.__anext__()

    assert raw_stream.num_chunks_yielded == 2
    # Closing the failed response closes the underlying stream once.
    await response.aclose()
    assert raw_stream.num_closed == 1


@pytest.mark.anyio
async def test_async_gzip_raw_length_not_decoded_length() -> None:
    body = "你好，世界".encode("utf-8")
    compressed = gzip.compress(body)

    def handler(request: httpx.Request) -> httpx.Response:
        return streaming_response(
            chunks=[compressed],
            content_length=str(len(compressed)),
            headers=[("Content-Encoding", "gzip")],
            async_stream=True,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = await client.get("https://example.org/")
    assert response.text == "你好，世界"


@pytest.mark.anyio
async def test_async_head_with_length_passes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return streaming_response(
            method=request.method,
            chunks=[],
            content_length="1234",
            async_stream=True,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        response = await client.head("https://example.org/")
    assert response.content == b""
    assert response.headers["Content-Length"] == "1234"


@pytest.mark.anyio
async def test_async_abandoned_iteration_closes_stream_once() -> None:
    raw_stream = CountingAsyncStream([b"ab", b"cde", b"f"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "6"},
            stream=raw_stream,
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        async with client.stream("GET", "https://example.org/") as response:
            seen = 0
            aiterator = response.aiter_raw()
            async for chunk in aiterator:
                seen += len(chunk)
                if seen >= 2:
                    break
            # The caller abandoned the stream; close the async iterator
            # explicitly (mirrors how a generator-returning sync iterator
            # gets closed on its way out).
            await aiterator.aclose()
    assert raw_stream.num_closed == 1


@pytest.mark.anyio
async def test_async_duplicate_conflicting_length_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Length", "5"), ("Content-Length", "7")],
            stream=httpx.ByteStream(b"hello"),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, event_hooks={"response": [enforce_content_length]}
    ) as client:
        with pytest.raises(httpx.InvalidContentLength):
            await client.get("https://example.org/")
