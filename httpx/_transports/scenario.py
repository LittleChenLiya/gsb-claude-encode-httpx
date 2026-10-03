"""
ScenarioTransport: a local transport that matches requests in declaration
order and returns preset response streams, without making any real network
requests.

Each *step* in the scenario defines:

* A request matcher (method, URL, and selected headers).
* A preset response (status code, headers, and a sequence of chunks).

Each chunk can either be yielded normally, pause at a *barrier* before being
yielded, or raise an explicit exception. The same scenario definition is used
for both sync and async clients.

If a request does not match the current step, a ScenarioMismatch is raised
reporting the expected and actual values, and the step is *not* consumed. The
next request will retry against the same step.

After a response is closed (or fails mid-stream), the step resources are
released and the transport records the close. A failed call never silently
advances to the next step.
"""

from __future__ import annotations

import asyncio
import threading
import typing

from .._models import Request, Response
from .._types import AsyncByteStream, SyncByteStream
from .._exceptions import StreamClosed
from .._urls import URL
from .base import AsyncBaseTransport, BaseTransport

__all__ = [
    "Barrier",
    "Chunk",
    "ScenarioMismatch",
    "ScenarioStep",
    "ScenarioTransport",
]


class Barrier:
    """
    A synchronization point that pauses the response stream until released.

    The stream sets the *reached* event when it arrives at the barrier, then
    waits for the *release* event. Test code can wait until the stream has
    reached the barrier (e.g. to cancel an in-flight async read) and then
    release the barrier.

    Both sync and async interfaces are provided so that a single scenario
    definition can drive either client.
    """

    def __init__(self) -> None:
        self._async_reached = asyncio.Event()
        self._async_release = asyncio.Event()
        self._sync_reached = threading.Event()
        self._sync_release = threading.Event()

    async def wait_async(self) -> None:
        """Called by the async stream before yielding a chunk."""
        self._async_reached.set()
        await self._async_release.wait()

    def wait_sync(self) -> None:
        """Called by the sync stream before yielding a chunk."""
        self._sync_reached.set()
        self._sync_release.wait()

    async def wait_until_reached_async(self) -> None:
        """Block until the async stream reaches the barrier."""
        await self._async_reached.wait()

    def wait_until_reached_sync(self) -> None:
        """Block until the sync stream reaches the barrier."""
        self._sync_reached.wait()

    def release(self) -> None:
        """Release the barrier, allowing the stream to continue."""
        self._async_release.set()
        self._sync_release.set()


class Chunk:
    """
    A single chunk of a preset response stream.

    * `data` is the bytes yielded for this chunk.
    * `barrier` (if set) is a `Barrier` that the stream waits at before
      yielding the chunk.
    * `error` (if set) is an exception instance that the stream raises
      instead of yielding the chunk.
    """

    def __init__(
        self,
        data: bytes = b"",
        *,
        barrier: Barrier | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.data = data
        self.barrier = barrier
        self.error = error


class ScenarioMismatch(Exception):
    """
    Raised when a request does not match the current scenario step.

    Reports the expected and actual values for method, URL, and any selected
    headers. The step is *not* consumed, so the next request retries against
    the same step.
    """

    def __init__(
        self,
        *,
        expected_method: str,
        actual_method: str,
        expected_url: URL,
        actual_url: URL,
        expected_headers: dict[str, str],
        actual_headers: dict[str, str],
    ) -> None:
        self.expected_method = expected_method
        self.actual_method = actual_method
        self.expected_url = expected_url
        self.actual_url = actual_url
        self.expected_headers = expected_headers
        self.actual_headers = actual_headers

        lines = [
            "Scenario step mismatch:",
            f"  method: expected {expected_method!r}, got {actual_method!r}",
            f"  url:    expected {expected_url!r}, got {actual_url!r}",
        ]
        for key, expected_value in expected_headers.items():
            actual_value = actual_headers.get(key)
            lines.append(
                f"  header {key!r}: expected {expected_value!r}, got {actual_value!r}"
            )
        super().__init__("\n".join(lines))


class ScenarioStep:
    """
    A single step in a scenario: a request matcher plus a preset response.

    The step matches on method, URL, and any selected headers. Only the
    headers explicitly provided are checked; other headers are ignored.
    """

    def __init__(
        self,
        method: str,
        url: str | URL,
        *,
        headers: typing.Mapping[str, str] | None = None,
        status_code: int = 200,
        response_headers: typing.Mapping[str, str] | None = None,
        chunks: typing.Sequence[Chunk] | None = None,
    ) -> None:
        self.method = method.upper()
        self.url = URL(url)
        self.headers: dict[str, str] = dict(headers) if headers else {}
        self.status_code = status_code
        self.response_headers: dict[str, str] = (
            dict(response_headers) if response_headers else {}
        )
        self.chunks: list[Chunk] = list(chunks) if chunks else []

    def matches(
        self, request: Request
    ) -> tuple[bool, dict[str, str], dict[str, str]]:
        """
        Check whether the request matches this step.

        Returns a tuple of (matched, expected_headers, actual_headers).
        """
        actual_headers: dict[str, str] = {}
        for key in self.headers:
            actual_headers[key] = request.headers.get(key)

        matched = (
            request.method == self.method
            and request.url == self.url
            and all(
                request.headers.get(key) == value
                for key, value in self.headers.items()
            )
        )
        return matched, dict(self.headers), actual_headers

    def build_response(
        self,
        request: Request,
        *,
        on_close: typing.Callable[[], None],
    ) -> Response:
        """
        Build a Response for this step with a stream over the preset chunks.
        """
        stream = _SyncScenarioStream(
            chunks=self.chunks,
            on_close=on_close,
        )
        return Response(
            self.status_code,
            headers=self.response_headers,
            stream=stream,
            request=request,
        )

    def build_async_response(
        self,
        request: Request,
        *,
        on_close: typing.Callable[[], typing.Awaitable[None]],
    ) -> Response:
        """
        Build a Response for this step with an async stream over the chunks.
        """
        stream = _AsyncScenarioStream(
            chunks=self.chunks,
            on_close=on_close,
        )
        return Response(
            self.status_code,
            headers=self.response_headers,
            stream=stream,
            request=request,
        )


class _SyncScenarioStream(SyncByteStream):
    """
    A sync byte stream that yields the preset chunks, waiting at barriers and
    raising errors as configured.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        on_close: typing.Callable[[], None],
    ) -> None:
        self._chunks = chunks
        self._on_close = on_close
        self._closed = False

    def __iter__(self) -> typing.Iterator[bytes]:
        for chunk in self._chunks:
            if self._closed:
                raise StreamClosed()
            if chunk.barrier is not None:
                chunk.barrier.wait_sync()
            if chunk.error is not None:
                raise chunk.error
            yield chunk.data

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            # Release any barrier the step may still be holding, so that a
            # partially-read response does not leave test code hanging.
            for chunk in self._chunks:
                if chunk.barrier is not None:
                    chunk.barrier.release()
            self._on_close()


class _AsyncScenarioStream(AsyncByteStream):
    """
    An async byte stream that yields the preset chunks, waiting at barriers
    and raising errors as configured.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        on_close: typing.Callable[[], typing.Awaitable[None]],
    ) -> None:
        self._chunks = chunks
        self._on_close = on_close
        self._closed = False

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        for chunk in self._chunks:
            if self._closed:
                raise StreamClosed()
            if chunk.barrier is not None:
                await chunk.barrier.wait_async()
            if chunk.error is not None:
                raise chunk.error
            yield chunk.data

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            # Release any barrier the step may still be holding, so that a
            # partially-read response does not leave test code hanging in the
            # background.
            for chunk in self._chunks:
                if chunk.barrier is not None:
                    chunk.barrier.release()
            self._on_close()


class ScenarioTransport(AsyncBaseTransport, BaseTransport):
    """
    A transport that matches requests in declaration order and returns preset
    response streams, without making any real network requests.

    The scenario is a list of `ScenarioStep` instances. Each request is matched
    against the current step. On mismatch a `ScenarioMismatch` is raised and
    the step is not consumed. Once a step matches, the transport advances to
    the next step for the following request.

    The transport tracks the number of times a response stream is closed
    (`close_count`) and the number of bytes delivered across all responses
    (`delivered_bytes`).
    """

    def __init__(self, steps: typing.Sequence[ScenarioStep]) -> None:
        self.steps: list[ScenarioStep] = list(steps)
        self._index = 0
        self._closed = False
        self.close_count = 0
        self.delivered_bytes = 0

    def _on_close(self) -> None:
        self.close_count += 1

    def _count_bytes(self, data: bytes) -> None:
        self.delivered_bytes += len(data)

    def handle_request(self, request: Request) -> Response:
        if self._closed:
            raise RuntimeError(
                "Cannot send a request, as the transport has been closed."
            )
        if self._index >= len(self.steps):
            raise RuntimeError(
                "Scenario exhausted: no more steps, but a request was made."
            )

        step = self.steps[self._index]
        matched, expected_headers, actual_headers = step.matches(request)
        if not matched:
            raise ScenarioMismatch(
                expected_method=step.method,
                actual_method=request.method,
                expected_url=step.url,
                actual_url=request.url,
                expected_headers=expected_headers,
                actual_headers=actual_headers,
            )

        self._index += 1
        response = step.build_response(request, on_close=self._on_close)
        # Wrap the stream so we can count delivered bytes as they are yielded.
        response.stream = _CountingSyncStream(
            response.stream, on_bytes=self._count_bytes
        )
        return response

    async def handle_async_request(self, request: Request) -> Response:
        if self._closed:
            raise RuntimeError(
                "Cannot send a request, as the transport has been closed."
            )
        if self._index >= len(self.steps):
            raise RuntimeError(
                "Scenario exhausted: no more steps, but a request was made."
            )

        step = self.steps[self._index]
        matched, expected_headers, actual_headers = step.matches(request)
        if not matched:
            raise ScenarioMismatch(
                expected_method=step.method,
                actual_method=request.method,
                expected_url=step.url,
                actual_url=request.url,
                expected_headers=expected_headers,
                actual_headers=actual_headers,
            )

        self._index += 1
        response = step.build_async_response(request, on_close=self._on_close)
        response.stream = _CountingAsyncStream(
            response.stream, on_bytes=self._count_bytes
        )
        return response

    def close(self) -> None:
        self._closed = True

    async def aclose(self) -> None:
        self._closed = True


class _CountingSyncStream(SyncByteStream):
    """Wraps a sync stream to count delivered bytes."""

    def __init__(
        self,
        stream: SyncByteStream,
        *,
        on_bytes: typing.Callable[[bytes], None],
    ) -> None:
        self._stream = stream
        self._on_bytes = on_bytes

    def __iter__(self) -> typing.Iterator[bytes]:
        for chunk in self._stream:
            self._on_bytes(chunk)
            yield chunk

    def close(self) -> None:
        self._stream.close()


class _CountingAsyncStream(AsyncByteStream):
    """Wraps an async stream to count delivered bytes."""

    def __init__(
        self,
        stream: AsyncByteStream,
        *,
        on_bytes: typing.Callable[[bytes], None],
    ) -> None:
        self._stream = stream
        self._on_bytes = on_bytes

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        async for chunk in self._stream:
            self._on_bytes(chunk)
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()
