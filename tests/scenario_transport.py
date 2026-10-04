"""
A scripted, offline transport for httpx client tests.

ScenarioTransport (sync) and AsyncScenarioTransport (async) share a single
scenario *definition*: an ordered list of Step objects. Each incoming request
is matched against the *current* step -- its method, URL and a selected
subset of headers are checked. A mismatch raises ScenarioMismatch describing
both the expected and the actual values *without consuming the step*, so a
corrected retry still meets the same step.

A step's response body is declared as a list of chunks. Before any chunk an
action may be injected via ``before={index: action}``:

* a Barrier -- the read blocks until the barrier is released, or
* an exception instance -- the read raises that exact exception.

Closing a transport forbids any further consumption: new requests raise
TransportClosed and still-open response streams are closed (which releases
the resources held by their step). Every delivered byte and every stream
close() is accounted on the step's StepRun records, so tests can assert
exactly what happened.
"""

from __future__ import annotations

import threading
import typing

import anyio

import httpx

__all__ = [
    "AsyncScenarioTransport",
    "Barrier",
    "ScenarioError",
    "ScenarioExhausted",
    "ScenarioMismatch",
    "ScenarioTransport",
    "Step",
    "StepRun",
    "TransportClosed",
]


class ScenarioError(AssertionError):
    """Base class for scenario transport failures."""


class ScenarioMismatch(ScenarioError):
    """A request did not match the current step. The step is NOT consumed."""


class ScenarioExhausted(ScenarioError):
    """A request arrived but the scenario has no steps left."""


class TransportClosed(ScenarioError):
    """The transport (or one of its streams) was used after close."""


class Barrier:
    """
    A named gate that blocks chunk delivery until release() is called.

    The same barrier object works for the sync transport (threading.Event)
    and the async transport (an anyio.Event created lazily inside the
    async context).
    """

    def __init__(self, name: str = "barrier") -> None:
        self.name = name
        self.released = False
        self._sync_event = threading.Event()
        self._async_event: anyio.Event | None = None

    def __repr__(self) -> str:  # pragma: no cover
        state = "released" if self.released else "pending"
        return f"<Barrier {self.name!r} {state}>"

    def wait(self) -> None:
        """Block a sync reader until the barrier is released."""
        self._sync_event.wait()

    async def wait_async(self) -> None:
        """Block an async reader until the barrier is released.

        This is a cancellation point: cancelling the waiter unwinds the
        response stream, which reclaims the step's resources inline.
        """
        if self.released:
            return
        if self._async_event is None:
            self._async_event = anyio.Event()
        await self._async_event.wait()

    def release(self) -> None:
        self.released = True
        self._sync_event.set()
        if self._async_event is not None:
            self._async_event.set()


# An action injected before a chunk: either a barrier to wait on, or an
# exception instance to raise.
Action = typing.Union[Barrier, BaseException]


class StepRun:
    """Runtime record for one matched step: delivered bytes and closes."""

    def __init__(self, step: Step) -> None:
        self.step = step
        self.delivered_bytes = 0
        self.close_count = 0
        self.closed = False

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<StepRun {self.step.method} {self.step.url} "
            f"delivered={self.delivered_bytes} closes={self.close_count}>"
        )


class Step:
    """
    One expected request and its scripted response.

    ``headers`` selects the request headers that must match (a subset).
    ``chunks`` is the response body, delivered piece by piece.
    ``before`` maps a chunk index to an action (or list of actions) to run
    *before* that chunk is delivered: a Barrier waits, an exception
    instance is raised.
    """

    def __init__(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        status: int = 200,
        response_headers: list[tuple[str, str]] | None = None,
        chunks: typing.Iterable[bytes] = (),
        before: dict[int, Action | list[Action]] | None = None,
    ) -> None:
        self.method = method
        self.url = url
        self.headers = dict(headers or {})
        self.status = status
        self.response_headers = list(response_headers or [])
        self.chunks = [bytes(chunk) for chunk in chunks]
        self.before: dict[int, list[Action]] = {}
        for index, actions in (before or {}).items():
            if not 0 <= index <= len(self.chunks):
                raise ValueError(f"before index {index} out of range for {self!r}")
            if isinstance(actions, (Barrier, BaseException)):
                actions = [actions]
            self.before[index] = list(actions)
        self.runs: list[StepRun] = []

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Step {self.method} {self.url}>"

    @property
    def delivered_bytes(self) -> int:
        return sum(run.delivered_bytes for run in self.runs)

    @property
    def close_count(self) -> int:
        return sum(run.close_count for run in self.runs)

    def check(self, request: httpx.Request, position: int) -> None:
        """Raise ScenarioMismatch if the request differs from this step."""
        problems = []
        if request.method != self.method:
            problems.append(
                f"  method: expected {self.method!r}, got {request.method!r}"
            )
        actual_url = str(request.url)
        if actual_url != self.url:
            problems.append(f"  url: expected {self.url!r}, got {actual_url!r}")
        for name, expected in self.headers.items():
            actual = request.headers.get(name)
            if actual != expected:
                problems.append(
                    f"  header {name!r}: expected {expected!r}, got {actual!r}"
                )
        if problems:
            summary = "\n".join(problems)
            raise ScenarioMismatch(
                f"request does not match step #{position} "
                f"({self.method} {self.url}):\n{summary}"
            )

    def actions(self) -> list[Barrier | BaseException | bytes]:
        """Flatten chunks and injected actions into one ordered program."""
        program: list[Barrier | BaseException | bytes] = []
        for index, chunk in enumerate(self.chunks):
            program.extend(self.before.get(index, ()))
            program.append(chunk)
        program.extend(self.before.get(len(self.chunks), ()))
        return program


class _Scenario:
    """Ordered matching state shared by the sync and async transports."""

    def __init__(self, steps: typing.Iterable[Step]) -> None:
        self.steps = list(steps)
        self.position = 0

    def match(self, request: httpx.Request) -> StepRun:
        if self.position >= len(self.steps):
            if not self.steps:
                detail = "the scenario is empty"
            else:
                detail = f"all {len(self.steps)} step(s) were already consumed"
            raise ScenarioExhausted(
                f"unexpected request {request.method} {request.url}: {detail}"
            )
        step = self.steps[self.position]
        # Raises ScenarioMismatch *without* advancing the position, so a
        # failed call never silently skips to the next step.
        step.check(request, self.position)
        self.position += 1
        run = StepRun(step)
        step.runs.append(run)
        return run


class _SyncStepStream(httpx.SyncByteStream):
    def __init__(
        self, transport: ScenarioTransport, run: StepRun, program: list[typing.Any]
    ) -> None:
        self._transport = transport
        self._run = run
        self._program = program
        self._index = 0

    def __iter__(self) -> typing.Iterator[bytes]:
        try:
            while self._index < len(self._program):
                if self._run.closed:
                    raise TransportClosed(
                        "cannot consume: stream was closed "
                        f"(step {self._run.step.method} {self._run.step.url})"
                    )
                action = self._program[self._index]
                self._index += 1
                if isinstance(action, Barrier):
                    action.wait()
                elif isinstance(action, BaseException):
                    raise action
                else:
                    self._run.delivered_bytes += len(action)
                    yield action
        finally:
            self.close()

    def close(self) -> None:
        run = self._run
        if not run.closed:
            run.closed = True
            run.close_count += 1
        self._transport._release(self)


class _AsyncStepStream(httpx.AsyncByteStream):
    def __init__(
        self, transport: AsyncScenarioTransport, run: StepRun, program: list[typing.Any]
    ) -> None:
        self._transport = transport
        self._run = run
        self._program = program
        self._index = 0

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        try:
            while self._index < len(self._program):
                if self._run.closed:
                    raise TransportClosed(
                        "cannot consume: stream was closed "
                        f"(step {self._run.step.method} {self._run.step.url})"
                    )
                action = self._program[self._index]
                self._index += 1
                if isinstance(action, Barrier):
                    await action.wait_async()
                elif isinstance(action, BaseException):
                    raise action
                else:
                    self._run.delivered_bytes += len(action)
                    yield action
        finally:
            # No checkpoint in here: cleanup also runs when the iteration is
            # cancelled while waiting on a barrier, so resources are reclaimed
            # inline and nothing is left hanging in the background.
            await self.aclose()

    async def aclose(self) -> None:
        run = self._run
        if not run.closed:
            run.closed = True
            run.close_count += 1
        self._transport._release(self)


class ScenarioTransport(httpx.BaseTransport):
    """A sync transport that replays a declared scenario, never the network."""

    def __init__(self, steps: typing.Iterable[Step]) -> None:
        self._scenario = _Scenario(steps)
        self._closed = False
        self._lock = threading.Lock()
        self._open_streams: set[_SyncStepStream] = set()

    @property
    def steps(self) -> list[Step]:
        return self._scenario.steps

    @property
    def close_count(self) -> int:
        return sum(step.close_count for step in self._scenario.steps)

    @property
    def open_stream_count(self) -> int:
        return len(self._open_streams)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        with self._lock:
            if self._closed:
                raise TransportClosed(
                    f"transport is closed; cannot send {request.method} {request.url}"
                )
            run = self._scenario.match(request)
            stream = _SyncStepStream(self, run, run.step.actions())
            self._open_streams.add(stream)
        return httpx.Response(
            status_code=run.step.status,
            headers=run.step.response_headers,
            stream=stream,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            streams = list(self._open_streams)
        # Unread responses must still release the resources their step holds.
        for stream in streams:
            stream.close()

    def _release(self, stream: _SyncStepStream) -> None:
        self._open_streams.discard(stream)


class AsyncScenarioTransport(httpx.AsyncBaseTransport):
    """The async twin of ScenarioTransport, same scenario definition."""

    def __init__(self, steps: typing.Iterable[Step]) -> None:
        self._scenario = _Scenario(steps)
        self._closed = False
        self._open_streams: set[_AsyncStepStream] = set()

    @property
    def steps(self) -> list[Step]:
        return self._scenario.steps

    @property
    def close_count(self) -> int:
        return sum(step.close_count for step in self._scenario.steps)

    @property
    def open_stream_count(self) -> int:
        return len(self._open_streams)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        if self._closed:
            raise TransportClosed(
                f"transport is closed; cannot send {request.method} {request.url}"
            )
        run = self._scenario.match(request)
        stream = _AsyncStepStream(self, run, run.step.actions())
        self._open_streams.add(stream)
        return httpx.Response(
            status_code=run.step.status,
            headers=run.step.response_headers,
            stream=stream,
        )

    async def aclose(self) -> None:
        self._closed = True
        streams = list(self._open_streams)
        # Unread responses must still release the resources their step holds.
        for stream in streams:
            await stream.aclose()

    def _release(self, stream: _AsyncStepStream) -> None:
        self._open_streams.discard(stream)
