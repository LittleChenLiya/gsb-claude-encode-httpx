"""
Tests for the scripted scenario transport (tests/scenario_transport.py).

The scenario transport never touches the network: it matches requests
against declared steps in order and replays preset response streams.
"""

import threading

import anyio
import pytest

import httpx
from tests.scenario_transport import (
    AsyncScenarioTransport,
    Barrier,
    ScenarioExhausted,
    ScenarioMismatch,
    ScenarioTransport,
    Step,
    TransportClosed,
)

URL_A = "http://example.org/a"
URL_B = "http://example.org/b"


def make_sync_transport() -> "tuple[ScenarioTransport, Step, Step]":
    step_a = Step("GET", URL_A, chunks=[b"Hello, ", b"world!"])
    step_b = Step(
        "GET",
        URL_B,
        chunks=[b"partial", b"never-sent"],
        before={1: RuntimeError("boom before chunk #1")},
    )
    return ScenarioTransport([step_a, step_b]), step_a, step_b


def test_sync_client_two_chunks_then_injected_failure():
    transport, step_a, step_b = make_sync_transport()
    with httpx.Client(transport=transport) as client:
        response = client.get(URL_A)
        assert response.status_code == 200
        assert response.content == b"Hello, world!"

        with pytest.raises(RuntimeError, match="boom before chunk #1"):
            client.get(URL_B)

    # GET /a delivered both chunks; GET /b failed before its second chunk.
    assert step_a.delivered_bytes == len(b"Hello, world!")
    assert step_b.delivered_bytes == len(b"partial")
    # Every response stream was closed exactly once.
    assert step_a.close_count == 1
    assert step_b.close_count == 1
    assert transport.close_count == 2
    assert transport.open_stream_count == 0


@pytest.mark.anyio
async def test_async_client_two_chunks_then_injected_failure():
    step_a = Step("GET", URL_A, chunks=[b"Hello, ", b"world!"])
    step_b = Step(
        "GET",
        URL_B,
        chunks=[b"partial", b"never-sent"],
        before={1: RuntimeError("boom before chunk #1")},
    )
    transport = AsyncScenarioTransport([step_a, step_b])
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get(URL_A)
        assert response.status_code == 200
        assert response.content == b"Hello, world!"

        with pytest.raises(RuntimeError, match="boom before chunk #1"):
            await client.get(URL_B)

    assert step_a.delivered_bytes == len(b"Hello, world!")
    assert step_b.delivered_bytes == len(b"partial")
    assert step_a.close_count == 1
    assert step_b.close_count == 1
    assert transport.close_count == 2
    assert transport.open_stream_count == 0


def test_method_mismatch_does_not_consume_step():
    step = Step("GET", URL_A, chunks=[b"ok"])
    transport = ScenarioTransport([step])
    with httpx.Client(transport=transport) as client:
        # Wrong method: reports expected vs actual, step stays current.
        with pytest.raises(ScenarioMismatch) as exc_info:
            client.post(URL_A)
        message = str(exc_info.value)
        assert "method: expected 'GET', got 'POST'" in message
        assert step.runs == []

        # Retrying with the correct method still uses the original step.
        response = client.get(URL_A)
        assert response.content == b"ok"
        assert len(step.runs) == 1

        # And a further request finds the scenario exhausted, not a skipped step.
        with pytest.raises(ScenarioExhausted, match="all 1 step"):
            client.get(URL_A)


def test_url_and_header_mismatch_reports_expected_and_actual():
    step = Step(
        "GET",
        URL_A,
        headers={"x-token": "secret", "x-mode": "fast"},
        chunks=[b"ok"],
    )
    transport = ScenarioTransport([step])
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ScenarioMismatch) as exc_info:
            client.get(
                "http://example.org/other",
                headers={"x-token": "wrong", "x-mode": "fast"},
            )
        message = str(exc_info.value)
        assert (
            "url: expected 'http://example.org/a', "
            "got 'http://example.org/other'" in message
        )
        assert "header 'x-token': expected 'secret', got 'wrong'" in message
        assert "x-mode" not in message  # matching headers are not reported
        assert step.runs == []

        # Correct URL and headers match the same (unconsumed) step.
        response = client.get(URL_A, headers={"x-token": "secret", "x-mode": "fast"})
        assert response.content == b"ok"


def test_empty_scenario_rejects_any_request():
    transport = ScenarioTransport([])
    with httpx.Client(transport=transport) as client:
        with pytest.raises(ScenarioExhausted, match="scenario is empty"):
            client.get(URL_A)


def test_extra_request_after_scenario_exhausted():
    step = Step("GET", URL_A, chunks=[b"ok"])
    transport = ScenarioTransport([step])
    with httpx.Client(transport=transport) as client:
        assert client.get(URL_A).content == b"ok"
        with pytest.raises(ScenarioExhausted) as exc_info:
            client.get(URL_B)
        assert "unexpected request GET http://example.org/b" in str(exc_info.value)
        assert "all 1 step(s) were already consumed" in str(exc_info.value)


def test_close_releases_unread_response_and_blocks_consumption():
    step = Step("GET", URL_A, chunks=[b"one", b"two"])
    transport = ScenarioTransport([step])
    client = httpx.Client(transport=transport)
    response = client.send(client.build_request("GET", URL_A), stream=True)
    assert transport.open_stream_count == 1

    # Closing the transport releases the unread response's resources.
    transport.close()
    assert step.close_count == 1
    assert transport.open_stream_count == 0

    # After close nothing can be consumed anymore.
    with pytest.raises(TransportClosed, match="cannot consume"):
        next(iter(response.stream))
    with pytest.raises(TransportClosed, match="transport is closed"):
        transport.handle_request(httpx.Request("GET", URL_A))

    # Closing again is harmless and does not double-count.
    transport.close()
    assert step.close_count == 1


def test_sync_barrier_blocks_until_released():
    barrier = Barrier("gate")
    step = Step("GET", URL_A, chunks=[b"one", b"two"], before={1: barrier})
    transport = ScenarioTransport([step])
    releaser = threading.Timer(0.1, barrier.release)
    releaser.start()
    with httpx.Client(transport=transport) as client:
        response = client.get(URL_A)
    releaser.join()
    assert response.content == b"onetwo"
    assert step.delivered_bytes == 6
    assert step.close_count == 1


@pytest.mark.anyio
async def test_async_cancel_at_barrier_reclaims_resources():
    barrier = Barrier("gate")
    step = Step(
        "GET",
        URL_A,
        chunks=[b"first", b"second"],
        before={1: barrier},
    )
    transport = AsyncScenarioTransport([step])
    async with httpx.AsyncClient(transport=transport) as client:
        with anyio.move_on_after(0.5) as scope:
            async with client.stream("GET", URL_A) as response:
                # Delivers b"first", then blocks on the barrier forever.
                await response.aread()
        assert scope.cancelled_caught

    # The cancelled read unwound the stream inline: the step's resources
    # were reclaimed and nothing is left hanging in the background.
    assert step.delivered_bytes == len(b"first")
    assert step.close_count == 1
    assert transport.open_stream_count == 0


@pytest.mark.anyio
async def test_async_close_releases_unread_response_and_blocks_consumption():
    step = Step("GET", URL_A, chunks=[b"one", b"two"])
    transport = AsyncScenarioTransport([step])
    client = httpx.AsyncClient(transport=transport)
    response = await client.send(client.build_request("GET", URL_A), stream=True)
    assert transport.open_stream_count == 1

    # Closing the transport releases the unread response's resources.
    await transport.aclose()
    assert step.close_count == 1
    assert transport.open_stream_count == 0

    # After close nothing can be consumed anymore.
    with pytest.raises(TransportClosed, match="cannot consume"):
        await response.stream.__aiter__().__anext__()
    with pytest.raises(TransportClosed, match="transport is closed"):
        await transport.handle_async_request(httpx.Request("GET", URL_A))

    # Closing again is harmless and does not double-count.
    await client.aclose()
    await transport.aclose()
    assert step.close_count == 1
