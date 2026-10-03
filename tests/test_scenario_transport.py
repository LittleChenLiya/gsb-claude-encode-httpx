"""
Tests for ScenarioTransport: a local transport that matches requests in
declaration order and returns preset response streams.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest


# === Sync tests ===


def test_get_a_returns_two_chunks() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/a",
            chunks=[httpx.Chunk(b"chunk1"), httpx.Chunk(b"chunk2")],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    client = httpx.Client(transport=transport)
    response = client.get("https://example.com/a")
    assert response.status_code == 200
    assert response.content == b"chunk1chunk2"
    assert transport.delivered_bytes == 12
    assert transport.close_count == 1
    client.close()


def test_get_b_throws_before_second_chunk() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/b",
            chunks=[
                httpx.Chunk(b"first"),
                httpx.Chunk(b"", error=RuntimeError("boom")),
            ],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    client = httpx.Client(transport=transport)
    with client.stream("GET", "https://example.com/b") as response:
        with pytest.raises(RuntimeError, match="boom"):
            response.read()
        print(f"Delivered bytes: {response.num_bytes_downloaded}")
        print(f"Close count: {transport.close_count}")
        assert response.num_bytes_downloaded == 5
    assert transport.close_count == 1
    client.close()


def test_retry_after_wrong_method_uses_original_step() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/a",
            chunks=[httpx.Chunk(b"chunk1"), httpx.Chunk(b"chunk2")],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    client = httpx.Client(transport=transport)
    with pytest.raises(httpx.ScenarioMismatch) as exc_info:
        client.post("https://example.com/a")
    print(f"Mismatch: {exc_info.value}")
    assert exc_info.value.expected_method == "GET"
    assert exc_info.value.actual_method == "POST"
    response = client.get("https://example.com/a")
    assert response.content == b"chunk1chunk2"
    client.close()


def test_empty_scenario_raises_on_first_request() -> None:
    transport = httpx.ScenarioTransport([])
    client = httpx.Client(transport=transport)
    with pytest.raises(RuntimeError, match="Scenario exhausted"):
        client.get("https://example.com/a")
    client.close()


def test_extra_request_after_steps_consumed() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET", "https://example.com/a", chunks=[httpx.Chunk(b"a")]
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    client = httpx.Client(transport=transport)
    response = client.get("https://example.com/a")
    assert response.content == b"a"
    with pytest.raises(RuntimeError, match="Scenario exhausted"):
        client.get("https://example.com/b")
    client.close()


def test_closed_transport_cannot_send_requests() -> None:
    transport = httpx.ScenarioTransport([])
    client = httpx.Client(transport=transport)
    client.close()
    with pytest.raises(RuntimeError, match="Cannot send a request"):
        client.get("https://example.com/a")


def test_header_matching() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/a",
            headers={"X-Custom": "value"},
            chunks=[httpx.Chunk(b"ok")],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    client = httpx.Client(transport=transport)
    with pytest.raises(httpx.ScenarioMismatch) as exc_info:
        client.get("https://example.com/a", headers={"X-Custom": "wrong"})
    print(f"Mismatch: {exc_info.value}")
    response = client.get(
        "https://example.com/a", headers={"X-Custom": "value"}
    )
    assert response.content == b"ok"
    client.close()


def test_close_after_partial_read_releases_resources() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/a",
            chunks=[httpx.Chunk(b"chunk1"), httpx.Chunk(b"chunk2")],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    client = httpx.Client(transport=transport)
    with client.stream("GET", "https://example.com/a") as response:
        chunks = response.iter_raw()
        first = next(chunks)
        assert first == b"chunk1"
        response.close()
    assert transport.close_count == 1
    with pytest.raises(httpx.StreamClosed):
        next(chunks)
    client.close()


# === Async tests ===


@pytest.mark.asyncio
async def test_get_a_returns_two_chunks_async() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/a",
            chunks=[httpx.Chunk(b"chunk1"), httpx.Chunk(b"chunk2")],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://example.com/a")
        assert response.status_code == 200
        assert response.content == b"chunk1chunk2"
        assert transport.delivered_bytes == 12
        assert transport.close_count == 1


@pytest.mark.asyncio
async def test_get_b_throws_before_second_chunk_async() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/b",
            chunks=[
                httpx.Chunk(b"first"),
                httpx.Chunk(b"", error=RuntimeError("boom")),
            ],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://example.com/b") as response:
            with pytest.raises(RuntimeError, match="boom"):
                await response.aread()
            print(f"Delivered bytes: {response.num_bytes_downloaded}")
            print(f"Close count: {transport.close_count}")
            assert response.num_bytes_downloaded == 5
        assert transport.close_count == 1


@pytest.mark.asyncio
async def test_retry_after_wrong_method_uses_original_step_async() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/a",
            chunks=[httpx.Chunk(b"chunk1"), httpx.Chunk(b"chunk2")],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ScenarioMismatch) as exc_info:
            await client.post("https://example.com/a")
        print(f"Mismatch: {exc_info.value}")
        assert exc_info.value.expected_method == "GET"
        assert exc_info.value.actual_method == "POST"
        response = await client.get("https://example.com/a")
        assert response.content == b"chunk1chunk2"


@pytest.mark.asyncio
async def test_empty_scenario_raises_on_first_request_async() -> None:
    transport = httpx.ScenarioTransport([])
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="Scenario exhausted"):
            await client.get("https://example.com/a")


@pytest.mark.asyncio
async def test_extra_request_after_steps_consumed_async() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET", "https://example.com/a", chunks=[httpx.Chunk(b"a")]
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://example.com/a")
        assert response.content == b"a"
        with pytest.raises(RuntimeError, match="Scenario exhausted"):
            await client.get("https://example.com/b")


@pytest.mark.asyncio
async def test_header_matching_async() -> None:
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/a",
            headers={"X-Custom": "value"},
            chunks=[httpx.Chunk(b"ok")],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ScenarioMismatch) as exc_info:
            await client.get(
                "https://example.com/a", headers={"X-Custom": "wrong"}
            )
        print(f"Mismatch: {exc_info.value}")
        response = await client.get(
            "https://example.com/a", headers={"X-Custom": "value"}
        )
        assert response.content == b"ok"


@pytest.mark.asyncio
async def test_cancel_waiting_barrier_async_read_reclaims_resources() -> None:
    barrier = httpx.Barrier()
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/slow",
            chunks=[
                httpx.Chunk(b"first"),
                httpx.Chunk(b"second", barrier=barrier),
            ],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://example.com/slow") as response:
            async def read_all() -> bytes:
                return await response.aread()

            task = asyncio.create_task(read_all())
            await barrier.wait_until_reached_async()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert transport.close_count == 1
        print(f"Close count: {transport.close_count}")


@pytest.mark.asyncio
async def test_barrier_allows_stream_to_continue_when_released() -> None:
    barrier = httpx.Barrier()
    scenario = [
        httpx.ScenarioStep(
            "GET",
            "https://example.com/slow",
            chunks=[
                httpx.Chunk(b"first"),
                httpx.Chunk(b"second", barrier=barrier),
            ],
        ),
    ]
    transport = httpx.ScenarioTransport(scenario)
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://example.com/slow") as response:
            async def read_all() -> bytes:
                return await response.aread()

            task = asyncio.create_task(read_all())
            await barrier.wait_until_reached_async()
            barrier.release()
            result = await task
            assert result == b"firstsecond"
        assert transport.close_count == 1
