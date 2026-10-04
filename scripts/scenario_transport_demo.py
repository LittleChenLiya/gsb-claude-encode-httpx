"""
Short demo of the scripted scenario transport (tests/scenario_transport.py).

Delivery notes
--------------
* ``tests/scenario_transport.py`` provides ``ScenarioTransport`` (sync) and
  ``AsyncScenarioTransport`` (async). Both replay one shared scenario
  definition: an ordered list of ``Step`` objects. No network is touched.
* Each step checks the request method, URL and a selected subset of headers.
  A mismatch raises ``ScenarioMismatch`` (expected vs actual) and the step is
  NOT consumed, so a corrected retry meets the same step.
* ``Step(..., chunks=[...], before={i: action})`` injects, before chunk ``i``,
  either a ``Barrier`` (read blocks until released) or an exception instance
  (raised as-is).
* Closing a transport forbids further consumption and closes still-open
  (unread) response streams, releasing the resources their steps hold.
  Delivered bytes and close counts are tracked per step.
* Cancelling an async read that waits on a barrier reclaims the step's
  resources inline -- nothing is left hanging in the background.

Run:  python scripts/scenario_transport_demo.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from tests.scenario_transport import (
    AsyncScenarioTransport,
    ScenarioMismatch,
    ScenarioTransport,
    Step,
)

URL_A = "http://example.org/a"
URL_B = "http://example.org/b"


def make_steps():
    return [
        Step("GET", URL_A, chunks=[b"Hello, ", b"world!"]),
        Step(
            "GET",
            URL_B,
            chunks=[b"partial", b"never-sent"],
            before={1: RuntimeError("boom before chunk #1")},
        ),
    ]


def report(tag, steps, transport):
    for step in steps:
        print(
            f"  [{tag}] {step.method} {step.url}: "
            f"delivered={step.delivered_bytes}B closes={step.close_count}"
        )
    print(f"  [{tag}] transport totals: closes={transport.close_count}")


def sync_demo():
    print("sync client")
    steps = make_steps()
    transport = ScenarioTransport(steps)
    with httpx.Client(transport=transport) as client:
        body = client.get(URL_A).content
        print(f"  GET /a -> {body!r} (two chunks, joined)")
        try:
            client.get(URL_B)
        except RuntimeError as exc:
            print(f"  GET /b -> raised {exc!r} before its second chunk")
    report("sync", steps, transport)


async def async_demo():
    print("async client (same scenario definition)")
    steps = make_steps()
    transport = AsyncScenarioTransport(steps)
    async with httpx.AsyncClient(transport=transport) as client:
        body = (await client.get(URL_A)).content
        print(f"  GET /a -> {body!r} (two chunks, joined)")
        try:
            await client.get(URL_B)
        except RuntimeError as exc:
            print(f"  GET /b -> raised {exc!r} before its second chunk")
    report("async", steps, transport)


def mismatch_demo():
    print("mismatch does not consume the step")
    step = Step("GET", URL_A, chunks=[b"ok"])
    transport = ScenarioTransport([step])
    with httpx.Client(transport=transport) as client:
        try:
            client.post(URL_A)  # wrong method on purpose
        except ScenarioMismatch as exc:
            print(f"  POST /a -> mismatch reported:\n    {exc}")
        print(f"  GET  /a -> {client.get(URL_A).content!r} (original step still used)")


if __name__ == "__main__":
    sync_demo()
    asyncio.run(async_demo())
    mismatch_demo()
