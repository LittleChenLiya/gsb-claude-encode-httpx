"""
Strict Content-Length validation - a local stream demo.

This script demonstrates the optional `StrictLengthTransport` /
`StrictLengthStream` wrappers using only in-memory streams
(`httpx.MockTransport` and `httpx.ByteStream`). No network is involved.

Run it with:

    python examples/strict_length_demo.py

The default behaviour of `httpx.Client` / `httpx.Response` is unchanged -
strict validation is opt-in by wrapping a transport.
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

# Use the local worktree copy of httpx when run directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


def divider(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def demo_short_read() -> None:
    """Declared 5 bytes but only 3 arrive -> structured short-read error."""
    divider("1. Short read (declared 5, body 'abc')")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "5"}, content=b"abc"
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        try:
            client.get("https://example.org")
        except httpx.ContentLengthError as exc:
            print(f"  reason           : {exc.reason}")
            print(f"  declared length  : {exc.declared_length}")
            print(f"  received length  : {exc.received_length}")
            print(f"  request attached : {exc.request is not None}")


def demo_gzip_uses_raw_length() -> None:
    """Validation uses the raw (compressed) length, not the decoded text."""
    divider("2. gzip - declared by compressed bytes, decoded text differs")

    compressed = gzip.compress(
        "你好，世界！这是一段用于测试的中文内容。".encode("utf-8")
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(compressed)),
                "Content-Encoding": "gzip",
            },
            content=compressed,
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        response = client.get("https://example.org")

    print(f"  compressed (declared) length : {len(compressed)}")
    print(f"  decoded text length          : {len(response.text)}")
    print(f"  decoded text                 : {response.text!r}")
    print("  -> validation passed using the raw compressed length")


def demo_head_no_body() -> None:
    """HEAD responses carry no body even when a length is declared."""
    divider("3. HEAD with Content-Length but no body")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        return httpx.Response(
            200, headers={"Content-Length": "5"}, content=b""
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        response = client.head("https://example.org")

    print(f"  status code : {response.status_code}")
    print(f"  body        : {response.content!r}")
    print("  -> passed (HEAD responses never carry a body)")


def demo_over_read_immediate() -> None:
    """A chunk that exceeds the declared length fails immediately."""
    divider("4. Over-read - fails immediately, not at the end")

    class OverReadStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.close_count = 0

        def __iter__(self):
            yield b"abcdef"  # 6 bytes > declared 5
            raise AssertionError("stream was resumed after the overshoot")

        def close(self) -> None:
            self.close_count += 1

    stream = OverReadStream()
    strict = httpx.StrictLengthStream(stream, content_length=5)

    iterator = iter(strict)
    try:
        next(iterator)
    except httpx.ContentLengthError as exc:
        print(f"  reason          : {exc.reason}")
        print(f"  declared length : {exc.declared_length}")
        print(f"  received length : {exc.received_length}")
        print(f"  close count     : {stream.close_count}")
        print("  -> failed on the first chunk, stream closed once")


def demo_conflicting_headers() -> None:
    """Conflicting duplicate Content-Length headers are rejected up front."""
    divider("5. Conflicting duplicate Content-Length headers")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Length", "5"), ("Content-Length", "7")],
            content=b"abcdefg",
        )

    transport = httpx.StrictLengthTransport(httpx.MockTransport(handler))
    with httpx.Client(transport=transport) as client:
        try:
            client.get("https://example.org")
        except httpx.ContentLengthError as exc:
            print(f"  reason : {exc.reason}")
            print(f"  detail : {exc}")


def demo_default_unchanged() -> None:
    """Without the strict transport, short reads do not raise."""
    divider("6. Default behaviour is unchanged (no strict transport)")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Length": "5"}, content=b"abc"
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = client.get("https://example.org")

    print(f"  body : {response.content!r} (no error raised)")


def main() -> None:
    print("httpx strict Content-Length validation - local stream demo")
    demo_short_read()
    demo_gzip_uses_raw_length()
    demo_head_no_body()
    demo_over_read_immediate()
    demo_conflicting_headers()
    demo_default_unchanged()
    print(f"\n{'=' * 60}\nDone.\n{'=' * 60}")


if __name__ == "__main__":
    main()
