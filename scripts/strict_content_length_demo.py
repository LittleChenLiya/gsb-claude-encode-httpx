"""
Local print-only demo for the optional strict Content-Length wrapper.

Run with:

    python scripts/strict_content_length_demo.py

Everything here runs in-process against a `MockTransport`, so no network
sockets, servers or Docker are involved.  Each scenario prints whether
the strict response hook (`httpx.enforce_content_length`) accepted or
structurally rejected the response.
"""

from __future__ import annotations

import gzip
import os
import sys

# Ensure the demo uses the in-tree HTTPX when run as
# `python scripts/strict_content_length_demo.py`, rather than any
# separately installed copy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402


class CountingStream(httpx.SyncByteStream):
    """A tiny stream that counts yielded chunks and `close()` calls."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.num_closed = 0

    def __iter__(self):
        yield from self.chunks

    def close(self) -> None:
        self.num_closed += 1


def make_client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        event_hooks={"response": [httpx.enforce_content_length]},
    )


def case(
    title: str,
    handler,
    method: str = "GET",
) -> None:
    print(f"\n=== {title} ===")
    with make_client(handler) as client:
        try:
            response = client.request(method, "https://example.test/")
            response.read()
        except httpx.ContentLengthError as exc:
            print(
                f"REJECTED ({type(exc).__name__}): expected={exc.expected} "
                f"received={exc.received}"
            )
        except httpx.InvalidContentLength as exc:
            print(f"REJECTED ({type(exc).__name__}): {exc.values}")
        else:
            print(
                f"OK: status={response.status_code} "
                f"raw_bytes={response.num_bytes_downloaded} "
                f"content={response.content!r}"
            )


def main() -> None:
    # 1. Declared length 5 but the body only gives "abc" -> short read.
    def short_read(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "5"},
            stream=CountingStream([b"abc"]),
            request=request,
        )

    case("1. Short read: Content-Length 5, body 'abc'", short_read)

    # 2. Gzip data whose declared length matches the *raw compressed*
    #    bytes, even though the decoded length differs.
    body = "你好，世界".encode("utf-8")
    compressed = gzip.compress(body)

    def gzip_ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(compressed)),
                "Content-Encoding": "gzip",
            },
            stream=CountingStream([compressed]),
            request=request,
        )

    case(
        f"2. Gzip: declares {len(compressed)} raw bytes, decodes to {len(body)} bytes",
        gzip_ok,
    )

    # 3. HEAD response carrying a length but no body is fine.
    def head_no_body(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "1234"},
            stream=httpx.ByteStream(b""),
            request=request,
        )

    case("3. HEAD: Content-Length 1234 but no body", head_no_body, method="HEAD")

    # 4. Extra bytes beyond the declared length -> rejected immediately
    #    while streaming.
    def too_long(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "3"},
            stream=CountingStream([b"abc", b"extra"]),
            request=request,
        )

    case("4. Too long: Content-Length 3, body 'abcextra'", too_long)

    # 5. Duplicate, disagreeing Content-Length values are rejected.
    def duplicate(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Content-Length", "5"), ("Content-Length", "6")],
            stream=httpx.ByteStream(b"hello"),
            request=request,
        )

    case("5. Duplicate disagreeing Content-Length (5 and 6)", duplicate)

    # 6. Exact length, delivered in several chunks -> accepted.
    def exact(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Length": "5"},
            stream=CountingStream([b"ab", b"c", b"de"]),
            request=request,
        )

    case("6. Exact length streamed as 'ab'/'c'/'de'", exact)

    print()


if __name__ == "__main__":
    main()
