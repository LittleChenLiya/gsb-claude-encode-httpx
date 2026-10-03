HTTPX allows you to register "event hooks" with the client, that are called
every time a particular type of event takes place.

There are currently two event hooks:

* `request` - Called after a request is fully prepared, but before it is sent to the network. Passed the `request` instance.
* `response` - Called after the response has been fetched from the network, but before it is returned to the caller. Passed the `response` instance.

These allow you to install client-wide functionality such as logging, monitoring or tracing.

```python
def log_request(request):
    print(f"Request event hook: {request.method} {request.url} - Waiting for response")

def log_response(response):
    request = response.request
    print(f"Response event hook: {request.method} {request.url} - Status {response.status_code}")

client = httpx.Client(event_hooks={'request': [log_request], 'response': [log_response]})
```

You can also use these hooks to install response processing code, such as this
example, which creates a client instance that always raises `httpx.HTTPStatusError`
on 4xx and 5xx responses.

```python
def raise_on_4xx_5xx(response):
    response.raise_for_status()

client = httpx.Client(event_hooks={'response': [raise_on_4xx_5xx]})
```

!!! note
    Response event hooks are called before determining if the response body
    should be read or not.

    If you need access to the response body inside an event hook, you'll
    need to call `response.read()`, or for AsyncClients, `response.aread()`.

The hooks are also allowed to modify `request` and `response` objects.

```python
def add_timestamp(request):
    request.headers['x-request-timestamp'] = datetime.now(tz=datetime.utc).isoformat()

client = httpx.Client(event_hooks={'request': [add_timestamp]})
```

Event hooks must always be set as a **list of callables**, and you may register
multiple event hooks for each type of event.

As well as being able to set event hooks on instantiating the client, there
is also an `.event_hooks` property, that allows you to inspect and modify
the installed hooks.

```python
client = httpx.Client()
client.event_hooks['request'] = [log_request]
client.event_hooks['response'] = [log_response, raise_on_4xx_5xx]
```

!!! note
    If you are using HTTPX's async support, then you need to be aware that
    hooks registered with `httpx.AsyncClient` MUST be async functions,
    rather than plain functions. (The built-in `httpx.enforce_content_length`
    hook works with both kinds of client.)

## Strict `Content-Length` enforcement

By default HTTPX is lenient about the length of a response body: if the
stream ends earlier than the declared `Content-Length`, or yields more
bytes than were declared, the bytes that *were* received are surfaced to
the caller unchanged. If you would rather reject such responses, register
the built-in `httpx.enforce_content_length` response hook:

```python
client = httpx.Client(event_hooks={"response": [httpx.enforce_content_length]})
```

The same hook can be registered with `httpx.AsyncClient`.

With the hook installed:

* Responses with multiple, disagreeing `Content-Length` headers (or a
  malformed value) raise `httpx.InvalidContentLength` as soon as the
  response is received. Repeated values that agree with each other are
  accepted.
* If the body ends before the declared number of bytes, reading the
  response raises `httpx.ContentLengthTooShort`.
* If the body yields more bytes than declared, streaming raises
  `httpx.ContentLengthTooLong` immediately when the surplus bytes
  arrive, rather than waiting for the whole body to be consumed.

The check counts the **raw** response bytes, *before* `Content-Encoding`
decompression. A gzip response whose declared length matches the
compressed body is accepted even if the decoded body has a different
length. The body is not buffered into memory in order to measure it.

Per HTTP semantics, the following responses are left untouched:

* responses to `HEAD` requests (which may carry a `Content-Length`
  describing the representation a `GET` would return, without a body);
* responses with bodyless status codes (`1xx`, `204`, `304`);
* responses without a `Content-Length` header, including bodies framed
  with `Transfer-Encoding: chunked`.

All of the raised errors subclass `httpx.RemoteProtocolError` (and thus
`httpx.RequestError`). `ContentLengthTooShort` and
`ContentLengthTooLong` share the `httpx.ContentLengthError` base class
and expose structured `expected` and `received` byte counts:

```python
try:
    client.get("https://example.com/")
except httpx.ContentLengthError as exc:
    print(f"body length mismatch: expected {exc.expected}, got {exc.received}")
```

