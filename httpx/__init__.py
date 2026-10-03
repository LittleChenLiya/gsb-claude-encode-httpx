from .__version__ import __description__, __title__, __version__
from ._api import *
from ._auth import *
from ._client import *
from ._config import *
from ._content import *
from ._exceptions import *
from ._models import *
from ._status_codes import *
from ._strict_length import *
from ._transports import *
from ._types import *
from ._urls import *

try:
    from ._main import main
except ImportError:  # pragma: no cover

    def main() -> None:  # type: ignore
        import sys

        print(
            "The httpx command line client could not run because the required "
            "dependencies were not installed.\nMake sure you've installed "
            "everything with: pip install 'httpx[cli]'"
        )
        sys.exit(1)


__all__ = [
    "__description__",
    "__title__",
    "__version__",
    "ASGITransport",
    "AsyncBaseTransport",
    "AsyncByteStream",
    "AsyncClient",
    "AsyncHTTPTransport",
    "AsyncStrictLengthStream",
    "AsyncStrictLengthTransport",
    "Auth",
    "BaseTransport",
    "BasicAuth",
    "ByteStream",
    "Client",
    "CloseError",
    "codes",
    "ConnectError",
    "ConnectTimeout",
    "ContentLengthError",
    "CookieConflict",
    "Cookies",
    "create_ssl_context",
    "DecodingError",
    "delete",
    "DigestAuth",
    "FunctionAuth",
    "get",
    "head",
    "Headers",
    "HTTPError",
    "HTTPStatusError",
    "HTTPTransport",
    "InvalidURL",
    "Limits",
    "LocalProtocolError",
    "main",
    "MockTransport",
    "NetRCAuth",
    "NetworkError",
    "options",
    "parse_content_length",
    "patch",
    "PoolTimeout",
    "post",
    "ProtocolError",
    "Proxy",
    "ProxyError",
    "put",
    "QueryParams",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "request",
    "Request",
    "RequestError",
    "RequestNotRead",
    "Response",
    "ResponseNotRead",
    "response_expects_body",
    "stream",
    "StreamClosed",
    "StreamConsumed",
    "StreamError",
    "StrictLengthStream",
    "StrictLengthTransport",
    "SyncByteStream",
    "Timeout",
    "TimeoutException",
    "TooManyRedirects",
    "TransportError",
    "UnsupportedProtocol",
    "URL",
    "USE_CLIENT_DEFAULT",
    "WriteError",
    "WriteTimeout",
    "WSGITransport",
]


__locals = locals()
for __name in __all__:
    if not __name.startswith("__"):
        setattr(__locals[__name], "__module__", "httpx")  # noqa
