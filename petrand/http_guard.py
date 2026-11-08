"""
petrand/http_guard.py
=====================
Request guards shared by PetRand's two local web servers (the console and Zoomies!).

- On a server bound to this computer only, the Host header must name this computer,
  which defeats DNS rebinding. A server bound to a network address on purpose skips it.
- API requests that a browser marks as coming from another site or another local port
  (Sec-Fetch-Site: cross-site or same-site) are refused, so other pages cannot draw,
  read or steer seeds. Tools such as curl send no such header and are unaffected.
- POSTs whose Origin is a different site are refused.
- Request bodies are capped, and a malformed Content-Length is rejected.
"""

from typing import Optional, Tuple

MAX_BODY_BYTES = 2 * 1024 * 1024        # a 4 s round from a 1,000 Hz mouse is about 0.3 MB
REQUEST_TIMEOUT_S = 5                   # an idle or half-sent request is dropped after this
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]", "::1")


def bound_to_loopback(server) -> bool:
    try:
        return server.server_address[0] in ("127.0.0.1", "::1", "localhost")
    except Exception:
        return True


def request_refusal(handler, is_post: bool) -> Optional[Tuple[int, str]]:
    """Return (status, message) when the request must be refused, otherwise None."""
    host = (handler.headers.get("Host") or "").strip().lower()
    if bound_to_loopback(handler.server):
        name = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        if name not in LOOPBACK_NAMES:
            return 403, "unexpected Host header"
    path = handler.path.split("?", 1)[0]
    site = (handler.headers.get("Sec-Fetch-Site") or "").strip().lower()
    if path.startswith("/api/") and site in ("cross-site", "same-site"):
        return 403, "request from another site refused"
    origin = handler.headers.get("Origin")
    if is_post and origin and origin.lower() != f"http://{host}":
        return 403, "cross-site request refused"
    return None


def read_body(handler) -> Tuple[Optional[str], Optional[Tuple[int, str]]]:
    """Read the request body with a size cap. Returns (text, None) or (None, (status, message))."""
    raw_len = (handler.headers.get("Content-Length") or "0").strip()
    try:
        length = int(raw_len)
    except ValueError:
        return None, (400, "bad Content-Length")
    if length < 0:
        return None, (400, "bad Content-Length")
    if length > MAX_BODY_BYTES:
        return None, (413, "request body too large")
    if length == 0:
        return "{}", None
    data = handler.rfile.read(length)
    if len(data) < length:
        return None, (400, "incomplete request body")
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, (400, "request body is not UTF-8")
