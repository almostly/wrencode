"""The `fetch` tool: a web page or URL as text, for any backend.

A GET with a size cap, HTML reduced to its text with headings, list markers
and link targets kept, JSON and plain text passed through, other content types
summarized. Pages longer than the cap come back in pieces through `offset`.
Fetching sends the URL to a server, so it asks for approval like a command
does; permission rules `fetch(host/path*)` apply, matched against the host,
path and query of the URL as approved. Redirects are followed on the same host
only; another host is a new fetch, so it is approved on its own. Addresses
that are not public (loopback, private networks, link-local and the cloud
metadata service) are refused unless WRENCODE_FETCH_LOCAL=1, since a page
could not otherwise direct the agent at services inside the machine's network.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

MAX_BYTES = int(os.environ.get("WRENCODE_FETCH_MAX_BYTES", str(4 * 1024 * 1024)))
MAX_CHARS = int(os.environ.get("WRENCODE_FETCH_MAX_CHARS", "40000"))
TIMEOUT = float(os.environ.get("WRENCODE_FETCH_TIMEOUT", "30"))
ALLOW_LOCAL = os.environ.get("WRENCODE_FETCH_LOCAL", "").lower() in ("1", "true", "yes")
MAX_REDIRECTS = 5
USER_AGENT = "wrencode/0.3 (+https://github.com/almostly/wrencode)"
_SKIP = {"script", "style", "noscript", "template", "svg"}
_BLOCK = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "header",
    "footer",
    "nav",
    "aside",
    "ul",
    "ol",
    "table",
    "tr",
    "blockquote",
    "pre",
    "br",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "dt",
    "dd",
    "figure",
    "figcaption",
}
_HEADINGS = {
    "h1": "# ",
    "h2": "## ",
    "h3": "### ",
    "h4": "#### ",
    "h5": "##### ",
    "h6": "###### ",
}


class _Text(HTMLParser):
    """Collect a page's readable text in reading order."""

    def __init__(self, base: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base = base
        self.out: list[str] = []
        self.skip = 0
        self.pre = 0
        self.href: str | None = None
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self.skip += 1
            return
        if tag == "title":
            self.in_title = True
        if tag in _HEADINGS:
            self.out.append("\n\n" + _HEADINGS[tag])
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n```\n")
        elif tag in _BLOCK:
            self.out.append("\n")
        elif tag == "a":
            self.href = next((v for k, v in attrs if k == "href" and v), None)
        elif tag == "img":
            alt = next((v for k, v in attrs if k == "alt" and v), "")
            if alt:
                self.out.append(f"[image: {alt}]")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self.skip = max(self.skip - 1, 0)
            return
        if tag == "title":
            self.in_title = False
        if tag == "pre":
            self.pre = max(self.pre - 1, 0)
            self.out.append("\n```\n")
        elif tag == "a" and self.href:
            href = urllib.parse.urljoin(self.base, self.href)
            if href.startswith(("http://", "https://")) and not href.startswith(
                self.base.rstrip("/") + "#"
            ):
                self.out.append(f" ({href})")
            self.href = None
        elif tag in _BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
            return
        if self.skip:
            return
        if self.pre:
            self.out.append(data)
        else:
            text = re.sub(r"\s+", " ", data)
            if text.strip():
                self.out.append(text)


def html_to_text(html: str, base: str = "") -> tuple[str, str]:
    """(title, text) for an HTML document: headings, lists, code and link targets kept."""
    parser = _Text(base)
    parser.feed(html)
    parser.close()
    text = "".join(parser.out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return parser.title.strip(), text.strip()


def subject(url: str) -> str:
    """What a permission rule is matched against and what the prompt shows: the
    host in lower case (the port only when it isn't the scheme's), the path and
    the query; no scheme, no user name."""
    parts = urllib.parse.urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    with contextlib.suppress(ValueError):
        port = parts.port
        if port and port != {"http": 80, "https": 443}.get(parts.scheme):
            host += f":{port}"
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{parts.path or '/'}{query}"


def _public(host: str) -> bool:
    """Whether every address `host` resolves to is a public one."""
    if ALLOW_LOCAL:
        return True
    if host.lower() in ("localhost", "localhost.localdomain"):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # let the request itself report the failure
    for info in infos:
        with contextlib.suppress(ValueError):
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_global and not addr.is_multicast:
                continue
            return False
    return True


class _Redirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect on the same host; stop at another host or a non-public one."""

    max_redirections = MAX_REDIRECTS

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        new = urllib.parse.urlsplit(newurl)
        old = urllib.parse.urlsplit(req.full_url)
        if (
            new.scheme not in ("http", "https")
            or (new.hostname or "").lower() != (old.hostname or "").lower()
            or not _public(new.hostname or "")
        ):
            raise _ElsewhereError(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _ElsewhereError(Exception):
    """A redirect to another host: a new fetch, to be approved on its own."""


_OPENER = urllib.request.build_opener(_Redirect)


def fetch(url: str, offset: int = 0) -> str:
    """Fetch `url` and return its text from `offset`, at most MAX_CHARS at a time."""
    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return f"error: not an http(s) URL: {url!r}"
    if not _public(parts.hostname or ""):
        return (
            f"error: {parts.hostname} is not a public address; wrencode fetches public "
            "pages only (WRENCODE_FETCH_LOCAL=1 allows local ones)"
        )
    req = urllib.request.Request(
        url.strip(),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/json,text/*;q=0.9,*/*;q=0.5",
            "Accept-Encoding": "gzip",
        },
    )
    try:
        with _OPENER.open(req, timeout=TIMEOUT) as resp:
            raw = resp.read(MAX_BYTES + 1)
            ctype = resp.headers.get("Content-Type", "")
            final = resp.geturl()
            if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(MAX_BYTES + 1)
    except _ElsewhereError as err:
        return (
            f"{url} redirects to {err.args[0]}; fetch that URL to follow it "
            "(redirects are followed on the same host only)"
        )
    except urllib.error.HTTPError as err:
        return f"error: HTTP {err.code} fetching {url}"
    except (urllib.error.URLError, OSError, ValueError, EOFError) as err:
        return f"error: could not fetch {url}: {getattr(err, 'reason', err)}"
    clipped = len(raw) > MAX_BYTES
    raw = raw[:MAX_BYTES]
    kind = ctype.split(";")[0].strip().lower()
    charset = re.search(r"charset=([\w-]+)", ctype)
    enc = charset.group(1) if charset else "utf-8"
    if kind in ("text/html", "application/xhtml+xml") or (
        not kind and raw.lstrip()[:1] == b"<"
    ):
        title, text = html_to_text(raw.decode(enc, errors="replace"), final)
        head = f"{title}\n{final}\n\n" if title else f"{final}\n\n"
    elif kind == "application/json" or kind.endswith("+json"):
        body = raw.decode(enc, errors="replace")
        with contextlib.suppress(ValueError):
            body = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
        head, text = f"{final}\n\n", body
    elif kind.startswith("text/") or kind in (
        "application/xml",
        "application/javascript",
    ):
        head, text = f"{final}\n\n", raw.decode(enc, errors="replace")
    else:
        return f"{final}\n[{kind or 'unknown type'}, {len(raw)} bytes{' or more' if clipped else ''}; not text]"
    offset = max(int(offset or 0), 0)
    piece = text[offset : offset + MAX_CHARS]
    note = ""
    if offset + MAX_CHARS < len(text):
        note = f"\n\n[{len(text) - offset - MAX_CHARS} more characters; fetch again with offset={offset + MAX_CHARS}]"
    elif clipped:
        note = "\n\n[the page was longer than the size cap; the rest was not read]"
    if offset and offset >= len(text):
        return f"error: offset {offset} is past the end ({len(text)} characters)"
    return head + piece + note
