"""Provider-independent public web discovery and source reading (never physics evidence)."""

from __future__ import annotations

import io
import ipaddress
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from html.parser import HTMLParser
from threading import RLock

_TIMEOUT = 20
_MAX_BYTES = 20 * 1024 * 1024
_MAX_TEXT = 2_000_000
_SEARCH_ENDPOINTS = (
    "https://lite.duckduckgo.com/lite/?q=",
    "https://html.duckduckgo.com/html/?q=",
)


class WebResearchError(ValueError):
    """A source could not be safely retrieved or interpreted."""


class SourceUnavailable(WebResearchError):
    """An external access limitation, not a program or numerical failure."""


_BLOCKED: OrderedDict[str, tuple[float, str]] = OrderedDict()
_BLOCKED_LOCK = RLock()
_BLOCKED_TTL = 600


def _remember_unavailable(url: str, reason: str) -> None:
    with _BLOCKED_LOCK:
        _BLOCKED[url] = (time.monotonic() + _BLOCKED_TTL, reason)
        _BLOCKED.move_to_end(url)
        while len(_BLOCKED) > 256:
            _BLOCKED.popitem(last=False)


def _public_url(url: str, *, resolve: bool = True) -> str:
    if not isinstance(url, str) or len(url) > 8192 or re.search(r"[\x00-\x20\\]", url):
        raise WebResearchError("Expected a public HTTP(S) URL without control characters.")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebResearchError("Only public HTTP(S) URLs are supported.")
    if parsed.username is not None or parsed.password is not None:
        raise WebResearchError("Credentials in source URLs are not allowed.")
    host = parsed.hostname.rstrip(".")
    if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local")):
        raise WebResearchError("Local/private source URLs are not allowed.")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        addresses = []
    if resolve:
        try:
            addresses = [ipaddress.ip_address(row[4][0]) for row in socket.getaddrinfo(
                host, parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )]
        except OSError as exc:
            raise WebResearchError(f"Cannot resolve public source host {host}: {exc}") from exc
    if any(not address.is_global or address.is_multicast or address.is_reserved
           or address.is_unspecified for address in addresses):
        raise WebResearchError("Local/private source addresses are not allowed.")
    return urllib.parse.urlunsplit(parsed._replace(fragment=""))


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if req is not None and req.has_header("X-subscription-token"):
            raise WebResearchError("Authenticated search endpoint redirected; credentials were not forwarded.")
        return super().redirect_request(req, fp, code, msg, headers, _public_url(newurl))


def _fetch(url: str, *, headers: dict | None = None) -> tuple[bytes, str, str, str]:
    url = _public_url(url)
    with _BLOCKED_LOCK:
        blocked = _BLOCKED.get(url)
        if blocked and blocked[0] > time.monotonic():
            raise SourceUnavailable(f"{blocked[1]} Repeated request suppressed for up to 10 minutes; use another source.")
        _BLOCKED.pop(url, None)
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; Brilliance/0.8; scientific research)",
        "Accept": "text/html, application/pdf, text/plain;q=0.9, */*;q=0.1",
        "Accept-Encoding": "identity",
        **(headers or {}),
    })
    deadline = time.monotonic() + _TIMEOUT
    try:
        with urllib.request.build_opener(_PublicRedirect()).open(request, timeout=_TIMEOUT) as response:
            final_url = _public_url(response.url)
            length = response.headers.get("Content-Length")
            if length and int(length) > _MAX_BYTES:
                raise WebResearchError("Source exceeds the 20 MiB download limit.")
            chunks, size = [], 0
            while True:
                if time.monotonic() > deadline:
                    raise WebResearchError("Public source download exceeded its time budget.")
                chunk = response.read(min(65536, _MAX_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > _MAX_BYTES:
                    raise WebResearchError("Source exceeds the 20 MiB download limit.")
            return (b"".join(chunks), final_url, response.headers.get_content_type(),
                    response.headers.get_content_charset() or "utf-8")
    except urllib.error.HTTPError as exc:
        reason = f"Source unavailable: HTTP {exc.code} {exc.reason} ({url}). No article content was read."
        if exc.code in {401, 403, 404, 406, 410, 429}:
            _remember_unavailable(url, reason)
        raise SourceUnavailable(reason) from exc
    except (OSError, urllib.error.URLError, ValueError) as exc:
        if isinstance(exc, WebResearchError):
            raise
        if isinstance(exc, (OSError, urllib.error.URLError)):
            raise SourceUnavailable(f"Public source retrieval failed: {exc}") from exc
        raise WebResearchError(f"Public source retrieval failed: {exc}") from exc


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _SearchParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.capture = None
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if tag == "a" and classes & {"result-link", "result__a"}:
            href = urllib.parse.urljoin("https://duckduckgo.com", attrs.get("href", ""))
            href = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query).get("uddg", [href])[0]
            self.results.append({"title": "", "url": href, "snippet": ""})
            self.capture, self.chunks = (tag, "title"), []
        elif classes & {"result-snippet", "result__snippet"} and self.results:
            self.capture, self.chunks = (tag, "snippet"), []

    def handle_data(self, data):
        if self.capture:
            self.chunks.append(data)

    def handle_endtag(self, tag):
        if self.capture and tag == self.capture[0]:
            self.results[-1][self.capture[1]] = _clean(" ".join(self.chunks))
            self.capture = None


def _search_duckduckgo(query: str, *, limit: int = 5) -> dict:
    """Search public DuckDuckGo pages, returning source links and honest service failures."""
    if not isinstance(query, str) or not query.strip() or len(query) > 2048:
        raise WebResearchError("Search query must contain 1–2048 characters.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise WebResearchError("Search limit must be an integer from 1 to 10.")
    errors = []
    for endpoint in _SEARCH_ENDPOINTS:
        try:
            data, url, _, encoding = _fetch(endpoint + urllib.parse.quote_plus(query.strip()))
            document = data.decode(encoding, errors="replace")
            if "anomaly.js" in document or "challenge-form" in document:
                # Both endpoints are the same service, not independent fallbacks.
                raise SourceUnavailable("DuckDuckGo returned a CAPTCHA; no results obtained.")
            parser = _SearchParser()
            parser.feed(document)
            results, seen = [], set()
            for item in parser.results:
                try:
                    target = _public_url(item["url"], resolve=False)
                except WebResearchError:
                    continue
                if target not in seen and item["title"]:
                    results.append({**item, "url": target})
                    seen.add(target)
            if not results and not re.search(r"no results|no more results", document, re.I):
                raise WebResearchError("Search response had no recognizable results; service may be blocked.")
            return {"query": query.strip(), "provider": "DuckDuckGo", "search_url": url,
                    "retrieved_at": _now(), "results": results[:limit],
                    "warnings": errors, "source_kind": "untrusted_public_web"}
        except SourceUnavailable:
            raise
        except WebResearchError as exc:
            errors.append(str(exc))
    raise SourceUnavailable("Web search unavailable: " + "; ".join(errors))


def search_configuration() -> dict:
    from .model_settings import read_settings
    config = read_settings().get("search", {})
    providers = config.get("providers", ["brave", "crossref", "duckduckgo"])
    if not isinstance(providers, list) or not providers or any(p not in {"brave", "crossref", "duckduckgo"} for p in providers):
        raise WebResearchError("search.providers must list brave, crossref and/or duckduckgo")
    key_env = config.get("brave_api_key_env", "BRAVE_SEARCH_API_KEY")
    if not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
        raise WebResearchError("Invalid search.brave_api_key_env")
    return {"providers": list(dict.fromkeys(providers)), "brave_api_key_env": key_env,
            "brave_configured": bool(os.environ.get(key_env))}


def _search_api(provider: str, query: str, limit: int, config: dict) -> dict:
    if provider == "brave":
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": limit})
        data, _, _, _ = _fetch(url, headers={"Accept": "application/json",
            "X-Subscription-Token": os.environ[config["brave_api_key_env"]]})
        rows = json.loads(data).get("web", {}).get("results", [])
        results = [{"title": row.get("title", ""), "url": row.get("url", ""),
                    "snippet": row.get("description", "")} for row in rows]
        scope = "web_links_and_snippets_not_full_text"
    else:
        # Crossref is a scholarly metadata index, NOT a general web engine.
        # It cannot implement site: filters; disclose that instead of silently removing them.
        if re.search(r"\b(?:site|filetype):", query):
            raise SourceUnavailable("Crossref cannot enforce site:/filetype: filters; use a plain scholarly query or a web provider.")
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": query, "rows": limit})
        data, _, _, _ = _fetch(url)
        rows = json.loads(data)["message"]["items"]
        results = [{"title": " ".join(row.get("title", [])), "url": row.get("URL", ""),
                    "snippet": _clean(re.sub(r"<[^>]+>", " ", row.get("abstract", "")))[:1200]}
                   for row in rows]
        scope = "scholarly_metadata_not_full_text_or_general_web"
    safe, seen = [], set()
    for row in results:
        try:
            row["url"] = _public_url(row["url"], resolve=False)
        except WebResearchError:
            continue
        if row["title"] and row["url"] not in seen:
            seen.add(row["url"])
            safe.append(row)
    return {"query": query, "provider": provider, "search_url": url, "scope": scope,
            "results": safe[:limit], "retrieved_at": _now(), "source_kind": "untrusted_public_web",
            "warnings": ["Discovery only: read the linked source before claiming that the paper supports a design decision."]}


def search_web(query: str, *, limit: int = 5) -> dict:
    if not isinstance(query, str) or not query.strip() or len(query) > 2048:
        raise WebResearchError("Search query must contain 1–2048 characters.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise WebResearchError("Search limit must be an integer from 1 to 10.")
    config, errors, empty = search_configuration(), [], None
    for provider in config["providers"]:
        if provider == "brave" and not config["brave_configured"]:
            errors.append("Brave not configured; add its API key for general web search.")
            continue
        cooldown = "search-provider:" + provider
        with _BLOCKED_LOCK:
            blocked = _BLOCKED.get(cooldown)
        if blocked and blocked[0] > time.monotonic():
            errors.append(f"{provider}: temporarily unavailable; repeated service requests suppressed.")
            continue
        try:
            result = (_search_duckduckgo(query, limit=limit) if provider == "duckduckgo"
                      else _search_api(provider, query.strip(), limit, config))
            if not result["results"]:
                empty = result
                errors.append(f"{provider}: no matching results returned.")
                continue
            result["warnings"] = errors + result.get("warnings", [])
            return result
        except (WebResearchError, ValueError, KeyError, TypeError) as exc:
            # No raw authenticated HTTP error/body, URL tokens or API credentials.
            reason = str(exc) if isinstance(exc, WebResearchError) else "Invalid search service response"
            secret = os.environ.get(config["brave_api_key_env"], "")
            if secret:
                reason = reason.replace(secret, "[redacted]")
            errors.append(f"{provider}: {reason}")
            if not (provider == "crossref" and "filters" in reason):
                _remember_unavailable(cooldown, reason)
    if empty is not None:
        empty["warnings"] = errors
        return empty
    raise SourceUnavailable("Web search unavailable: " + "; ".join(errors))


class _PageParser(HTMLParser):
    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    _HIDDEN = {"script", "style", "noscript", "svg", "nav", "aside", "template"}
    _BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "section", "article", "main"}

    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url, self.text, self.title, self.links = url, [], [], []
        self.stack, self.sections = [], []

    def _append(self, value):
        self.text.append(value)
        for frame in self.stack:
            if frame["section"] is not None:
                frame["section"]["text"].append(value)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        role = str(attrs.get("role") or "").lower()
        in_content = any(frame["section"] is not None for frame in self.stack)
        # Article headers can contain the paper title/authors. Suppress page
        # chrome, not scientific headers belonging to the selected article.
        hidden = (bool(self.stack and self.stack[-1]["hidden"])
                  or tag in self._HIDDEN or role in {"navigation", "complementary"}
                  or "hidden" in attrs or attrs.get("aria-hidden") == "true"
                  or (tag in {"header", "footer"} and not in_content))
        kind = "main" if tag == "main" or role == "main" else "article" if tag == "article" else None
        section = None
        if kind and not hidden:
            section = {"kind": kind, "text": [], "links": [],
                       "nested": any(frame["section"] and frame["section"]["kind"] == kind
                                     for frame in self.stack)}
            self.sections.append(section)
        frame = {"tag": tag, "hidden": hidden, "section": section, "link": None}
        if tag not in self._VOID:
            self.stack.append(frame)
        if hidden:
            return
        if tag in self._BLOCK:
            self._append("\n")
        if tag in {"td", "th"}:
            self._append("\t")
        if tag == "a" and attrs.get("href"):
            try:
                link = {"url": _public_url(urllib.parse.urljoin(self.url, attrs["href"]),
                                          resolve=False), "title": ""}
                frame["link"] = link
                if len(self.links) < 80:
                    self.links.append(link)
                # A large navigation preceding main must not consume the
                # article's link allowance (PDFs/references often occur last).
                for ancestor in self.stack:
                    selected = ancestor["section"]
                    if selected is not None and len(selected["links"]) < 80:
                        selected["links"].append(link)
            except WebResearchError:
                pass

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                if not self.stack[index]["hidden"] and tag in self._BLOCK:
                    self._append("\n")
                del self.stack[index:]
                break

    def handle_data(self, data):
        if not self.stack or not self.stack[-1]["hidden"]:
            if any(frame["tag"] == "title" for frame in self.stack):
                self.title.append(data)
                return
            self._append(data)
            for frame in reversed(self.stack):
                if frame["link"] is not None:
                    frame["link"]["title"] += data
                    break

    def content(self):
        """Prefer semantic primary content; empty landmarks must not hide a page."""
        for kind in ("main", "article"):
            selected = [part for part in self.sections if part["kind"] == kind
                        and not part["nested"] and _clean("".join(part["text"]))]
            if selected:
                return ([item for part in selected for item in part["text"]],
                        [link for part in selected for link in part["links"]][:80], kind)
        return self.text, self.links, "body"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_web_page(url: str, *, offset: int = 0, limit: int = 12000) -> dict:
    """Read bounded public HTML, text, or a text-bearing PDF; offsets are characters."""
    if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 40000):
        raise WebResearchError("Offset must be nonnegative; character limit must be 1–40000.")
    data, final_url, content_type, encoding = _fetch(url)
    links, title, warnings, extraction = [], "", [], None
    if content_type == "application/pdf" or data.startswith(b"%PDF-"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            title = str((reader.metadata or {}).get("/Title", ""))
            pages, size = [], 0
            for number, page in enumerate(reader.pages, 1):
                part = f"\n[Page {number}]\n" + (page.extract_text() or "")
                pages.append(part)
                size += len(part)
                if size >= _MAX_TEXT:
                    break
            text = "\n".join(pages)
            warnings.append("PDF text extraction only; figures, scanned text and equation layout may be missing.")
        except ImportError as exc:
            raise WebResearchError("Reading research PDFs requires the pypdf package.") from exc
        except Exception as exc:
            raise WebResearchError(f"Could not extract PDF text: {exc}") from exc
    elif content_type in {"text/html", "application/xhtml+xml"}:
        document = data.decode(encoding, errors="replace")
        if re.search(r"anomaly\.js|challenge-form|cf-chl-|challenge-platform|"
                     r"<title>\s*(?:just a moment|access denied|verify you are human|checking your browser)",
                     document, re.I):
            reason = "Source returned an access challenge, not readable article content."
            _remember_unavailable(_public_url(url, resolve=False), reason)
            raise SourceUnavailable(reason)
        parser = _PageParser(final_url)
        parser.feed(document)
        parser.close()
        parts, links, extraction = parser.content()
        text = "\n".join(line for item in "".join(parts).splitlines() if (line := _clean(item)))
        title = _clean(" ".join(parser.title))
        for link in links:
            link["title"] = _clean(link["title"])
    elif content_type.startswith("text/") or content_type in {"application/json", "application/xml"}:
        text = data.decode(encoding, errors="replace")
    else:
        raise WebResearchError(f"Unsupported source content type: {content_type}.")
    source_truncated = len(text) >= _MAX_TEXT
    text = text[:_MAX_TEXT]
    end = min(len(text), offset + limit)
    return {"url": url, "final_url": final_url, "title": title, "content_type": content_type,
            "retrieved_at": _now(), "text": text[offset:end], "offset": offset,
            "next_offset": end if end < len(text) else None, "total_characters": len(text),
            "source_truncated": source_truncated, "links": links, "warnings": warnings,
            **({"html_content_scope": extraction} if extraction else {}),
            "source_kind": "untrusted_public_web"}


def read_web_page(url: str, *, offset: int = 0, limit: int = 12000) -> dict:
    """Read a source; unavailable arXiv HTML may fall back once to its official PDF.

    The resolved URL and fallback warning identify what was actually read.
    No paywall/challenge bypass or guessed publisher mirror is attempted.
    """
    try:
        return _read_web_page(url, offset=offset, limit=limit)
    except SourceUnavailable as exc:
        if re.search(r"HTTP (?:401|429)\b", str(exc)):
            # Do not try another endpoint to evade authentication/rate limits.
            raise
        parsed = urllib.parse.urlsplit(url)
        match = re.fullmatch(r"/html/((?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?)/?", parsed.path)
        if parsed.hostname not in {"arxiv.org", "www.arxiv.org"} or not match:
            raise
        alternative = "https://arxiv.org/pdf/" + match[1]
        try:
            result = _read_web_page(alternative, offset=offset, limit=limit)
        except WebResearchError as fallback:
            raise SourceUnavailable(f"{exc} Official PDF alternative also unavailable: {fallback}") from fallback
        result["url"] = url
        result["fallback_from"] = url
        result["warnings"].insert(0, f"Requested HTML unavailable ({exc}); reading the official PDF instead. Offsets refer to PDF text.")
        return result
