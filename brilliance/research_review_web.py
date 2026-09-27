"""Audit recorded online research without fetching URLs or judging relevance.

Tool success, returned search results, and readable source content are separate
observations. Provider warnings are not an HTTP-request log. Text coverage refers
only to the saved extraction, never to reading an entire scientific paper.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def _text(value, limit=500):
    return str(value)[:limit] if value is not None else ""


def _integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _url(value):
    """Keep distinct endpoints/queries; remove fragments and credential fields."""
    if not isinstance(value, str):
        return ""
    try:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return ""
        host = parts.hostname.lower()
        if ":" in host:
            host = "[" + host + "]"
        if parts.port:
            host += ":" + str(parts.port)
        query = parts.query
        if any(re.search(r"(?:api[-_]?key|access[-_]?token|password|secret|authorization)", key, re.I)
               for key, _ in parse_qsl(query, keep_blank_values=True)):
            query = urlencode([(key, "[redacted]" if re.search(
                r"(?:api[-_]?key|access[-_]?token|password|secret|authorization)", key, re.I) else val)
                for key, val in parse_qsl(query, keep_blank_values=True)])
        return urlunsplit((parts.scheme, host, parts.path, query, ""))
    except ValueError:
        return ""


def _domain(value):
    return urlsplit(value).hostname or "" if value else ""


def _status(result):
    if not isinstance(result, dict):
        return "unknown"
    if str(result.get("error") or "").startswith("Not executed:"):
        return "skipped"
    if result.get("ok") is True:
        return "success"
    if result.get("ok") is False or result.get("isError") is True:
        return "failure"
    return "unknown"


def _error_class(error, result, content_returned):
    recorded = {"Timeout": "timeout", "RateLimit": "rate_limit", "AuthenticationError": "authentication",
                "PermissionError": "permission", "ConnectionError": "connection",
                "Interrupted": "interrupted", "ToolError": "tool_error"}.get(str(result.get("error_class") or ""))
    if recorded:
        return recorded
    match = re.search(r"\bHTTP\s*[:=]?\s*(\d{3})\b", error, re.I)
    if match:
        return "http_" + match[1]
    if re.search(r"timed?\s*out|timeout", error, re.I):
        return "timeout"
    if re.search(r"access challenge|access denied|verify you are human", error, re.I):
        return "access_challenge"
    if result.get("source_unavailable") is True or re.search(r"unavailable", error, re.I):
        return "unavailable"
    if error:
        return "tool_error"
    return "no_content" if content_returned is False else ""


def _request(call):
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    tool = str(call.get("tool") or "")
    leaf = tool.rsplit(".", 1)[-1]
    if leaf == "search_web":
        queries = [_text(args.get("query"), 2048)] if args.get("query") else []
        return "search", queries, [], args
    if leaf == "read_web_page":
        return "read", [], [_url(args.get("url"))] if _url(args.get("url")) else [], args
    queries, urls = [], []
    action = args.get("action") if isinstance(args.get("action"), dict) else args
    searches = args.get("search_query")
    opens = args.get("open")
    if isinstance(searches, list):
        queries = [_text(row.get("q"), 2048) for row in searches if isinstance(row, dict) and row.get("q")]
    if isinstance(opens, list):
        urls = [_url(row.get("ref_id")) for row in opens if isinstance(row, dict) and _url(row.get("ref_id"))]
    if action.get("type") == "search":
        recorded = action.get("queries")
        queries = [_text(q, 2048) for q in recorded if isinstance(q, str)] if isinstance(recorded, list) else []
        if action.get("query") and str(action["query"]) not in queries:
            queries.append(_text(action["query"], 2048))
    if action.get("type") == "openPage" and _url(action.get("url")):
        urls = [_url(action["url"])]
    has_search = isinstance(searches, list) or action.get("type") == "search"
    has_read = isinstance(opens, list) or action.get("type") == "openPage"
    kind = "search" if has_search and not has_read else "read" if has_read and not has_search else "other"
    return kind, queries, urls, args


def _counter():
    return {"calls": 0, "succeeded": 0, "failed": 0, "unknown": 0, "skipped": 0}


def _count(row, status):
    row["calls"] += 1
    row[{"success": "succeeded", "failure": "failed"}.get(status, status)] += 1


def _rate(numerator, denominator):
    return round(100 * numerator / denominator, 4) if denominator else None


def _finish_counter(row):
    row["completed"] = row["succeeded"] + row["failed"]
    row["success_rate_percent"] = _rate(row["succeeded"], row["completed"])


def _union_length(intervals):
    end, total = -1, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize_web_calls(calls: list[dict]) -> dict:
    """Summarize already paired canonical calls; do not read saved-source copies."""
    rows, warnings = [], []
    provider_rows = {}
    source_rows = {}
    intervals = defaultdict(list)
    totals = defaultdict(set)

    def provider(name):
        if name not in provider_rows:
            provider_rows[name] = {"provider": name, **_counter(), "result_count": 0,
                "results_known": 0, "nonempty_results": 0, "warning_mentions": 0,
                "unconfigured_warnings": 0, "suppressed_warnings": 0, "http_attempts": None}
        return provider_rows[name]

    for call in calls:
        if not isinstance(call, dict):
            continue
        kind, queries, requested_urls, args = _request(call)
        result = call.get("result") if isinstance(call.get("result"), dict) else {}
        status = _status(call.get("result"))
        content = result.get("text")
        recorded_characters = _integer(result.get("content_characters"))
        content_known = isinstance(content, str) or recorded_characters is not None or status == "failure"
        returned = len(content) if isinstance(content, str) else recorded_characters if recorded_characters is not None else 0 if status == "failure" else None
        content_returned = (bool(content.strip()) if isinstance(content, str) else
                            result.get("content_returned") if isinstance(result.get("content_returned"), bool) else
                            bool(recorded_characters) if recorded_characters is not None else
                            False if status == "failure" else None)
        if status == "failure":
            content_returned = False
        found = result.get("results")
        results_known = isinstance(found, list) or status == "failure"
        result_count = len(found) if isinstance(found, list) else 0 if status == "failure" else None
        results = [{"url": _url(item.get("url")), "title": _text(item.get("title"), 500)}
                   for item in found if isinstance(item, dict)] if isinstance(found, list) else []
        provider_name = _text(result.get("provider") or args.get("provider"), 100) or "unknown"
        request_url = requested_urls[0] if len(requested_urls) == 1 else _url(result.get("url"))
        final_url = _url(result.get("final_url"))
        if result.get("query") and not queries:
            queries = [_text(result["query"], 2048)]
        error = _text(result.get("error"), 500)
        recorded_warnings = result.get("warnings")
        call_warnings = [_text(value, 500) for value in recorded_warnings if isinstance(value, str)] if isinstance(recorded_warnings, list) else []
        row = {"id": _text(call.get("id"), 500), "runtime": _text(call.get("runtime"), 100),
            "tool": _text(call.get("tool"), 200), "kind": kind,
            "source": _text(call.get("source"), 1000), "event_index": call.get("event_index"),
            "result_event_index": call.get("result_event_index"), "status": status,
            "query": " | ".join(queries), "queries": queries, "provider": provider_name,
            "scope": _text(result.get("scope"), 200), "retrieved_at": _text(result.get("retrieved_at"), 100),
            "result_count": result_count if kind == "search" else None,
            "nonempty_results": bool(result_count) and status != "failure" if kind == "search" and results_known else None,
            "results": results if kind == "search" else [], "requested_url": request_url,
            "requested_urls": requested_urls, "final_url": final_url,
            "domain": _domain(final_url or request_url), "title": _text(result.get("title"), 500),
            "content_type": _text(result.get("content_type"), 100),
            "html_content_scope": _text(result.get("html_content_scope"), 100),
            "returned_characters": returned if kind == "read" else None,
            "content_returned": content_returned if kind == "read" else None,
            "requested_offset": _integer(args.get("offset")), "requested_limit": _integer(args.get("limit")),
            "offset": _integer(result.get("offset")), "next_offset": _integer(result.get("next_offset")),
            "total_characters": _integer(result.get("total_characters")),
            "source_truncated": result.get("source_truncated") if isinstance(result.get("source_truncated"), bool) else None,
            "fallback_from": _url(result.get("fallback_from")), "error": error,
            "error_class": _error_class(error, result, content_returned if kind == "read" else None),
            "warnings": call_warnings, "request_detail_known": bool(queries or requested_urls)}
        rows.append(row)
        if kind == "search":
            aggregate = provider(provider_name)
            _count(aggregate, status)
            if results_known:
                aggregate["results_known"] += 1
                aggregate["result_count"] += result_count
                aggregate["nonempty_results"] += row["nonempty_results"] is True
            for warning in call_warnings:
                match = re.match(r"(brave|crossref|duckduckgo)\b", warning, re.I)
                if match:
                    mentioned = provider(match[1].lower())
                    mentioned["warning_mentions"] += 1
                    mentioned["unconfigured_warnings"] += bool(re.search(r"not configured", warning, re.I))
                    mentioned["suppressed_warnings"] += bool(re.search(r"suppressed", warning, re.I))
        if kind != "read":
            continue
        source_url = final_url or request_url
        if not source_url:
            continue
        if source_url not in source_rows:
            source_rows[source_url] = {"url": source_url, "domain": _domain(source_url),
                "requested_urls": [], **_counter(), "content_calls": 0, "content_known": 0,
                "returned_characters": 0, "interval_calls": 0, "html_content_scopes": [],
                "content_types": [], "fallback_calls": 0, "source_truncated": False,
                "title": "", "error_classes": {}}
        aggregate = source_rows[source_url]
        _count(aggregate, status)
        for field, item in (("requested_urls", request_url), ("html_content_scopes", row["html_content_scope"]),
                            ("content_types", row["content_type"])):
            if item and item not in aggregate[field]:
                aggregate[field].append(item)
        aggregate["title"] = aggregate["title"] or row["title"]
        aggregate["content_known"] += content_known
        aggregate["content_calls"] += content_returned is True
        aggregate["returned_characters"] += returned or 0
        aggregate["fallback_calls"] += bool(row["fallback_from"])
        aggregate["source_truncated"] = aggregate["source_truncated"] or row["source_truncated"] is True
        if row["error_class"]:
            errors = aggregate["error_classes"]
            errors[row["error_class"]] = errors.get(row["error_class"], 0) + 1
        if row["offset"] is not None and returned is not None:
            intervals[source_url].append((row["offset"], row["offset"] + returned))
            aggregate["interval_calls"] += 1
        if row["total_characters"] is not None:
            totals[source_url].add(row["total_characters"])

    for aggregate in provider_rows.values():
        _finish_counter(aggregate)
        aggregate["nonempty_result_rate_percent"] = _rate(aggregate["nonempty_results"], aggregate["results_known"])
        if not aggregate["results_known"]:
            aggregate["result_count"] = None
    for source_url, aggregate in source_rows.items():
        _finish_counter(aggregate)
        aggregate["unique_returned_characters"] = _union_length(intervals[source_url]) if aggregate["interval_calls"] else None
        recorded_totals = sorted(totals[source_url])
        aggregate["recorded_total_characters"] = recorded_totals[0] if len(recorded_totals) == 1 else None
        known_total = aggregate["recorded_total_characters"]
        observed = aggregate["unique_returned_characters"]
        aggregate["extracted_text_coverage_percent"] = _rate(observed, known_total) if observed is not None and known_total and observed <= known_total else None
        aggregate["content_rate_percent"] = _rate(aggregate["content_calls"], aggregate["content_known"])
        if not aggregate["content_known"]:
            aggregate["returned_characters"] = None
        aggregate["coverage"] = (f"Text length recorded for {aggregate['content_known']}/{aggregate['calls']} outcomes; "
            f"offset intervals for {aggregate['interval_calls']}/{aggregate['calls']} reads. "
            "Union of saved extracted-text offsets; does not establish full-paper reading or comprehension.")
        if len(recorded_totals) > 1:
            aggregate["coverage"] += " Extraction lengths differ between reads; text coverage percentage is unavailable."

    metrics = []

    def metric(identity, label, value, unit, definition, coverage="recorded canonical tool evidence", numerator=None, denominator=None):
        item = {"id": identity, "label": label, "value": value, "unit": unit,
                "definition": definition, "coverage": coverage, "group": "Online research"}
        if unit == "percent":
            item.update(numerator=numerator, denominator=denominator)
        metrics.append(item)

    for kind, label in (("search", "Search"), ("read", "Source read")):
        selected = [row for row in rows if row["kind"] == kind]
        count = Counter(row["status"] for row in selected)
        completed = count["success"] + count["failure"]
        coverage = f"Known technical outcomes {completed}/{len(selected)}; unknown {count['unknown']}; skipped {count['skipped']}."
        prefix = "web_" + kind + "_"
        for key, value, suffix in (("calls", len(selected), "calls"), ("succeeded", count["success"], "technical successes"),
                                  ("failed", count["failure"], "technical failures"), ("unknown", count["unknown"], "unknown outcomes"),
                                  ("skipped", count["skipped"], "skipped calls"), ("completed", completed, "known outcomes")):
            metric(prefix + key, label + " " + suffix, value, "count",
                   "One recorded tool invocation; provider retries/fallback HTTP requests are not separately counted.", coverage)
        metric(prefix + "success_rate", label + " technical success rate", _rate(count["success"], completed), "percent",
               "Explicit successful outcomes / (explicit successes + failures); unknown and skipped calls excluded. This is not relevance or scientific acceptance.",
               coverage, count["success"], completed)
        field = "nonempty_results" if kind == "search" else "content_returned"
        known = [row for row in selected if row[field] is not None]
        positive = sum(row[field] is True for row in known)
        metric(prefix + ("nonempty_result_rate" if kind == "search" else "content_rate"),
               label + (" non-empty-result rate" if kind == "search" else " readable-content return rate"),
               _rate(positive, len(known)), "percent",
               ("Calls returning at least one result / calls with a saved result list or explicit failure; empty successful searches and failures count as zero."
                if kind == "search" else "Calls returning non-whitespace extracted text / calls with saved text or explicit failure; explicit failures count as zero even if a partial payload is retained. Technical completion alone does not establish content returned."),
               f"Content/result outcome known for {len(known)}/{len(selected)} calls; missing details excluded.", positive, len(known))

    searches = [row for row in rows if row["kind"] == "search"]
    reads = [row for row in rows if row["kind"] == "read"]
    unique_queries = {query for row in searches for query in row["queries"]}
    result_urls = {item["url"] for row in searches for item in row["results"] if item["url"]}
    metric("web_search_unique_queries", "Distinct recorded search queries", len(unique_queries), "count", "Exact recorded query strings; repeated identical queries count once.")
    metric("web_search_result_count", "Returned search-result entries", sum(row["result_count"] or 0 for row in searches), "count", "Observed result entries, including repeats between queries; a search hit is not a read or endorsement.")
    metric("web_search_unique_result_urls", "Distinct search-result URLs", len(result_urls), "count", "Result URLs with fragments removed; abstract, HTML and PDF endpoints remain distinct.")
    discovered_by_runtime = defaultdict(set)
    requested_after_search, content_after_search = set(), set()
    for row in rows:
        discovered = discovered_by_runtime[row["runtime"]]
        if row["kind"] == "search":
            discovered.update(item["url"] for item in row["results"] if item["url"])
        if row["kind"] == "read":
            matching = discovered.intersection(row["requested_urls"] + [row["requested_url"], row["final_url"]])
            requested_after_search.update(matching)
            if row["content_returned"] is True:
                content_after_search.update(matching)
    for suffix, label, observed in (("requested", "subsequently requested", requested_after_search),
                                     ("with_content", "subsequently yielding content", content_after_search)):
        metric("web_search_result_urls_" + suffix, "Search-result URLs " + label, len(observed), "count",
               "Exact URL match against a later requested/final source URL within the same runtime's recorded order; fragments removed. Retrieval follow-through, not citation, relevance or paper identity. No ordering is inferred across runtimes.")
    metric("web_read_unique_urls", "Distinct source URLs attempted", len(source_rows), "count", "Final resolved URL where recorded, otherwise requested URL; no inference that distinct URLs mean distinct papers.")
    metric("web_read_returned_characters", "Source characters returned", sum(row["returned_characters"] or 0 for row in reads), "characters", "Saved extracted-text character counts; repeated and overlapping chunks remain in this transfer total.")
    metric("web_read_fallback_calls", "Observed source fallback calls", sum(bool(row["fallback_from"]) for row in reads), "count", "Read calls explicitly recording fallback_from; hidden retries and unrecorded fallbacks are unknown.")
    if rows:
        warnings.append("Search technical success and non-empty results do not establish relevance, source support, or scientific correctness.")
        warnings.append("Provider warnings (including unconfigured keys and suppressed requests) are not HTTP attempts; provider HTTP-attempt counts/rates are unavailable.")
        warnings.append("Source coverage measures saved extracted text only. Abstract pages, PDF extraction limits, figures, equations and comprehension are not full-paper reading.")
    if any(row["kind"] == "other" for row in rows):
        warnings.append("Some retained web calls have unknown or mixed action types; they are listed but excluded from separate search/read rates.")
    if any(row["status"] == "unknown" for row in rows):
        warnings.append("Missing result details remain unknown; SDK completion events alone are not proof of retrieval success.")
    return {"calls": rows, "metrics": metrics,
            "providers": sorted(provider_rows.values(), key=lambda row: row["provider"]),
            "sources": sorted(source_rows.values(), key=lambda row: row["url"]), "warnings": warnings}
