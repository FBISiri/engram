#!/usr/bin/env python3
"""engram-cli — a thin shell wrapper over the Engram REST API.

This CLI reinvents nothing: every subcommand maps 1:1 onto an existing REST
route served by pkg/server/http.go. It uses only the Python standard library
(urllib/argparse/json) so it runs on a bare host with no pip installs.

Config:
  ENGRAM_URL      base URL of the engram HTTP server (default http://localhost:8080)
  ENGRAM_API_KEY  bearer token for the Authorization header (optional)
Both are overridable with --url / --api-key.

Output:
  raw JSON to stdout by default; --pretty for indented human-readable JSON.
  On an HTTP error the response body is printed to stderr and the process
  exits non-zero.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8080"


def _die(msg, code=1):
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.exit(code)


def request(args, method, path, body=None):
    """Perform an HTTP request against the engram server.

    Returns the raw response text (str). On a non-2xx status or a transport
    error, prints the error to stderr and exits non-zero.
    """
    base = (args.url or os.environ.get("ENGRAM_URL") or DEFAULT_URL).rstrip("/")
    url = base + path

    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    api_key = args.api_key or os.environ.get("ENGRAM_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    caller_type = getattr(args, "caller_type", None)
    if caller_type:
        headers["X-Caller-Type"] = caller_type

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            pass
        _die("HTTP %d %s: %s" % (e.code, e.reason, detail.strip()))
    except urllib.error.URLError as e:
        _die("request error: %s" % (e.reason,))


def emit(args, text):
    """Print a response body to stdout, pretty-printing JSON if requested."""
    if getattr(args, "pretty", False):
        try:
            parsed = json.loads(text)
            sys.stdout.write(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n")
            return
        except (ValueError, TypeError):
            pass
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


# ─────────────────────────────────────────────────────────────
# Subcommand handlers — each maps onto exactly one REST route.
# ─────────────────────────────────────────────────────────────

def cmd_add(args):
    body = {"content": args.content}
    if args.type:
        body["type"] = args.type
    if args.importance is not None:
        body["importance"] = args.importance
    if args.tags:
        body["tags"] = args.tags
    if args.source:
        body["source"] = args.source
    if args.valid_until is not None:
        body["valid_until"] = args.valid_until
    if args.metadata:
        body["metadata"] = _parse_json_obj(args.metadata, "--metadata")
    if args.collection:
        path = "/collections/%s/memories" % args.collection
    else:
        path = "/memories"
    return request(args, "POST", path, body)


def cmd_search(args):
    body = {"query": args.query}
    if args.limit is not None:
        body["limit"] = args.limit
    if args.include_archived:
        body["include_archived"] = True
    if args.types:
        body["types"] = args.types
    if args.tags:
        body["tags"] = args.tags
    if args.collection:
        # The legacy /memories/search route accepts an explicit `collection`
        # field (see handleSearchMemories) — no caller-type ownership needed.
        body["collection"] = args.collection
    return request(args, "POST", "/memories/search", body)


def cmd_get(args):
    if args.collection:
        path = "/collections/%s/memories/%s" % (args.collection, args.id)
    else:
        path = "/memories/%s" % args.id
    return request(args, "GET", path)


def cmd_update(args):
    # If --content is given we must PUT (content is re-embedded and cannot be
    # PATCHed). Otherwise a partial PATCH of metadata-style fields.
    if args.collection:
        path = "/collections/%s/memories/%s" % (args.collection, args.id)
    else:
        path = "/memories/%s" % args.id

    if args.content is not None:
        body = {"content": args.content}
        if args.type:
            body["type"] = args.type
        if args.importance is not None:
            body["importance"] = args.importance
        if args.tags:
            body["tags"] = args.tags
        if args.source:
            body["source"] = args.source
        if args.valid_until is not None:
            body["valid_until"] = args.valid_until
        if args.metadata:
            body["metadata"] = _parse_json_obj(args.metadata, "--metadata")
        return request(args, "PUT", path, body)

    body = {}
    if args.importance is not None:
        body["importance"] = args.importance
    if args.tags:
        body["tags"] = args.tags
    if args.source:
        body["source"] = args.source
    if args.lifecycle_status:
        body["lifecycle_status"] = args.lifecycle_status
    if args.metadata:
        body["metadata"] = _parse_json_obj(args.metadata, "--metadata")
    if not body:
        _die("update: nothing to change — provide --content (PUT) or at least "
             "one of --importance/--tags/--source/--lifecycle-status/--metadata (PATCH)")
    return request(args, "PATCH", path, body)


def cmd_delete(args):
    if args.collection:
        path = "/collections/%s/memories/%s" % (args.collection, args.id)
    else:
        path = "/memories/%s" % args.id
    return request(args, "DELETE", path)


def cmd_reset(args):
    if args.collection:
        path = "/collections/%s/memories/%s/reset" % (args.collection, args.id)
    else:
        path = "/memories/%s/reset" % args.id
    return request(args, "POST", path)


def cmd_list_collections(args):
    return request(args, "GET", "/collections")


def cmd_create_collection(args):
    body = {"name": args.name}
    if args.ttl:
        body["ttl"] = args.ttl
    return request(args, "POST", "/collections", body)


def cmd_cross_search(args):
    body = {"query": args.query, "collections": args.collections}
    if args.limit is not None:
        body["limit"] = args.limit
    if args.include_archived:
        body["include_archived"] = True
    if args.types:
        body["types"] = args.types
    if args.tags:
        body["tags"] = args.tags
    return request(args, "POST", "/memories/cross-search", body)


def cmd_reflect(args):
    body = {"dry_run": True} if args.dry_run else {}
    return request(args, "POST", "/reflect", body)


def cmd_reflect_check(args):
    return request(args, "GET", "/reflect/check")


def cmd_health(args):
    return request(args, "GET", "/health")


def _parse_json_obj(s, flag):
    try:
        v = json.loads(s)
    except ValueError as e:
        _die("%s: invalid JSON: %s" % (flag, e))
    if not isinstance(v, dict):
        _die("%s: expected a JSON object" % flag)
    return v


# ─────────────────────────────────────────────────────────────
# Argument wiring
# ─────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="engram-cli",
        description="Thin CLI wrapper over the Engram REST API.",
    )
    p.add_argument("--url", help="engram base URL (default $ENGRAM_URL or %s)" % DEFAULT_URL)
    p.add_argument("--api-key", help="bearer token (default $ENGRAM_API_KEY)")
    p.add_argument("--caller-type", choices=["user", "agent-self", "reflection", "pigo"],
                   help="X-Caller-Type header (required for --collection ownership routes)")
    p.add_argument("--pretty", action="store_true", help="pretty-print JSON output")

    sub = p.add_subparsers(dest="command")
    sub.required = True  # py3.6 compat: not accepted as add_subparsers kwarg

    sp = sub.add_parser("add", help="create a memory (POST /memories)")
    sp.add_argument("content")
    sp.add_argument("--type", choices=["identity", "event", "insight", "directive"])
    sp.add_argument("--importance", type=float)
    sp.add_argument("--tags", nargs="+")
    sp.add_argument("--source")
    sp.add_argument("--valid-until", type=float, dest="valid_until")
    sp.add_argument("--metadata", help="JSON object string")
    sp.add_argument("--collection")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("search", help="vector search (POST /memories/search)")
    sp.add_argument("query")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--include-archived", action="store_true", dest="include_archived")
    sp.add_argument("--types", nargs="+")
    sp.add_argument("--tags", nargs="+")
    sp.add_argument("--collection")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("get", help="fetch a memory by id (GET /memories/{id})")
    sp.add_argument("id")
    sp.add_argument("--collection")
    sp.set_defaults(func=cmd_get)

    sp = sub.add_parser("update", help="update a memory (PATCH, or PUT with --content)")
    sp.add_argument("id")
    sp.add_argument("--content", help="new content — switches to PUT (re-embeds)")
    sp.add_argument("--type", choices=["identity", "event", "insight", "directive"])
    sp.add_argument("--importance", type=float)
    sp.add_argument("--tags", nargs="+")
    sp.add_argument("--source")
    sp.add_argument("--lifecycle-status", dest="lifecycle_status",
                    choices=["active", "deprecated", "archived"])
    sp.add_argument("--valid-until", type=float, dest="valid_until")
    sp.add_argument("--metadata", help="JSON object string")
    sp.add_argument("--collection")
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser("delete", help="soft-delete a memory (DELETE /memories/{id})")
    sp.add_argument("id")
    sp.add_argument("--collection")
    sp.set_defaults(func=cmd_delete)

    sp = sub.add_parser("reset", help="restore archived/deprecated → active (POST /memories/{id}/reset)")
    sp.add_argument("id")
    sp.add_argument("--collection")
    sp.set_defaults(func=cmd_reset)

    sp = sub.add_parser("list-collections", help="list collections (GET /collections)")
    sp.set_defaults(func=cmd_list_collections)

    sp = sub.add_parser("create-collection", help="register a collection (POST /collections)")
    sp.add_argument("name")
    sp.add_argument("--ttl", help='e.g. "30d" or "720h"')
    sp.set_defaults(func=cmd_create_collection)

    sp = sub.add_parser("cross-search", help="cross-collection search (POST /memories/cross-search)")
    sp.add_argument("query")
    sp.add_argument("--collections", nargs="+", required=True)
    sp.add_argument("--limit", type=int)
    sp.add_argument("--include-archived", action="store_true", dest="include_archived")
    sp.add_argument("--types", nargs="+")
    sp.add_argument("--tags", nargs="+")
    sp.set_defaults(func=cmd_cross_search)

    sp = sub.add_parser("reflect", help="run a reflection cycle (POST /reflect)")
    sp.add_argument("--dry-run", action="store_true", dest="dry_run")
    sp.set_defaults(func=cmd_reflect)

    sp = sub.add_parser("reflect-check", help="check reflection triggers (GET /reflect/check)")
    sp.set_defaults(func=cmd_reflect_check)

    sp = sub.add_parser("health", help="liveness probe (GET /health)")
    sp.set_defaults(func=cmd_health)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    text = args.func(args)
    emit(args, text)


if __name__ == "__main__":
    main()
