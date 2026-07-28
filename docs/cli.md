# Engram CLI (`engram-cli`)

> A thin shell wrapper over the Engram **REST API**. It reinvents no logic —
> every subcommand maps 1:1 onto an existing HTTP route (see
> [REST API Endpoints](api.md#rest-api-endpoints)). Stdlib-only Python 3
> (urllib/argparse/json), so it runs on a bare host with no `pip install`.

## Install

```bash
# from the repo root
chmod +x scripts/engram_cli.py

# optional: symlink to /usr/local/bin/engram-cli (idempotent)
sudo ./scripts/install-cli.sh
# or install elsewhere:
ENGRAM_CLI_DEST=~/.local/bin/engram-cli ./scripts/install-cli.sh
```

If you don't install it, invoke directly: `python3 scripts/engram_cli.py ...`
or `./scripts/engram_cli.py ...`.

## Configuration

| Env var | Flag | Default | Meaning |
|---------|------|---------|---------|
| `ENGRAM_URL` | `--url` | `http://localhost:8080` | base URL of the engram HTTP server |
| `ENGRAM_API_KEY` | `--api-key` | *(none)* | bearer token → `Authorization: Bearer <key>` |
| — | `--caller-type` | *(none)* | `X-Caller-Type` header: `user` \| `agent-self` \| `reflection` |
| — | `--pretty` | off | pretty-print JSON output |

Output is **raw JSON to stdout** by default. On an HTTP error the response body
is printed to **stderr** and the process exits **non-zero**.

Collection ownership: the per-collection routes (`--collection` on
`add`/`get`/`update`/`delete`/`reset`) enforce that `--caller-type` resolves to
the target collection (`user`→`engram_user`, `agent-self`→`engram_agent_self`,
`reflection`→`engram_reflection`); a mismatch returns HTTP 403.

## Subcommands

Each block below is copy-pasteable (assumes `ENGRAM_URL`/`ENGRAM_API_KEY` set).

### add — `POST /memories`
```bash
engram-cli add "Frank prefers dark mode" --type insight --importance 7 --tags ui prefs
# target a collection (needs matching --caller-type):
engram-cli --caller-type reflection add "synthesized insight" \
  --collection engram_reflection
```

### search — `POST /memories/search`
```bash
engram-cli search "what does Frank prefer" --limit 5
engram-cli search "ui prefs" --types insight --tags prefs --collection engram_user --pretty
```

### get — `GET /memories/{id}`
```bash
engram-cli get 3f9a1c2e-...
```

### update — `PATCH /memories/{id}` (or `PUT` with `--content`)
```bash
# partial update (PATCH): change metadata-style fields
engram-cli update 3f9a1c2e-... --importance 9 --tags prefs ui archived
engram-cli update 3f9a1c2e-... --lifecycle-status deprecated
# replace content (PUT — re-embeds):
engram-cli update 3f9a1c2e-... --content "Frank now prefers light mode" --type insight
```

### delete — `DELETE /memories/{id}` (soft delete → archived)
```bash
engram-cli delete 3f9a1c2e-...
```

### reset — `POST /memories/{id}/reset` (archived/deprecated → active)
```bash
engram-cli reset 3f9a1c2e-...
```

### list-collections — `GET /collections`
```bash
engram-cli list-collections --pretty
```

### create-collection — `POST /collections`
```bash
engram-cli create-collection engram_project --ttl 30d
```

### cross-search — `POST /memories/cross-search`
```bash
engram-cli cross-search "project status" \
  --collections engram_user engram_reflection --limit 10
```

### reflect — `POST /reflect`
```bash
engram-cli reflect            # run a cycle
engram-cli reflect --dry-run  # simulate, no writes
```

### reflect-check — `GET /reflect/check`
```bash
engram-cli reflect-check --pretty
```

### health — `GET /health` (no auth required)
```bash
engram-cli health
```

## Tests

No network / live Qdrant needed — an in-process `http.server` stub is used:

```bash
python3 scripts/test_engram_cli.py
```
