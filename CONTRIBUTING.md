# Contributing to Engram

Thank you for your interest in contributing to Engram! This document covers how to set up your development environment, code conventions, and the PR workflow.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Code Conventions](#code-conventions)
- [PR Workflow](#pr-workflow)
- [Reporting Issues](#reporting-issues)

---

## Development Setup

### Prerequisites

- **Go 1.24+** — [Install](https://go.dev/doc/install)
- **Docker** — for running Qdrant locally
- **golangci-lint** — for linting (`go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest`)
- An **OpenAI API key** (or Voyage AI key if using the Voyage embedder)

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/FBISiri/engram.git
cd engram
go mod tidy
```

### 2. Start Qdrant

Engram uses [Qdrant](https://qdrant.tech/) as its vector store. The easiest way to run it locally:

```bash
docker run -d --name engram-qdrant \
  --security-opt seccomp=unconfined \
  -p 6333:6333 -p 6334:6334 \
  -v engram_qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.9.7
```

Or via Docker Compose (also starts Engram itself):

```bash
docker-compose up -d qdrant
```

### 3. Configure Environment

Engram is configured entirely via environment variables. For local development, create a `.env` file (not committed) or export directly:

```bash
export ENGRAM_QDRANT_URL=localhost:6334
export ENGRAM_OPENAI_API_KEY=sk-...
# Optional overrides:
export ENGRAM_COLLECTION_NAME=engram_dev   # use a separate collection for dev
export ENGRAM_DEDUP_THRESHOLD=0.92
export ENGRAM_TRANSPORT=stdio
```

See `pkg/config/config.go` for the full list of environment variables and their defaults.

### 4. Build and Run

```bash
# Build binary
make build

# Run as MCP server (stdio transport)
./engram serve

# Run as HTTP server
ENGRAM_TRANSPORT=http ./engram serve
```

---

## Project Structure

```
engram/
├── cmd/engram/          CLI entry point (main.go)
├── pkg/
│   ├── config/          Configuration loading from env vars
│   ├── memory/          Core types, scoring (relevance×recency×importance), dedup, MMR
│   ├── embedding/       Embedder interface + OpenAI + Voyage AI implementations
│   ├── qdrant/          Qdrant vector store implementation
│   ├── server/          MCP server (stdio + HTTP transports)
│   ├── reflection/      Optional async reflection engine
│   ├── dream/           Dream Engine (async background tasks)
│   └── sync/            Memory sync utilities
├── Dockerfile           Multi-stage build
├── docker-compose.yml   Qdrant + Engram stack
├── Makefile             Build, test, lint targets
└── integration_test.sh  End-to-end MCP test (requires live Qdrant)
```

**Key design principle**: No LLM in the hot path. `memory_add` and `memory_search` are pure vector operations. Reflection and Dream Engine run async/optionally.

---

## Running Tests

### Unit Tests

```bash
make test
# or: go test -v -race ./...
```

Unit tests are located alongside source files (`*_test.go`). They do **not** require a running Qdrant instance.

### Integration Tests

The integration test suite runs all 4 MCP tools (search, add, update, delete) end-to-end against a live Qdrant + OpenAI setup:

```bash
# Ensure Qdrant is running on localhost:6333/6334
ENGRAM_OPENAI_API_KEY=sk-... make integration-test
# or: ENGRAM_OPENAI_API_KEY=sk-... ./integration_test.sh
```

**Note**: Integration tests write real data to Qdrant. Use a dedicated collection (`ENGRAM_COLLECTION_NAME=engram_test`) to avoid polluting production data.

### Linting

```bash
make lint
```

All lint errors must pass before a PR can be merged. The project uses `golangci-lint` with default settings.

---

## Code Conventions

### Go Style

- Follow standard Go conventions — `gofmt`, `go vet`, `golangci-lint` must all pass.
- **Exported identifiers** must have GoDoc comments.
- **Error handling**: always check errors explicitly; no `_` discards for errors that could affect correctness.
- **No panics** in library code (`pkg/`). Panics are acceptable only in `main()` for unrecoverable startup failures.

### Package Boundaries

| Package | Allowed to import | Must NOT import |
|---------|------------------|-----------------|
| `memory` | stdlib only | `embedding`, `qdrant`, `server` |
| `embedding` | `memory`, stdlib | `qdrant`, `server` |
| `qdrant` | `memory`, `embedding`, stdlib | `server` |
| `server` | anything | — |
| `config` | `memory`, stdlib | `server`, `qdrant` |

This layering keeps `memory` as the pure domain model with no external dependencies.

### Naming Conventions

- Types: `PascalCase`
- Functions/methods: `PascalCase` (exported), `camelCase` (unexported)
- Test files: `*_test.go` in same package (prefer white-box tests)
- Integration tests: suffix `_integration_test.go` or shell scripts

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]
[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`

Examples:
```
feat(embedding): add Voyage AI embedder
fix(dedup): fix off-by-one in cosine similarity threshold
docs(readme): update docker-compose quickstart
test(memory): add scoring edge cases for zero importance
```

---

## PR Workflow

### Before Opening a PR

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Write tests** for any new logic. PRs that lower test coverage without justification will be asked to add tests.

3. **Run the full test suite** locally:
   ```bash
   make test && make lint
   ```

4. **Update documentation** if your change affects:
   - Public API (MCP tools or REST endpoints)
   - Environment variables / configuration
   - Architecture or data flow

### Opening the PR

- **Title**: follow the commit message convention (`feat(scope): description`)
- **Description**: fill in what changed, why, and any testing notes
- **Link issues**: reference any related GitHub issues with `Closes #N` or `Related to #N`
- **Draft PRs** are welcome for early feedback — prefix title with `[WIP]` or mark as Draft

### Review Process

- At least **1 approving review** is required to merge
- CI must pass (tests + lint)
- Squash merge is preferred for feature branches to keep `main` history clean
- Force-pushing to `main` is not allowed

### What Makes a Good PR

- **Small and focused** — one logical change per PR is easier to review
- **Tests included** — especially for bug fixes (add a test that would have caught the bug)
- **No unrelated changes** — avoid formatting changes or refactors mixed with feature work
- **Clear description** — explain the "why", not just the "what"

---

## Reporting Issues

Use [GitHub Issues](https://github.com/FBISiri/engram/issues) to report bugs or request features.

For **bugs**, include:
- Engram version / commit hash
- Go version (`go version`)
- Qdrant version
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

For **feature requests**, describe the use case and why existing functionality doesn't cover it.

---

## Questions?

Open a GitHub Discussion or file an issue with the `question` label. 

We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Be kind.
