# === Build stage ===
FROM golang:1.25-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w -X main.version=0.1.0" \
    -o /engram ./cmd/engram/

# === Runtime stage ===
FROM alpine:3.21

# Prefer alpine's musl-built py3-numpy / py3-requests apk packages over pip:
# musl wheels are the reliable path (many PyPI wheels are glibc-only).
RUN apk add --no-cache ca-certificates python3 py3-numpy py3-requests

COPY --from=builder /engram /usr/local/bin/engram
# Consolidation CLI source (Stage 0 hook shells out to `python3 -m consolidation.main`).
COPY cmd/consolidation /opt/consolidation

# MCP stdio is the default transport
ENTRYPOINT ["engram"]
CMD ["serve"]
