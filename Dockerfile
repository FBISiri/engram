# === Build stage ===
FROM golang:1.24-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w -X main.version=0.1.0" \
    -o /engram ./cmd/engram/

# === Runtime stage ===
FROM alpine:3.21

RUN apk add --no-cache ca-certificates

COPY --from=builder /engram /usr/local/bin/engram

# MCP stdio is the default transport
ENTRYPOINT ["engram"]
CMD ["serve"]
