.PHONY: build test lint clean docker integration-test

BINARY=engram
VERSION=0.1.0
GIT_COMMIT=$(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME=$(shell date -u '+%Y-%m-%dT%H:%M:%SZ')

build:
	go build -ldflags "-X main.version=$(VERSION) -X main.gitCommit=$(GIT_COMMIT) -X main.buildTime=$(BUILD_TIME)" -o $(BINARY) ./cmd/engram/

test:
	go test -v -race ./...

lint:
	golangci-lint run ./...

clean:
	rm -f $(BINARY)

tidy:
	go mod tidy

# Docker targets
docker:
	docker build -t engram:$(VERSION) .

docker-up:
	docker-compose up -d qdrant

docker-down:
	docker-compose down

# Integration test (requires ENGRAM_OPENAI_API_KEY and Qdrant running)
integration-test: build
	./integration_test.sh
