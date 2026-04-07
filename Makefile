.PHONY: build test lint clean

BINARY=engram
VERSION=0.1.0

build:
	go build -ldflags "-X main.version=$(VERSION)" -o $(BINARY) ./cmd/engram/

test:
	go test -v -race ./...

lint:
	golangci-lint run ./...

clean:
	rm -f $(BINARY)

tidy:
	go mod tidy
