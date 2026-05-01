package otel

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func TestFileExporter_WritesJSONL(t *testing.T) {
	dir := t.TempDir()
	exp, err := NewFileExporter(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer exp.Shutdown(context.Background())

	stubs := tracetest.SpanStubs{
		{Name: "span-1", SpanKind: trace.SpanKindInternal},
		{Name: "span-2", SpanKind: trace.SpanKindServer},
		{Name: "span-3", SpanKind: trace.SpanKindClient},
	}
	spans := stubs.Snapshots()

	if err := exp.ExportSpans(context.Background(), spans); err != nil {
		t.Fatalf("ExportSpans: %v", err)
	}

	today := time.Now().Format("2006-01-02")
	path := filepath.Join(dir, "engram-traces-"+today+".jsonl")

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open trace file: %v", err)
	}
	defer f.Close()

	var lineCount int
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal(line, &obj); err != nil {
			t.Errorf("line %d is not valid JSON: %v", lineCount+1, err)
		}
		lineCount++
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}

	if lineCount != 3 {
		t.Errorf("expected 3 JSONL lines, got %d", lineCount)
	}
}

func TestFileExporter_FileNaming(t *testing.T) {
	dir := t.TempDir()
	exp, err := NewFileExporter(dir)
	if err != nil {
		t.Fatal(err)
	}

	stubs := tracetest.SpanStubs{{Name: "test"}}
	if err := exp.ExportSpans(context.Background(), stubs.Snapshots()); err != nil {
		t.Fatal(err)
	}
	exp.Shutdown(context.Background())

	today := time.Now().Format("2006-01-02")
	expected := "engram-traces-" + today + ".jsonl"

	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("expected 1 file, got %d", len(entries))
	}
	if entries[0].Name() != expected {
		t.Errorf("expected file %s, got %s", expected, entries[0].Name())
	}
}

func TestFileExporter_EmptyBatch(t *testing.T) {
	dir := t.TempDir()
	exp, err := NewFileExporter(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer exp.Shutdown(context.Background())

	if err := exp.ExportSpans(context.Background(), nil); err != nil {
		t.Fatalf("ExportSpans with nil should not error: %v", err)
	}

	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Errorf("expected no files for empty export, got %d", len(entries))
	}
}
