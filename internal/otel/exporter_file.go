package otel

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// FileExporter writes spans as JSONL to daily-rotated files using the SDK's
// JSON encoder (stdouttrace) with a rotating file writer underneath.
type FileExporter struct {
	inner  sdktrace.SpanExporter
	writer *rotatingWriter
}

type rotatingWriter struct {
	mu      sync.Mutex
	dir     string
	curDate string
	file    *os.File
}

func NewFileExporter(dir string) (*FileExporter, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create trace dir: %w", err)
	}

	w := &rotatingWriter{dir: dir}
	exp, err := stdouttrace.New(stdouttrace.WithWriter(w))
	if err != nil {
		return nil, fmt.Errorf("create stdout exporter for file: %w", err)
	}

	return &FileExporter{inner: exp, writer: w}, nil
}

func (e *FileExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	return e.inner.ExportSpans(ctx, spans)
}

func (e *FileExporter) Shutdown(ctx context.Context) error {
	err := e.inner.Shutdown(ctx)
	closeErr := e.writer.close()
	if err != nil {
		return err
	}
	return closeErr
}

func (w *rotatingWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	today := time.Now().Format("2006-01-02")
	if today != w.curDate || w.file == nil {
		if err := w.rotate(today); err != nil {
			return 0, err
		}
	}
	return w.file.Write(p)
}

func (w *rotatingWriter) rotate(date string) error {
	if w.file != nil {
		w.file.Close()
	}
	path := filepath.Join(w.dir, fmt.Sprintf("engram-traces-%s.jsonl", date))
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("open trace file: %w", err)
	}
	w.file = f
	w.curDate = date
	return nil
}

func (w *rotatingWriter) close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.file != nil {
		err := w.file.Close()
		w.file = nil
		return err
	}
	return nil
}
