package otel

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace/noop"
)

func TestNewTracerProvider_Disabled(t *testing.T) {
	cfg := Config{Enabled: false}
	tp, err := NewTracerProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if tp != nil {
		t.Error("expected nil TracerProvider when disabled")
		tp.Shutdown(context.Background())
	}

	_, isNoop := otel.GetTracerProvider().(noop.TracerProvider)
	if !isNoop {
		t.Error("expected global provider to be noop when disabled")
	}
}

func TestNewTracerProvider_ExporterNone(t *testing.T) {
	cfg := Config{Enabled: true, Exporter: "none"}
	tp, err := NewTracerProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if tp != nil {
		t.Error("expected nil TracerProvider for exporter=none")
		tp.Shutdown(context.Background())
	}

	_, isNoop := otel.GetTracerProvider().(noop.TracerProvider)
	if !isNoop {
		t.Error("expected global provider to be noop for exporter=none")
	}
}

func TestNewTracerProvider_File(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{
		Enabled:     true,
		Exporter:    "file",
		FileDir:     dir,
		SampleRatio: 1.0,
	}

	tp, err := NewTracerProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if tp == nil {
		t.Fatal("expected non-nil TracerProvider for file exporter")
	}
	defer tp.Shutdown(context.Background())

	_, isNoop := otel.GetTracerProvider().(noop.TracerProvider)
	if isNoop {
		t.Error("global provider should not be noop for file exporter")
	}
}
