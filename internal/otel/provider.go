package otel

import (
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

// NewTracerProvider creates and globally registers a TracerProvider.
// Returns nil when tracing is disabled (noop provider is still registered globally).
func NewTracerProvider(cfg Config) (*sdktrace.TracerProvider, error) {
	if !cfg.Enabled || cfg.Exporter == "none" {
		otel.SetTracerProvider(noop.NewTracerProvider())
		return nil, nil
	}

	var exporter sdktrace.SpanExporter
	var err error

	switch cfg.Exporter {
	case "file":
		exporter, err = NewFileExporter(cfg.FileDir)
	case "stdout":
		exporter, err = NewStdoutExporter()
	default:
		return nil, fmt.Errorf("unknown exporter: %s", cfg.Exporter)
	}
	if err != nil {
		return nil, fmt.Errorf("create exporter: %w", err)
	}

	res := resource.NewSchemaless(
		attribute.String("service.name", "engram"),
	)

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter,
			sdktrace.WithBatchTimeout(5*time.Second),
			sdktrace.WithMaxExportBatchSize(512),
		),
		sdktrace.WithSampler(sdktrace.TraceIDRatioBased(cfg.SampleRatio)),
		sdktrace.WithResource(res),
	)

	otel.SetTracerProvider(tp)
	return tp, nil
}
