package otel

import (
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// NewStdoutExporter creates an exporter that writes structured JSON to stdout.
// WithPrettyPrint is off by default (structured output for post-processing).
func NewStdoutExporter() (sdktrace.SpanExporter, error) {
	return stdouttrace.New()
}
