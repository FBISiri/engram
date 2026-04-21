package reflection

import (
	"context"
	"testing"
	"time"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// TestValidUntilFields_Positive — all stored insights have non-zero TTL.
func TestValidUntilFields_Positive(t *testing.T) {
	ttl := float64(time.Now().Add(30 * 24 * time.Hour).Unix())
	r := &RunResult{}
	setValidUntilFields(r, []float64{ttl})

	if !r.ValidUntilSet {
		t.Fatal("expected ValidUntilSet=true")
	}
	if r.ValidUntil == nil {
		t.Fatal("expected ValidUntil non-nil")
	}
	parsed, err := time.Parse(time.RFC3339, *r.ValidUntil)
	if err != nil {
		t.Fatalf("ValidUntil not RFC3339: %v", err)
	}
	diff := parsed.Sub(time.Now())
	if diff < 29*24*time.Hour || diff > 31*24*time.Hour {
		t.Errorf("ValidUntil should be ~30d from now, got %v", diff)
	}
}

// TestValidUntilFields_Negative — a zero TTL means the option was not wired.
func TestValidUntilFields_Negative(t *testing.T) {
	r := &RunResult{}
	setValidUntilFields(r, []float64{0})

	if r.ValidUntilSet {
		t.Error("expected ValidUntilSet=false when TTL is zero")
	}
	if r.ValidUntil != nil {
		t.Errorf("expected ValidUntil=nil, got %q", *r.ValidUntil)
	}
}

// TestValidUntilFields_Empty — no insights produced → set=false, null.
func TestValidUntilFields_Empty(t *testing.T) {
	r := &RunResult{}
	setValidUntilFields(r, nil)

	if r.ValidUntilSet {
		t.Error("expected ValidUntilSet=false when no insights")
	}
	if r.ValidUntil != nil {
		t.Errorf("expected ValidUntil=nil, got %q", *r.ValidUntil)
	}
}

// TestValidUntilFields_PartialMissing — one good TTL + one zero → false.
func TestValidUntilFields_PartialMissing(t *testing.T) {
	good := float64(time.Now().Add(30 * 24 * time.Hour).Unix())
	r := &RunResult{}
	setValidUntilFields(r, []float64{good, 0})

	if r.ValidUntilSet {
		t.Error("expected ValidUntilSet=false when any TTL is zero")
	}
	if r.ValidUntil != nil {
		t.Errorf("expected ValidUntil=nil, got %q", *r.ValidUntil)
	}
}

// TestValidUntilFields_MinAggregation — with multiple TTLs, picks the earliest.
func TestValidUntilFields_MinAggregation(t *testing.T) {
	now := time.Now()
	early := float64(now.Add(10 * 24 * time.Hour).Unix())
	late := float64(now.Add(30 * 24 * time.Hour).Unix())

	r := &RunResult{}
	setValidUntilFields(r, []float64{late, early, late})

	if !r.ValidUntilSet {
		t.Fatal("expected ValidUntilSet=true")
	}
	if r.ValidUntil == nil {
		t.Fatal("expected ValidUntil non-nil")
	}

	parsed, err := time.Parse(time.RFC3339, *r.ValidUntil)
	if err != nil {
		t.Fatalf("ValidUntil not RFC3339: %v", err)
	}
	expected := time.Unix(int64(early), 0).UTC()
	if !parsed.Equal(expected) {
		t.Errorf("expected min TTL %v, got %v", expected, parsed)
	}
}

// TestRunSpanAttributes_OTel — verifies span attributes are set correctly.
func TestRunSpanAttributes_OTel(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	defer tp.Shutdown(context.Background())

	tr := tp.Tracer("test")

	t.Run("set=true", func(t *testing.T) {
		exporter.Reset()
		_, span := tr.Start(context.Background(), "test.run")
		ttl := float64(time.Now().Add(30 * 24 * time.Hour).Unix())
		r := &RunResult{}
		setValidUntilFields(r, []float64{ttl})
		setRunSpanAttributes(span, r)
		span.End()

		spans := exporter.GetSpans()
		if len(spans) == 0 {
			t.Fatal("no spans recorded")
		}
		attrs := spans[0].Attributes
		assertAttr(t, attrs, "engram.memory.valid_until_set", true)
		assertAttrNonEmpty(t, attrs, "engram.memory.valid_until")
	})

	t.Run("set=false", func(t *testing.T) {
		exporter.Reset()
		_, span := tr.Start(context.Background(), "test.run")
		r := &RunResult{}
		setValidUntilFields(r, nil)
		setRunSpanAttributes(span, r)
		span.End()

		spans := exporter.GetSpans()
		if len(spans) == 0 {
			t.Fatal("no spans recorded")
		}
		attrs := spans[0].Attributes
		assertAttr(t, attrs, "engram.memory.valid_until_set", false)
		assertAttrEmpty(t, attrs, "engram.memory.valid_until")
	})
}

func assertAttr(t *testing.T, attrs []attribute.KeyValue, key string, want bool) {
	t.Helper()
	for _, a := range attrs {
		if string(a.Key) == key {
			if a.Value.AsBool() != want {
				t.Errorf("attribute %s: got %v, want %v", key, a.Value.AsBool(), want)
			}
			return
		}
	}
	t.Errorf("attribute %s not found", key)
}

func assertAttrNonEmpty(t *testing.T, attrs []attribute.KeyValue, key string) {
	t.Helper()
	for _, a := range attrs {
		if string(a.Key) == key {
			if a.Value.AsString() == "" {
				t.Errorf("attribute %s is empty, expected non-empty", key)
			}
			return
		}
	}
	t.Errorf("attribute %s not found", key)
}

func assertAttrEmpty(t *testing.T, attrs []attribute.KeyValue, key string) {
	t.Helper()
	for _, a := range attrs {
		if string(a.Key) == key {
			if a.Value.AsString() != "" {
				t.Errorf("attribute %s should be empty for null, got %q", key, a.Value.AsString())
			}
			return
		}
	}
	t.Errorf("attribute %s not found", key)
}
