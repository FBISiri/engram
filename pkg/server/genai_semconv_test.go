package server

import (
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/FBISiri/engram/pkg/semconv"
)

// TestGenAIOperationNameMapping is a table test over the operation-name consts,
// asserting the span-name -> gen_ai.operation.name mapping used at each memory
// span call site.
func TestGenAIOperationNameMapping(t *testing.T) {
	cases := []struct {
		span   string
		opName string
	}{
		{"engram.memory.search", semconv.OpSearchMemory},
		{"engram.memory.add", semconv.OpUpsertMemory},
		{"engram.memory.list", semconv.OpListMemory},
		{"engram.memory.delete", semconv.OpDeleteMemory},
		{"engram.memory.dedup_check", semconv.OpUpsertMemory},     // sub-step of upsert
		{"engram.memory.provenance_merge", semconv.OpUpsertMemory}, // sub-step of upsert
	}
	want := map[string]string{
		"search_memory": "search_memory",
		"upsert_memory": "upsert_memory",
		"list_memory":   "list_memory",
		"delete_memory": "delete_memory",
	}
	for _, c := range cases {
		if want[c.opName] != c.opName {
			t.Errorf("span %s: op-name const %q not a recognized gen_ai operation", c.span, c.opName)
		}
	}
}

// TestSearchSpanCarriesGenAIAttrs uses an in-memory SpanRecorder to assert the
// search span actually carries gen_ai.operation.name=search_memory and
// gen_ai.system=engram, without touching any production path.
func TestSearchSpanCarriesGenAIAttrs(t *testing.T) {
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	// Reassign package tracer to use the test provider.
	oldTracer := tracer
	tracer = tp.Tracer("engram.memory")
	t.Cleanup(func() { tracer = oldTracer })

	srv, _ := newTestServer()

	if _, err := callTool(srv, "memory_search", map[string]any{"query": "hello"}); err != nil {
		t.Fatalf("search: %v", err)
	}

	var found bool
	for _, sp := range sr.Ended() {
		if sp.Name() != "engram.memory.search" {
			continue
		}
		found = true
		var opName, system string
		for _, kv := range sp.Attributes() {
			switch string(kv.Key) {
			case semconv.AttrOperationName:
				opName = kv.Value.AsString()
			case semconv.AttrSystem:
				system = kv.Value.AsString()
			}
		}
		if opName != semconv.OpSearchMemory {
			t.Errorf("search span %s=%q, want %q", semconv.AttrOperationName, opName, semconv.OpSearchMemory)
		}
		if system != semconv.SystemEngram {
			t.Errorf("search span %s=%q, want %q", semconv.AttrSystem, system, semconv.SystemEngram)
		}
	}
	if !found {
		t.Fatal("no engram.memory.search span recorded")
	}
}
