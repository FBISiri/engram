package server

// C1 provenance audit-trail tests (P2, EU AI Act deadline 2026-08-02).
//
// source_type is soft-required: omitting it must still succeed, but the server
// must emit a WARN-level audit log whenever a memory is stored without
// provenance. These tests capture stdlib log output via log.SetOutput and
// assert the presence/absence of the "[WARN]" audit line. They reuse the same
// test infra as source_type_compat_test.go (newTestServer / callTool /
// injectMemory / extractText).

import (
	"bytes"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// captureLog redirects the stdlib logger to a buffer for the duration of fn and
// returns everything written to it.
func captureLog(fn func()) string {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	fn()
	return buf.String()
}

// memory_add WITHOUT source_type must succeed AND emit a WARN audit line.
func TestWarn_AddWithoutSourceTypeLogsWarn(t *testing.T) {
	srv, _ := newTestServer()

	out := captureLog(func() {
		result, err := callTool(srv, "memory_add", map[string]any{
			"content": "The deploy pipeline runs nightly",
		})
		if err != nil {
			t.Fatalf("add failed: %v", err)
		}
		if result.IsError {
			t.Fatalf("add returned error: %s", extractText(result))
		}
	})

	if !strings.Contains(out, "[WARN]") {
		t.Fatalf("expected a [WARN] audit log for add without source_type, got: %q", out)
	}
	if !strings.Contains(out, "defaulting to 'unknown'") {
		t.Errorf("expected defaulting message in warn log, got: %q", out)
	}
}

// memory_update WITHOUT source_type must succeed AND emit a WARN audit line.
func TestWarn_UpdateWithoutSourceTypeLogsWarn(t *testing.T) {
	srv, st := newTestServer()

	injectMemory(t, srv, st, "The staging host is host-alpha", memory.TypeEvent, nil, 0)

	out := captureLog(func() {
		result, err := callTool(srv, "memory_update", map[string]any{
			"old_content":          "The staging host is host-alpha",
			"new_content":          "The staging host is host-bravo",
			"similarity_threshold": float64(0.92),
		})
		if err != nil {
			t.Fatalf("update failed: %v", err)
		}
		if result.IsError {
			t.Fatalf("update returned error: %s", extractText(result))
		}
	})

	if !strings.Contains(out, "[WARN]") {
		t.Fatalf("expected a [WARN] audit log for update without source_type, got: %q", out)
	}
	if !strings.Contains(out, "memory_update") {
		t.Errorf("expected memory_update in warn log, got: %q", out)
	}
}

// memory_add WITHOUT source_type must default metadata.source_type to reflection.
func TestDefault_AddWithoutSourceTypeDefaultsReflection(t *testing.T) {
	srv, st := newTestServer()

	result, err := callTool(srv, "memory_add", map[string]any{
		"content": "The nightly job compacts the WAL",
	})
	if err != nil {
		t.Fatalf("add failed: %v", err)
	}
	if result.IsError {
		t.Fatalf("add returned error: %s", extractText(result))
	}

	mems := st.all()
	if len(mems) != 1 {
		t.Fatalf("expected 1 stored memory, got %d", len(mems))
	}
	if got, _ := mems[0].Metadata["source_type"].(string); got != string(memory.DefaultSourceType) {
		t.Errorf("expected defaulted source_type=%q, got %v", memory.DefaultSourceType, mems[0].Metadata["source_type"])
	}
}

// memory_add WITH a valid source_type must NOT emit a WARN audit line.
func TestWarn_AddWithSourceTypeNoWarn(t *testing.T) {
	srv, _ := newTestServer()

	out := captureLog(func() {
		result, err := callTool(srv, "memory_add", map[string]any{
			"content":     "Berlin is the capital of Germany",
			"source_type": "web_search",
		})
		if err != nil {
			t.Fatalf("add failed: %v", err)
		}
		if result.IsError {
			t.Fatalf("add returned error: %s", extractText(result))
		}
	})

	if strings.Contains(out, "[WARN]") {
		t.Errorf("expected no [WARN] audit log when source_type is provided, got: %q", out)
	}
}
