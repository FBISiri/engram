package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestHTTPCreate_DedupHit verifies that REST POST /memories runs the same
// server-side 0.92 dedup as the MCP add path (C1). A near-identical second
// write must be rejected with 409 Conflict + a "duplicate" body.
func TestHTTPCreate_DedupHit(t *testing.T) {
	ts := buildHTTPTestServer(t, "")

	content := `The Consolidation Agent writes memories via REST POST /memories.`
	first := bytes.NewBufferString(`{"type":"event","content":"` + content + `"}`)
	resp, err := http.Post(ts.URL+"/memories", "application/json", first)
	if err != nil {
		t.Fatalf("first POST: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("first POST: want 201, got %d", resp.StatusCode)
	}

	// Same content → cosine 1.0 > 0.92 threshold → duplicate.
	second := bytes.NewBufferString(`{"type":"event","content":"` + content + `"}`)
	resp2, err := http.Post(ts.URL+"/memories", "application/json", second)
	if err != nil {
		t.Fatalf("second POST: %v", err)
	}
	defer func() { _ = resp2.Body.Close() }()
	if resp2.StatusCode != http.StatusConflict {
		t.Fatalf("second POST: want 409, got %d", resp2.StatusCode)
	}
	var body map[string]any
	if err := json.NewDecoder(resp2.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body["status"] != "duplicate" {
		t.Fatalf("want status=duplicate, got %v", body["status"])
	}
	if id, ok := body["existing_id"].(string); !ok || id == "" {
		t.Fatalf("want non-empty existing_id, got %v", body["existing_id"])
	}
	if sim, ok := body["similarity"].(float64); !ok || sim < 0.92 {
		t.Fatalf("want similarity >= 0.92, got %v", body["similarity"])
	}
}

// TestHTTPCreate_DedupHitEmitsThresholdLabel drives the REST POST /memories
// dedup-hit path (crud.go) through an A-MAC server whose directive dedup
// threshold is 0.90 (!= the legacy 0.92), then enumerates the emitted DedupHits
// series. The dedup_type label MUST be "server_side_090" and NOT the old
// hardcoded "server_side_092" — this assertion fails if crud.go were
// re-hardcoded to "server_side_092".
func TestHTTPCreate_DedupHitEmitsThresholdLabel(t *testing.T) {
	srv, _ := newAMACServer()
	h := NewHTTPServer(srv, 0, "")
	// NewHTTPServer installs its own metrics via SetMetrics, so read them back
	// from the server rather than injecting our own (which would be clobbered).
	m := srv.Metrics()
	ts := httptest.NewServer(h.Handler())
	t.Cleanup(ts.Close)

	content := `Never overwrite the production database without an explicit backup.`
	body := `{"type":"directive","content":"` + content + `"}`

	// First POST: created. Second identical POST: dedup_rejected -> emits metric.
	first, err := http.Post(ts.URL+"/memories", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatalf("first POST: %v", err)
	}
	_ = first.Body.Close()
	if first.StatusCode != http.StatusCreated {
		t.Fatalf("first POST: want 201, got %d", first.StatusCode)
	}
	second, err := http.Post(ts.URL+"/memories", "application/json", bytes.NewBufferString(body))
	if err != nil {
		t.Fatalf("second POST: %v", err)
	}
	_ = second.Body.Close()
	if second.StatusCode != http.StatusConflict {
		t.Fatalf("second POST: want 409, got %d", second.StatusCode)
	}

	labels := collectDedupTypeLabels(t, m.DedupHits)
	if len(labels) != 1 {
		t.Fatalf("expected exactly one emitted dedup_type series, got %v", labels)
	}
	if got := labels["server_side_090"]; got != 1 {
		t.Errorf("dedup_hits{dedup_type=server_side_090} = %v, want 1", got)
	}
	if _, ok := labels["server_side_092"]; ok {
		t.Errorf("old hardcoded label server_side_092 must NOT be emitted for a 0.90 threshold; got series %v", labels)
	}
}

// TestHTTPCreate_NovelContent verifies novel content still returns 201.
func TestHTTPCreate_NovelContent(t *testing.T) {
	ts := buildHTTPTestServer(t, "")

	first := bytes.NewBufferString(`{"type":"event","content":"Hi."}`)
	resp, err := http.Post(ts.URL+"/memories", "application/json", first)
	if err != nil {
		t.Fatalf("first POST: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("first POST: want 201, got %d", resp.StatusCode)
	}

	second := bytes.NewBufferString(`{"type":"event","content":"The quarterly financial report shows growth."}`)
	resp2, err := http.Post(ts.URL+"/memories", "application/json", second)
	if err != nil {
		t.Fatalf("second POST: %v", err)
	}
	defer func() { _ = resp2.Body.Close() }()
	if resp2.StatusCode != http.StatusCreated {
		t.Fatalf("novel POST: want 201, got %d", resp2.StatusCode)
	}
}
