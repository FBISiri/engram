package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
)

// buildHTTPTestServer creates an HTTPServer backed by the same mock
// store/embedder used in server_test.go (same package, so helpers accessible).
func buildHTTPTestServer(t *testing.T, apiKey string) *httptest.Server {
	t.Helper()
	srv, _ := newTestServer()
	h := NewHTTPServer(srv, 0, apiKey)
	ts := httptest.NewServer(h.Handler())
	t.Cleanup(ts.Close)
	return ts
}

// failingStore wraps mockStore but makes Stats() return an error.
type failingStore struct {
	mockStore
}

func (f *failingStore) Stats(_ context.Context) (*memory.CollectionStats, error) {
	return nil, errors.New("connection refused")
}

// ─────────────────────────────────────────────────────────────
// /health
// ─────────────────────────────────────────────────────────────

func TestHTTPHealth(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var body healthResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.Status != "ok" {
		t.Fatalf("want status=ok, got %q", body.Status)
	}
	if body.Qdrant != "green" {
		t.Fatalf("want qdrant=green, got %q", body.Qdrant)
	}
}

func TestHTTPHealth_NoAuthRequired(t *testing.T) {
	// /health must be accessible without auth even when apiKey is set.
	ts := buildHTTPTestServer(t, "secret-key")
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200 (no auth needed), got %d", resp.StatusCode)
	}
}

func TestHTTPHealth_Degraded(t *testing.T) {
	// When Qdrant is unreachable, /health should return 503.
	store := &failingStore{}
	embedder := newMockEmbedder()
	cfg := &config.Config{
		Weights:        memory.DefaultScoringWeights(),
		Decay:          memory.DefaultDecayConfig(),
		MMRLambda:      0.5,
		DedupThreshold: 0.92,
	}
	srv := NewServer(store, embedder, cfg)
	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("want 503, got %d", resp.StatusCode)
	}
	var body healthResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.Status != "degraded" {
		t.Fatalf("want status=degraded, got %q", body.Status)
	}
	if body.Error == "" {
		t.Fatal("want non-empty error field when degraded")
	}
}

// ─────────────────────────────────────────────────────────────
// GET /reflect/check
// ─────────────────────────────────────────────────────────────

func TestHTTPReflectCheck_ReturnsValidJSON(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	resp, err := http.Get(ts.URL + "/reflect/check")
	if err != nil {
		t.Fatalf("GET /reflect/check: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := body["should_trigger"]; !ok {
		t.Fatalf("response missing 'should_trigger' field: %v", body)
	}
}

func TestHTTPReflectCheck_WrongMethod(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	resp, err := http.Post(ts.URL+"/reflect/check", "application/json", nil)
	if err != nil {
		t.Fatalf("POST /reflect/check: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", resp.StatusCode)
	}
}

// ─────────────────────────────────────────────────────────────
// POST /reflect
// ─────────────────────────────────────────────────────────────

func TestHTTPReflect_DryRun(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := bytes.NewBufferString(`{"dry_run":true}`)
	resp, err := http.Post(ts.URL+"/reflect", "application/json", body)
	if err != nil {
		t.Fatalf("POST /reflect: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := result["triggered"]; !ok {
		t.Fatalf("response missing 'triggered' field: %v", result)
	}
}

func TestHTTPReflect_WrongMethod(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	resp, err := http.Get(ts.URL + "/reflect")
	if err != nil {
		t.Fatalf("GET /reflect: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", resp.StatusCode)
	}
}

// ─────────────────────────────────────────────────────────────
// Auth middleware
// ─────────────────────────────────────────────────────────────

func TestHTTPAuth_MissingToken(t *testing.T) {
	ts := buildHTTPTestServer(t, "secret-key")
	// /health is exempt from auth, so test with /reflect/check instead.
	resp, err := http.Get(ts.URL + "/reflect/check")
	if err != nil {
		t.Fatalf("GET /reflect/check: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", resp.StatusCode)
	}
}

func TestHTTPAuth_ValidToken(t *testing.T) {
	ts := buildHTTPTestServer(t, "secret-key")
	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/reflect/check", nil)
	req.Header.Set("Authorization", "Bearer secret-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
}

// ─────────────────────────────────────────────────────────────
// ListenAndServe / context cancellation
// ─────────────────────────────────────────────────────────────

func TestHTTPListenAndServe_GracefulShutdown(t *testing.T) {
	srv, _ := newTestServer()
	h := NewHTTPServer(srv, 0, "") // port=0 → OS picks a free port

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)
	go func() {
		errCh <- h.ListenAndServe(ctx)
	}()

	time.Sleep(50 * time.Millisecond)
	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("ListenAndServe returned error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("ListenAndServe did not shut down within 3s")
	}
}
