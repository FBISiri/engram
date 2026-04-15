package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
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
	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body["status"] != "ok" {
		t.Fatalf("want status=ok, got %q", body["status"])
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
	resp, err := http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("want 401, got %d", resp.StatusCode)
	}
}

func TestHTTPAuth_ValidToken(t *testing.T) {
	ts := buildHTTPTestServer(t, "secret-key")
	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/health", nil)
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
