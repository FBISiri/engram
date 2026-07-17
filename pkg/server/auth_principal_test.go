// auth_principal_test.go — per-principal API key auth (withAuth).
package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/FBISiri/engram/pkg/collection"
)

// serveAuthed runs a request through CallerTypeMiddleware + withAuth and
// reports the HTTP status and the caller type the handler observed.
func serveAuthed(t *testing.T, h *HTTPServer, bearer, headerCT string) (int, string) {
	t.Helper()
	var gotCT string
	handler := CallerTypeMiddleware(http.HandlerFunc(h.withAuth(func(_ http.ResponseWriter, r *http.Request) {
		gotCT = CallerTypeFromContext(r.Context())
	})))
	req := httptest.NewRequest("POST", "/memories", nil)
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	if headerCT != "" {
		req.Header.Set("X-Caller-Type", headerCT)
	}
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec.Code, gotCT
}

func TestWithAuth_PrincipalKeys(t *testing.T) {
	collection.DefaultRegistry.Init()
	h := &HTTPServer{
		apiKey:        "legacy-key",
		principalKeys: map[string]string{"pigo": "pigo-key", "reflection": "refl-key"},
	}

	// Principal key wins: caller type derived from key, header ignored.
	if code, ct := serveAuthed(t, h, "pigo-key", "user"); code != http.StatusOK || ct != "pigo" {
		t.Errorf("pigo key: got code=%d ct=%q, want 200/pigo", code, ct)
	}
	if code, ct := serveAuthed(t, h, "refl-key", "pigo"); code != http.StatusOK || ct != "reflection" {
		t.Errorf("reflection key: got code=%d ct=%q, want 200/reflection", code, ct)
	}

	// Legacy key keeps header-based self-declaration.
	if code, ct := serveAuthed(t, h, "legacy-key", "pigo"); code != http.StatusOK || ct != "pigo" {
		t.Errorf("legacy key + pigo header: got code=%d ct=%q, want 200/pigo", code, ct)
	}
	if code, ct := serveAuthed(t, h, "legacy-key", ""); code != http.StatusOK || ct != "user" {
		t.Errorf("legacy key no header: got code=%d ct=%q, want 200/user", code, ct)
	}

	// Bad / missing credentials → 401.
	if code, _ := serveAuthed(t, h, "wrong-key", ""); code != http.StatusUnauthorized {
		t.Errorf("wrong key: got %d, want 401", code)
	}
	if code, _ := serveAuthed(t, h, "", ""); code != http.StatusUnauthorized {
		t.Errorf("no auth: got %d, want 401", code)
	}
}

func TestWithAuth_NoKeysConfigured_Open(t *testing.T) {
	h := &HTTPServer{}
	if code, _ := serveAuthed(t, h, "", ""); code != http.StatusOK {
		t.Errorf("open server: got %d, want 200", code)
	}
}

func TestWithAuth_PrincipalOnly_NoLegacy(t *testing.T) {
	h := &HTTPServer{principalKeys: map[string]string{"pigo": "pigo-key"}}
	if code, ct := serveAuthed(t, h, "pigo-key", ""); code != http.StatusOK || ct != "pigo" {
		t.Errorf("pigo key: got code=%d ct=%q, want 200/pigo", code, ct)
	}
	if code, _ := serveAuthed(t, h, "anything", ""); code != http.StatusUnauthorized {
		t.Errorf("unknown key: got %d, want 401", code)
	}
}
