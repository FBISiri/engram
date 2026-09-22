package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestStatusError_ErrorPrefix(t *testing.T) {
	e := &StatusError{StatusCode: 402, RetryAfter: "3", Body: "region blocked"}
	got := e.Error()
	if !strings.HasPrefix(got, "llm returned status 402: ") {
		t.Errorf("prefix broken: %q", got)
	}
	if !strings.Contains(got, "region blocked") {
		t.Errorf("body excerpt missing: %q", got)
	}
}

func TestIsTransientStatus(t *testing.T) {
	cases := map[int]bool{429: true, 500: true, 503: true, 599: true, 400: false, 401: false, 200: false, 404: false}
	for code, want := range cases {
		if got := IsTransientStatus(code); got != want {
			t.Errorf("IsTransientStatus(%d) = %v, want %v", code, got, want)
		}
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 9, 22, 12, 0, 0, 0, time.UTC)

	if d, ok := ParseRetryAfter("2", now); !ok || d != 2*time.Second {
		t.Errorf("delta-seconds: got %v %v", d, ok)
	}
	if _, ok := ParseRetryAfter("", now); ok {
		t.Error("empty must be (0,false)")
	}
	if _, ok := ParseRetryAfter("-5", now); ok {
		t.Error("negative must be (0,false)")
	}
	if _, ok := ParseRetryAfter("garbage", now); ok {
		t.Error("unparseable must be (0,false)")
	}
	// HTTP-date form 5s in the future.
	future := now.Add(5 * time.Second).UTC().Format(http.TimeFormat)
	if d, ok := ParseRetryAfter(future, now); !ok || d != 5*time.Second {
		t.Errorf("http-date: got %v %v", d, ok)
	}
	// HTTP-date in the past clamps to 0.
	past := now.Add(-time.Hour).UTC().Format(http.TimeFormat)
	if d, ok := ParseRetryAfter(past, now); !ok || d != 0 {
		t.Errorf("http-date past: got %v %v", d, ok)
	}
}

// Anthropic non-200 returns *StatusError with the Retry-After header populated.
func TestCallWithBudget_Anthropic_Non200ReturnsStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Retry-After", "7")
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"error":"slow down"}`))
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")
	t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "")

	_, _, err := CallWithBudget(context.Background(), "hi", 1500)
	var se *StatusError
	if !errors.As(err, &se) {
		t.Fatalf("expected *StatusError, got %T: %v", err, err)
	}
	if se.StatusCode != 429 || se.RetryAfter != "7" {
		t.Errorf("StatusError = %+v, want 429 + Retry-After 7", se)
	}
	if !strings.Contains(err.Error(), "llm returned status 429") {
		t.Errorf("prefix broken: %q", err.Error())
	}
}

// OpenRouter non-200 returns *StatusError (aligned to the colon prefix) with
// Retry-After populated.
func TestCallWithBudget_OpenRouter_Non200ReturnsStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Retry-After", "4")
		w.WriteHeader(503)
		_, _ = w.Write([]byte(`upstream unavailable`))
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_PROVIDER", "")
	t.Setenv("ENGRAM_LLM_API_KEY", "k")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ENGRAM_LLM_MODEL", "")

	_, _, err := CallWithBudget(context.Background(), "hi", 1500)
	var se *StatusError
	if !errors.As(err, &se) {
		t.Fatalf("expected *StatusError, got %T: %v", err, err)
	}
	if se.StatusCode != 503 || se.RetryAfter != "4" {
		t.Errorf("StatusError = %+v, want 503 + Retry-After 4", se)
	}
	if !strings.Contains(err.Error(), "llm returned status 503") {
		t.Errorf("prefix broken: %q", err.Error())
	}
}
