package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLoadConfig_Defaults(t *testing.T) {
	t.Setenv("ENGRAM_LLM_API_KEY", "test-key")
	t.Setenv("ENGRAM_LLM_BASE_URL", "")
	t.Setenv("ENGRAM_LLM_MODEL", "")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.BaseURL != defaultBaseURL {
		t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, defaultBaseURL)
	}
	if cfg.Model != defaultModel {
		t.Errorf("Model = %q, want %q", cfg.Model, defaultModel)
	}
	if cfg.APIKey != "test-key" {
		t.Errorf("APIKey = %q, want %q", cfg.APIKey, "test-key")
	}
}

func TestLoadConfig_Overrides(t *testing.T) {
	t.Setenv("ENGRAM_LLM_API_KEY", "k")
	t.Setenv("ENGRAM_LLM_BASE_URL", "https://example.test/v1/")
	t.Setenv("ENGRAM_LLM_MODEL", "openai/gpt-4o")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.BaseURL != "https://example.test/v1" { // trailing slash trimmed
		t.Errorf("BaseURL = %q", cfg.BaseURL)
	}
	if cfg.Model != "openai/gpt-4o" {
		t.Errorf("Model = %q", cfg.Model)
	}
}

func TestResolveMaxTokens(t *testing.T) {
	cases := []struct {
		name string
		set  bool
		val  string
		want int
	}{
		{"unset default", false, "", 1500},
		{"valid override", true, "800", 800},
		{"non-numeric", true, "abc", 1500},
		{"zero", true, "0", 1500},
		{"negative", true, "-5", 1500},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.set {
				t.Setenv("ENGRAM_LLM_MAX_TOKENS", tc.val)
			} else {
				t.Setenv("ENGRAM_LLM_MAX_TOKENS", "")
			}
			if got := resolveMaxTokens(); got != tc.want {
				t.Errorf("resolveMaxTokens() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestLoadConfig_KeyRequired(t *testing.T) {
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	if _, err := loadConfig(); err == nil {
		t.Fatal("expected error when ENGRAM_LLM_API_KEY unset")
	}
}

func TestCall_NoKeyReturnsError(t *testing.T) {
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	_, err := Call(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected error when key unset")
	}
	if !strings.Contains(err.Error(), "no LLM API key configured") {
		t.Errorf("error = %q, want it to mention missing key", err.Error())
	}
}

func TestResolveDialecticMaxTokens(t *testing.T) {
	cases := []struct {
		name string
		val  string
		want int
	}{
		{"unset default", "", 4000},
		{"valid override", "5000", 5000},
		{"non-numeric", "abc", 4000},
		{"zero", "0", 4000},
		{"negative", "-1", 4000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("ENGRAM_LLM_DIALECTIC_MAX_TOKENS", tc.val)
			if got := resolveDialecticMaxTokens(); got != tc.want {
				t.Errorf("resolveDialecticMaxTokens() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestResolveMaxTokensCeiling(t *testing.T) {
	cases := []struct {
		name string
		val  string
		want int
	}{
		{"unset default", "", 8000},
		{"valid override", "16000", 16000},
		{"non-numeric", "xyz", 8000},
		{"zero", "0", 8000},
		{"negative", "-9", 8000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("ENGRAM_LLM_MAX_TOKENS_CEILING", tc.val)
			if got := resolveMaxTokensCeiling(); got != tc.want {
				t.Errorf("resolveMaxTokensCeiling() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestCallWithBudget_SendsMaxTokensAndPopulatesMeta(t *testing.T) {
	var gotMaxTokens float64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotMaxTokens, _ = req["max_tokens"].(float64)
		_, _ = io.WriteString(w, `{
			"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
			"usage": {
				"prompt_tokens": 120,
				"completion_tokens": 4000,
				"total_tokens": 4120,
				"completion_tokens_details": {"reasoning_tokens": 3900}
			}
		}`)
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_API_KEY", "dummy")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ENGRAM_LLM_MODEL", "")

	content, meta, err := CallWithBudget(context.Background(), "hi", 4000)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content != "partial" {
		t.Errorf("content = %q", content)
	}
	if gotMaxTokens != 4000 {
		t.Errorf("request max_tokens = %v, want 4000", gotMaxTokens)
	}
	if meta.FinishReason != "length" {
		t.Errorf("FinishReason = %q", meta.FinishReason)
	}
	if meta.MaxTokens != 4000 {
		t.Errorf("MaxTokens = %d, want 4000", meta.MaxTokens)
	}
	if meta.PromptTokens != 120 {
		t.Errorf("PromptTokens = %d, want 120", meta.PromptTokens)
	}
	if meta.CompletionTokens != 4000 {
		t.Errorf("CompletionTokens = %d, want 4000", meta.CompletionTokens)
	}
	if meta.TotalTokens != 4120 {
		t.Errorf("TotalTokens = %d, want 4120", meta.TotalTokens)
	}
	if meta.ReasoningTokens != 3900 {
		t.Errorf("ReasoningTokens = %d, want 3900", meta.ReasoningTokens)
	}
}

func TestCallWithBudget_UsageAbsentSentinels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}`)
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_API_KEY", "dummy")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ENGRAM_LLM_MODEL", "")

	_, meta, err := CallWithBudget(context.Background(), "hi", 1500)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.PromptTokens != -1 || meta.CompletionTokens != -1 || meta.TotalTokens != -1 || meta.ReasoningTokens != -1 {
		t.Errorf("expected -1 sentinels when usage absent, got %+v", meta)
	}
	if meta.MaxTokens != 1500 {
		t.Errorf("MaxTokens = %d, want 1500", meta.MaxTokens)
	}
}
