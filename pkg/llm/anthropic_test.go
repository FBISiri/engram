package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- R1: no behaviour change when ENGRAM_LLM_PROVIDER is unset ---

func TestResolveProvider_DefaultsToOpenRouter(t *testing.T) {
	cases := []struct {
		name string
		val  string
		want string
	}{
		{"unset", "", providerOpenRouter},
		{"empty", "", providerOpenRouter},
		{"unknown", "gemini", providerOpenRouter},
		{"openrouter explicit", "openrouter", providerOpenRouter},
		{"anthropic", "anthropic", providerAnthropic},
		{"anthropic mixed case", "Anthropic", providerAnthropic},
		{"anthropic padded", "  anthropic  ", providerAnthropic},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("ENGRAM_LLM_PROVIDER", tc.val)
			if got := resolveProvider(); got != tc.want {
				t.Errorf("resolveProvider() = %q, want %q", got, tc.want)
			}
		})
	}
}

// R1: with the provider var unset, loadConfig must behave exactly as before —
// openrouter defaults and ENGRAM_LLM_API_KEY still required.
func TestLoadConfig_ProviderUnset_IsOpenRouterUnchanged(t *testing.T) {
	t.Setenv("ENGRAM_LLM_PROVIDER", "")
	t.Setenv("ENGRAM_LLM_BASE_URL", "")
	t.Setenv("ENGRAM_LLM_MODEL", "")

	t.Setenv("ENGRAM_LLM_API_KEY", "")
	if _, err := loadConfig(); err == nil {
		t.Fatal("expected error when key unset on openrouter path")
	}

	t.Setenv("ENGRAM_LLM_API_KEY", "k")
	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Provider != providerOpenRouter {
		t.Errorf("Provider = %q, want openrouter", cfg.Provider)
	}
	if cfg.BaseURL != defaultBaseURL || cfg.Model != defaultModel {
		t.Errorf("openrouter defaults changed: base=%q model=%q", cfg.BaseURL, cfg.Model)
	}
}

// --- R2: anthropic config resolution ---

func TestLoadConfig_Anthropic_DefaultsAndKeyNotRequired(t *testing.T) {
	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "") // must NOT be required
	t.Setenv("ENGRAM_LLM_BASE_URL", "")
	t.Setenv("ENGRAM_LLM_MODEL", "")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("anthropic path must not require ENGRAM_LLM_API_KEY: %v", err)
	}
	if cfg.Provider != providerAnthropic {
		t.Errorf("Provider = %q, want anthropic", cfg.Provider)
	}
	if cfg.BaseURL != defaultAnthropicBaseURL {
		t.Errorf("BaseURL = %q, want %q", cfg.BaseURL, defaultAnthropicBaseURL)
	}
	if cfg.Model != defaultAnthropicModel {
		t.Errorf("Model = %q, want %q", cfg.Model, defaultAnthropicModel)
	}
}

func TestLoadConfig_Anthropic_StripsProviderPrefixAndTrimsBase(t *testing.T) {
	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	t.Setenv("ENGRAM_LLM_BASE_URL", "https://api.anthropic.com/")
	t.Setenv("ENGRAM_LLM_MODEL", "anthropic/claude-sonnet-5")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Model != "claude-sonnet-5" {
		t.Errorf("Model = %q, want claude-sonnet-5 (prefix stripped)", cfg.Model)
	}
	if cfg.BaseURL != "https://api.anthropic.com" {
		t.Errorf("BaseURL = %q, want trailing slash trimmed", cfg.BaseURL)
	}
}

// --- R3 + R4: response mapping and budget flow on the anthropic path ---

func TestCallWithBudget_Anthropic_MapsMetaAndSendsBudget(t *testing.T) {
	var gotPath, gotAPIKey, gotAuth, gotBeta, gotVersion string
	var gotMaxTokens float64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAPIKey = r.Header.Get("X-Api-Key")
		gotAuth = r.Header.Get("Authorization")
		gotBeta = r.Header.Get("Anthropic-Beta")
		gotVersion = r.Header.Get("Anthropic-Version")
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotMaxTokens, _ = req["max_tokens"].(float64)
		_, _ = io.WriteString(w, `{
			"content": [{"type": "text", "text": "What is the core tension in today's work?"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 210, "output_tokens": 42}
		}`)
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ENGRAM_LLM_MODEL", "")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test") // exercise X-Api-Key path
	t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "")

	content, meta, err := CallWithBudget(context.Background(), "focal prompt", 1500)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotPath != "/v1/messages" {
		t.Errorf("path = %q, want /v1/messages", gotPath)
	}
	if gotMaxTokens != 1500 {
		t.Errorf("max_tokens on wire = %v, want 1500", gotMaxTokens)
	}
	if gotAPIKey != "sk-ant-test" {
		t.Errorf("X-Api-Key not set from ANTHROPIC_API_KEY")
	}
	if gotAuth != "" || gotBeta != "" {
		t.Errorf("api-key path must not send Authorization/Anthropic-Beta; got auth=%q beta=%q", gotAuth, gotBeta)
	}
	if gotVersion != "2023-06-01" {
		t.Errorf("Anthropic-Version = %q", gotVersion)
	}
	if content != "What is the core tension in today's work?" {
		t.Errorf("content = %q", content)
	}
	if meta.FinishReason != "stop" { // end_turn -> stop
		t.Errorf("FinishReason = %q, want stop", meta.FinishReason)
	}
	if meta.PromptTokens != 210 || meta.CompletionTokens != 42 || meta.TotalTokens != 252 {
		t.Errorf("token mapping wrong: %+v", meta)
	}
	if meta.ReasoningTokens != -1 {
		t.Errorf("ReasoningTokens = %d, want -1 (unreported)", meta.ReasoningTokens)
	}
	if meta.MaxTokens != 1500 {
		t.Errorf("meta.MaxTokens = %d, want 1500", meta.MaxTokens)
	}
}

// R3: stop_reason "max_tokens" must normalise to finish_reason "length" so
// reflection's truncation retry fires.
func TestCallWithBudget_Anthropic_MaxTokensMapsToLength(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{
			"content": [{"type": "text", "text": "partial"}],
			"stop_reason": "max_tokens",
			"usage": {"input_tokens": 10, "output_tokens": 1500}
		}`)
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")
	t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "")

	_, meta, err := CallWithBudget(context.Background(), "hi", 1500)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.FinishReason != "length" {
		t.Errorf("FinishReason = %q, want length", meta.FinishReason)
	}
}

// R3: usage absent -> -1 sentinels preserved.
func TestCallWithBudget_Anthropic_UsageAbsentSentinels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}`)
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")
	t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "")

	_, meta, err := CallWithBudget(context.Background(), "hi", 1500)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.PromptTokens != -1 || meta.CompletionTokens != -1 || meta.TotalTokens != -1 || meta.ReasoningTokens != -1 {
		t.Errorf("expected -1 sentinels when usage absent, got %+v", meta)
	}
}

// R2: OAuth token path (env) sets Bearer + Anthropic-Beta, not X-Api-Key.
func TestCallWithBudget_Anthropic_OAuthTokenHeaders(t *testing.T) {
	var gotAuth, gotBeta, gotAPIKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotBeta = r.Header.Get("Anthropic-Beta")
		gotAPIKey = r.Header.Get("X-Api-Key")
		_, _ = io.WriteString(w, `{"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}`)
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ANTHROPIC_API_KEY", "") // force OAuth env path
	t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-xyz")

	if _, _, err := CallWithBudget(context.Background(), "hi", 1500); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gotAuth != "Bearer sk-ant-oat-xyz" {
		t.Errorf("Authorization = %q, want Bearer token", gotAuth)
	}
	if gotBeta != "claude-code-20250219,oauth-2025-04-20" {
		t.Errorf("Anthropic-Beta = %q", gotBeta)
	}
	if gotAPIKey != "" {
		t.Errorf("OAuth path must not send X-Api-Key; got %q", gotAPIKey)
	}
}

// Fake-429 fix: the OAuth path must send the Claude Code identity system
// prompt; the X-Api-Key path must not send any system field.
func TestCallWithBudget_Anthropic_SystemPromptOAuthOnly(t *testing.T) {
	capture := func(t *testing.T, oauth bool) (systemField any, present bool) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			systemField, present = req["system"]
			_, _ = io.WriteString(w, `{"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}`)
		}))
		defer srv.Close()

		t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
		t.Setenv("ENGRAM_LLM_API_KEY", "")
		t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
		if oauth {
			t.Setenv("ANTHROPIC_API_KEY", "")
			t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-xyz")
		} else {
			t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")
			t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "")
		}
		if _, _, err := CallWithBudget(context.Background(), "focal prompt", 1500); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return systemField, present
	}

	t.Run("oauth path sends system==const", func(t *testing.T) {
		sys, present := capture(t, true)
		if !present {
			t.Fatal("OAuth path must include a top-level system field")
		}
		if s, _ := sys.(string); s != claudeCodeSystemPrompt {
			t.Errorf("system = %q, want %q", s, claudeCodeSystemPrompt)
		}
	})
	t.Run("api-key path sends no system", func(t *testing.T) {
		_, present := capture(t, false)
		if present {
			t.Error("X-Api-Key path must NOT send a system field")
		}
	})
}

// R5: non-200 keeps "llm returned status %d" and appends a body excerpt.
func TestCallWithBudget_Anthropic_Non200IncludesStatusAndExcerpt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(402)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"billing","message":"region blocked"}}`)
	}))
	defer srv.Close()

	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	t.Setenv("ENGRAM_LLM_BASE_URL", srv.URL)
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")
	t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "")

	_, _, err := CallWithBudget(context.Background(), "hi", 1500)
	if err == nil {
		t.Fatal("expected error on 402")
	}
	if !strings.Contains(err.Error(), "llm returned status 402") {
		t.Errorf("error must keep 'llm returned status 402' shape: %q", err.Error())
	}
	if !strings.Contains(err.Error(), "region blocked") {
		t.Errorf("error must include body excerpt: %q", err.Error())
	}
}

// R2: no credentials at all -> clear error, does not hard-fail on missing
// ENGRAM_LLM_API_KEY.
func TestCallWithBudget_Anthropic_NoCredentials(t *testing.T) {
	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("CLAUDE_CODE_OAUTH_TOKEN", "")
	// Point the creds file at a nonexistent path so no host file leaks in.
	orig := claudeCredentialsPath
	claudeCredentialsPath = filepath.Join(t.TempDir(), "nope.json")
	defer func() { claudeCredentialsPath = orig }()

	_, _, err := CallWithBudget(context.Background(), "hi", 1500)
	if err == nil {
		t.Fatal("expected error when no anthropic credentials available")
	}
	if !strings.Contains(err.Error(), "no Anthropic credentials") {
		t.Errorf("error = %q", err.Error())
	}
}

// readClaudeOAuthToken: valid token returned; expired token skipped. Uses a
// temp file (no /root access), so no root guard needed.
func TestReadClaudeOAuthToken_ExpiryHandling(t *testing.T) {
	orig := claudeCredentialsPath
	defer func() { claudeCredentialsPath = orig }()

	write := func(tok string, expMs int64) {
		p := filepath.Join(t.TempDir(), "creds.json")
		body, _ := json.Marshal(map[string]any{
			"claudeAiOauth": map[string]any{"accessToken": tok, "expiresAt": expMs},
		})
		if err := os.WriteFile(p, body, 0o600); err != nil {
			t.Fatal(err)
		}
		claudeCredentialsPath = p
	}

	future := time.Now().Add(time.Hour).UnixMilli()
	write("sk-ant-oat-live", future)
	if got := readClaudeOAuthToken(); got != "sk-ant-oat-live" {
		t.Errorf("valid token: got %q", got)
	}

	past := time.Now().Add(-time.Hour).UnixMilli()
	write("sk-ant-oat-stale", past)
	if got := readClaudeOAuthToken(); got != "" {
		t.Errorf("expired token must be skipped, got %q", got)
	}

	claudeCredentialsPath = filepath.Join(t.TempDir(), "missing.json")
	if got := readClaudeOAuthToken(); got != "" {
		t.Errorf("missing file must return empty, got %q", got)
	}
}
