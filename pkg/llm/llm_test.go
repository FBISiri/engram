package llm

import (
	"context"
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
