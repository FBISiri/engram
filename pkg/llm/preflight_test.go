package llm

import "testing"

func TestPreflight(t *testing.T) {
	// provider unset & key unset -> openrouter needs a key -> error
	t.Setenv("ENGRAM_LLM_PROVIDER", "")
	t.Setenv("ENGRAM_LLM_API_KEY", "")
	if err := Preflight(); err == nil {
		t.Fatal("expected error when provider unset and key unset")
	}

	// provider=anthropic -> no key needed -> nil
	t.Setenv("ENGRAM_LLM_PROVIDER", "anthropic")
	if err := Preflight(); err != nil {
		t.Fatalf("anthropic should preflight clean: %v", err)
	}

	// openrouter with key -> nil
	t.Setenv("ENGRAM_LLM_PROVIDER", "openrouter")
	t.Setenv("ENGRAM_LLM_API_KEY", "k")
	if err := Preflight(); err != nil {
		t.Fatalf("openrouter+key should preflight clean: %v", err)
	}
}
