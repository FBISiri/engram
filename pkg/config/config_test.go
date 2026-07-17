package config

import (
	"os"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// clearEngramEnv removes every ENGRAM_* variable for the duration of the test so
// that Load() observes a pristine environment. t.Setenv restores originals on
// cleanup, so tests remain hermetic and order-independent.
func clearEngramEnv(t *testing.T) {
	t.Helper()
	for _, kv := range os.Environ() {
		key := kv[:strings.IndexByte(kv, '=')]
		if strings.HasPrefix(key, "ENGRAM_") {
			t.Setenv(key, "")
			os.Unsetenv(key)
		}
	}
}

func TestLoad_Defaults(t *testing.T) {
	clearEngramEnv(t)
	c := Load()

	if c == nil {
		t.Fatal("Load returned nil")
	}
	// Spot-check one field of each type / section against documented defaults.
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"QdrantURL", c.QdrantURL, "localhost:6334"},
		{"QdrantUseTLS", c.QdrantUseTLS, false},
		{"EmbedderProvider", c.EmbedderProvider, "openai"},
		{"EmbeddingModel", c.EmbeddingModel, "text-embedding-3-small"},
		{"EmbeddingDimension", c.EmbeddingDimension, 1536},
		{"OpenAIBaseURL", c.OpenAIBaseURL, "https://api.openai.com/v1"},
		{"WeightRelevance", c.Weights.Relevance, 1.0},
		{"WeightRecency", c.Weights.Recency, 0.5},
		{"WeightImportance", c.Weights.Importance, 0.3},
		{"MMRLambda", c.MMRLambda, 0.5},
		{"DedupThreshold", c.DedupThreshold, 0.92},
		{"Transport", c.Transport, "stdio"},
		{"HTTPPort", c.HTTPPort, 8080},
		{"ReflectionEnabled", c.ReflectionEnabled, false},
		{"ReflectionTrigger", c.ReflectionTrigger, "count"},
		{"ReflectionCount", c.ReflectionCount, 10},
		{"ReflectionModel", c.ReflectionModel, "claude-sonnet-4-20250514"},
	}
	for _, ck := range checks {
		if ck.got != ck.want {
			t.Errorf("%s default = %v, want %v", ck.name, ck.got, ck.want)
		}
	}
	// Decay must equal the memory package default.
	if c.Decay != memory.DefaultDecayConfig() {
		t.Errorf("Decay default = %+v, want %+v", c.Decay, memory.DefaultDecayConfig())
	}
}

func TestLoad_EnvOverride(t *testing.T) {
	clearEngramEnv(t)
	env := map[string]string{
		"ENGRAM_QDRANT_URL":          "qdrant.internal:7000",
		"ENGRAM_QDRANT_USE_TLS":      "true",
		"ENGRAM_EMBEDDER_PROVIDER":   "voyage",
		"ENGRAM_EMBEDDING_DIMENSION": "1024",
		"ENGRAM_WEIGHT_RELEVANCE":    "2.5",
		"ENGRAM_MMR_LAMBDA":          "0.7",
		"ENGRAM_DEDUP_THRESHOLD":     "0.70",
		"ENGRAM_HTTP_PORT":           "9090",
		"ENGRAM_REFLECTION_ENABLED":  "1",
		"ENGRAM_REFLECTION_COUNT":    "25",
	}
	for k, v := range env {
		t.Setenv(k, v)
	}
	c := Load()

	if c.QdrantURL != "qdrant.internal:7000" {
		t.Errorf("QdrantURL = %q", c.QdrantURL)
	}
	if !c.QdrantUseTLS {
		t.Error("QdrantUseTLS should be true")
	}
	if c.EmbedderProvider != "voyage" {
		t.Errorf("EmbedderProvider = %q", c.EmbedderProvider)
	}
	if c.EmbeddingDimension != 1024 {
		t.Errorf("EmbeddingDimension = %d", c.EmbeddingDimension)
	}
	if c.Weights.Relevance != 2.5 {
		t.Errorf("Weights.Relevance = %v", c.Weights.Relevance)
	}
	if c.MMRLambda != 0.7 {
		t.Errorf("MMRLambda = %v", c.MMRLambda)
	}
	if c.DedupThreshold != 0.70 {
		t.Errorf("DedupThreshold = %v", c.DedupThreshold)
	}
	if c.HTTPPort != 9090 {
		t.Errorf("HTTPPort = %d", c.HTTPPort)
	}
	if !c.ReflectionEnabled {
		t.Error("ReflectionEnabled should be true (env=1)")
	}
	if c.ReflectionCount != 25 {
		t.Errorf("ReflectionCount = %d", c.ReflectionCount)
	}
}

func TestEnvStr(t *testing.T) {
	tests := []struct {
		name       string
		set        bool
		val        string
		defaultVal string
		want       string
	}{
		{"unset returns default", false, "", "def", "def"},
		{"empty returns default", true, "", "def", "def"},
		{"set returns value", true, "custom", "def", "custom"},
		{"whitespace is a value", true, " ", "def", " "},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "ENGRAM_TEST_STR"
			os.Unsetenv(key)
			if tt.set {
				t.Setenv(key, tt.val)
			}
			if got := envStr(key, tt.defaultVal); got != tt.want {
				t.Errorf("envStr = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestEnvInt(t *testing.T) {
	tests := []struct {
		name       string
		set        bool
		val        string
		defaultVal int
		want       int
	}{
		{"unset returns default", false, "", 7, 7},
		{"empty returns default", true, "", 7, 7},
		{"valid int", true, "42", 7, 42},
		{"negative int", true, "-3", 7, -3},
		{"malformed returns default", true, "notanint", 7, 7},
		{"float string returns default", true, "3.14", 7, 7},
		{"overflow returns default", true, "99999999999999999999999", 7, 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "ENGRAM_TEST_INT"
			os.Unsetenv(key)
			if tt.set {
				t.Setenv(key, tt.val)
			}
			if got := envInt(key, tt.defaultVal); got != tt.want {
				t.Errorf("envInt = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestEnvFloat(t *testing.T) {
	tests := []struct {
		name       string
		set        bool
		val        string
		defaultVal float64
		want       float64
	}{
		{"unset returns default", false, "", 0.5, 0.5},
		{"empty returns default", true, "", 0.5, 0.5},
		{"valid float", true, "0.82", 0.5, 0.82},
		{"integer string", true, "2", 0.5, 2.0},
		{"scientific notation", true, "1e-2", 0.5, 0.01},
		{"malformed returns default", true, "abc", 0.5, 0.5},
		{"trailing junk returns default", true, "0.7x", 0.5, 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "ENGRAM_TEST_FLOAT"
			os.Unsetenv(key)
			if tt.set {
				t.Setenv(key, tt.val)
			}
			if got := envFloat(key, tt.defaultVal); got != tt.want {
				t.Errorf("envFloat = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEnvBool(t *testing.T) {
	tests := []struct {
		name       string
		set        bool
		val        string
		defaultVal bool
		want       bool
	}{
		{"unset returns default true", false, "", true, true},
		{"unset returns default false", false, "", false, false},
		{"empty returns default", true, "", true, true},
		{"true literal", true, "true", false, true},
		{"1 is true", true, "1", false, true},
		{"0 is false", true, "0", true, false},
		{"false literal", true, "false", true, false},
		{"TRUE uppercase", true, "TRUE", false, true},
		{"malformed returns default true", true, "yes", true, true},
		{"malformed returns default false", true, "nope", false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "ENGRAM_TEST_BOOL"
			os.Unsetenv(key)
			if tt.set {
				t.Setenv(key, tt.val)
			}
			if got := envBool(key, tt.defaultVal); got != tt.want {
				t.Errorf("envBool = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParsePrincipalKeys(t *testing.T) {
	cases := []struct {
		in   string
		want map[string]string
	}{
		{"", nil},
		{"pigo:abc", map[string]string{"pigo": "abc"}},
		{"pigo:abc,reflection:def", map[string]string{"pigo": "abc", "reflection": "def"}},
		{" pigo : abc , ", map[string]string{"pigo": "abc"}},
		{"malformed", nil},
		{":nokey,notype:", nil},
		// invalid/typo caller-type is skipped (must not default to engram_user).
		{"reflectionn:key", nil},
		{"pigo:abc,reflectionn:bad", map[string]string{"pigo": "abc"}},
		{"agent-self:xyz", map[string]string{"agent-self": "xyz"}},
	}
	for _, tc := range cases {
		got := parsePrincipalKeys(tc.in)
		if len(got) != len(tc.want) {
			t.Errorf("parsePrincipalKeys(%q) = %v, want %v", tc.in, got, tc.want)
			continue
		}
		for k, v := range tc.want {
			if got[k] != v {
				t.Errorf("parsePrincipalKeys(%q)[%s] = %q, want %q", tc.in, k, got[k], v)
			}
		}
	}
}
