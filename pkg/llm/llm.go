// Package llm is the single shared LLM client for engram. It is used by both
// the dream and reflection engines.
//
// Two providers are selectable via ENGRAM_LLM_PROVIDER:
//   - "openrouter" (default): OpenAI-compatible POST {base}/chat/completions
//     authenticated with ENGRAM_LLM_API_KEY. This is the original behaviour and
//     is used bit-for-bit when the var is unset or unknown.
//   - "anthropic": direct Anthropic Messages API POST {base}/v1/messages. This
//     is a TEMPORARY workaround (2026-09-21) for OpenRouter's region block on
//     Anthropic models; it reuses the Claude Code OAuth credential and reads it
//     per-call so a rotating access token is always picked up fresh.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultBaseURL = "https://openrouter.ai/api/v1"
	defaultModel   = "anthropic/claude-sonnet-5"

	// providerOpenRouter is the default backend (unchanged behaviour).
	providerOpenRouter = "openrouter"
	// providerAnthropic is the direct Anthropic Messages API backend.
	providerAnthropic = "anthropic"

	// defaultAnthropicBaseURL / defaultAnthropicModel apply when
	// ENGRAM_LLM_PROVIDER=anthropic and the corresponding var is unset.
	defaultAnthropicBaseURL = "https://api.anthropic.com"
	defaultAnthropicModel   = "claude-sonnet-5"

	// maxTokens is the response ceiling for a single call. It unifies the two
	// prior call-site values (dream 1024, reflection 1500) to the larger one so
	// neither path truncates.
	maxTokens = 1500

	// defaultDialecticMaxTokens is the fallback budget for the dialectic stage,
	// which needs more headroom than the global default because reasoning tokens
	// (claude-sonnet-5 via OpenRouter) can otherwise exhaust the budget.
	defaultDialecticMaxTokens = 4000
	// defaultMaxTokensCeiling caps the retry escalation so a single call cannot
	// grow unbounded.
	defaultMaxTokensCeiling = 8000

	requestTimeout = 60 * time.Second
)

// config holds the resolved LLM client configuration.
type config struct {
	Provider string
	APIKey   string // openrouter bearer key; unused on the anthropic path
	BaseURL  string
	Model    string
}

// resolveProvider reads ENGRAM_LLM_PROVIDER. Unset or unknown => openrouter, so
// the default behaviour is preserved bit-for-bit.
func resolveProvider() string {
	if strings.ToLower(strings.TrimSpace(os.Getenv("ENGRAM_LLM_PROVIDER"))) == providerAnthropic {
		return providerAnthropic
	}
	return providerOpenRouter
}

// loadConfig resolves configuration from ENGRAM_LLM_* env vars.
//
// openrouter (default): ENGRAM_LLM_API_KEY is required; base URL and model fall
// back to OpenRouter / Sonnet-5 defaults.
//
// anthropic: ENGRAM_LLM_API_KEY is NOT required (credentials are resolved
// per-call, see anthropic.go); base URL falls back to the Anthropic API and the
// model falls back to claude-sonnet-5, with any leading "anthropic/" provider
// prefix stripped so an OpenRouter-style ENGRAM_LLM_MODEL resolves cleanly.
func loadConfig() (*config, error) {
	provider := resolveProvider()
	if provider == providerAnthropic {
		baseURL := os.Getenv("ENGRAM_LLM_BASE_URL")
		if baseURL == "" {
			baseURL = defaultAnthropicBaseURL
		}
		baseURL = strings.TrimRight(baseURL, "/")
		model := os.Getenv("ENGRAM_LLM_MODEL")
		if model == "" {
			model = defaultAnthropicModel
		}
		model = strings.TrimPrefix(model, "anthropic/")
		return &config{Provider: provider, BaseURL: baseURL, Model: model}, nil
	}

	key := os.Getenv("ENGRAM_LLM_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("no LLM API key configured (set ENGRAM_LLM_API_KEY)")
	}
	baseURL := os.Getenv("ENGRAM_LLM_BASE_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	model := os.Getenv("ENGRAM_LLM_MODEL")
	if model == "" {
		model = defaultModel
	}
	return &config{Provider: provider, APIKey: key, BaseURL: baseURL, Model: model}, nil
}

// Meta carries observability metadata about a single LLM call. Token counts use
// -1 as a sentinel meaning "provider did not report this value".
type Meta struct {
	FinishReason     string // choices[0].finish_reason from the provider
	RawLen           int    // byte length of the assistant content (len(content))
	MaxTokens        int    // max_tokens actually sent in the request
	PromptTokens     int    // usage.prompt_tokens (-1 if unreported)
	CompletionTokens int    // usage.completion_tokens (-1 if unreported)
	TotalTokens      int    // usage.total_tokens (-1 if unreported)
	ReasoningTokens  int    // usage.completion_tokens_details.reasoning_tokens (-1 if unreported)
}

// resolveMaxTokens reads ENGRAM_LLM_MAX_TOKENS. Unset, non-numeric, or <= 0
// falls back to the maxTokens default (single source of truth).
func resolveMaxTokens() int {
	v := os.Getenv("ENGRAM_LLM_MAX_TOKENS")
	if v == "" {
		return maxTokens
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return maxTokens
	}
	return n
}

// resolveDialecticMaxTokens reads ENGRAM_LLM_DIALECTIC_MAX_TOKENS. Unset,
// non-numeric, or <= 0 falls back to the defaultDialecticMaxTokens default.
func resolveDialecticMaxTokens() int {
	v := os.Getenv("ENGRAM_LLM_DIALECTIC_MAX_TOKENS")
	if v == "" {
		return defaultDialecticMaxTokens
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return defaultDialecticMaxTokens
	}
	return n
}

// resolveMaxTokensCeiling reads ENGRAM_LLM_MAX_TOKENS_CEILING. Unset,
// non-numeric, or <= 0 falls back to the defaultMaxTokensCeiling default.
func resolveMaxTokensCeiling() int {
	v := os.Getenv("ENGRAM_LLM_MAX_TOKENS_CEILING")
	if v == "" {
		return defaultMaxTokensCeiling
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return defaultMaxTokensCeiling
	}
	return n
}

// MaxTokens returns the resolved global response budget (ENGRAM_LLM_MAX_TOKENS).
func MaxTokens() int { return resolveMaxTokens() }

// DialecticMaxTokens returns the resolved dialectic-stage budget
// (ENGRAM_LLM_DIALECTIC_MAX_TOKENS).
func DialecticMaxTokens() int { return resolveDialecticMaxTokens() }

// MaxTokensCeiling returns the resolved retry escalation cap
// (ENGRAM_LLM_MAX_TOKENS_CEILING).
func MaxTokensCeiling() int { return resolveMaxTokensCeiling() }

// Call sends a single-turn user prompt and returns the assistant text. It
// delegates to CallWithMeta and discards the metadata, staying backward
// compatible with existing callers.
func Call(ctx context.Context, prompt string) (string, error) {
	content, _, err := CallWithMeta(ctx, prompt)
	return content, err
}

// CallWithMeta sends a single-turn user prompt at the resolved global budget and
// returns the assistant text along with observability metadata. It delegates to
// CallWithBudget.
func CallWithMeta(ctx context.Context, prompt string) (string, Meta, error) {
	return CallWithBudget(ctx, prompt, resolveMaxTokens())
}

// CallWithBudget sends a single-turn user prompt at the given max_tokens budget
// and returns the assistant text along with observability metadata
// (finish_reason, raw byte length, and usage token counts when reported). A
// non-positive budget falls back to the resolved global default.
func CallWithBudget(ctx context.Context, prompt string, maxTokens int) (string, Meta, error) {
	if maxTokens <= 0 {
		maxTokens = resolveMaxTokens()
	}
	cfg, err := loadConfig()
	if err != nil {
		return "", Meta{}, err
	}
	if cfg.Provider == providerAnthropic {
		return callAnthropic(ctx, cfg, prompt, maxTokens)
	}
	return callOpenRouter(ctx, cfg, prompt, maxTokens)
}

// callOpenRouter performs the OpenAI-compatible chat/completions request. This
// is the original CallWithBudget body, unchanged, so ENGRAM_LLM_PROVIDER unset
// (or unknown) behaves bit-for-bit as before.
func callOpenRouter(ctx context.Context, cfg *config, prompt string, maxTokens int) (string, Meta, error) {
	reqBody, err := json.Marshal(map[string]any{
		"model":      cfg.Model,
		"max_tokens": maxTokens,
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
	})
	if err != nil {
		return "", Meta{}, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", cfg.BaseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return "", Meta{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	client := &http.Client{Timeout: requestTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", Meta{}, fmt.Errorf("llm request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		// Read a short body excerpt so the typed StatusError is diagnosable, and
		// align to the "llm returned status %d: " prefix the anthropic path uses.
		body, _ := io.ReadAll(resp.Body)
		return "", Meta{}, &StatusError{StatusCode: resp.StatusCode, RetryAfter: resp.Header.Get("Retry-After"), Body: bodyExcerpt(body)}
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", Meta{}, fmt.Errorf("read llm response: %w", err)
	}

	var apiResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens            *int `json:"prompt_tokens"`
			CompletionTokens        *int `json:"completion_tokens"`
			TotalTokens             *int `json:"total_tokens"`
			CompletionTokensDetails *struct {
				ReasoningTokens *int `json:"reasoning_tokens"`
			} `json:"completion_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", Meta{}, fmt.Errorf("decode llm response: %w", err)
	}

	if len(apiResp.Choices) == 0 {
		return "", Meta{}, fmt.Errorf("no choices in llm response")
	}
	content := strings.TrimSpace(apiResp.Choices[0].Message.Content)
	meta := Meta{
		FinishReason:     apiResp.Choices[0].FinishReason,
		RawLen:           len(content),
		MaxTokens:        maxTokens,
		PromptTokens:     -1,
		CompletionTokens: -1,
		TotalTokens:      -1,
		ReasoningTokens:  -1,
	}
	if u := apiResp.Usage; u != nil {
		if u.PromptTokens != nil {
			meta.PromptTokens = *u.PromptTokens
		}
		if u.CompletionTokens != nil {
			meta.CompletionTokens = *u.CompletionTokens
		}
		if u.TotalTokens != nil {
			meta.TotalTokens = *u.TotalTokens
		}
		if d := u.CompletionTokensDetails; d != nil && d.ReasoningTokens != nil {
			meta.ReasoningTokens = *d.ReasoningTokens
		}
	}
	return content, meta, nil
}
