// Package llm is the single shared LLM client for engram. It speaks the
// OpenAI-compatible chat/completions protocol and is used by both the dream
// and reflection engines.
//
// Migration note: this package replaces the previous per-package Anthropic
// /v1/messages clients that read Claude Code OAuth credentials. There is no
// credential-file / OAuth fallback anymore — configuration comes solely from
// the ENGRAM_LLM_* env vars below.
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

	// maxTokens is the response ceiling for a single call. It unifies the two
	// prior call-site values (dream 1024, reflection 1500) to the larger one so
	// neither path truncates.
	maxTokens = 1500

	requestTimeout = 60 * time.Second
)

// config holds the resolved LLM client configuration.
type config struct {
	APIKey  string
	BaseURL string
	Model   string
}

// loadConfig resolves configuration from ENGRAM_LLM_* env vars. The API key is
// required; base URL and model fall back to OpenRouter / Sonnet-5 defaults.
func loadConfig() (*config, error) {
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
	return &config{APIKey: key, BaseURL: baseURL, Model: model}, nil
}

// Meta carries observability metadata about a single LLM call.
type Meta struct {
	FinishReason string // choices[0].finish_reason from the provider
	RawLen       int    // byte length of the assistant content (len(content))
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

// Call sends a single-turn user prompt and returns the assistant text. It
// delegates to CallWithMeta and discards the metadata, staying backward
// compatible with existing callers.
func Call(ctx context.Context, prompt string) (string, error) {
	content, _, err := CallWithMeta(ctx, prompt)
	return content, err
}

// CallWithMeta sends a single-turn user prompt and returns the assistant text
// along with observability metadata (finish_reason, raw byte length).
func CallWithMeta(ctx context.Context, prompt string) (string, Meta, error) {
	cfg, err := loadConfig()
	if err != nil {
		return "", Meta{}, err
	}

	reqBody, err := json.Marshal(map[string]any{
		"model":      cfg.Model,
		"max_tokens": resolveMaxTokens(),
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
		return "", Meta{}, fmt.Errorf("llm returned status %d", resp.StatusCode)
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
	}
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", Meta{}, fmt.Errorf("decode llm response: %w", err)
	}

	if len(apiResp.Choices) == 0 {
		return "", Meta{}, fmt.Errorf("no choices in llm response")
	}
	content := strings.TrimSpace(apiResp.Choices[0].Message.Content)
	meta := Meta{
		FinishReason: apiResp.Choices[0].FinishReason,
		RawLen:       len(content),
	}
	return content, meta, nil
}
