package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// claudeCredentialsPath is the on-disk Claude Code OAuth credentials file. It is
// a package var (not const) only so tests can point it at a temp file; in
// production it is always the path below. The token here rotates (short TTL,
// refreshed externally by the master process), so it is read PER-CALL — never
// cached at process start — and skipped when expired.
var claudeCredentialsPath = "/root/.claude/.credentials.json"

// anthropicCreds is a resolved credential for a single Anthropic call. Exactly
// one of the fields is set. It is never logged.
type anthropicCreds struct {
	apiKey string // ANTHROPIC_API_KEY -> X-Api-Key header
	token  string // OAuth access token -> Authorization: Bearer + Anthropic-Beta
}

// resolveAnthropicCreds resolves credentials in priority order:
//  1. ANTHROPIC_API_KEY env (X-Api-Key path)
//  2. CLAUDE_CODE_OAUTH_TOKEN env (OAuth bearer path)
//  3. /root/.claude/.credentials.json claudeAiOauth.accessToken (OAuth bearer
//     path), read fresh on every call and skipped if expiresAt is in the past.
func resolveAnthropicCreds() (anthropicCreds, error) {
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		return anthropicCreds{apiKey: key}, nil
	}
	if token := os.Getenv("CLAUDE_CODE_OAUTH_TOKEN"); token != "" {
		return anthropicCreds{token: token}, nil
	}
	if token := readClaudeOAuthToken(); token != "" {
		return anthropicCreds{token: token}, nil
	}
	return anthropicCreds{}, fmt.Errorf("no Anthropic credentials available (set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN, or provide %s)", claudeCredentialsPath)
}

// readClaudeOAuthToken reads a non-expired OAuth access token from the Claude
// Code credentials file. Returns "" if the file is missing, unreadable, has no
// token, or the token has expired.
func readClaudeOAuthToken() string {
	var creds struct {
		ClaudeAiOauth struct {
			AccessToken string `json:"accessToken"`
			ExpiresAt   int64  `json:"expiresAt"`
		} `json:"claudeAiOauth"`
	}
	data, err := os.ReadFile(claudeCredentialsPath)
	if err != nil {
		return ""
	}
	if err := json.Unmarshal(data, &creds); err != nil {
		return ""
	}
	token := creds.ClaudeAiOauth.AccessToken
	if token == "" {
		return ""
	}
	if exp := creds.ClaudeAiOauth.ExpiresAt; exp > 0 && exp < time.Now().UnixMilli() {
		return "" // token expired
	}
	return token
}

// bodyExcerpt returns a trimmed, length-capped excerpt of a response body for
// diagnostics. The body is the provider's response, never our credential.
func bodyExcerpt(b []byte) string {
	const max = 200
	s := strings.TrimSpace(string(b))
	if len(s) > max {
		s = s[:max]
	}
	return s
}

// callAnthropic performs a POST {base}/v1/messages request against the direct
// Anthropic Messages API and normalises the response into the shared Meta shape
// (stop_reason -> finish_reason, input/output tokens -> prompt/completion
// tokens) so truncation detection in reflection keeps working unchanged.
func callAnthropic(ctx context.Context, cfg *config, prompt string, maxTokens int) (string, Meta, error) {
	creds, err := resolveAnthropicCreds()
	if err != nil {
		return "", Meta{}, err
	}

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

	req, err := http.NewRequestWithContext(ctx, "POST", cfg.BaseURL+"/v1/messages", bytes.NewReader(reqBody))
	if err != nil {
		return "", Meta{}, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	if creds.token != "" {
		req.Header.Set("Authorization", "Bearer "+creds.token)
		req.Header.Set("Anthropic-Beta", "claude-code-20250219,oauth-2025-04-20")
	} else {
		req.Header.Set("X-Api-Key", creds.apiKey)
	}

	client := &http.Client{Timeout: requestTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", Meta{}, fmt.Errorf("llm request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", Meta{}, fmt.Errorf("read llm response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Keep the "llm returned status %d" shape reflection/alerting keys off,
		// plus a short token-free body excerpt so the failure is diagnosable.
		return "", Meta{}, fmt.Errorf("llm returned status %d: %s", resp.StatusCode, bodyExcerpt(body))
	}

	var apiResp struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      *struct {
			InputTokens  *int `json:"input_tokens"`
			OutputTokens *int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", Meta{}, fmt.Errorf("decode llm response: %w", err)
	}

	var content string
	for _, block := range apiResp.Content {
		if block.Type == "text" {
			content = strings.TrimSpace(block.Text)
			break
		}
	}

	// Normalise stop_reason to the OpenAI-style finish_reason that
	// reflection/llm_retry.go branches on.
	finish := apiResp.StopReason
	switch apiResp.StopReason {
	case "max_tokens":
		finish = "length"
	case "end_turn":
		finish = "stop"
	}

	meta := Meta{
		FinishReason:     finish,
		RawLen:           len(content),
		MaxTokens:        maxTokens,
		PromptTokens:     -1,
		CompletionTokens: -1,
		TotalTokens:      -1,
		ReasoningTokens:  -1, // Anthropic /v1/messages does not report this
	}
	if u := apiResp.Usage; u != nil {
		if u.InputTokens != nil {
			meta.PromptTokens = *u.InputTokens
		}
		if u.OutputTokens != nil {
			meta.CompletionTokens = *u.OutputTokens
		}
		if u.InputTokens != nil && u.OutputTokens != nil {
			meta.TotalTokens = *u.InputTokens + *u.OutputTokens
		}
	}
	return content, meta, nil
}
