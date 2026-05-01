package dream

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// haikuConfig holds credentials for Haiku API calls.
type haikuConfig struct {
	APIKey  string
	BaseURL string
	Model   string
	IsOAuth bool
}

// getHaikuConfig returns the best available Haiku configuration.
// Priority: CLAUDE_CODE_OAUTH_TOKEN env → /mnt/bmo/.credentials.json → ANTHROPIC_API_KEY.
func getHaikuConfig() *haikuConfig {
	model := os.Getenv("ANTHROPIC_LIGHT_MODEL")
	if model == "" {
		model = "claude-haiku-4-5-20251001"
	}

	// 1. Claude Code OAuth token from env
	if token := os.Getenv("CLAUDE_CODE_OAUTH_TOKEN"); token != "" {
		return &haikuConfig{
			APIKey:  token,
			BaseURL: "https://api.anthropic.com",
			Model:   model,
			IsOAuth: true,
		}
	}

	// 2. Claude Code credentials file
	if token := readClaudeOAuthToken(); token != "" {
		return &haikuConfig{
			APIKey:  token,
			BaseURL: "https://api.anthropic.com",
			Model:   model,
			IsOAuth: true,
		}
	}

	// 3. Direct Anthropic API key
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		return &haikuConfig{
			APIKey:  key,
			BaseURL: "https://api.anthropic.com",
			Model:   model,
			IsOAuth: false,
		}
	}

	return nil
}

// readClaudeOAuthToken reads the OAuth access token from /mnt/bmo/.credentials.json.
func readClaudeOAuthToken() string {
	data, err := os.ReadFile("/mnt/bmo/.credentials.json")
	if err != nil {
		return ""
	}
	var creds struct {
		ClaudeAiOauth struct {
			AccessToken string `json:"accessToken"`
		} `json:"claudeAiOauth"`
	}
	if err := json.Unmarshal(data, &creds); err != nil {
		return ""
	}
	return creds.ClaudeAiOauth.AccessToken
}

// callHaiku sends a prompt to Claude Haiku and returns the text response.
func callHaiku(ctx context.Context, prompt string) (string, error) {
	cfg := getHaikuConfig()
	if cfg == nil {
		return "", fmt.Errorf("no Haiku API credentials available")
	}

	reqBody, err := json.Marshal(map[string]any{
		"model":      cfg.Model,
		"max_tokens": 1024,
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
	})
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", cfg.BaseURL+"/v1/messages", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	if cfg.IsOAuth {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		req.Header.Set("Anthropic-Beta", "claude-code-20250219,oauth-2025-04-20")
	} else {
		req.Header.Set("X-Api-Key", cfg.APIKey)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("haiku request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("haiku returned status %d", resp.StatusCode)
	}

	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode haiku response: %w", err)
	}

	for _, block := range result.Content {
		if block.Type == "text" {
			return strings.TrimSpace(block.Text), nil
		}
	}
	return "", fmt.Errorf("no text content in haiku response")
}
