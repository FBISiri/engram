package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAI implements the Embedder interface using OpenAI's embedding API.
type OpenAI struct {
	apiKey    string
	model     string
	baseURL   string
	dimension int
	client    *http.Client
}

// OpenAIConfig configures the OpenAI embedder.
type OpenAIConfig struct {
	APIKey    string
	Model     string // default: text-embedding-3-small
	BaseURL   string // default: https://api.openai.com/v1
	Dimension int    // default: 1536
}

// NewOpenAI creates a new OpenAI embedder.
func NewOpenAI(cfg OpenAIConfig) *OpenAI {
	if cfg.Model == "" {
		cfg.Model = "text-embedding-3-small"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.Dimension == 0 {
		cfg.Dimension = 1536
	}
	return &OpenAI{
		apiKey:    cfg.APIKey,
		model:     cfg.Model,
		baseURL:   cfg.BaseURL,
		dimension: cfg.Dimension,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// embeddingRequest is the request body for the OpenAI embeddings API.
type embeddingRequest struct {
	Input      []string `json:"input"`
	Model      string   `json:"model"`
	Dimensions int      `json:"dimensions,omitempty"`
}

// embeddingResponse is the response from the OpenAI embeddings API.
type embeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// embeddingError is an error response from the OpenAI API.
type embeddingError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

// Embed returns the embedding vector for a single text input.
func (o *OpenAI) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := o.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("openai: empty embedding response")
	}
	return vecs[0], nil
}

// EmbedBatch returns embedding vectors for multiple texts.
// The OpenAI API supports batch embedding natively.
func (o *OpenAI) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody := embeddingRequest{
		Input: texts,
		Model: o.model,
	}
	// Only send dimensions for models that natively support it (text-embedding-3-*).
	// When using a proxy like OpenRouter, the model name has a provider prefix
	// (e.g. "openai/text-embedding-3-small"), and the dimensions parameter may
	// not be forwarded correctly, resulting in empty embeddings.
	modelSuffix := o.model
	if idx := strings.LastIndex(o.model, "/"); idx >= 0 {
		modelSuffix = o.model[idx+1:]
	}
	if o.dimension > 0 && strings.HasPrefix(modelSuffix, "text-embedding-3-") {
		reqBody.Dimensions = o.dimension
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	url := o.baseURL + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: send request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiErr embeddingError
		if json.Unmarshal(respBody, &apiErr) == nil && apiErr.Error.Message != "" {
			return nil, fmt.Errorf("openai: API error (HTTP %d): %s [%s]",
				resp.StatusCode, apiErr.Error.Message, apiErr.Error.Type)
		}
		return nil, fmt.Errorf("openai: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var embResp embeddingResponse
	if err := json.Unmarshal(respBody, &embResp); err != nil {
		return nil, fmt.Errorf("openai: unmarshal response: %w", err)
	}

	if len(embResp.Data) != len(texts) {
		return nil, fmt.Errorf("openai: expected %d embeddings, got %d", len(texts), len(embResp.Data))
	}

	// The API may return embeddings out of order, so sort by index.
	vectors := make([][]float32, len(texts))
	for _, d := range embResp.Data {
		if d.Index < 0 || d.Index >= len(texts) {
			return nil, fmt.Errorf("openai: embedding index %d out of range [0, %d)", d.Index, len(texts))
		}
		vectors[d.Index] = d.Embedding
	}

	// Validate all slots were filled and non-empty.
	for i, v := range vectors {
		if len(v) == 0 {
			return nil, fmt.Errorf("openai: missing or empty embedding at index %d", i)
		}
	}

	return vectors, nil
}

// Dimension returns the dimensionality of the embedding vectors.
func (o *OpenAI) Dimension() int {
	return o.dimension
}
