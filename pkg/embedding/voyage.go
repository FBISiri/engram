package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Voyage implements the Embedder interface using Voyage AI's embedding API.
// Voyage AI provides high-quality embeddings optimized for retrieval tasks.
type Voyage struct {
	apiKey    string
	model     string
	baseURL   string
	dimension int
	client    *http.Client
}

// VoyageConfig configures the Voyage embedder.
type VoyageConfig struct {
	APIKey    string
	Model     string // default: voyage-3
	BaseURL   string // default: https://api.voyageai.com/v1
	Dimension int    // default: 1024
}

// NewVoyage creates a new Voyage AI embedder.
func NewVoyage(cfg VoyageConfig) *Voyage {
	if cfg.Model == "" {
		cfg.Model = "voyage-3"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.voyageai.com/v1"
	}
	if cfg.Dimension == 0 {
		cfg.Dimension = 1024
	}
	return &Voyage{
		apiKey:    cfg.APIKey,
		model:     cfg.Model,
		baseURL:   cfg.BaseURL,
		dimension: cfg.Dimension,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// voyageEmbeddingRequest is the request body for the Voyage embeddings API.
type voyageEmbeddingRequest struct {
	Input     []string `json:"input"`
	Model     string   `json:"model"`
	InputType string   `json:"input_type,omitempty"` // "query" or "document"
}

// voyageEmbeddingResponse is the response from the Voyage embeddings API.
type voyageEmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// voyageError is an error response from the Voyage API.
type voyageError struct {
	Detail string `json:"detail"`
}

// Embed returns the embedding vector for a single text input.
func (v *Voyage) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := v.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("voyage: empty embedding response")
	}
	return vecs[0], nil
}

// EmbedBatch returns embedding vectors for multiple texts.
func (v *Voyage) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody := voyageEmbeddingRequest{
		Input: texts,
		Model: v.model,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("voyage: marshal request: %w", err)
	}

	url := v.baseURL + "/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("voyage: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+v.apiKey)

	resp, err := v.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("voyage: send request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("voyage: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiErr voyageError
		if json.Unmarshal(respBody, &apiErr) == nil && apiErr.Detail != "" {
			return nil, fmt.Errorf("voyage: API error (HTTP %d): %s", resp.StatusCode, apiErr.Detail)
		}
		return nil, fmt.Errorf("voyage: unexpected status %d: %s", resp.StatusCode, string(respBody))
	}

	var embResp voyageEmbeddingResponse
	if err := json.Unmarshal(respBody, &embResp); err != nil {
		return nil, fmt.Errorf("voyage: unmarshal response: %w", err)
	}

	if len(embResp.Data) != len(texts) {
		return nil, fmt.Errorf("voyage: expected %d embeddings, got %d", len(texts), len(embResp.Data))
	}

	// Sort by index (API may return out of order).
	vectors := make([][]float32, len(texts))
	for _, d := range embResp.Data {
		if d.Index < 0 || d.Index >= len(texts) {
			return nil, fmt.Errorf("voyage: embedding index %d out of range [0, %d)", d.Index, len(texts))
		}
		vectors[d.Index] = d.Embedding
	}

	for i, vec := range vectors {
		if vec == nil {
			return nil, fmt.Errorf("voyage: missing embedding at index %d", i)
		}
	}

	return vectors, nil
}

// Dimension returns the dimensionality of the embedding vectors.
func (v *Voyage) Dimension() int {
	return v.dimension
}
