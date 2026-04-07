package embedding

import (
	"context"
	"fmt"
)

// OpenAI implements the Embedder interface using OpenAI's embedding API.
type OpenAI struct {
	apiKey    string
	model     string
	baseURL   string
	dimension int
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
	}
}

func (o *OpenAI) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := o.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

func (o *OpenAI) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	// TODO: Implement OpenAI embedding API call
	// Use github.com/sashabaranov/go-openai or direct HTTP
	return nil, fmt.Errorf("OpenAI embedding not yet implemented")
}

func (o *OpenAI) Dimension() int {
	return o.dimension
}
