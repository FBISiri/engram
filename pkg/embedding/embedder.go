// Package embedding defines the Embedder interface and implementations.
package embedding

import "context"

// Embedder converts text into dense vector representations.
type Embedder interface {
	// Embed returns the embedding vector for a single text input.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch returns embedding vectors for multiple texts.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimension returns the dimensionality of the embedding vectors.
	Dimension() int
}
