package embedding

import (
	"context"

	"github.com/FBISiri/engram/pkg/memory"
)

// CachingEmbedder wraps an Embedder with an EmbedCache.
// All Embed and EmbedBatch calls check the cache before hitting the underlying
// embedder API, reducing cost for repeated texts.
type CachingEmbedder struct {
	inner    Embedder
	cache    memory.EmbedCache
	modelKey string
}

// NewCachingEmbedder returns a CachingEmbedder wrapping inner.
// modelKey should encode the model name and version (e.g. "voyage-3/v1");
// changing it invalidates all existing cache entries for this instance.
func NewCachingEmbedder(inner Embedder, cache memory.EmbedCache, modelKey string) *CachingEmbedder {
	return &CachingEmbedder{inner: inner, cache: cache, modelKey: modelKey}
}

func (c *CachingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	key := memory.ContentHash(text, c.modelKey)
	if vec, ok := c.cache.Get(key); ok {
		return vec, nil
	}
	vec, err := c.inner.Embed(ctx, text)
	if err != nil {
		return nil, err
	}
	c.cache.Put(key, vec)
	return vec, nil
}

func (c *CachingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	var missIdx []int
	var missTexts []string

	for i, text := range texts {
		key := memory.ContentHash(text, c.modelKey)
		if vec, ok := c.cache.Get(key); ok {
			results[i] = vec
		} else {
			missIdx = append(missIdx, i)
			missTexts = append(missTexts, text)
		}
	}

	if len(missTexts) == 0 {
		return results, nil
	}

	vecs, err := c.inner.EmbedBatch(ctx, missTexts)
	if err != nil {
		return nil, err
	}

	for j, idx := range missIdx {
		results[idx] = vecs[j]
		key := memory.ContentHash(texts[idx], c.modelKey)
		c.cache.Put(key, vecs[j])
	}

	return results, nil
}

func (c *CachingEmbedder) Dimension() int {
	return c.inner.Dimension()
}
