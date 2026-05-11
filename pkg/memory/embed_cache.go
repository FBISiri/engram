package memory

import (
	"container/list"
	"crypto/sha256"
	"fmt"
	"sync"
	"sync/atomic"
)

// EmbedCache caches content hash → embedding vector pairs.
// Embeddings are deterministic (same content + model always produces the same vector),
// so entries never expire.
type EmbedCache interface {
	Get(contentHash string) ([]float32, bool)
	Put(contentHash string, vector []float32)
	// Stats returns cumulative hit and miss counts since creation.
	Stats() (hits, misses int64)
}

// ContentHash computes a 32-hex-char cache key for an embedding request.
// modelKey should encode the model name and version (e.g. "voyage-3/v1") so
// that model upgrades automatically invalidate stale cache entries.
func ContentHash(content, modelKey string) string {
	h := sha256.Sum256([]byte(content + "\x00" + modelKey))
	return fmt.Sprintf("%x", h[:16])
}

// NewEmbedCache returns an in-memory LRU embed cache.
// capacity is the max number of entries; 0 defaults to 10000 (~40 MB at 1024-dim float32).
func NewEmbedCache(capacity int) EmbedCache {
	if capacity <= 0 {
		capacity = 10000
	}
	return &lruCache{
		capacity: capacity,
		items:    make(map[string]*list.Element, capacity),
		order:    list.New(),
	}
}

type lruEntry struct {
	key    string
	vector []float32
}

type lruCache struct {
	mu       sync.Mutex
	capacity int
	items    map[string]*list.Element
	order    *list.List
	hits     atomic.Int64
	misses   atomic.Int64
}

func (c *lruCache) Get(contentHash string) ([]float32, bool) {
	c.mu.Lock()
	el, ok := c.items[contentHash]
	if ok {
		c.order.MoveToFront(el)
		vec := el.Value.(*lruEntry).vector
		c.mu.Unlock()
		c.hits.Add(1)
		return vec, true
	}
	c.mu.Unlock()
	c.misses.Add(1)
	return nil, false
}

func (c *lruCache) Put(contentHash string, vector []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[contentHash]; ok {
		c.order.MoveToFront(el)
		el.Value.(*lruEntry).vector = vector
		return
	}
	if c.order.Len() >= c.capacity {
		oldest := c.order.Back()
		if oldest != nil {
			c.order.Remove(oldest)
			delete(c.items, oldest.Value.(*lruEntry).key)
		}
	}
	el := c.order.PushFront(&lruEntry{key: contentHash, vector: vector})
	c.items[contentHash] = el
}

func (c *lruCache) Stats() (hits, misses int64) {
	return c.hits.Load(), c.misses.Load()
}
