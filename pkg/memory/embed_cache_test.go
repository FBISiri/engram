package memory

import (
	"testing"
)

func TestEmbedCache_HitMiss(t *testing.T) {
	c := NewEmbedCache(10)

	vec := []float32{0.1, 0.2, 0.3}
	key := ContentHash("hello", "model/v1")

	// Miss on empty cache
	if _, ok := c.Get(key); ok {
		t.Fatal("expected miss, got hit")
	}

	c.Put(key, vec)

	// Hit after put
	got, ok := c.Get(key)
	if !ok {
		t.Fatal("expected hit, got miss")
	}
	if len(got) != len(vec) || got[0] != vec[0] {
		t.Fatalf("got vector %v, want %v", got, vec)
	}

	hits, misses := c.Stats()
	if hits != 1 {
		t.Errorf("hits = %d, want 1", hits)
	}
	if misses != 1 {
		t.Errorf("misses = %d, want 1", misses)
	}
}

func TestEmbedCache_Eviction(t *testing.T) {
	c := NewEmbedCache(3)

	put := func(k, v string) {
		c.Put(ContentHash(k, "m"), []float32{float32(v[0])})
	}
	get := func(k string) bool {
		_, ok := c.Get(ContentHash(k, "m"))
		return ok
	}

	put("a", "a")
	put("b", "b")
	put("c", "c")

	// Access "a" to make it recently used
	get("a")

	// Adding "d" should evict LRU entry: "b" (oldest untouched after "a" was promoted)
	put("d", "d")

	if !get("a") {
		t.Error("a should still be in cache")
	}
	if !get("c") {
		t.Error("c should still be in cache")
	}
	if !get("d") {
		t.Error("d should be in cache")
	}
	if get("b") {
		t.Error("b should have been evicted")
	}
}

func TestContentHash_ModelIsolation(t *testing.T) {
	h1 := ContentHash("text", "voyage-3/v1")
	h2 := ContentHash("text", "voyage-3/v2")
	if h1 == h2 {
		t.Error("different model versions should produce different hashes")
	}
}

func TestEmbedCache_UpdateExisting(t *testing.T) {
	c := NewEmbedCache(5)
	key := ContentHash("x", "m")
	c.Put(key, []float32{1.0})
	c.Put(key, []float32{2.0})

	got, ok := c.Get(key)
	if !ok {
		t.Fatal("expected hit")
	}
	if got[0] != 2.0 {
		t.Errorf("got %v, want [2.0]", got)
	}
}
