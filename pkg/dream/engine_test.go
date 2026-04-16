package dream

import (
	"context"
	"fmt"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// fakeStore is a minimal memory.Store implementation for testing scrollAll
// pagination behaviour. Only Scroll is exercised.
type fakeStore struct {
	all       []memory.Memory
	callCount int
}

func (f *fakeStore) Insert(ctx context.Context, mem *memory.Memory, vector []float32) error {
	return nil
}
func (f *fakeStore) Search(ctx context.Context, vector []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return nil, nil
}
func (f *fakeStore) SearchByIDs(ctx context.Context, ids []string) ([]memory.Memory, error) {
	return nil, nil
}
func (f *fakeStore) Delete(ctx context.Context, ids []string) (int, error) { return 0, nil }
func (f *fakeStore) Update(ctx context.Context, id string, fields map[string]any) error {
	return nil
}
func (f *fakeStore) DeleteExpired(ctx context.Context) (int, error) { return 0, nil }
func (f *fakeStore) EnsureCollection(ctx context.Context) error     { return nil }
func (f *fakeStore) Stats(ctx context.Context) (*memory.CollectionStats, error) {
	return &memory.CollectionStats{PointCount: uint64(len(f.all))}, nil
}

// Scroll honours opts.Limit as page size, and uses opts.Offset as the starting
// ID (linear scan for test simplicity). Returns the last page's tail ID as
// nextOffset when the page is full.
func (f *fakeStore) Scroll(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	f.callCount++
	limit := opts.Limit
	if limit == 0 {
		limit = 50
	}

	start := 0
	if opts.Offset != "" {
		for i, m := range f.all {
			if m.ID == opts.Offset {
				start = i + 1
				break
			}
		}
	}

	end := start + limit
	if end > len(f.all) {
		end = len(f.all)
	}
	page := f.all[start:end]

	var nextOffset string
	if len(page) == limit && end < len(f.all) {
		nextOffset = page[len(page)-1].ID
	}
	return page, nextOffset, nil
}

func buildFakeStore(n int) *fakeStore {
	mems := make([]memory.Memory, n)
	for i := 0; i < n; i++ {
		mems[i] = memory.Memory{ID: fmt.Sprintf("id-%04d", i)}
	}
	return &fakeStore{all: mems}
}

func TestScrollAll_SinglePage(t *testing.T) {
	store := buildFakeStore(50)
	got, err := scrollAll(context.Background(), store, memory.ScrollOptions{Limit: 100}, 0)
	if err != nil {
		t.Fatalf("scrollAll: %v", err)
	}
	if len(got) != 50 {
		t.Errorf("expected 50 memories, got %d", len(got))
	}
	if store.callCount != 1 {
		t.Errorf("expected 1 scroll call, got %d", store.callCount)
	}
}

func TestScrollAll_MultiPage(t *testing.T) {
	store := buildFakeStore(750) // W16 is at 783+, simulates pagination need
	got, err := scrollAll(context.Background(), store, memory.ScrollOptions{Limit: 200}, 0)
	if err != nil {
		t.Fatalf("scrollAll: %v", err)
	}
	if len(got) != 750 {
		t.Errorf("expected 750 memories, got %d", len(got))
	}
	// 750 / 200 = 4 full pages (200+200+200+150), so at least 4 calls.
	if store.callCount < 4 {
		t.Errorf("expected at least 4 scroll calls for pagination, got %d", store.callCount)
	}
}

func TestScrollAll_MaxTotal(t *testing.T) {
	store := buildFakeStore(1000)
	got, err := scrollAll(context.Background(), store, memory.ScrollOptions{Limit: 200}, 500)
	if err != nil {
		t.Fatalf("scrollAll: %v", err)
	}
	if len(got) != 500 {
		t.Errorf("maxTotal cap broken: expected 500, got %d", len(got))
	}
}

func TestScrollAll_EmptyStore(t *testing.T) {
	store := buildFakeStore(0)
	got, err := scrollAll(context.Background(), store, memory.ScrollOptions{Limit: 100}, 0)
	if err != nil {
		t.Fatalf("scrollAll: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected 0 memories, got %d", len(got))
	}
	if store.callCount != 1 {
		t.Errorf("expected 1 scroll call even for empty store, got %d", store.callCount)
	}
}

func TestScrollAll_DefaultLimit(t *testing.T) {
	store := buildFakeStore(300)
	got, err := scrollAll(context.Background(), store, memory.ScrollOptions{}, 0)
	if err != nil {
		t.Fatalf("scrollAll: %v", err)
	}
	if len(got) != 300 {
		t.Errorf("expected 300 memories, got %d", len(got))
	}
	// Default page size is 200, so 300 items should need exactly 2 calls.
	if store.callCount != 2 {
		t.Errorf("expected 2 scroll calls (300/200), got %d", store.callCount)
	}
}
