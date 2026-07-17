package qdrant

import (
	"sort"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// buildTestMultiStore builds a MultiStore with one empty *Store per name. The
// targeting logic only reads map keys and pointer identity, so no Qdrant
// connection is needed. Returns the store and a pointer→name reverse map.
func buildTestMultiStore(names ...string) (*MultiStore, map[*Store]string) {
	stores := make(map[string]*Store, len(names))
	rev := make(map[*Store]string, len(names))
	for _, n := range names {
		s := &Store{}
		stores[n] = s
		rev[s] = n
	}
	return NewMultiStore(stores, "engram_user"), rev
}

func targetNames(targets []*Store, rev map[*Store]string) []string {
	out := make([]string, 0, len(targets))
	for _, s := range targets {
		out = append(out, rev[s])
	}
	sort.Strings(out)
	return out
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// (a) default (unfiltered) fan-out must EXCLUDE isolated collections.
func TestDefaultFanOutExcludesIsolated(t *testing.T) {
	m, rev := buildTestMultiStore("engram_user", "engram_agent_self", "engram_reflection", "engram_pigo")
	m.SetIsolatedCollections("engram_pigo")

	want := []string{"engram_agent_self", "engram_reflection", "engram_user"}

	// searchTargets(nil) — no filter → default fan-out.
	if got := targetNames(m.searchTargets(nil), rev); !equalStrings(got, want) {
		t.Errorf("searchTargets(nil) = %v, want %v", got, want)
	}
	// defaultTargets directly.
	if got := targetNames(m.defaultTargets(), rev); !equalStrings(got, want) {
		t.Errorf("defaultTargets() = %v, want %v", got, want)
	}
	for _, s := range m.searchTargets(nil) {
		if rev[s] == "engram_pigo" {
			t.Fatal("searchTargets(nil) leaked isolated engram_pigo")
		}
	}
}

// (a) Scroll fan-out uses the same default target set (excludes isolated).
func TestScrollFanOutExcludesIsolated(t *testing.T) {
	m, rev := buildTestMultiStore("engram_user", "engram_agent_self", "engram_reflection", "engram_pigo")
	m.SetIsolatedCollections("engram_pigo")

	// The Scroll fan-out branch targets defaultTargets(); assert it directly.
	want := []string{"engram_agent_self", "engram_reflection", "engram_user"}
	if got := targetNames(m.defaultTargets(), rev); !equalStrings(got, want) {
		t.Errorf("Scroll fan-out targets = %v, want %v", got, want)
	}
}

// (b) explicit collection filter on engram_pigo selects ONLY that store.
func TestExplicitFilterSelectsIsolatedOnly(t *testing.T) {
	m, rev := buildTestMultiStore("engram_user", "engram_agent_self", "engram_reflection", "engram_pigo")
	m.SetIsolatedCollections("engram_pigo")

	filters := []memory.Filter{{Field: "collection", Op: memory.OpIn, Value: []string{"engram_pigo"}}}
	got := targetNames(m.searchTargets(filters), rev)
	if !equalStrings(got, []string{"engram_pigo"}) {
		t.Errorf("searchTargets(pigo filter) = %v, want [engram_pigo]", got)
	}
}

// Without any isolated registration, default fan-out includes every store
// (back-compat with existing callers).
func TestDefaultFanOutNoIsolated(t *testing.T) {
	m, rev := buildTestMultiStore("engram_user", "engram_agent_self", "engram_reflection", "engram_pigo")
	want := []string{"engram_agent_self", "engram_pigo", "engram_reflection", "engram_user"}
	if got := targetNames(m.searchTargets(nil), rev); !equalStrings(got, want) {
		t.Errorf("searchTargets(nil) without isolation = %v, want %v", got, want)
	}
}
