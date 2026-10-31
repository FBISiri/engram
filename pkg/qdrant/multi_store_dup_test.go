package qdrant

import (
	"context"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// idLookupOrder is the fixed by-id precedence: user, agent_self, reflection,
// then remaining stores by name — identical on every call (no map order).
func TestIDLookupOrder_DeterministicPrecedence(t *testing.T) {
	m, _ := buildTestMultiStore("engram_pigo", "engram_reflection", "engram_zzz", "engram_agent_self", "engram_user", "engram_aaa")
	want := []string{"engram_user", "engram_agent_self", "engram_reflection", "engram_aaa", "engram_pigo", "engram_zzz"}
	for i := 0; i < 100; i++ {
		if got := m.idLookupOrder(); !equalStrings(got, want) {
			t.Fatalf("call %d: idLookupOrder() = %v, want %v", i, got, want)
		}
	}
}

func TestIDLookupOrder_MissingPrecedenceStoresSkipped(t *testing.T) {
	m, _ := buildTestMultiStore("engram_agent_self", "engram_pigo")
	want := []string{"engram_agent_self", "engram_pigo"}
	if got := m.idLookupOrder(); !equalStrings(got, want) {
		t.Fatalf("idLookupOrder() = %v, want %v", got, want)
	}
}

// Insert honors mem.Collection: a known collection routes to its own store;
// empty or unknown falls back to defaultCol (original behaviour, kept so e.g.
// eval restore never drops a record).
func TestInsertTarget_HonorsCollection(t *testing.T) {
	m, rev := buildTestMultiStore("engram_user", "engram_agent_self", "engram_reflection")

	col, s := m.insertTarget("engram_agent_self")
	if col != "engram_agent_self" || rev[s] != "engram_agent_self" {
		t.Fatalf("insertTarget(agent_self) = %q,%q", col, rev[s])
	}
	col, s = m.insertTarget("")
	if col != "engram_user" || rev[s] != "engram_user" {
		t.Fatalf("insertTarget(\"\") = %q,%q", col, rev[s])
	}
	col, s = m.insertTarget("engram_bogus")
	if col != "engram_user" || rev[s] != "engram_user" {
		t.Fatalf("insertTarget(unknown) = %q,%q, want default fallback", col, rev[s])
	}
}

func TestSearchByIDsIn_UnknownCollectionErrors(t *testing.T) {
	m, _ := buildTestMultiStore("engram_user")
	if _, err := m.SearchByIDsIn(context.Background(), []string{"x"}, "engram_bogus"); err == nil {
		t.Fatal("SearchByIDsIn(unknown) must error")
	}
}

func TestStampCollection_PhysicalWins(t *testing.T) {
	in := []memory.Memory{{ID: "a"}, {ID: "b", Collection: "engram_user"}}
	out := stampCollection(in, "engram_agent_self")
	for _, m := range out {
		if m.Collection != "engram_agent_self" {
			t.Fatalf("%s: Collection = %q, want physical engram_agent_self", m.ID, m.Collection)
		}
	}
}

// TestSearchByIDsAndUpdate_RealMethods_DeterministicOrder drives the REAL
// MultiStore.SearchByIDs / Update (per-store calls swapped via the test seam)
// with the same id in engram_user AND engram_agent_self. With `range m.stores`
// the visiting order is randomized per call, so 200 iterations would fail with
// overwhelming probability.
func TestSearchByIDsAndUpdate_RealMethods_DeterministicOrder(t *testing.T) {
	m, rev := buildTestMultiStore("engram_user", "engram_agent_self", "engram_reflection", "engram_pigo")
	holders := map[string]bool{"engram_user": true, "engram_agent_self": true}
	var visited []string

	origS, origU := storeSearchByIDs, storeUpdate
	t.Cleanup(func() { storeSearchByIDs, storeUpdate = origS, origU })
	storeSearchByIDs = func(_ context.Context, s *Store, ids []string) ([]memory.Memory, error) {
		name := rev[s]
		visited = append(visited, name)
		if !holders[name] {
			return nil, nil
		}
		return []memory.Memory{{ID: ids[0], Content: name}}, nil
	}
	storeUpdate = func(_ context.Context, s *Store, _ string, _ map[string]any) error {
		name := rev[s]
		visited = append(visited, name)
		if !holders[name] {
			return status.Error(codes.NotFound, "not found")
		}
		return nil
	}

	wantOrder := []string{"engram_user", "engram_agent_self", "engram_reflection", "engram_pigo"}
	for i := 0; i < 200; i++ {
		visited = nil
		got, err := m.SearchByIDs(context.Background(), []string{"dup"})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 2 || got[0].Content != "engram_user" || got[0].Collection != "engram_user" ||
			got[1].Collection != "engram_agent_self" {
			t.Fatalf("call %d: SearchByIDs = %+v, want [user, agent_self]", i, got)
		}
		if !equalStrings(visited, wantOrder) {
			t.Fatalf("call %d: SearchByIDs visit order = %v, want %v", i, visited, wantOrder)
		}

		visited = nil
		if err := m.Update(context.Background(), "dup", map[string]any{"x": 1}); err != nil {
			t.Fatal(err)
		}
		if !equalStrings(visited, []string{"engram_user"}) {
			t.Fatalf("call %d: Update visited %v, want only engram_user (first copy)", i, visited)
		}
	}

	// Update falls through the fixed order to the first holder.
	holders = map[string]bool{"engram_reflection": true}
	visited = nil
	if err := m.Update(context.Background(), "dup", map[string]any{"x": 1}); err != nil {
		t.Fatal(err)
	}
	if !equalStrings(visited, []string{"engram_user", "engram_agent_self", "engram_reflection"}) {
		t.Fatalf("Update fall-through order = %v", visited)
	}
}
