package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
)

// dupMultiStore is an in-memory stand-in for qdrant.MultiStore: one map per
// physical collection, Insert routed by mem.Collection (empty → engram_user,
// unknown → error), and by-id reads/updates in the same fixed order as
// MultiStore.idLookupOrder. It implements the optional targetedUpdater and
// scopedIDReader capabilities the server type-asserts for.
type dupMultiStore struct {
	mu   sync.Mutex
	cols map[string]map[string]memory.Memory

	targetedUpdates []string // collections passed to UpdateInCollection
	failCol         string   // UpdateInCollection on this collection errors
	fanoutUpdates   int
}

var dupOrder = []string{
	collection.CollectionUser,
	collection.CollectionAgentSelf,
	collection.CollectionReflection,
	collection.CollectionPigo,
}

func newDupMultiStore() *dupMultiStore {
	d := &dupMultiStore{cols: map[string]map[string]memory.Memory{}}
	for _, c := range dupOrder {
		d.cols[c] = map[string]memory.Memory{}
	}
	return d
}

func (d *dupMultiStore) put(col string, m memory.Memory) {
	d.mu.Lock()
	defer d.mu.Unlock()
	m.Collection = col
	d.cols[col][m.ID] = m
}

func (d *dupMultiStore) get(col, id string) (memory.Memory, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	m, ok := d.cols[col][id]
	return m, ok
}

func (d *dupMultiStore) Insert(_ context.Context, mem *memory.Memory, _ []float32) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	col := mem.Collection
	if col == "" {
		col = collection.CollectionUser
	}
	c, ok := d.cols[col]
	if !ok {
		return fmt.Errorf("unknown collection %q", col)
	}
	mem.Collection = col
	c[mem.ID] = *mem
	return nil
}

func (d *dupMultiStore) Search(context.Context, []float32, memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return nil, nil
}

func (d *dupMultiStore) Delete(_ context.Context, ids []string) (int, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for _, c := range d.cols {
		for _, id := range ids {
			delete(c, id)
		}
	}
	return len(ids), nil
}

func applyFields(m *memory.Memory, fields map[string]any) {
	if v, ok := fields["lifecycle_status"].(string); ok {
		m.LifecycleStatus = v
	}
	if v, ok := fields["archived_at"].(float64); ok {
		m.ArchivedAt = v
	}
	if v, ok := fields["updated_at"].(float64); ok {
		m.UpdatedAt = v
	}
	if v, ok := fields["importance"].(float64); ok {
		m.Importance = v
	}
}

func (d *dupMultiStore) Update(_ context.Context, id string, fields map[string]any) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.fanoutUpdates++
	for _, col := range dupOrder {
		if m, ok := d.cols[col][id]; ok {
			applyFields(&m, fields)
			d.cols[col][id] = m
			return nil
		}
	}
	return nil
}

func (d *dupMultiStore) UpdateInCollection(_ context.Context, id string, fields map[string]any, col string) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.targetedUpdates = append(d.targetedUpdates, col)
	if col == d.failCol {
		return fmt.Errorf("injected failure in %s", col)
	}
	m, ok := d.cols[col][id]
	if !ok {
		return fmt.Errorf("not found in %s", col)
	}
	applyFields(&m, fields)
	d.cols[col][id] = m
	return nil
}

func (d *dupMultiStore) SearchByIDs(_ context.Context, ids []string) ([]memory.Memory, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	var out []memory.Memory
	for _, col := range dupOrder {
		for _, id := range ids {
			if m, ok := d.cols[col][id]; ok {
				out = append(out, m)
			}
		}
	}
	return out, nil
}

func (d *dupMultiStore) SearchByIDsIn(_ context.Context, ids []string, col string) ([]memory.Memory, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	c, ok := d.cols[col]
	if !ok {
		return nil, fmt.Errorf("unknown collection %q", col)
	}
	var out []memory.Memory
	for _, id := range ids {
		if m, ok := c[id]; ok {
			out = append(out, m)
		}
	}
	return out, nil
}

func (d *dupMultiStore) EnsureCollection(context.Context) error { return nil }
func (d *dupMultiStore) Scroll(context.Context, memory.ScrollOptions) ([]memory.Memory, string, error) {
	return nil, "", nil
}
func (d *dupMultiStore) Stats(context.Context) (*memory.CollectionStats, error) {
	return &memory.CollectionStats{Status: "green"}, nil
}
func (d *dupMultiStore) DeleteExpired(context.Context) (int, error) { return 0, nil }

func newDupServer(t *testing.T) (*Server, *dupMultiStore) {
	t.Helper()
	collection.DefaultRegistry.Init()
	store := newDupMultiStore()
	cfg := &config.Config{
		Weights:        memory.DefaultScoringWeights(),
		Decay:          memory.DefaultDecayConfig(),
		MMRLambda:      0.5,
		DedupThreshold: 0.92,
	}
	return NewServer(store, newMockEmbedder(), cfg), store
}

func dupSeed(store *dupMultiStore, col, id, content string) {
	m := memory.New(content, memory.WithType(memory.TypeEvent))
	m.ID = id
	m.LifecycleStatus = memory.LifecycleActive
	store.put(col, *m)
}

// assertInPlace checks the D1 invariant: agent_self holds "B", user has no copy.
func assertInPlace(t *testing.T, store *dupMultiStore, id string) {
	t.Helper()
	got, ok := store.get(collection.CollectionAgentSelf, id)
	if !ok || got.Content != "B" {
		t.Fatalf("agent_self copy = %+v (ok=%v), want content B", got, ok)
	}
	if u, ok := store.get(collection.CollectionUser, id); ok {
		t.Fatalf("engram_user must have NO copy of %s, got %+v", id, u)
	}
}

// ── D1: every by-id update entry path replaces in place ─────────────────────

func TestDupUUID_UpdateByID_MCP_InPlace(t *testing.T) {
	for _, ct := range []string{"", "user", "agent-self"} {
		t.Run("caller="+ct, func(t *testing.T) {
			srv, store := newDupServer(t)
			dupSeed(store, collection.CollectionAgentSelf, "X", "A")
			ctx := context.Background()
			if ct != "" {
				ctx = WithCallerType(ctx, ct)
			}
			res, err := srv.handleUpdateByID(ctx, toolReq(map[string]any{"id": "X", "content": "B"}))
			if err != nil || res.IsError {
				t.Fatalf("update_by_id: err=%v res=%s", err, extractText(res))
			}
			assertInPlace(t, store, "X")
		})
	}
}

func TestDupUUID_MemoryUpdateWithID_MCP_InPlace(t *testing.T) {
	srv, store := newDupServer(t)
	dupSeed(store, collection.CollectionAgentSelf, "X", "A")
	res, err := srv.handleUpdate(context.Background(), toolReq(map[string]any{"id": "X", "new_content": "B"}))
	if err != nil || res.IsError {
		t.Fatalf("memory_update(id=): err=%v res=%s", err, extractText(res))
	}
	assertInPlace(t, store, "X")
}

func TestDupUUID_RESTPut_InPlace(t *testing.T) {
	cases := []struct{ name, path, caller string }{
		{"PUT /memories/{id}", "/memories/X", "user"},
		{"PUT /collections/{name}/memories/{id}", "/collections/engram_agent_self/memories/X", "agent-self"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, store := newDupServer(t)
			dupSeed(store, collection.CollectionAgentSelf, "X", "A")
			ts := httptest.NewServer(NewHTTPServer(srv, 0, "").Handler())
			t.Cleanup(ts.Close)
			resp := doJSON(t, ts.URL, "PUT", tc.path, tc.caller, `{"content":"B"}`)
			_ = resp.Body.Close()
			if resp.StatusCode != 200 {
				t.Fatalf("status %d", resp.StatusCode)
			}
			assertInPlace(t, store, "X")
		})
	}
}

// ── D2: deterministic copy selection + delete archives every copy ───────────

func seedDup(store *dupMultiStore, id string) {
	dupSeed(store, collection.CollectionUser, id, "user copy")
	dupSeed(store, collection.CollectionAgentSelf, id, "agent copy")
}

func TestDupUUID_DeleteByID_ArchivesEveryCopy(t *testing.T) {
	srv, store := newDupServer(t)
	seedDup(store, "D")
	if _, err := srv.deleteByID(context.Background(), "D"); err != nil {
		t.Fatalf("deleteByID: %v", err)
	}
	for _, col := range []string{collection.CollectionUser, collection.CollectionAgentSelf} {
		m, _ := store.get(col, "D")
		if m.LifecycleStatus != memory.LifecycleArchived {
			t.Fatalf("%s copy not archived: %q", col, m.LifecycleStatus)
		}
	}
	sort.Strings(store.targetedUpdates)
	want := []string{collection.CollectionAgentSelf, collection.CollectionUser}
	if fmt.Sprint(store.targetedUpdates) != fmt.Sprint(want) || store.fanoutUpdates != 0 {
		t.Fatalf("targeted=%v fanout=%d, want targeted %v and no fan-out", store.targetedUpdates, store.fanoutUpdates, want)
	}
}

func TestDupUUID_RESTPatchAndReset_TargetFirstCopy(t *testing.T) {
	srv, store := newDupServer(t)
	seedDup(store, "P")
	ts := httptest.NewServer(NewHTTPServer(srv, 0, "").Handler())
	t.Cleanup(ts.Close)

	resp := doJSON(t, ts.URL, "PATCH", "/memories/P", "user", `{"lifecycle_status":"archived"}`)
	_ = resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("PATCH status %d", resp.StatusCode)
	}
	resp = doJSON(t, ts.URL, "POST", "/memories/P/reset", "user", ``)
	_ = resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("reset status %d", resp.StatusCode)
	}
	want := []string{collection.CollectionUser, collection.CollectionUser}
	if fmt.Sprint(store.targetedUpdates) != fmt.Sprint(want) || store.fanoutUpdates != 0 {
		t.Fatalf("targeted=%v fanout=%d, want %v", store.targetedUpdates, store.fanoutUpdates, want)
	}
	if m, _ := store.get(collection.CollectionAgentSelf, "P"); m.LifecycleStatus != memory.LifecycleActive {
		t.Fatalf("agent_self copy must be untouched, got %q", m.LifecycleStatus)
	}
}

// ── D3: memory_get_by_id collection / also_in / isolation ───────────────────

type getByIDResp struct {
	ID               string   `json:"id"`
	Content          string   `json:"content"`
	SourceCollection string   `json:"source_collection"`
	AlsoIn           []string `json:"also_in"`
}

func getByID(t *testing.T, srv *Server, ctx context.Context, args map[string]any) (*getByIDResp, bool) {
	t.Helper()
	res, err := srv.handleGetByID(ctx, toolReq(args))
	if err != nil {
		t.Fatalf("handleGetByID: %v", err)
	}
	if res.IsError {
		return nil, true
	}
	txt := extractText(res)
	if txt == "null" {
		return nil, false
	}
	var r getByIDResp
	if err := json.Unmarshal([]byte(txt), &r); err != nil {
		t.Fatalf("parse %s: %v", txt, err)
	}
	return &r, false
}

func TestGetByID_DefaultDeterministic_AlsoIn(t *testing.T) {
	srv, store := newDupServer(t)
	seedDup(store, "G")
	for i := 0; i < 100; i++ {
		r, _ := getByID(t, srv, context.Background(), map[string]any{"id": "G"})
		if r == nil || r.SourceCollection != collection.CollectionUser || r.Content != "user copy" {
			t.Fatalf("call %d: got %+v, want user copy", i, r)
		}
		if fmt.Sprint(r.AlsoIn) != "[engram_agent_self]" {
			t.Fatalf("also_in = %v", r.AlsoIn)
		}
	}
	// agent-self caller gets its own copy first.
	r, _ := getByID(t, srv, WithCallerType(context.Background(), "agent-self"), map[string]any{"id": "G"})
	if r == nil || r.SourceCollection != collection.CollectionAgentSelf || fmt.Sprint(r.AlsoIn) != "[engram_user]" {
		t.Fatalf("agent-self caller got %+v", r)
	}
}

func TestGetByID_CollectionParam(t *testing.T) {
	srv, store := newDupServer(t)
	seedDup(store, "G")
	dupSeed(store, collection.CollectionUser, "U", "only user")

	r, _ := getByID(t, srv, context.Background(), map[string]any{"id": "G", "collection": collection.CollectionAgentSelf})
	if r == nil || r.Content != "agent copy" || fmt.Sprint(r.AlsoIn) != "[engram_user]" {
		t.Fatalf("collection=agent_self got %+v", r)
	}
	if r, _ := getByID(t, srv, context.Background(), map[string]any{"id": "U", "collection": collection.CollectionAgentSelf}); r != nil {
		t.Fatalf("id absent in requested collection must be null, got %+v", r)
	}
	r, _ = getByID(t, srv, context.Background(), map[string]any{"id": "U"})
	if r == nil || r.AlsoIn != nil {
		t.Fatalf("single-copy id must omit also_in, got %+v", r)
	}
	if _, isErr := getByID(t, srv, context.Background(), map[string]any{"id": "G", "collection": "engram_nope"}); !isErr {
		t.Fatal("unknown collection must be an error result")
	}
}

func TestGetByID_IsolatedCaller(t *testing.T) {
	srv, store := newDupServer(t)
	seedDup(store, "G")
	dupSeed(store, collection.CollectionPigo, "G", "pigo copy")
	dupSeed(store, collection.CollectionUser, "U", "only user")
	pigo := WithCallerType(context.Background(), "pigo")

	if r, _ := getByID(t, srv, pigo, map[string]any{"id": "U", "collection": collection.CollectionUser}); r != nil {
		t.Fatalf("pigo passing engram_user must get null, got %+v", r)
	}
	if r, _ := getByID(t, srv, pigo, map[string]any{"id": "U"}); r != nil {
		t.Fatalf("pigo default on foreign id must get null, got %+v", r)
	}
	res, _ := srv.handleGetByID(pigo, toolReq(map[string]any{"id": "G"}))
	txt := extractText(res)
	var raw map[string]any
	if err := json.Unmarshal([]byte(txt), &raw); err != nil {
		t.Fatalf("parse %s: %v", txt, err)
	}
	if raw["content"] != "pigo copy" {
		t.Fatalf("pigo should read its own copy, got %v", raw["content"])
	}
	if _, leak := raw["also_in"]; leak {
		t.Fatalf("also_in leaked to isolated caller: %s", txt)
	}
	if r, _ := getByID(t, srv, pigo, map[string]any{"id": "G", "collection": collection.CollectionPigo}); r == nil || r.Content != "pigo copy" || r.AlsoIn != nil {
		t.Fatalf("pigo own collection explicit got %+v", r)
	}
}

// ── read/write same-copy selection (pickCopy) ───────────────────────────────

func TestDupUUID_AgentSelfPutOnDup_UpdatesAgentSelfCopy(t *testing.T) {
	srv, store := newDupServer(t)
	seedDup(store, "D")
	ts := httptest.NewServer(NewHTTPServer(srv, 0, "").Handler())
	t.Cleanup(ts.Close)
	resp := doJSON(t, ts.URL, "PUT", "/collections/engram_agent_self/memories/D", "agent-self", `{"content":"B"}`)
	_ = resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if m, _ := store.get(collection.CollectionAgentSelf, "D"); m.Content != "B" {
		t.Fatalf("agent_self copy = %q, want B", m.Content)
	}
	if m, _ := store.get(collection.CollectionUser, "D"); m.Content != "user copy" {
		t.Fatalf("user copy must be untouched, got %q", m.Content)
	}
}

func TestDupUUID_IsolatedCaller_OwnCopyWithStrayUserCopy(t *testing.T) {
	pigo := WithCallerType(context.Background(), "pigo")

	t.Run("update_by_id", func(t *testing.T) {
		srv, store := newDupServer(t)
		dupSeed(store, collection.CollectionUser, "P", "stray user")
		dupSeed(store, collection.CollectionPigo, "P", "pigo own")
		res, err := srv.handleUpdateByID(pigo, toolReq(map[string]any{"id": "P", "content": "pigo new"}))
		if err != nil || mustStatus(t, res) != "updated" {
			t.Fatalf("update: err=%v res=%s", err, extractText(res))
		}
		if m, _ := store.get(collection.CollectionPigo, "P"); m.Content != "pigo new" {
			t.Fatalf("pigo copy = %q, want pigo new", m.Content)
		}
		if m, _ := store.get(collection.CollectionUser, "P"); m.Content != "stray user" {
			t.Fatalf("stray user copy touched: %q", m.Content)
		}
	})
	t.Run("delete_by_id", func(t *testing.T) {
		srv, store := newDupServer(t)
		dupSeed(store, collection.CollectionUser, "P", "stray user")
		dupSeed(store, collection.CollectionPigo, "P", "pigo own")
		res, err := srv.handleDeleteByID(pigo, toolReq(map[string]any{"id": "P"}))
		if err != nil || mustStatus(t, res) != "archived" {
			t.Fatalf("delete: err=%v res=%s", err, extractText(res))
		}
		if m, _ := store.get(collection.CollectionPigo, "P"); m.LifecycleStatus != memory.LifecycleArchived {
			t.Fatalf("pigo copy not archived: %q", m.LifecycleStatus)
		}
		if m, _ := store.get(collection.CollectionUser, "P"); m.LifecycleStatus != memory.LifecycleActive {
			t.Fatalf("stray user copy touched: %q", m.LifecycleStatus)
		}
		if fmt.Sprint(store.targetedUpdates) != "[engram_pigo]" {
			t.Fatalf("targeted updates = %v, want only engram_pigo", store.targetedUpdates)
		}
	})
}

func TestDupUUID_DeleteByID_TriesAllCopiesAndJoinsErrors(t *testing.T) {
	srv, store := newDupServer(t)
	seedDup(store, "E")
	store.failCol = collection.CollectionUser // first copy in lookup order fails
	_, err := srv.deleteByID(context.Background(), "E")
	if err == nil || !strings.Contains(err.Error(), "engram_user") {
		t.Fatalf("err = %v, want joined error naming engram_user", err)
	}
	if m, _ := store.get(collection.CollectionAgentSelf, "E"); m.LifecycleStatus != memory.LifecycleArchived {
		t.Fatalf("agent_self copy must still be archived after user failure, got %q", m.LifecycleStatus)
	}
}

func TestDupUUID_DeleteByID_NoEligibleCopy_NotFound(t *testing.T) {
	srv, store := newDupServer(t)
	dupSeed(store, collection.CollectionUser, "F", "foreign")
	// Call the core helper directly (bypassing byIDAllowed): an isolated caller
	// with no copy in its own collection archives nothing → not found, not 200.
	_, err := srv.deleteByID(WithCallerType(context.Background(), "pigo"), "F")
	if !errors.Is(err, errMemoryNotFound) {
		t.Fatalf("err = %v, want errMemoryNotFound", err)
	}
	if m, _ := store.get(collection.CollectionUser, "F"); m.LifecycleStatus != memory.LifecycleActive {
		t.Fatalf("foreign copy touched: %q", m.LifecycleStatus)
	}
	if len(store.targetedUpdates) != 0 || store.fanoutUpdates != 0 {
		t.Fatalf("unexpected writes: targeted=%v fanout=%d", store.targetedUpdates, store.fanoutUpdates)
	}
}
