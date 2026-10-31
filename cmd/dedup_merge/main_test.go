package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeQdrant is an in-memory Qdrant REST fake that records every request.
type fakeQdrant struct {
	mu        sync.Mutex
	cols      map[string]map[string]map[string]any // col → id(json) → point json obj
	calls     []string                             // "METHOD path"
	snapFail  string                               // collection whose snapshot POST fails
	snapHide  string                               // collection whose snapshot is not listed
	snaps     map[string][]string
	pageLimit int
	// hooks
	repeatOffset bool                // scroll always returns the same next offset
	onReRead     func(f *fakeQdrant) // runs (locked) on the first per-id re-read
	hookFired    bool
}

func newFake() *fakeQdrant {
	return &fakeQdrant{
		cols: map[string]map[string]map[string]any{userCol: {}, agentCol: {},
			"engram_reflection": {}, "engram_pigo": {}},
		snaps:     map[string][]string{},
		pageLimit: 1, // force pagination
	}
}

func (f *fakeQdrant) add(col, id string, vec []float64, payload map[string]any) {
	p := map[string]any{"id": id, "payload": payload}
	if vec != nil {
		p["vector"] = vec
	}
	b, _ := json.Marshal(p)
	var obj map[string]any
	_ = json.Unmarshal(b, &obj)
	idj, _ := json.Marshal(id)
	f.cols[col][string(idj)] = obj
}

func (f *fakeQdrant) get(col, id string) map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	idj, _ := json.Marshal(id)
	return f.cols[col][string(idj)]
}

func (f *fakeQdrant) writes() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var w []string
	for _, c := range f.calls {
		if !isRead(c) {
			w = append(w, c)
		}
	}
	return w
}

func isRead(call string) bool {
	return strings.HasSuffix(call, "/points/scroll") && strings.HasPrefix(call, "POST ") ||
		strings.HasPrefix(call, "GET ") && strings.HasSuffix(call, "/snapshots") ||
		strings.HasPrefix(call, "POST ") && strings.HasSuffix(call, "/points")
}

func (f *fakeQdrant) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, r.Method+" "+r.URL.Path)
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/collections/"), "/")
	col := parts[0]
	rest := strings.Join(parts[1:], "/")
	body, _ := io.ReadAll(r.Body)
	var req map[string]any
	_ = json.Unmarshal(body, &req)
	reply := func(v any) { _ = json.NewEncoder(w).Encode(map[string]any{"result": v, "status": "ok"}) }
	store := f.cols[col]
	strip := func(p map[string]any, withVec bool) map[string]any {
		out := map[string]any{"id": p["id"], "payload": p["payload"]}
		if withVec && p["vector"] != nil {
			out["vector"] = p["vector"]
		}
		return out
	}
	switch {
	case r.Method == "POST" && rest == "points/scroll":
		keys := sortedKeys(store)
		start := 0
		if off, ok := req["offset"]; ok {
			ob, _ := json.Marshal(off)
			for i, k := range keys {
				if k == string(ob) {
					start = i
				}
			}
		}
		withVec, _ := req["with_vector"].(bool)
		end := start + f.pageLimit
		var next any
		if f.repeatOffset {
			reply(map[string]any{"points": []any{}, "next_page_offset": "same"})
			return
		}
		if end < len(keys) {
			_ = json.Unmarshal([]byte(keys[end]), &next)
		} else {
			end = len(keys)
		}
		pts := []any{}
		for _, k := range keys[start:end] {
			pts = append(pts, strip(store[k], withVec))
		}
		reply(map[string]any{"points": pts, "next_page_offset": next})
	case r.Method == "POST" && rest == "points":
		if col == userCol && f.onReRead != nil && !f.hookFired {
			f.hookFired = true
			f.onReRead(f)
		}
		withVec, _ := req["with_vector"].(bool)
		pts := []any{}
		for _, id := range req["ids"].([]any) {
			ib, _ := json.Marshal(id)
			if p, ok := store[string(ib)]; ok {
				pts = append(pts, strip(p, withVec))
			}
		}
		reply(pts)
	case r.Method == "PUT" && rest == "points":
		for _, p := range req["points"].([]any) {
			pm := p.(map[string]any)
			ib, _ := json.Marshal(pm["id"])
			store[string(ib)] = pm
		}
		reply(map[string]any{"status": "completed"})
	case r.Method == "POST" && rest == "points/delete":
		for _, id := range req["points"].([]any) {
			ib, _ := json.Marshal(id)
			delete(store, string(ib))
		}
		reply(map[string]any{"status": "completed"})
	case r.Method == "POST" && rest == "snapshots":
		if f.snapFail == col {
			http.Error(w, `{"status":{"error":"boom"}}`, 500)
			return
		}
		name := fmt.Sprintf("%s-snap-%d.snapshot", col, len(f.snaps[col]))
		if f.snapHide != col {
			f.snaps[col] = append(f.snaps[col], name)
		}
		reply(map[string]any{"name": name})
	case r.Method == "GET" && rest == "snapshots":
		out := []any{}
		for _, n := range f.snaps[col] {
			out = append(out, map[string]any{"name": n})
		}
		reply(out)
	default:
		http.Error(w, "unexpected "+r.Method+" "+r.URL.Path, 404)
	}
}

func sortedKeys(m map[string]map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	for i := range keys {
		for j := i + 1; j < len(keys); j++ {
			if keys[j] < keys[i] {
				keys[i], keys[j] = keys[j], keys[i]
			}
		}
	}
	return keys
}

// seed builds a fixture:
//
//	ok-1      mergeable dup (user newer)
//	flag-new  agent_self newer → flagged
//	flag-type type differs → flagged
//	solo-u    only in user, solo-a only in agent_self (not dups)
func seed(f *fakeQdrant) {
	f.add(userCol, "ok-1", []float64{1, 0}, map[string]any{
		"content": "new user content", "type": "insight", "tags": []any{"u"}, "updated_at": 200, "created_at": 10,
		"access_count": 3, "last_accessed_at": 150, "lifecycle_status": "archived", "archived_at": 190,
		"collection": userCol, "source": "agent",
		"metadata": map[string]any{"source_type": "user_input", "provenance_history": []any{
			map[string]any{"source_type": "a", "merged_at": 1, "content_score": 0.9},
			map[string]any{"source_type": "b", "merged_at": 2, "content_score": 0.8},
		}},
	})
	f.add(agentCol, "ok-1", []float64{0, 1}, map[string]any{
		"content": "old agent content", "type": "insight", "tags": []any{"a"}, "updated_at": 100, "created_at": 10,
		"access_count": 5, "last_accessed_at": 180, "lifecycle_status": "active", "importance": 7,
		"collection": agentCol, "source": "agent", "reflected_at": 55,
		"metadata": map[string]any{"source_type": "reflection", "keep": "me", "provenance_history": []any{
			map[string]any{"source_type": "a", "merged_at": 1, "content_score": 0.9},
		}},
	})
	f.add(userCol, "flag-new", []float64{1}, map[string]any{"content": "u", "type": "event", "updated_at": 100})
	f.add(agentCol, "flag-new", []float64{1}, map[string]any{"content": "a", "type": "event", "updated_at": 300})
	f.add(userCol, "flag-type", []float64{1}, map[string]any{"content": "u", "type": "event", "updated_at": 300})
	f.add(agentCol, "flag-type", []float64{1}, map[string]any{"content": "a", "type": "insight", "updated_at": 100})
	f.add(userCol, "solo-u", []float64{1}, map[string]any{"content": "s", "type": "event", "updated_at": 1})
	f.add(agentCol, "solo-a", []float64{1}, map[string]any{"content": "s", "type": "event", "updated_at": 1})
}

func startFake(t *testing.T) (*fakeQdrant, *httptest.Server) {
	t.Helper()
	f := newFake()
	seed(f)
	ts := httptest.NewServer(f)
	t.Cleanup(ts.Close)
	return f, ts
}

func TestFlagsDefaultToDryRun(t *testing.T) {
	opts, err := parseFlags(nil, io.Discard)
	if err != nil || opts.execute {
		t.Fatalf("default: opts=%+v err=%v, want dry-run", opts, err)
	}
	if _, err := parseFlags([]string{"--execute"}, io.Discard); err == nil {
		t.Fatal("--execute alone must be a usage error")
	}
	if _, err := parseFlags([]string{"--dry-run=false"}, io.Discard); err == nil {
		t.Fatal("--dry-run=false alone must be a usage error")
	}
	if _, err := parseFlags([]string{"--execute", "--dry-run=true"}, io.Discard); err == nil {
		t.Fatal("--execute --dry-run=true must be a usage error")
	}
	opts, err = parseFlags([]string{"--execute", "--dry-run=false"}, io.Discard)
	if err != nil || !opts.execute {
		t.Fatalf("explicit execute: opts=%+v err=%v", opts, err)
	}
}

func TestRestURLFromConfig(t *testing.T) {
	cases := map[string]string{
		"localhost:6334":      "http://localhost:6333",
		"10.0.0.1:7000":       "http://10.0.0.1:7000",
		"http://q:6333":       "http://q:6333",
		"qdrant.example:6334": "http://qdrant.example:6333",
	}
	for in, want := range cases {
		if got := restURLFromConfig(in, false); got != want {
			t.Errorf("%s → %s, want %s", in, got, want)
		}
	}
	if got := restURLFromConfig("h:6334", true); got != "https://h:6333" {
		t.Errorf("tls → %s", got)
	}
}

func TestDryRun_ReadOnly_PlanAndCounts(t *testing.T) {
	f, ts := startFake(t)
	out := filepath.Join(t.TempDir(), "plan.jsonl")
	var stdout, stderr bytes.Buffer
	if code := realMain([]string{"--qdrant-url", ts.URL, "--out", out}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	if w := f.writes(); len(w) != 0 {
		t.Fatalf("dry-run issued non-read calls: %v", w)
	}
	for _, c := range f.calls {
		if !isRead(c) {
			t.Fatalf("non-read call %s", c)
		}
	}
	s := stdout.String()
	if !strings.Contains(s, `"dups":3`) || !strings.Contains(s, `"mergeable":1`) || !strings.Contains(s, `"flagged":2`) || !strings.Contains(s, `"merged":0`) {
		t.Fatalf("summary wrong:\n%s", s)
	}
	if !strings.Contains(s, "flag-new\tflagged\tagent_self_newer") || !strings.Contains(s, "flag-type\tflagged\ttype_differs") || !strings.Contains(s, "ok-1\tmerge") {
		t.Fatalf("plan lines wrong:\n%s", s)
	}
	data, err := os.ReadFile(out)
	if err != nil || strings.Count(string(data), "\n") != 3 {
		t.Fatalf("plan jsonl: %v %q", err, data)
	}
	// Both copies untouched.
	if f.get(userCol, "ok-1") == nil || f.get(agentCol, "ok-1")["payload"].(map[string]any)["content"] != "old agent content" {
		t.Fatal("dry-run mutated data")
	}
}

func TestExecute_RefusesWithoutSnapshot(t *testing.T) {
	for _, mode := range []string{"fail-user", "fail-agent", "hide-agent"} {
		t.Run(mode, func(t *testing.T) {
			f, ts := startFake(t)
			switch mode {
			case "fail-user":
				f.snapFail = userCol
			case "fail-agent":
				f.snapFail = agentCol
			case "hide-agent":
				f.snapHide = agentCol
			}
			var stdout, stderr bytes.Buffer
			code := realMain([]string{"--qdrant-url", ts.URL, "--execute", "--dry-run=false"}, &stdout, &stderr)
			if code == 0 {
				t.Fatal("execute must fail without a verified snapshot")
			}
			for _, c := range f.writes() {
				if !strings.HasSuffix(c, "/snapshots") {
					t.Fatalf("data write %q despite snapshot failure", c)
				}
			}
			if f.get(userCol, "ok-1") == nil {
				t.Fatal("user copy deleted despite snapshot failure")
			}
		})
	}
}

func TestExecute_OrderAndMergeRules(t *testing.T) {
	f, ts := startFake(t)
	var stdout, stderr bytes.Buffer
	if code := realMain([]string{"--qdrant-url", ts.URL, "--execute", "--dry-run=false"}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s\n%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), `"merged":1`) {
		t.Fatalf("summary: %s", stdout.String())
	}

	want := []string{
		"POST /collections/engram_user/snapshots",
		"GET /collections/engram_user/snapshots",
		"POST /collections/engram_agent_self/snapshots",
		"GET /collections/engram_agent_self/snapshots",
		"POST /collections/engram_user/points",
		"POST /collections/engram_agent_self/points",
		"PUT /collections/engram_agent_self/points",
		"POST /collections/engram_agent_self/points",
		"POST /collections/engram_user/points/delete",
	}
	var got []string
	for _, c := range f.calls {
		if !strings.HasSuffix(c, "/points/scroll") {
			got = append(got, c)
		}
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("call order:\n%s\nwant:\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
	for _, c := range f.calls {
		if strings.Contains(c, agentCol) && strings.HasSuffix(c, "/delete") {
			t.Fatalf("must never delete from agent_self: %s", c)
		}
	}

	// user copy of ok-1 gone; flagged dups untouched in both.
	if f.get(userCol, "ok-1") != nil {
		t.Fatal("user copy of merged id should be deleted")
	}
	for _, id := range []string{"flag-new", "flag-type"} {
		if f.get(userCol, id) == nil || f.get(agentCol, id) == nil {
			t.Fatalf("flagged %s must not be touched", id)
		}
	}

	m := f.get(agentCol, "ok-1")
	p := m["payload"].(map[string]any)
	if vec := fmt.Sprint(m["vector"]); vec != "[1 0]" {
		t.Errorf("vector = %s, want user vector [1 0]", vec)
	}
	checks := map[string]string{
		"content":          "new user content",
		"type":             "insight",
		"tags":             "[u]",
		"updated_at":       "200",
		"access_count":     "5", // max(3,5)
		"last_accessed_at": "180",
		"lifecycle_status": "archived",
		"archived_at":      "190",
		"collection":       agentCol,
		"importance":       "7",
		"reflected_at":     "55",
		"created_at":       "10",
	}
	for k, v := range checks {
		if got := fmt.Sprint(p[k]); got != v {
			t.Errorf("%s = %s, want %s", k, got, v)
		}
	}
	meta := p["metadata"].(map[string]any)
	if meta["source_type"] != "user_input" || meta["keep"] != "me" {
		t.Errorf("metadata = %v", meta)
	}
	if ph, _ := meta["provenance_history"].([]any); len(ph) != 2 {
		t.Errorf("provenance_history union = %v, want 2 deduped entries", meta["provenance_history"])
	}
}

func TestFlagReasons_MissingVectorExecuteOnly(t *testing.T) {
	u := &point{Vector: json.RawMessage(`[1]`), Payload: map[string]any{"type": "event", "updated_at": json.Number("2")}}
	a := &point{Payload: map[string]any{"type": "event", "updated_at": json.Number("1")}}
	if r := flagReasons(u, a, false); len(r) != 0 {
		t.Fatalf("dry-run must not flag missing vector: %v", r)
	}
	if r := flagReasons(u, a, true); len(r) != 1 || r[0] != "missing_vector" {
		t.Fatalf("execute flags = %v, want missing_vector", r)
	}
}

func pt(payload map[string]any) *point {
	return &point{ID: json.RawMessage(`"x"`), Vector: json.RawMessage(`[1]`), Payload: payload}
}

func TestMergePair_OneSidedFieldsSurvive(t *testing.T) {
	// Gap-fillable: survive from whichever copy has them.
	fillable := map[string]any{
		"reflected_at": json.Number("77"),
		"importance":   json.Number("9"),
		"confidence":   json.Number("0.7"),
	}
	// User-authoritative: survive only from the user copy; an agent-only value
	// must NOT leak in (updateByID recomputed/cleared it on purpose).
	authoritative := map[string]any{
		"superseded_by":      "succ",
		"lifecycle_status":   "deprecated",
		"valid_until":        json.Number("100"), // in the past
		"expiry_eligible_at": json.Number("4000"),
		"archive_reason":     "evaporated",
		"tags":               []any{"t"},
	}
	bare := func() map[string]any {
		return map[string]any{"content": "c", "type": "event", "updated_at": json.Number("1"),
			"lifecycle_status": "active", "metadata": map[string]any{}}
	}
	withSided := func() map[string]any {
		p := bare()
		for _, src := range []map[string]any{fillable, authoritative} {
			for k, v := range src {
				p[k] = v
			}
		}
		p["metadata"] = map[string]any{"only_meta": "yes"}
		return p
	}
	check := func(t *testing.T, m map[string]any, want map[string]any) {
		t.Helper()
		for k, v := range want {
			if fmt.Sprint(m[k]) != fmt.Sprint(v) {
				t.Errorf("%s = %v, want %v", k, m[k], v)
			}
		}
		if asMap(m["metadata"])["only_meta"] != "yes" {
			t.Errorf("metadata key lost: %v", m["metadata"])
		}
		if m["collection"] != agentCol {
			t.Errorf("collection = %v", m["collection"])
		}
	}

	t.Run("user_only", func(t *testing.T) {
		m := mergePair(pt(withSided()), pt(bare())).Payload
		check(t, m, fillable)
		check(t, m, authoritative)
	})
	t.Run("agent_only", func(t *testing.T) {
		u, a := pt(bare()), pt(withSided())
		// Such a pair is flagged (never auto-merged); mergePair is still
		// checked here for the field rules it would apply.
		if r := flagReasons(u, a, true); fmt.Sprint(r) != "[agent_superseded_only agent_deprecated_not_revived]" {
			t.Fatalf("flags = %v", r)
		}
		m := mergePair(u, a).Payload
		check(t, m, fillable)
		if m["lifecycle_status"] != "active" {
			t.Errorf("lifecycle_status = %v, want user's active (revive)", m["lifecycle_status"])
		}
		for _, k := range []string{"superseded_by", "valid_until", "expiry_eligible_at", "archive_reason", "tags"} {
			if v, ok := m[k]; ok {
				t.Errorf("%s = %v leaked from agent_self; user copy is authoritative (absent)", k, v)
			}
		}
	})
}

func TestMergePair_Lifecycle(t *testing.T) {
	type lc = map[string]any
	cases := []struct {
		name       string
		user, agt  lc
		wantStatus string
		wantAt     string // archived_at, "" = absent
		wantReason string
	}{
		{"agent_archived_user_active",
			lc{"lifecycle_status": "active"},
			lc{"lifecycle_status": "archived", "archived_at": json.Number("99"), "archive_reason": "a-reason"},
			"archived", "99", "a-reason"},
		{"agent_archived_user_deprecated",
			lc{"lifecycle_status": "deprecated", "archive_reason": "u-reason"},
			lc{"lifecycle_status": "archived", "archived_at": json.Number("99"), "archive_reason": "a-reason"},
			"archived", "99", "a-reason"},
		{"user_archived_agent_active",
			lc{"lifecycle_status": "archived", "archived_at": json.Number("50"), "archive_reason": "u-reason"},
			lc{"lifecycle_status": "active", "archived_at": json.Number("7")},
			"archived", "50", "u-reason"},
		{"revive_user_active_agent_deprecated",
			lc{"lifecycle_status": "active"},
			lc{"lifecycle_status": "deprecated", "archive_reason": "evaporated"},
			"active", "", ""},
		{"user_deprecated_agent_active",
			lc{"lifecycle_status": "deprecated"},
			lc{"lifecycle_status": "active"},
			"deprecated", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.user["type"], tc.user["updated_at"] = "event", json.Number("2")
			tc.agt["type"], tc.agt["updated_at"] = "event", json.Number("1")
			u, a := pt(tc.user), pt(tc.agt)
			if r := flagReasons(u, a, true); len(r) != 0 {
				t.Fatalf("unexpected flags %v", r)
			}
			m := mergePair(u, a).Payload
			if m["lifecycle_status"] != tc.wantStatus {
				t.Errorf("lifecycle_status = %v, want %s", m["lifecycle_status"], tc.wantStatus)
			}
			gotAt := ""
			if v, ok := m["archived_at"]; ok {
				gotAt = fmt.Sprint(v)
			}
			if gotAt != tc.wantAt || str(m["archive_reason"]) != tc.wantReason {
				t.Errorf("archived_at=%q archive_reason=%q, want %q %q", gotAt, str(m["archive_reason"]), tc.wantAt, tc.wantReason)
			}
		})
	}
}

func TestMergePair_SupersededByUserAuthoritative(t *testing.T) {
	base := func(sb string) map[string]any {
		p := map[string]any{"type": "event", "updated_at": json.Number("1")}
		if sb != "" {
			p["superseded_by"] = sb
		}
		return p
	}
	if m := mergePair(pt(base("s1")), pt(base(""))).Payload; m["superseded_by"] != "s1" {
		t.Errorf("user-only superseded_by = %v, want s1", m["superseded_by"])
	}
	if m := mergePair(pt(base("")), pt(base("s2"))).Payload; m["superseded_by"] != nil {
		t.Errorf("agent-only superseded_by = %v, want absent (user authoritative)", m["superseded_by"])
	}
	if m := mergePair(pt(base("s1")), pt(base("s1"))).Payload; m["superseded_by"] != "s1" {
		t.Errorf("equal superseded_by = %v", m["superseded_by"])
	}
}

func TestMergePair_AccessCountMax(t *testing.T) {
	u := pt(map[string]any{"type": "event", "access_count": json.Number("18")})
	a := pt(map[string]any{"type": "event", "access_count": json.Number("32")})
	if m := mergePair(u, a).Payload; fmt.Sprint(m["access_count"]) != "32" {
		t.Fatalf("access_count = %v, want max 32 (shared baseline, no sum)", m["access_count"])
	}
}

func TestFlagReasons_SupersededConflict(t *testing.T) {
	u := pt(map[string]any{"type": "event", "superseded_by": "s1", "updated_at": json.Number("2")})
	a := pt(map[string]any{"type": "event", "superseded_by": "s2", "updated_at": json.Number("1")})
	if r := flagReasons(u, a, false); len(r) != 1 || r[0] != "superseded_by_conflict" {
		t.Fatalf("flags = %v", r)
	}
	a.Payload["superseded_by"] = "s1"
	if r := flagReasons(u, a, false); len(r) != 0 {
		t.Fatalf("equal superseded_by must not flag: %v", r)
	}
}

func TestOtherPair_ReportedNeverWritten(t *testing.T) {
	f := newFake()
	seed(f)
	f.add("engram_reflection", "ok-1", []float64{1}, map[string]any{"content": "r", "type": "insight", "updated_at": 1}) // 3-way
	f.add("engram_pigo", "solo-a", []float64{1}, map[string]any{"content": "p", "type": "event", "updated_at": 1})       // agent+pigo
	ts := httptest.NewServer(f)
	t.Cleanup(ts.Close)

	var stdout, stderr bytes.Buffer
	if code := realMain([]string{"--qdrant-url", ts.URL}, &stdout, &stderr); code != 0 {
		t.Fatalf("dry-run exit %d: %s", code, stderr.String())
	}
	s := stdout.String()
	if !strings.Contains(s, "ok-1\tflagged\tother_pair:engram_user+engram_agent_self+engram_reflection") ||
		!strings.Contains(s, "solo-a\tflagged\tother_pair:engram_agent_self+engram_pigo") ||
		!strings.Contains(s, `"other_pair":2`) || !strings.Contains(s, `"mergeable":0`) {
		t.Fatalf("other_pair not reported:\n%s", s)
	}

	stdout.Reset()
	if code := realMain([]string{"--qdrant-url", ts.URL, "--execute", "--dry-run=false"}, &stdout, &stderr); code != 0 {
		t.Fatalf("execute exit %d: %s", code, stderr.String())
	}
	for _, c := range f.writes() {
		if !strings.HasSuffix(c, "/snapshots") {
			t.Fatalf("write %q on other_pair-only fixture", c)
		}
	}
}

func TestExecute_TOCTOU_ReReadSkipsChanged(t *testing.T) {
	cases := map[string]func(f *fakeQdrant){
		"agent_now_newer": func(f *fakeQdrant) {
			idj, _ := json.Marshal("ok-1")
			f.cols[agentCol][string(idj)]["payload"].(map[string]any)["updated_at"] = 999
		},
		"user_copy_gone": func(f *fakeQdrant) {
			idj, _ := json.Marshal("ok-1")
			delete(f.cols[userCol], string(idj))
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			f, ts := startFake(t)
			// Mutate during the per-id re-read, i.e. after the scan planned ok-1 as mergeable.
			f.onReRead = mutate
			var stdout, stderr bytes.Buffer
			if code := realMain([]string{"--qdrant-url", ts.URL, "--execute", "--dry-run=false"}, &stdout, &stderr); code != 0 {
				t.Fatalf("exit %d: %s", code, stderr.String())
			}
			for _, c := range f.writes() {
				if !strings.HasSuffix(c, "/snapshots") {
					t.Fatalf("write %q despite changed data", c)
				}
			}
			if !strings.Contains(stdout.String(), `"merged":0`) || !strings.Contains(stdout.String(), `"skipped_changed":1`) {
				t.Fatalf("summary:\n%s", stdout.String())
			}
		})
	}
}

func TestScrollAll_RepeatedOffsetErrors(t *testing.T) {
	f := newFake()
	f.repeatOffset = true
	ts := httptest.NewServer(f)
	t.Cleanup(ts.Close)
	done := make(chan error, 1)
	go func() {
		_, err := newRESTClient(ts.URL, "").ScrollAll(context.Background(), userCol, false)
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "repeated next_page_offset") {
			t.Fatalf("err = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("ScrollAll looped forever on a repeated offset")
	}
}

func TestExecute_PerEntryRecordSurvivesMidRunError(t *testing.T) {
	f := newFake()
	seed(f)
	// second mergeable id so ok-1 merges, then ok-2's delete fails.
	f.add(userCol, "ok-2", []float64{1}, map[string]any{"content": "u2", "type": "event", "updated_at": 5})
	f.add(agentCol, "ok-2", []float64{1}, map[string]any{"content": "a2", "type": "event", "updated_at": 1})
	ts := httptest.NewServer(f)
	t.Cleanup(ts.Close)
	out := filepath.Join(t.TempDir(), "plan.jsonl")

	var stdout, stderr bytes.Buffer
	// Fail the delete of the second merge only.
	deletes := 0
	wrapped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/points/delete") {
			deletes++
			if deletes == 2 {
				http.Error(w, "boom", 500)
				return
			}
		}
		f.ServeHTTP(w, r)
	})
	ts.Config.Handler = wrapped
	code := realMain([]string{"--qdrant-url", ts.URL, "--execute", "--dry-run=false", "--out", out}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("expected failure")
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `{"id":"ok-1","action":"merge","result":"merged"}`) {
		t.Fatalf("merged ok-1 not recorded before failure:\n%s", data)
	}
	if !strings.Contains(string(data), `"id":"ok-2"`) || !strings.Contains(string(data), `"result":"error: `) {
		t.Fatalf("failing id not recorded:\n%s", data)
	}
}

func TestFlagReasons_AgentSupersededOnly(t *testing.T) {
	u := pt(map[string]any{"type": "event", "updated_at": json.Number("5")})
	a := pt(map[string]any{"type": "event", "updated_at": json.Number("1"), "superseded_by": "succ"})
	if r := flagReasons(u, a, false); fmt.Sprint(r) != "[agent_superseded_only]" {
		t.Fatalf("flags = %v, want [agent_superseded_only]", r)
	}
	u.Payload["superseded_by"] = "succ"
	if r := flagReasons(u, a, false); len(r) != 0 {
		t.Fatalf("same superseded_by on both must not flag: %v", r)
	}
}

func TestFlagReasons_AgentDeprecatedRevive(t *testing.T) {
	mk := func(userUpd, agentUpd, userLC string) []string {
		u := pt(map[string]any{"type": "event", "updated_at": json.Number(userUpd)})
		if userLC != "" {
			u.Payload["lifecycle_status"] = userLC
		}
		a := pt(map[string]any{"type": "event", "updated_at": json.Number(agentUpd), "lifecycle_status": "deprecated"})
		return flagReasons(u, a, false)
	}
	if r := mk("5", "5", "active"); fmt.Sprint(r) != "[agent_deprecated_not_revived]" {
		t.Fatalf("equal updated_at: flags = %v", r)
	}
	if r := mk("5", "5", ""); fmt.Sprint(r) != "[agent_deprecated_not_revived]" {
		t.Fatalf("empty user lifecycle: flags = %v", r)
	}
	if r := mk("6", "5", "active"); len(r) != 0 {
		t.Fatalf("user strictly newer is a revive, must not flag: %v", r)
	}
	if r := mk("5", "6", "active"); fmt.Sprint(r) != "[agent_self_newer agent_deprecated_not_revived]" {
		t.Fatalf("agent newer: flags = %v", r)
	}
}

// runExecuteWith runs --execute against the fake with an injection hook
// (return true = request handled by the hook).
func runExecuteWith(t *testing.T, inject func(w http.ResponseWriter, r *http.Request, f *fakeQdrant) bool) (*fakeQdrant, int, string) {
	t.Helper()
	f := newFake()
	seed(f)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if inject(w, r, f) {
			return
		}
		f.ServeHTTP(w, r)
	}))
	t.Cleanup(ts.Close)
	var stdout, stderr bytes.Buffer
	code := realMain([]string{"--qdrant-url", ts.URL, "--execute", "--dry-run=false"}, &stdout, &stderr)
	return f, code, stdout.String() + stderr.String()
}

func assertNoDelete(t *testing.T, f *fakeQdrant) {
	t.Helper()
	for _, c := range f.calls {
		if strings.HasSuffix(c, "/points/delete") {
			t.Fatalf("delete issued: %s", c)
		}
	}
	if f.get(userCol, "ok-1") == nil {
		t.Fatal("user copy deleted")
	}
}

func TestExecute_UpsertFailure_NoDelete(t *testing.T) {
	f, code, out := runExecuteWith(t, func(w http.ResponseWriter, r *http.Request, _ *fakeQdrant) bool {
		if r.Method == "PUT" && strings.HasSuffix(r.URL.Path, "/points") {
			http.Error(w, "boom", 500)
			return true
		}
		return false
	})
	if code == 0 || !strings.Contains(out, "upsert ok-1") {
		t.Fatalf("exit %d, out:\n%s", code, out)
	}
	assertNoDelete(t, f)
}

func TestExecute_VerifyMismatch_NoDelete(t *testing.T) {
	for _, field := range []string{"content", "type", "tags"} {
		t.Run(field, func(t *testing.T) {
			f, code, out := runExecuteWith(t, func(w http.ResponseWriter, r *http.Request, f *fakeQdrant) bool {
				if r.Method == "PUT" && strings.HasSuffix(r.URL.Path, "/collections/"+agentCol+"/points") {
					// Accept the upsert but persist a corrupted field.
					f.ServeHTTP(w, r)
					f.mu.Lock()
					idj, _ := json.Marshal("ok-1")
					p := f.cols[agentCol][string(idj)]["payload"].(map[string]any)
					switch field {
					case "content":
						p["content"] = "corrupted"
					case "type":
						p["type"] = "corrupted"
					case "tags":
						p["tags"] = []any{"corrupted"}
					}
					f.mu.Unlock()
					return true
				}
				return false
			})
			if code == 0 || !strings.Contains(out, "verify ok-1") {
				t.Fatalf("exit %d, out:\n%s", code, out)
			}
			assertNoDelete(t, f)
		})
	}
}

func TestExecute_ScanNeverFetchesVectors(t *testing.T) {
	var scrollVec, retrieveVec []bool
	_, code, out := runExecuteWith(t, func(_ http.ResponseWriter, r *http.Request, _ *fakeQdrant) bool {
		if r.Method != "POST" {
			return false
		}
		body, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(body))
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		wv, _ := req["with_vector"].(bool)
		switch {
		case strings.HasSuffix(r.URL.Path, "/points/scroll"):
			scrollVec = append(scrollVec, wv)
		case strings.HasSuffix(r.URL.Path, "/points"):
			retrieveVec = append(retrieveVec, wv)
		}
		return false
	})
	if code != 0 {
		t.Fatalf("exit %d:\n%s", code, out)
	}
	for _, v := range scrollVec {
		if v {
			t.Fatal("execute scan fetched vectors")
		}
	}
	// ok-1: re-read user (vec), re-read agent (vec), verify read-back (no vec).
	if fmt.Sprint(retrieveVec) != "[true true false]" {
		t.Fatalf("retrieve with_vector sequence = %v", retrieveVec)
	}
	if !strings.Contains(out, `"merged":1`) {
		t.Fatalf("missing_vector must be judged on re-retrieved data; out:\n%s", out)
	}
}
