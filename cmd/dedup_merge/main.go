// cmd/dedup_merge — one-shot merge of same-UUID double-stored memories.
//
// Background: before the by-id update fix, updateByID re-inserted an
// engram_agent_self memory without its Collection, so MultiStore.Insert wrote a
// second point with the SAME id into engram_user and left the stale agent_self
// copy behind. This tool finds ids present in BOTH engram_user and
// engram_agent_self and merges each pair into the agent_self home.
//
// Merge rules (payload field names as written by pkg/qdrant memoryToPoint):
//
//	survivor   the engram_agent_self point id (same id); vector from the USER copy
//	base       the USER copy payload, top-level AND metadata key-wise; any key
//	           missing or empty in user is filled from agent_self, EXCEPT the
//	           user-authoritative keys valid_until, expiry_eligible_at, tags,
//	           superseded_by, lifecycle_status, archived_at, archive_reason
//	           (updateByID recomputes/clears these: user value wins even when
//	           absent or empty)
//	overrides  collection        = engram_agent_self
//	           access_count      = max (both copies share the prev baseline)
//	           last_accessed_at  = max; reflected_at = max
//	           metadata.provenance_history = union (deduplicated)
//	           lifecycle_status  = archived if either copy is archived (with
//	                               that copy's archived_at/archive_reason),
//	                               else the user value
//
// FLAGGED pairs are reported and never merged automatically:
//   - agent_self updated_at > user updated_at (agent copy has newer edits)
//   - type differs between the copies
//   - superseded_by non-empty on both and different
//   - agent_superseded_only: superseded_by set only on agent_self
//   - agent_deprecated_not_revived: agent_self deprecated, user active, and
//     user updated_at <= agent_self updated_at
//   - execute mode only, on the per-id re-retrieve: either copy lacks a vector
//   - other_pair: an id stored in any collection set other than exactly
//     {engram_user, engram_agent_self} across engram_user, engram_agent_self,
//     engram_reflection, engram_pigo (report-only, never written)
//
// Modes:
//
//	default / --dry-run=true   read-only: scroll + plan + summary, ZERO writes
//	--execute --dry-run=false  (both required, explicitly) real run:
//	    0. snapshot engram_user and engram_agent_self and verify each snapshot is
//	       listed; abort (exit 1) with no writes if either fails
//	    1. per id: re-retrieve BOTH copies (with vectors), recompute merge and
//	       flags from that fresh data (skip if now flagged / no longer a dup),
//	       upsert merged point into engram_agent_self (wait=true), retrieve it
//	       back and verify content, then delete the id from engram_user ONLY
//	       (never from agent_self). Stop on first error.
//	    Stop engram (or make sure it is quiet) during --execute: concurrent
//	    writes between the re-read and the upsert would be lost.
//
// Each plan/result entry is written to stdout and --out (JSONL, synced) as soon
// as that id completes, so an aborted run still records what was merged.
//
// Flags:
//
//	--qdrant-url  Qdrant REST base URL (default: derived from ENGRAM_QDRANT_URL,
//	              gRPC port 6334 → REST port 6333; ENGRAM_QDRANT_USE_TLS → https)
//	--dry-run     (default true)
//	--execute     (default false)
//	--out         optional JSONL path for the per-id plan
//
// Exit codes: 0 success, 1 runtime error, 2 usage error.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/config"
)

const (
	userCol  = collection.CollectionUser
	agentCol = collection.CollectionAgentSelf
)

// extraCols are scrolled read-only (no vectors) only to detect other_pair ids.
var extraCols = []string{collection.CollectionReflection, collection.CollectionPigo}

// Payload field names (mirror pkg/qdrant/qdrant.go field* constants).
const (
	fContent          = "content"
	fType             = "type"
	fTags             = "tags"
	fUpdatedAt        = "updated_at"
	fMetadata         = "metadata"
	fAccessCount      = "access_count"
	fLastAccessedAt   = "last_accessed_at"
	fLifecycleStatus  = "lifecycle_status"
	fArchivedAt       = "archived_at"
	fArchiveReason    = "archive_reason"
	fReflectedAt      = "reflected_at"
	fSupersededBy     = "superseded_by"
	fValidUntil       = "valid_until"
	fExpiryEligibleAt = "expiry_eligible_at"
	fCollection       = "collection"
	mSourceType       = "source_type"
	mProvHistory      = "provenance_history"
	lifecycleArchive  = "archived"
)

// point is a Qdrant REST point. ID and Vector are kept raw so they round-trip
// unchanged (uuid or integer ids; unnamed or named vectors).
type point struct {
	ID      json.RawMessage `json:"id"`
	Vector  json.RawMessage `json:"vector,omitempty"`
	Payload map[string]any  `json:"payload"`
}

func (p *point) key() string { return string(p.ID) }

func (p *point) hasVector() bool {
	v := bytes.TrimSpace(p.Vector)
	return len(v) > 0 && !bytes.Equal(v, []byte("null")) && !bytes.Equal(v, []byte("[]")) && !bytes.Equal(v, []byte("{}"))
}

// qdrantAPI is the narrow Qdrant REST surface this tool uses. Tests drive the
// real restClient against an httptest fake.
type qdrantAPI interface {
	ScrollAll(ctx context.Context, col string, withVector bool) ([]point, error)
	Retrieve(ctx context.Context, col string, id json.RawMessage, withVector bool) (*point, error)
	CreateSnapshot(ctx context.Context, col string) (string, error)
	ListSnapshots(ctx context.Context, col string) ([]string, error)
	Upsert(ctx context.Context, col string, p point) error
	DeletePoint(ctx context.Context, col string, id json.RawMessage) error
}

// ─── REST client ───────────────────────────────────────────────────────────

type restClient struct {
	base   string
	apiKey string
	http   *http.Client
}

func newRESTClient(base, apiKey string) *restClient {
	return &restClient{base: strings.TrimRight(base, "/"), apiKey: apiKey, http: &http.Client{Timeout: 120 * time.Second}}
}

func (c *restClient) do(ctx context.Context, method, path string, body any, out any) error {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, rd)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("api-key", c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("%s %s: HTTP %d: %s", method, path, resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if out == nil {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	return dec.Decode(out)
}

func colPath(col string) string { return "/collections/" + url.PathEscape(col) }

func (c *restClient) ScrollAll(ctx context.Context, col string, withVector bool) ([]point, error) {
	var all []point
	var offset any
	seen := map[string]struct{}{}
	for {
		body := map[string]any{"limit": 256, "with_payload": true, "with_vector": withVector}
		if offset != nil {
			body["offset"] = offset
		}
		var resp struct {
			Result struct {
				Points         []point `json:"points"`
				NextPageOffset any     `json:"next_page_offset"`
			} `json:"result"`
		}
		if err := c.do(ctx, http.MethodPost, colPath(col)+"/points/scroll", body, &resp); err != nil {
			return nil, err
		}
		all = append(all, resp.Result.Points...)
		if resp.Result.NextPageOffset == nil {
			return all, nil
		}
		ob, _ := json.Marshal(resp.Result.NextPageOffset)
		if _, dup := seen[string(ob)]; dup {
			return nil, fmt.Errorf("scroll %s: repeated next_page_offset %s", col, ob)
		}
		seen[string(ob)] = struct{}{}
		offset = resp.Result.NextPageOffset
	}
}

func (c *restClient) Retrieve(ctx context.Context, col string, id json.RawMessage, withVector bool) (*point, error) {
	body := map[string]any{"ids": []json.RawMessage{id}, "with_payload": true, "with_vector": withVector}
	var resp struct {
		Result []point `json:"result"`
	}
	if err := c.do(ctx, http.MethodPost, colPath(col)+"/points", body, &resp); err != nil {
		return nil, err
	}
	if len(resp.Result) == 0 {
		return nil, nil
	}
	return &resp.Result[0], nil
}

func (c *restClient) CreateSnapshot(ctx context.Context, col string) (string, error) {
	var resp struct {
		Result struct {
			Name string `json:"name"`
		} `json:"result"`
	}
	if err := c.do(ctx, http.MethodPost, colPath(col)+"/snapshots?wait=true", nil, &resp); err != nil {
		return "", err
	}
	if resp.Result.Name == "" {
		return "", fmt.Errorf("snapshot %s: empty snapshot name in response", col)
	}
	return resp.Result.Name, nil
}

func (c *restClient) ListSnapshots(ctx context.Context, col string) ([]string, error) {
	var resp struct {
		Result []struct {
			Name string `json:"name"`
		} `json:"result"`
	}
	if err := c.do(ctx, http.MethodGet, colPath(col)+"/snapshots", nil, &resp); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(resp.Result))
	for _, s := range resp.Result {
		out = append(out, s.Name)
	}
	return out, nil
}

func (c *restClient) Upsert(ctx context.Context, col string, p point) error {
	body := map[string]any{"points": []point{p}}
	return c.do(ctx, http.MethodPut, colPath(col)+"/points?wait=true", body, nil)
}

func (c *restClient) DeletePoint(ctx context.Context, col string, id json.RawMessage) error {
	body := map[string]any{"points": []json.RawMessage{id}}
	return c.do(ctx, http.MethodPost, colPath(col)+"/points/delete?wait=true", body, nil)
}

// ─── merge logic ───────────────────────────────────────────────────────────

func num(v any) float64 {
	switch x := v.(type) {
	case json.Number:
		f, _ := x.Float64()
		return f
	case float64:
		return x
	case int:
		return float64(x)
	case int64:
		return float64(x)
	}
	return 0
}

// numOut renders a float as an integer json.Number when integral.
func numOut(f float64) json.Number {
	if f == float64(int64(f)) {
		return json.Number(fmt.Sprintf("%d", int64(f)))
	}
	return json.Number(fmt.Sprintf("%v", f))
}

func str(v any) string {
	s, _ := v.(string)
	return s
}

func asMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

func copyMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// flagReasons returns why a pair must NOT be auto-merged (empty → mergeable).
// checkVector is true only for the per-id re-retrieve in execute mode (the
// initial scroll never fetches vectors).
func flagReasons(user, agent *point, checkVector bool) []string {
	var r []string
	if num(agent.Payload[fUpdatedAt]) > num(user.Payload[fUpdatedAt]) {
		r = append(r, "agent_self_newer")
	}
	if str(user.Payload[fType]) != str(agent.Payload[fType]) {
		r = append(r, "type_differs")
	}
	us, as := str(user.Payload[fSupersededBy]), str(agent.Payload[fSupersededBy])
	if us != "" && as != "" && us != as {
		r = append(r, "superseded_by_conflict")
	}
	// superseded_by only on agent_self: a pre-fix fan-out Update (e.g. dream)
	// may have landed there without bumping updated_at. User-authoritative
	// merge would drop it and resurrect a superseded memory → manual review.
	if us == "" && as != "" {
		r = append(r, "agent_superseded_only")
	}
	// agent_self deprecated, user active: only a revive when the user copy is
	// strictly newer; otherwise the deprecation may be the later event.
	ul := str(user.Payload[fLifecycleStatus])
	if str(agent.Payload[fLifecycleStatus]) == "deprecated" && (ul == "" || ul == "active") &&
		num(user.Payload[fUpdatedAt]) <= num(agent.Payload[fUpdatedAt]) {
		r = append(r, "agent_deprecated_not_revived")
	}
	if checkVector && (!user.hasVector() || !agent.hasVector()) {
		r = append(r, "missing_vector")
	}
	return r
}

// isEmpty reports whether a payload value counts as "missing or empty" for
// the fill-from-agent_self rule.
func isEmpty(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return x == ""
	case json.Number, float64, int, int64:
		return num(x) == 0
	case []any:
		return len(x) == 0
	case map[string]any:
		return len(x) == 0
	}
	return false
}

// fillFrom returns base with every key missing/empty in base filled from fill.
func fillFrom(base, fill map[string]any) map[string]any {
	out := copyMap(base)
	for k, v := range fill {
		if isEmpty(out[k]) {
			out[k] = v
		}
	}
	return out
}

// userAuthoritative payload keys take the USER copy value verbatim (absent or
// empty included) — never gap-filled from agent_self.
var userAuthoritative = []string{
	fValidUntil, fExpiryEligibleAt, fTags, fSupersededBy,
	fLifecycleStatus, fArchivedAt, fArchiveReason,
}

// unionProvenance merges two provenance_history lists, agent entries first,
// dropping exact duplicates (canonical JSON equality).
func unionProvenance(agent, user any) []any {
	out := []any{}
	seen := map[string]struct{}{}
	for _, src := range []any{agent, user} {
		list, _ := src.([]any)
		for _, e := range list {
			b, _ := json.Marshal(e)
			if _, dup := seen[string(b)]; dup {
				continue
			}
			seen[string(b)] = struct{}{}
			out = append(out, e)
		}
	}
	return out
}

func maxField(p, user, agent map[string]any, f string) {
	if _, uok := user[f]; !uok {
		if _, aok := agent[f]; !aok {
			return
		}
	}
	if num(agent[f]) > num(user[f]) {
		p[f] = agent[f]
	} else if _, ok := user[f]; ok {
		p[f] = user[f]
	}
}

// mergePair builds the merged survivor point (agent_self id/home): USER copy
// as base (top-level and metadata key-wise), gaps filled from agent_self, then
// the overrides documented in the package comment.
func mergePair(user, agent *point) point {
	up, ap := user.Payload, agent.Payload
	p := fillFrom(up, ap)
	// User-authoritative keys: updateByID recomputes or deliberately clears
	// these, so the user value wins even when absent or empty (no gap-fill).
	for _, f := range userAuthoritative {
		if v, ok := up[f]; ok {
			p[f] = v
		} else {
			delete(p, f)
		}
	}

	um, am := asMap(up[fMetadata]), asMap(ap[fMetadata])
	if um != nil || am != nil {
		meta := fillFrom(um, am)
		_, ah := am[mProvHistory]
		_, uh := um[mProvHistory]
		if ah || uh {
			meta[mProvHistory] = unionProvenance(am[mProvHistory], um[mProvHistory])
		}
		p[fMetadata] = meta
	}

	// access_count = MAX, not sum: updateByID created the user copy with
	// AccessCount: prev.AccessCount, so both copies share the same baseline
	// and summing would double-count it.
	p[fAccessCount] = numOut(math.Max(num(up[fAccessCount]), num(ap[fAccessCount])))
	maxField(p, up, ap, fLastAccessedAt)
	maxField(p, up, ap, fReflectedAt)

	// lifecycle: archived if either copy is archived (carrying that copy's
	// archive fields); otherwise the user value (incl. an active user copy
	// reviving a deprecated agent_self copy). The user-archived case is
	// already covered by the user-authoritative copy above.
	if str(up[fLifecycleStatus]) != lifecycleArchive && str(ap[fLifecycleStatus]) == lifecycleArchive {
		p[fLifecycleStatus] = lifecycleArchive
		for _, f := range []string{fArchivedAt, fArchiveReason} {
			if v, ok := ap[f]; ok {
				p[f] = v
			} else {
				delete(p, f)
			}
		}
	}

	p[fCollection] = agentCol
	return point{ID: agent.ID, Vector: user.Vector, Payload: p}
}

// sameFields reports whether a and b hold JSON-equal values for every field.
func sameFields(a, b map[string]any, fields ...string) bool {
	for _, f := range fields {
		ja, _ := json.Marshal(a[f])
		jb, _ := json.Marshal(b[f])
		if !bytes.Equal(ja, jb) {
			return false
		}
	}
	return true
}

// ─── run ───────────────────────────────────────────────────────────────────

type planEntry struct {
	ID      string   `json:"id"`
	Action  string   `json:"action"` // merge | flagged
	Reasons []string `json:"reasons,omitempty"`
	Result  string   `json:"result,omitempty"` // execute: merged | skipped
}

type summary struct {
	Execute    bool `json:"execute"`
	UserPoints int  `json:"user_points"`
	AgentPts   int  `json:"agent_self_points"`
	Dups       int  `json:"dups"`
	Mergeable  int  `json:"mergeable"`
	Flagged    int  `json:"flagged"`
	OtherPair  int  `json:"other_pair"`
	Merged     int  `json:"merged"`
	// SkippedChanged counts mergeable ids skipped in execute because the
	// fresh re-read showed them flagged or no longer duplicated.
	SkippedChanged int `json:"skipped_changed"`
}

type options struct {
	execute   bool
	qdrantURL string
	out       string
}

func idToString(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return string(raw)
}

// planWriter emits each plan/result entry immediately to stdout and, when
// set, the --out JSONL (synced per entry when it supports Sync).
type planWriter struct {
	stdout  io.Writer
	planOut io.Writer
}

func (w planWriter) emit(e planEntry) {
	line := fmt.Sprintf("%s\t%s", e.ID, e.Action)
	if len(e.Reasons) > 0 {
		line += "\t" + strings.Join(e.Reasons, ",")
	}
	if e.Result != "" {
		line += "\t" + e.Result
	}
	_, _ = fmt.Fprintln(w.stdout, line)
	if w.planOut != nil {
		b, _ := json.Marshal(e)
		_, _ = fmt.Fprintln(w.planOut, string(b))
		if s, ok := w.planOut.(interface{ Sync() error }); ok {
			_ = s.Sync()
		}
	}
}

func run(ctx context.Context, api qdrantAPI, execute bool, stdout io.Writer, planOut io.Writer) (*summary, error) {
	sum := &summary{Execute: execute}
	out := planWriter{stdout: stdout, planOut: planOut}
	// Never fetch vectors in the scan (memory); execute re-retrieves per id.
	users, err := api.ScrollAll(ctx, userCol, false)
	if err != nil {
		return sum, fmt.Errorf("scroll %s: %w", userCol, err)
	}
	agents, err := api.ScrollAll(ctx, agentCol, false)
	if err != nil {
		return sum, fmt.Errorf("scroll %s: %w", agentCol, err)
	}
	sum.UserPoints, sum.AgentPts = len(users), len(agents)

	// id → collections holding it, across all four collections.
	where := map[string][]string{}
	for _, u := range users {
		where[u.key()] = append(where[u.key()], userCol)
	}
	for _, a := range agents {
		where[a.key()] = append(where[a.key()], agentCol)
	}
	for _, col := range extraCols {
		pts, err := api.ScrollAll(ctx, col, false)
		if err != nil {
			return sum, fmt.Errorf("scroll %s: %w", col, err)
		}
		for _, p := range pts {
			where[p.key()] = append(where[p.key()], col)
		}
	}
	isPair := func(k string) bool {
		c := where[k]
		return len(c) == 2 && c[0] == userCol && c[1] == agentCol
	}

	// other_pair: multi-collection ids that are not exactly user∩agent_self.
	var others []string
	for k, cols := range where {
		if len(cols) > 1 && !isPair(k) {
			others = append(others, k)
		}
	}
	sort.Strings(others)

	byKey := make(map[string]*point, len(agents))
	for i := range agents {
		byKey[agents[i].key()] = &agents[i]
	}
	type pair struct{ user, agent *point }
	var pairs []pair
	for i := range users {
		if a, ok := byKey[users[i].key()]; ok && isPair(users[i].key()) {
			pairs = append(pairs, pair{&users[i], a})
		}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].user.key() < pairs[j].user.key() })
	sum.Dups = len(pairs)

	plan := make([]planEntry, len(pairs))
	for i, pr := range pairs {
		e := planEntry{ID: idToString(pr.user.ID), Action: "merge"}
		if r := flagReasons(pr.user, pr.agent, false); len(r) > 0 {
			e.Action, e.Reasons = "flagged", r
			sum.Flagged++
		} else {
			sum.Mergeable++
		}
		plan[i] = e
	}
	for _, k := range others {
		sum.OtherPair++
		e := planEntry{ID: idToString(json.RawMessage(k)), Action: "flagged", Reasons: []string{"other_pair:" + strings.Join(where[k], "+")}}
		if execute {
			e.Result = "skipped"
		}
		out.emit(e)
	}

	if !execute {
		for _, e := range plan {
			out.emit(e)
		}
		return sum, nil
	}

	// STEP 0: snapshots of both collections, verified, before ANY write.
	for _, col := range []string{userCol, agentCol} {
		name, err := api.CreateSnapshot(ctx, col)
		if err != nil {
			return sum, fmt.Errorf("snapshot %s: %w (aborted, no writes)", col, err)
		}
		names, err := api.ListSnapshots(ctx, col)
		if err != nil {
			return sum, fmt.Errorf("list snapshots %s: %w (aborted, no writes)", col, err)
		}
		found := false
		for _, n := range names {
			if n == name {
				found = true
				break
			}
		}
		if !found {
			return sum, fmt.Errorf("snapshot %s %q not listed (aborted, no writes)", col, name)
		}
		_, _ = fmt.Fprintf(stdout, "snapshot ok: %s %s\n", col, name)
	}

	for i, pr := range pairs {
		e := plan[i]
		if e.Action != "merge" {
			e.Result = "skipped"
			out.emit(e)
			continue
		}
		fail := func(err error) (*summary, error) {
			e.Result = "error: " + err.Error()
			out.emit(e)
			return sum, err
		}
		// TOCTOU: re-read both copies right before writing and decide again.
		fu, err := api.Retrieve(ctx, userCol, pr.user.ID, true)
		if err != nil {
			return fail(fmt.Errorf("re-read %s/%s: %w", userCol, e.ID, err))
		}
		fa, err := api.Retrieve(ctx, agentCol, pr.agent.ID, true)
		if err != nil {
			return fail(fmt.Errorf("re-read %s/%s: %w", agentCol, e.ID, err))
		}
		if fu == nil || fa == nil {
			e.Result = "skipped_no_longer_dup"
			sum.SkippedChanged++
			out.emit(e)
			continue
		}
		if r := flagReasons(fu, fa, true); len(r) > 0 {
			e.Action, e.Reasons, e.Result = "flagged", r, "skipped_changed"
			sum.SkippedChanged++
			out.emit(e)
			continue
		}
		merged := mergePair(fu, fa)
		if err := api.Upsert(ctx, agentCol, merged); err != nil {
			return fail(fmt.Errorf("upsert %s: %w", e.ID, err))
		}
		got, err := api.Retrieve(ctx, agentCol, merged.ID, false)
		if err != nil {
			return fail(fmt.Errorf("verify %s: %w", e.ID, err))
		}
		if got == nil || !sameFields(got.Payload, merged.Payload, fContent, fType, fTags) {
			return fail(fmt.Errorf("verify %s: merged content/type/tags not read back from %s; user copy NOT deleted", e.ID, agentCol))
		}
		if err := api.DeletePoint(ctx, userCol, fu.ID); err != nil {
			return fail(fmt.Errorf("delete %s from %s: %w", e.ID, userCol, err))
		}
		e.Result = "merged"
		sum.Merged++
		out.emit(e)
	}
	return sum, nil
}

// ─── flags / main ──────────────────────────────────────────────────────────

var errUsage = errors.New("usage")

// parseFlags parses args. A real (writing) run requires BOTH --execute and an
// explicit --dry-run=false; any other combination is dry-run or a usage error.
func parseFlags(args []string, stderr io.Writer) (options, error) {
	fs := flag.NewFlagSet("dedup_merge", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dryRun := fs.Bool("dry-run", true, "read-only plan (default); pass --dry-run=false together with --execute to write")
	execute := fs.Bool("execute", false, "perform the merge (requires --dry-run=false); stop engram (or keep it quiet) while executing")
	qurl := fs.String("qdrant-url", "", "Qdrant REST base URL (default derived from ENGRAM_QDRANT_URL)")
	out := fs.String("out", "", "optional JSONL path for the per-id plan")
	if err := fs.Parse(args); err != nil {
		return options{}, fmt.Errorf("%w: %v", errUsage, err)
	}
	dryExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "dry-run" {
			dryExplicit = true
		}
	})
	opts := options{qdrantURL: *qurl, out: *out}
	switch {
	case *execute && dryExplicit && !*dryRun:
		opts.execute = true
	case *execute:
		return opts, fmt.Errorf("%w: --execute also requires an explicit --dry-run=false", errUsage)
	case !*dryRun:
		return opts, fmt.Errorf("%w: --dry-run=false also requires --execute", errUsage)
	}
	return opts, nil
}

// restURLFromConfig derives the REST base URL from the gRPC-style
// ENGRAM_QDRANT_URL (host:6334 → http(s)://host:6333).
func restURLFromConfig(grpcURL string, useTLS bool) string {
	if strings.HasPrefix(grpcURL, "http://") || strings.HasPrefix(grpcURL, "https://") {
		return grpcURL
	}
	scheme := "http"
	if useTLS {
		scheme = "https"
	}
	host, port, err := net.SplitHostPort(grpcURL)
	if err != nil {
		return scheme + "://" + grpcURL
	}
	if port == "6334" {
		port = "6333"
	}
	return scheme + "://" + net.JoinHostPort(host, port)
}

func main() {
	os.Exit(realMain(os.Args[1:], os.Stdout, os.Stderr))
}

func realMain(args []string, stdout, stderr io.Writer) int {
	opts, err := parseFlags(args, stderr)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 2
	}
	cfg := config.Load()
	base := opts.qdrantURL
	if base == "" {
		base = restURLFromConfig(cfg.QdrantURL, cfg.QdrantUseTLS)
	}
	var planOut io.Writer
	if opts.out != "" {
		f, err := os.Create(opts.out)
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "open --out: %v\n", err)
			return 1
		}
		defer func() { _ = f.Close() }()
		planOut = f
	}
	mode := "DRY-RUN (read-only)"
	if opts.execute {
		mode = "EXECUTE"
	}
	_, _ = fmt.Fprintf(stdout, "dedup_merge mode=%s qdrant=%s\n", mode, base)
	sum, err := run(context.Background(), newRESTClient(base, cfg.QdrantAPIKey), opts.execute, stdout, planOut)
	b, _ := json.Marshal(sum)
	_, _ = fmt.Fprintf(stdout, "summary %s\n", b)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	return 0
}
