package server

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/metrics"
	"github.com/FBISiri/engram/pkg/qdrant"
)

// detailedStore implements memory.Store (via the embedded expiryObsStore) plus
// the optional detailedExpirer interface, returning a fixed set of
// ExpiryRecords so runExpiryTick's detailed branch can be exercised offline.
type detailedStore struct {
	*expiryObsStore
	records []qdrant.ExpiryRecord
}

func (s *detailedStore) DeleteExpiredDetailed(context.Context) ([]qdrant.ExpiryRecord, error) {
	return append([]qdrant.ExpiryRecord(nil), s.records...), nil
}

// TestRunExpiryTickDetailedMetricExcludesProtectedHealed verifies the
// EvaporationHardDeletedTotal counter only advances for points ACTUALLY
// deleted (intersected with the evaporation-reason pre-scan set), NOT for
// protected or healed imp>=8 evaporation-reason points.
func TestRunExpiryTickDetailedMetricExcludesProtectedHealed(t *testing.T) {
	t.Setenv("ENGRAM_DELETE_AUDIT_PATH", filepath.Join(t.TempDir(), "deletes.jsonl"))
	past := float64(time.Now().Add(-time.Hour).Unix())
	evap := func(id string, ty memory.MemoryType) memory.Memory {
		return memory.Memory{ID: id, Type: ty, Collection: "engram_user", ValidUntil: past,
			Metadata: map[string]any{"deprecated_reason": "evaporation"}}
	}

	store := &detailedStore{
		// Pre-scan set: all three are evaporation-reason expired candidates.
		expiryObsStore: &expiryObsStore{mems: []memory.Memory{
			evap("del-evap", memory.TypeEvent),
			evap("prot-evap", memory.TypeEvent),
			evap("heal-evap", memory.TypeInsight),
		}},
		// Actual outcomes: only del-evap is physically deleted.
		records: []qdrant.ExpiryRecord{
			{ID: "del-evap", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionDeleted, Importance: 2},
			{ID: "prot-evap", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionProtected, Importance: 9},
			{ID: "heal-evap", Collection: "engram_user", Type: "insight", Action: qdrant.ExpiryActionHealed, Importance: 10},
		},
	}

	m := metrics.New(nil, nil)
	runExpiryTick(context.Background(), store, m, time.Now())

	if got := counterValue(t, m.EvaporationHardDeletedTotal.WithLabelValues("event")); got != 1 {
		t.Fatalf("EvaporationHardDeletedTotal{event}=%v, want 1 (only del-evap)", got)
	}
	if got := counterValue(t, m.EvaporationHardDeletedTotal.WithLabelValues("insight")); got != 0 {
		t.Fatalf("EvaporationHardDeletedTotal{insight}=%v, want 0 (healed, not deleted)", got)
	}
}

// TestRunExpiryTickDetailedAudits verifies the detailed branch: deleted and
// healed outcomes are written to the audit JSONL, protected outcomes are NOT
// (only surfaced as counts), and the log line carries ids=/healed=/protected=.
func TestRunExpiryTickDetailedAudits(t *testing.T) {
	auditPath := filepath.Join(t.TempDir(), "deletes.jsonl")
	t.Setenv("ENGRAM_DELETE_AUDIT_PATH", auditPath)

	store := &detailedStore{
		expiryObsStore: &expiryObsStore{},
		records: []qdrant.ExpiryRecord{
			{ID: "del-1", Collection: "engram_user", Type: "insight", Action: qdrant.ExpiryActionDeleted, Importance: 3, ContentPrefix: "low value note"},
			{ID: "heal-1", Collection: "engram_user", Type: "insight", Action: qdrant.ExpiryActionHealed, Importance: 10, ContentPrefix: "P17 insight"},
			{ID: "prot-1", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionProtected, Importance: 9},
			{ID: "prot-2", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionProtected, Importance: 8},
		},
	}

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	prev := setStderrWriter(w)
	runExpiryTick(context.Background(), store, nil, time.Now())
	setStderrWriter(prev)
	_ = w.Close()
	logBytes, _ := io.ReadAll(r)
	out := string(logBytes)

	if !strings.Contains(out, "[expiry] deleted 1 expired memories") {
		t.Fatalf("log missing deleted line: %q", out)
	}
	for _, want := range []string{"ids=[del-1]", "healed=1 [heal-1]", "protected=2 [prot-1 prot-2]"} {
		if !strings.Contains(out, want) {
			t.Fatalf("log missing %q; got %q", want, out)
		}
	}

	f, err := os.Open(auditPath)
	if err != nil {
		t.Fatalf("audit file not written: %v", err)
	}
	defer func() { _ = f.Close() }()

	var ids []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var rec deleteAuditRecord
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			t.Fatalf("bad audit line %q: %v", sc.Text(), err)
		}
		if rec.Path != "expiry_tick" {
			t.Fatalf("audit path=%q, want expiry_tick", rec.Path)
		}
		ids = append(ids, rec.ID)
	}
	if len(ids) != 2 {
		t.Fatalf("want 2 audit records (deleted+healed), got %d: %v", len(ids), ids)
	}
	got := strings.Join(ids, ",")
	if !strings.Contains(got, "del-1") || !strings.Contains(got, "heal-1") {
		t.Fatalf("audit ids=%v, want del-1 and heal-1", ids)
	}
	if strings.Contains(got, "prot-") {
		t.Fatalf("protected ids must NOT be audited; got %v", ids)
	}
}

// TestWriteDeleteAuditAppendsAndPerms verifies append-only JSONL behaviour and
// 0600/0700 permissions using the env override under t.TempDir().
func TestWriteDeleteAuditAppendsAndPerms(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "siri")
	auditPath := filepath.Join(dir, "deletes.jsonl")
	t.Setenv("ENGRAM_DELETE_AUDIT_PATH", auditPath)

	writeDeleteAudit(deleteAuditRecord{Path: "memory_delete", ID: "a"})
	writeDeleteAudit(deleteAuditRecord{Path: "memory_update", ID: "b"})

	b, err := os.ReadFile(auditPath)
	if err != nil {
		t.Fatalf("read audit: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 appended lines, got %d: %q", len(lines), string(b))
	}

	fi, err := os.Stat(auditPath)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm() != 0o600 {
		t.Fatalf("file perm=%v, want 0600", fi.Mode().Perm())
	}
	di, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if di.Mode().Perm() != 0o700 {
		t.Fatalf("dir perm=%v, want 0700", di.Mode().Perm())
	}
}

// TestScoredDeleteAudits maps ScoredMemory to audit records for the API delete
// paths.
func TestScoredDeleteAudits(t *testing.T) {
	mems := []memory.ScoredMemory{
		{Memory: memory.Memory{ID: "x", Type: memory.TypeInsight, Collection: "engram_user", Importance: 7, Content: "one\ntwo"}},
	}
	recs := scoredDeleteAudits("memory_delete", mems)
	if len(recs) != 1 {
		t.Fatalf("want 1 record, got %d", len(recs))
	}
	if recs[0].Path != "memory_delete" || recs[0].ID != "x" || recs[0].ContentPrefix != "one two" {
		t.Fatalf("unexpected record: %+v", recs[0])
	}
}
