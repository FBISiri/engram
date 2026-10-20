package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/statedir"
)

// deleteAuditRecord is one JSON line in the append-only id-level delete audit
// log (P17 R3/P1). One record is written per physically deleted memory and per
// one-time self-heal event; "protected" skips are NOT written (they would spam
// the file every tick) and are surfaced as counts in the expiry log line instead.
type deleteAuditRecord struct {
	TS            string  `json:"ts"`                       // RFC3339
	Path          string  `json:"path"`                     // expiry_tick | memory_delete | memory_update
	ID            string  `json:"id"`
	Collection    string  `json:"collection,omitempty"`
	Type          string  `json:"type,omitempty"`
	Importance    float64 `json:"importance"`
	CreatedAt     float64 `json:"created_at"`
	ValidUntil    float64 `json:"valid_until"`
	ContentPrefix string  `json:"content_prefix,omitempty"`
	Reason        string  `json:"reason,omitempty"`
	Action        string  `json:"action,omitempty"`
}

var deleteAuditMu sync.Mutex

// deleteAuditPath resolves the audit file path: ENGRAM_DELETE_AUDIT_PATH when
// set, else <statedir.Dir()>/engram_deletes.jsonl.
//
// It returns an error (instead of a bare $HOME fallback) so that under `go
// test` statedir.Dir() refuses the /root/.siri fallback and the audit write is
// skipped rather than clobbering the production audit log. Tests must set
// ENGRAM_DELETE_AUDIT_PATH or ENGRAM_STATE_DIR (e.g. t.TempDir()).
func deleteAuditPath() (string, error) {
	if p := os.Getenv("ENGRAM_DELETE_AUDIT_PATH"); p != "" {
		return p, nil
	}
	dir, err := statedir.Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "engram_deletes.jsonl"), nil
}

// writeDeleteAudit appends records to the audit log as JSONL. It is
// concurrency-safe and NEVER aborts a delete/expiry tick: any failure (mkdir,
// open, encode) is logged via stderrf and swallowed.
func writeDeleteAudit(records ...deleteAuditRecord) {
	if len(records) == 0 {
		return
	}
	path, err := deleteAuditPath()
	if err != nil {
		stderrf("[delete-audit] resolve path: %v\n", err)
		return
	}

	deleteAuditMu.Lock()
	defer deleteAuditMu.Unlock()

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		stderrf("[delete-audit] mkdir %s: %v\n", filepath.Dir(path), err)
		return
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		stderrf("[delete-audit] open %s: %v\n", path, err)
		return
	}
	defer func() { _ = f.Close() }()

	enc := json.NewEncoder(f)
	for i := range records {
		if records[i].TS == "" {
			records[i].TS = time.Now().Format(time.RFC3339)
		}
		if err := enc.Encode(&records[i]); err != nil {
			stderrf("[delete-audit] encode: %v\n", err)
			return
		}
	}
}

// auditContentPrefix returns a single-line, <=120-rune prefix of content.
func auditContentPrefix(content string) string {
	s := strings.ReplaceAll(content, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	r := []rune(s)
	if len(r) > 120 {
		r = r[:120]
	}
	return strings.TrimSpace(string(r))
}

// scoredDeleteAudits builds audit records for a set of memories removed by a
// hard-delete API path (memory_delete / memory_update old-id delete).
func scoredDeleteAudits(path string, mems []memory.ScoredMemory) []deleteAuditRecord {
	ts := time.Now().Format(time.RFC3339)
	out := make([]deleteAuditRecord, 0, len(mems))
	for i := range mems {
		m := &mems[i]
		out = append(out, deleteAuditRecord{
			TS:            ts,
			Path:          path,
			ID:            m.ID,
			Collection:    m.Collection,
			Type:          string(m.Type),
			Importance:    m.Importance,
			CreatedAt:     m.CreatedAt,
			ValidUntil:    m.ValidUntil,
			ContentPrefix: auditContentPrefix(m.Content),
			Action:        "deleted",
		})
	}
	return out
}
