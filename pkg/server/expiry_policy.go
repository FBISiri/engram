package server

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// expiryPolicy defines the type-based deletion eligibility criteria.
// A memory must satisfy BOTH importance and age conditions to be a candidate.
type expiryPolicy struct {
	minImportance float64 // importance < this → eligible (unless overridden)
	maxAgeDays    float64 // age > this days → eligible
}

// defaultPolicies maps memory type to its policy.
// TypeIdentity is absent — identity memories are never auto-deleted.
var defaultPolicies = map[memory.MemoryType]expiryPolicy{
	memory.TypeEvent:     {minImportance: 4, maxAgeDays: 30},
	memory.TypeInsight:   {minImportance: 5, maxAgeDays: 90},
	memory.TypeDirective: {minImportance: 6, maxAgeDays: 180},
}

const maxExpiryPerRun = 50

// ExpiryCandidate is a memory eligible for policy-based deletion.
type ExpiryCandidate struct {
	ID             string            `json:"id"`
	Type           memory.MemoryType `json:"type"`
	Importance     float64           `json:"importance"`
	AgeDays        float64           `json:"age_days"`
	ContentPreview string            `json:"content_preview"`
	Tags           []string          `json:"tags"`
}

func hasTag(tags []string, target string) bool {
	for _, t := range tags {
		if t == target {
			return true
		}
	}
	return false
}

// isExpiryCandidate returns true if m is eligible for policy-based deletion.
func isExpiryCandidate(m *memory.Memory, now time.Time) bool {
	policy, ok := defaultPolicies[m.Type]
	if !ok {
		return false // identity or unknown type: never delete
	}

	// Safety: importance >= 8 is always protected
	if m.Importance >= 8 {
		return false
	}

	// Must be below the importance threshold for its type
	if m.Importance >= policy.minImportance {
		return false
	}

	// Compute age threshold (tag overrides lift it to 365 days)
	ageThreshold := policy.maxAgeDays
	if hasTag(m.Tags, "frank-feedback") || hasTag(m.Tags, "directive") {
		ageThreshold = 365
	}

	ageDays := now.Sub(time.Unix(int64(m.CreatedAt), 0)).Hours() / 24
	return ageDays > ageThreshold
}

func contentPreview(s string) string {
	const limit = 100
	if len(s) <= limit {
		return s
	}
	return s[:limit] + "..."
}

// findExpiryCandidates scrolls all active memories and returns up to
// maxExpiryPerRun candidates eligible for policy-based deletion.
func (h *HTTPServer) findExpiryCandidates(ctx *http.Request) ([]ExpiryCandidate, error) {
	now := time.Now()
	var candidates []ExpiryCandidate
	var offset string

	for {
		mems, nextOffset, err := h.srv.store.Scroll(ctx.Context(), memory.ScrollOptions{
			Limit:  100,
			Offset: offset,
		})
		if err != nil {
			return nil, fmt.Errorf("scroll: %w", err)
		}

		for _, m := range mems {
			if !isExpiryCandidate(&m, now) {
				continue
			}
			ageDays := now.Sub(time.Unix(int64(m.CreatedAt), 0)).Hours() / 24
			candidates = append(candidates, ExpiryCandidate{
				ID:             m.ID,
				Type:           m.Type,
				Importance:     m.Importance,
				AgeDays:        ageDays,
				ContentPreview: contentPreview(m.Content),
				Tags:           m.Tags,
			})
			if len(candidates) >= maxExpiryPerRun {
				return candidates, nil
			}
		}

		if nextOffset == "" {
			break
		}
		offset = nextOffset
	}

	return candidates, nil
}

// writeExpirySnapshot saves a markdown snapshot of the deletion candidates
// to /mnt/data/siri-vault/Engram/expiry-snapshots/YYYY-MM-DD.md.
func writeExpirySnapshot(candidates []ExpiryCandidate) (string, error) {
	dir := "/mnt/data/siri-vault/Engram/expiry-snapshots"
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("mkdir %s: %w", dir, err)
	}

	now := time.Now().UTC()
	path := filepath.Join(dir, now.Format("2006-01-02")+".md")

	var sb strings.Builder
	fmt.Fprintf(&sb, "# Engram Expiry Snapshot — %s\n\n", now.Format("2006-01-02"))
	fmt.Fprintf(&sb, "Generated: %s  \n", now.Format(time.RFC3339))
	fmt.Fprintf(&sb, "Total deleted: %d\n\n", len(candidates))
	sb.WriteString("| ID | Type | Importance | Age (days) | Tags | Preview |\n")
	sb.WriteString("|---|---|---|---|---|---|\n")
	for _, c := range candidates {
		tags := strings.Join(c.Tags, ", ")
		preview := strings.ReplaceAll(c.ContentPreview, "|", "\\|")
		fmt.Fprintf(&sb, "| %s | %s | %.1f | %.0f | %s | %s |\n",
			c.ID, c.Type, c.Importance, c.AgeDays, tags, preview)
	}

	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		return "", fmt.Errorf("write snapshot: %w", err)
	}
	return path, nil
}

// handleExpiryCandidates handles GET /memories/expiry-candidates.
// Returns up to 50 memories eligible for policy-based deletion.
func (h *HTTPServer) handleExpiryCandidates(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed — use GET"})
		return
	}

	candidates, err := h.findExpiryCandidates(r)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	if candidates == nil {
		candidates = []ExpiryCandidate{}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"candidates": candidates,
		"count":      len(candidates),
	})
}

// expireResult is the response body for DELETE /memories/expired.
type expireResult struct {
	DeletedCount int    `json:"deleted_count"`
	SkippedCount int    `json:"skipped_count"`
	SnapshotPath string `json:"snapshot_path,omitempty"`
	DryRun       bool   `json:"dry_run"`
}

// handleDeleteExpired handles DELETE /memories/expired.
//
// Query params:
//
//	dry_run=true  (default) — report candidates without deleting
//	dry_run=false&confirm=true — execute deletion with snapshot
func (h *HTTPServer) handleDeleteExpired(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed — use DELETE"})
		return
	}

	q := r.URL.Query()
	dryRun := q.Get("dry_run") != "false"
	confirm := q.Get("confirm") == "true"

	if !dryRun && !confirm {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "confirm=true required when dry_run=false",
		})
		return
	}

	candidates, err := h.findExpiryCandidates(r)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	if dryRun {
		writeJSON(w, http.StatusOK, expireResult{
			DeletedCount: 0,
			SkippedCount: len(candidates),
			DryRun:       true,
		})
		return
	}

	// Write snapshot before deletion so it exists even if delete partially fails.
	snapshotPath, snapErr := writeExpirySnapshot(candidates)
	if snapErr != nil {
		fmt.Fprintf(os.Stderr, "expiry: snapshot write error: %v\n", snapErr)
	}

	ids := make([]string, len(candidates))
	for i, c := range candidates {
		ids[i] = c.ID
	}

	n, err := h.srv.store.Delete(r.Context(), ids)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, expireResult{
		DeletedCount: n,
		SkippedCount: len(candidates) - n,
		SnapshotPath: snapshotPath,
		DryRun:       false,
	})
}
