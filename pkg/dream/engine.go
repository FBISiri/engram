package dream

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/embedding"
	"github.com/FBISiri/engram/pkg/llm"
	"github.com/FBISiri/engram/pkg/memory"
)

// Phase identifies a single dream engine phase.
type Phase string

const (
	PhaseOrient      Phase = "orient"
	PhaseGather      Phase = "gather"
	PhaseConsolidate Phase = "consolidate"
	PhasePrune       Phase = "prune"
)

// AllPhases in execution order.
var AllPhases = []Phase{PhaseOrient, PhaseGather, PhaseConsolidate, PhasePrune}

// skillDiffHoldThreshold is the maximum tolerated skill-diff failure rate. When
// the fraction of failed per-skill proposals exceeds this value the run is
// considered degraded and last_run is HELD (the gate budget is NOT consumed), so
// a mostly-empty run does not waste the 20h window.
const skillDiffHoldThreshold = 0.5

// ErrSkillDiffPartial is returned by Run when one or more skill-diff proposals
// failed. It is a sentinel so main() can map it to a dedicated non-zero exit
// code — a LIVE dream-run whose skill-diff sub-step was (partially) broken must
// not be able to exit 0.
var ErrSkillDiffPartial = errors.New("skill-diff partial failure")

// Config holds Dream Engine runtime configuration.
type Config struct {
	DryRun bool   // Read-only mode: no writes to memory store or files.
	Phase  string // Run only this phase (empty = all phases).
}

// Engine orchestrates the four-phase dream cycle.
type Engine struct {
	store    memory.Store
	embedder embedding.Embedder
	cfg      Config
	log      *Log
}

// Log collects structured output from each phase.
type Log struct {
	StartedAt     string     `json:"started_at"`
	DryRun        bool       `json:"dry_run"`
	Phases        []PhaseLog `json:"phases"`
	Summary       string     `json:"summary,omitempty"`
	SkillDiffPath string     `json:"skill_diff_path,omitempty"` // path to skill diff draft file
	HasSkillDiff  bool       `json:"has_skill_diff,omitempty"`  // true if diff was generated

	// Skill-diff observability (R1/R3): counts and last_run decision.
	SkillDiffRan         bool     `json:"skill_diff_ran,omitempty"`          // true if the per-skill LLM loop actually ran
	SkillDiffOK          int      `json:"skill_diff_ok,omitempty"`           // proposals successfully generated
	SkillDiffFailed      int      `json:"skill_diff_failed,omitempty"`       // proposals that failed (e.g. llm error)
	SkillDiffTotal       int      `json:"skill_diff_total,omitempty"`        // candidate skills attempted (ok+failed)
	SkillDiffFailedNames []string `json:"skill_diff_failed_names,omitempty"` // names of skills whose proposal failed
	LastRunAdvanced      bool     `json:"last_run_advanced,omitempty"`       // whether UpdateRunTimestamp was called
}

// PhaseLog records a single phase's execution.
type PhaseLog struct {
	Name      string   `json:"name"`
	StartedAt string   `json:"started_at"`
	Duration  string   `json:"duration"`
	Items     []string `json:"items,omitempty"` // human-readable action items
	Error     string   `json:"error,omitempty"`
}

// NewEngine creates a Dream Engine instance.
func NewEngine(store memory.Store, embedder embedding.Embedder, cfg Config) *Engine {
	return &Engine{
		store:    store,
		embedder: embedder,
		cfg:      cfg,
		log: &Log{
			StartedAt: time.Now().UTC().Format(time.RFC3339),
			DryRun:    cfg.DryRun,
		},
	}
}

// Run executes the dream cycle (all phases or a single phase).
func (e *Engine) Run(ctx context.Context) error {
	phases := AllPhases
	if e.cfg.Phase != "" {
		p := Phase(e.cfg.Phase)
		valid := false
		for _, ap := range AllPhases {
			if ap == p {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("unknown phase: %s (valid: orient, gather, consolidate, prune)", e.cfg.Phase)
		}
		phases = []Phase{p}
	}

	if !e.cfg.DryRun {
		if err := WritePIDLock(); err != nil {
			return fmt.Errorf("write pid lock: %w", err)
		}
		defer ReleasePIDLock()
	}

	for _, p := range phases {
		plog := PhaseLog{
			Name:      string(p),
			StartedAt: time.Now().UTC().Format(time.RFC3339),
		}
		start := time.Now()

		var err error
		switch p {
		case PhaseOrient:
			plog.Items, err = e.orient(ctx)
		case PhaseGather:
			plog.Items, err = e.gather(ctx)
		case PhaseConsolidate:
			plog.Items, err = e.consolidate(ctx)
		case PhasePrune:
			plog.Items, err = e.prune(ctx)
		}

		plog.Duration = time.Since(start).Truncate(time.Millisecond).String()
		if err != nil {
			plog.Error = err.Error()
		}
		e.log.Phases = append(e.log.Phases, plog)

		if err != nil {
			return fmt.Errorf("phase %s: %w", p, err)
		}
	}

	// Skill diff — runs after all phases (or only when phase == "prune" or "").
	// Generates a skill improvement draft and writes it to workspace.
	shouldRunSkillDiff := e.cfg.Phase == "" || e.cfg.Phase == "prune"
	if shouldRunSkillDiff {
		res, diffItems, diffErr := e.skillDiff(ctx)
		if diffErr != nil {
			fmt.Fprintf(os.Stderr, "warning: skill diff failed: %v\n", diffErr)
		} else if res != nil {
			if res.draftPath != "" {
				e.log.SkillDiffPath = res.draftPath
				e.log.HasSkillDiff = true
			}
			if res.ran {
				e.log.SkillDiffRan = true
				e.log.SkillDiffOK = res.ok
				e.log.SkillDiffFailed = res.failed
				e.log.SkillDiffTotal = res.total
				e.log.SkillDiffFailedNames = res.failedNames
			}
		}
		// Log skill diff items in the prune phase log if it exists, else append a new entry.
		if len(diffItems) > 0 {
			for i := range e.log.Phases {
				if e.log.Phases[i].Name == "prune" {
					e.log.Phases[i].Items = append(e.log.Phases[i].Items, diffItems...)
					break
				}
			}
		}
	}

	// Update run timestamp (only if full run, not single-phase, and not dry-run).
	// R3: HOLD last_run (skip the write) when skill-diff was too broken to justify
	// consuming the gate budget, so a degraded run doesn't waste the 20h window.
	if e.cfg.Phase == "" && !e.cfg.DryRun {
		hold := shouldHoldLastRun(e.log.SkillDiffRan, e.log.SkillDiffFailed, e.log.SkillDiffTotal)
		if hold {
			fmt.Fprintf(os.Stderr, "last_run HELD: skill-diff %d/%d failed (>%.0f%% threshold); gate budget NOT consumed\n",
				e.log.SkillDiffFailed, e.log.SkillDiffTotal, skillDiffHoldThreshold*100)
			e.log.LastRunAdvanced = false
		} else {
			if err := UpdateRunTimestamp(); err != nil {
				return fmt.Errorf("update timestamp: %w", err)
			}
			e.log.LastRunAdvanced = true
		}
	}

	e.log.Summary = e.buildSummary()

	// Write report to workspace.
	if err := e.writeReport(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not write report: %v\n", err)
	}

	// R2: surface skill-diff partial failure as a non-zero exit signal. The
	// report and log have already been written above so observability output is
	// preserved even though we return a (sentinel) error here. Gated on !DryRun:
	// a --dry-run is a read-only preview and keeps its always-succeed contract
	// (its degradation is still visible in the summary counts and the draft
	// Coverage header) — the exit-3 signal is reserved for LIVE runs, the
	// dangerous case that advances last_run.
	if !e.cfg.DryRun && e.log.SkillDiffFailed > 0 {
		return fmt.Errorf("%w: %d of %d skill-diff proposals failed", ErrSkillDiffPartial, e.log.SkillDiffFailed, e.log.SkillDiffTotal)
	}

	return nil
}

// PrintLog outputs the structured log as JSON to stdout.
func (e *Engine) PrintLog() error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(e.log)
}

// orient scans memory health and produces a snapshot.
func (e *Engine) orient(ctx context.Context) ([]string, error) {
	var items []string

	// Collection stats.
	stats, err := e.store.Stats(ctx)
	if err != nil {
		return nil, fmt.Errorf("get stats: %w", err)
	}
	items = append(items, fmt.Sprintf("total memories: %d (indexed: %d, status: %s)",
		stats.PointCount, stats.IndexedCount, stats.Status))

	// Count by type.
	for _, t := range []memory.MemoryType{memory.TypeIdentity, memory.TypeEvent, memory.TypeInsight, memory.TypeDirective} {
		allOfType, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
			Limit:   500,
			Filters: []memory.Filter{{Field: "type", Op: memory.OpEq, Value: string(t)}},
		})
		if err != nil {
			items = append(items, fmt.Sprintf("type %s: error (%v)", t, err))
			continue
		}
		items = append(items, fmt.Sprintf("type %s: %d memories", t, len(allOfType)))
	}

	// Recent activity: memories created in last 7 days.
	sevenDaysAgo := float64(time.Now().Add(-7 * 24 * time.Hour).Unix())
	recent, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit:   100,
		Filters: []memory.Filter{{Field: "created_at", Op: memory.OpGte, Value: sevenDaysAgo}},
	})
	if err != nil {
		items = append(items, fmt.Sprintf("recent 7d: error (%v)", err))
	} else {
		items = append(items, fmt.Sprintf("recent 7d: %d memories", len(recent)))
	}

	return items, nil
}

// gather finds consolidation candidates.
func (e *Engine) gather(ctx context.Context) ([]string, error) {
	var items []string

	// 1. Recent events (last 7 days) — candidates for consolidation.
	sevenDaysAgo := float64(time.Now().Add(-7 * 24 * time.Hour).Unix())
	recentEvents, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit: 200,
		Filters: []memory.Filter{
			{Field: "type", Op: memory.OpEq, Value: string(memory.TypeEvent)},
			{Field: "created_at", Op: memory.OpGte, Value: sevenDaysAgo},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("scroll recent events: %w", err)
	}
	items = append(items, fmt.Sprintf("recent events (7d): %d", len(recentEvents)))

	// 2. Low importance + old memories (candidates for pruning).
	thirtyDaysAgo := float64(time.Now().Add(-30 * 24 * time.Hour).Unix())
	lowImportance, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit: 200,
		Filters: []memory.Filter{
			{Field: "importance", Op: memory.OpLte, Value: 4.0},
			{Field: "created_at", Op: memory.OpLte, Value: thirtyDaysAgo},
		},
	})
	if err != nil {
		items = append(items, fmt.Sprintf("low importance (>30d): error (%v)", err))
	} else {
		items = append(items, fmt.Sprintf("low importance (<=4, >30d old): %d", len(lowImportance)))
	}

	// 3. Expired memories (valid_until > 0 and valid_until < now).
	// Note: Scroll already filters out expired, so we need a direct query.
	// For now, report that expired memories are auto-excluded by the store.
	items = append(items, "expired memories: auto-excluded by store filters")

	// 4. Never-accessed memories older than 14 days.
	fourteenDaysAgo := float64(time.Now().Add(-14 * 24 * time.Hour).Unix())
	neverAccessed, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit: 200,
		Filters: []memory.Filter{
			{Field: "access_count", Op: memory.OpEq, Value: int64(0)},
			{Field: "created_at", Op: memory.OpLte, Value: fourteenDaysAgo},
		},
	})
	if err != nil {
		items = append(items, fmt.Sprintf("never-accessed (>14d): error (%v)", err))
	} else {
		items = append(items, fmt.Sprintf("never-accessed (>14d old): %d", len(neverAccessed)))
		for i, m := range neverAccessed {
			if i >= 5 {
				items = append(items, fmt.Sprintf("  ... and %d more", len(neverAccessed)-5))
				break
			}
			summary := m.Content
			if len(summary) > 80 {
				summary = summary[:80] + "..."
			}
			items = append(items, fmt.Sprintf("  - [%s] %s (importance=%.0f)", m.ID[:8], summary, m.Importance))
		}
	}

	return items, nil
}

// consolidate merges events and generates insights using the LLM.
// In dry-run mode, only reports what would be done.
func (e *Engine) consolidate(ctx context.Context) ([]string, error) {
	var items []string

	// Scan recent events grouped by tags for potential merging.
	sevenDaysAgo := float64(time.Now().Add(-7 * 24 * time.Hour).Unix())
	events, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit: 200,
		Filters: []memory.Filter{
			{Field: "type", Op: memory.OpEq, Value: string(memory.TypeEvent)},
			{Field: "created_at", Op: memory.OpGte, Value: sevenDaysAgo},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("scroll events for consolidation: %w", err)
	}

	// Group events by primary tag.
	tagGroups := map[string][]memory.Memory{}
	for _, m := range events {
		// W17 v1.1: skip recent reflection-origin memories (boundary isolation).
		if isRecentReflection(m) {
			continue
		}
		tag := "untagged"
		if len(m.Tags) > 0 {
			tag = m.Tags[0]
		}
		tagGroups[tag] = append(tagGroups[tag], m)
	}

	// Sort tags for deterministic output.
	sortedTags := make([]string, 0, len(tagGroups))
	for tag := range tagGroups {
		sortedTags = append(sortedTags, tag)
	}
	sort.Strings(sortedTags)

	// Track consolidation stats.
	var mergedCount, newInsights, skipped int

	// Find groups with 3+ events (consolidation candidates).
	for _, tag := range sortedTags {
		group := tagGroups[tag]
		if len(group) < 3 {
			continue
		}

		items = append(items, fmt.Sprintf("consolidation candidate: tag=%q (%d events)", tag, len(group)))

		if e.cfg.DryRun {
			items = append(items, "  [dry-run] would merge into 1 insight")
			skipped++
			continue
		}

		// Build prompt for the LLM.
		insight, sourceIDs, err := e.generateInsight(ctx, tag, group)
		if err != nil {
			items = append(items, fmt.Sprintf("  error generating insight for tag=%q: %v", tag, err))
			skipped++
			continue
		}

		// Compute average importance of source events (capped at 8).
		var totalImportance float64
		for _, m := range group {
			totalImportance += m.Importance
		}
		avgImportance := totalImportance / float64(len(group))
		if avgImportance > 8 {
			avgImportance = 8
		}

		// Collect all tags from source events (dedup).
		allTagsMap := map[string]bool{tag: true}
		for _, m := range group {
			for _, t := range m.Tags {
				allTagsMap[t] = true
			}
		}
		allTags := make([]string, 0, len(allTagsMap))
		for t := range allTagsMap {
			allTags = append(allTags, t)
		}
		sort.Strings(allTags)

		// Build the new insight memory.
		insightContent := fmt.Sprintf(
			"[Consolidated from %d events, tag=%q] %s (source_ids: %s)",
			len(group), tag, insight, strings.Join(sourceIDs, ","),
		)

		newMem := memory.New(insightContent,
			memory.WithType(memory.TypeInsight),
			memory.WithSource("agent"),
			memory.WithImportance(avgImportance),
			memory.WithTags(allTags...),
			memory.WithMetadata(map[string]any{"source_type": "reflection"}),
		)

		// Embed and store the new insight.
		if e.embedder != nil {
			vec, embedErr := e.embedder.Embed(ctx, insightContent)
			if embedErr != nil {
				items = append(items, fmt.Sprintf("  embed error for tag=%q: %v (skipping)", tag, embedErr))
				skipped++
				continue
			}
			if storeErr := e.store.Insert(ctx, newMem, vec); storeErr != nil {
				items = append(items, fmt.Sprintf("  store error for tag=%q: %v", tag, storeErr))
				skipped++
				continue
			}
		} else {
			// No embedder available: skip insertion rather than storing a
			// zero vector. A zero vector with hardcoded dimension would fail
			// if the collection dimension differs (e.g. 1536 != 1024).
			items = append(items, fmt.Sprintf("  skipped (no embedder) for tag=%q", tag))
			skipped++
			continue
		}

		mergedCount += len(group)
		newInsights++
		items = append(items, fmt.Sprintf("  merged %d events → new insight (id=%s, importance=%.0f)", len(group), newMem.ID[:8], avgImportance))

		// Mark source events as superseded by the new insight to prevent
		// re-consolidation on subsequent dream runs.
		for _, srcID := range sourceIDs {
			if updateErr := e.store.Update(ctx, srcID, map[string]any{
				"superseded_by": newMem.ID,
			}); updateErr != nil {
				items = append(items, fmt.Sprintf("  warn: failed to mark source %s as superseded: %v", srcID[:8], updateErr))
			}
		}
	}

	if mergedCount == 0 && newInsights == 0 && skipped == 0 {
		items = append(items, "no consolidation candidates found")
	} else {
		items = append(items, fmt.Sprintf("consolidation result: merged_count=%d, new_insights=%d, skipped=%d", mergedCount, newInsights, skipped))
	}

	// Usage-frequency weighting (Cognee-inspired).
	// Paginate through the full collection — single Scroll(Limit=500) silently
	// truncates for collections > 500 points (W16 has 783+). Cap at a defensive
	// upper bound to avoid runaway memory on pathological collections.
	allMemories, err := scrollAll(ctx, e.store, memory.ScrollOptions{}, 5000)
	if err != nil {
		items = append(items, fmt.Sprintf("usage-frequency scan: error (%v)", err))
	} else {
		var highAccess, lowAccess int
		fourteenDaysAgo := float64(time.Now().Add(-14 * 24 * time.Hour).Unix())
		for _, m := range allMemories {
			// W17 v1.1: skip recent reflection-origin insights for boundary isolation.
			if isRecentReflection(m) {
				continue
			}
			if m.Type == memory.TypeInsight && m.AccessCount > 10 {
				highAccess++
				if !e.cfg.DryRun {
					newImportance := m.Importance + 1
					if newImportance > 10 {
						newImportance = 10
					}
					if newImportance != m.Importance {
						_ = e.store.Update(ctx, m.ID, map[string]any{"importance": newImportance})
					}
				}
			}
			if m.AccessCount == 0 && m.CreatedAt < fourteenDaysAgo {
				lowAccess++
				if !e.cfg.DryRun {
					newImportance := m.Importance - 1
					minImportance := m.Importance - 2
					if minImportance < 1 {
						minImportance = 1
					}
					if newImportance < minImportance {
						newImportance = minImportance
					}
					if newImportance != m.Importance {
						_ = e.store.Update(ctx, m.ID, map[string]any{"importance": newImportance})
					}
				}
			}
		}
		prefix := ""
		if e.cfg.DryRun {
			prefix = "[dry-run] would adjust: "
		}
		items = append(items, fmt.Sprintf("%susage-frequency: %d high-access insights (+importance), %d dormant memories (-importance)",
			prefix, highAccess, lowAccess))
	}

	return items, nil
}

// generateInsight calls the LLM to produce a consolidated insight string from a group of events.
// Returns the insight text and the list of source memory IDs.
func (e *Engine) generateInsight(ctx context.Context, tag string, group []memory.Memory) (string, []string, error) {

	// Build compact event summaries for the prompt.
	var sb strings.Builder
	fmt.Fprintf(&sb,
		"You are a memory consolidation AI. Below are %d related memory events tagged %q, "+
			"recorded over the past 7 days. Synthesize them into a single concise insight (2-4 sentences). "+
			"Focus on patterns, trends, or key conclusions — not just summarizing individual events. "+
			"Write in English, third-person style (e.g. \"Siri has been...\", \"Frank tends to...\").\n\n",
		len(group), tag,
	)
	sb.WriteString("Events:\n")
	for i, m := range group {
		content := m.Content
		if len(content) > 200 {
			content = content[:200] + "..."
		}
		fmt.Fprintf(&sb, "%d. [%s] %s\n", i+1, time.Unix(int64(m.CreatedAt), 0).Format("2006-01-02"), content)
	}
	sb.WriteString("\nInsight:")

	insight, err := llm.Call(ctx, sb.String())
	if err != nil {
		return "", nil, err
	}

	// Collect source IDs.
	sourceIDs := make([]string, len(group))
	for i, m := range group {
		sourceIDs[i] = m.ID
	}

	return insight, sourceIDs, nil
}

// pruneMinAge is the minimum age a memory must reach before prune may delete
// it. Freshly written memories are born with access_count=0, so without an age
// floor any low-importance memory written today (e.g. an inbound email record)
// would be deleted at the very next dream run. The gather phase already applies
// age floors; prune must too. See the 2026-09-05 8980c8a1 incident.
const pruneMinAge = 7 * 24 * time.Hour

// prune cleans up low-value memories.
func (e *Engine) prune(ctx context.Context) ([]string, error) {
	var items []string

	now := time.Now()
	cutoff := float64(now.Add(-pruneMinAge).Unix())

	// Find memories with importance <= 3 AND access_count = 0 AND age >= 7d.
	candidates, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit: 200,
		Filters: []memory.Filter{
			{Field: "importance", Op: memory.OpLte, Value: 3.0},
			{Field: "access_count", Op: memory.OpEq, Value: int64(0)},
			{Field: "created_at", Op: memory.OpLte, Value: cutoff},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("scroll prune candidates: %w", err)
	}

	// Defense in depth: a store whose filter support is incomplete must still
	// never delete a <7d memory. Re-filter in memory with the same cutoff.
	eligible := make([]memory.Memory, 0, len(candidates))
	skipped := 0
	for _, m := range candidates {
		if m.CreatedAt <= cutoff {
			eligible = append(eligible, m)
		} else {
			skipped++
		}
	}

	items = append(items, fmt.Sprintf("prune candidates (importance<=3, access_count=0, age>=7d): %d", len(eligible)))
	if skipped > 0 {
		items = append(items, fmt.Sprintf("prune skipped (age<7d): %d", skipped))
	}

	if len(eligible) > 0 {
		verb := "deleted"
		if e.cfg.DryRun {
			verb = "would-delete"
		}
		ids := make([]string, 0, len(eligible))
		for _, m := range eligible {
			ids = append(ids, m.ID)
			// Truncate content by runes (not bytes) so a multibyte UTF-8
			// character is never split.
			summary := m.Content
			if r := []rune(summary); len(r) > 60 {
				summary = string(r[:60])
			}
			ageDays := int(now.Sub(time.Unix(int64(m.CreatedAt), 0)).Hours() / 24)
			items = append(items, fmt.Sprintf("  - %s [%s] %s... (type=%s, importance=%.0f, age=%dd)",
				verb, m.ID, summary, m.Type, m.Importance, ageDays))
		}

		if e.cfg.DryRun {
			items = append(items, fmt.Sprintf("[dry-run] would delete %d memories", len(ids)))
		} else {
			deleted, err := e.store.Delete(ctx, ids)
			if err != nil {
				items = append(items, fmt.Sprintf("delete error: %v", err))
			} else {
				items = append(items, fmt.Sprintf("deleted %d memories", deleted))
			}
		}
	}

	return items, nil
}

// scrollAll paginates through the memory store's Scroll endpoint, collecting up
// to maxTotal memories. A single Scroll call is bounded by opts.Limit (default
// 50 in the store), so long-lived collections can exceed any fixed page size
// silently. scrollAll walks the cursor until either the store returns fewer
// than the page limit (end of data) or maxTotal is reached.
//
// opts.Limit sets page size; if unset, a sensible default (200) is used.
// Filters and other ScrollOptions fields are preserved across pages.
func scrollAll(ctx context.Context, store memory.Store, opts memory.ScrollOptions, maxTotal int) ([]memory.Memory, error) {
	if opts.Limit == 0 {
		opts.Limit = 200
	}
	pageLimit := opts.Limit
	var all []memory.Memory
	offset := opts.Offset

	for {
		pageOpts := opts
		pageOpts.Offset = offset
		page, nextOffset, err := store.Scroll(ctx, pageOpts)
		if err != nil {
			return nil, err
		}
		all = append(all, page...)

		if nextOffset == "" || len(page) < pageLimit {
			break
		}
		if maxTotal > 0 && len(all) >= maxTotal {
			if len(all) > maxTotal {
				all = all[:maxTotal]
			}
			break
		}
		offset = nextOffset
	}
	return all, nil
}

// buildSummary generates a human-readable summary of the dream run.
func (e *Engine) buildSummary() string {
	total := 0
	for _, p := range e.log.Phases {
		total += len(p.Items)
	}
	mode := "LIVE"
	if e.cfg.DryRun {
		mode = "DRY-RUN"
	}
	summary := fmt.Sprintf("Dream run completed (%s): %d phases, %d items logged", mode, len(e.log.Phases), total)
	if e.log.SkillDiffRan {
		summary += fmt.Sprintf("; skill-diff: %d/%d ok, %d failed", e.log.SkillDiffOK, e.log.SkillDiffTotal, e.log.SkillDiffFailed)
	}
	if e.cfg.Phase == "" && !e.cfg.DryRun {
		if e.log.LastRunAdvanced {
			summary += "; last_run advanced"
		} else {
			summary += "; last_run HELD (degraded run)"
		}
	}
	return summary
}

// shouldHoldLastRun reports whether last_run must be HELD (not advanced) because
// the skill-diff sub-step failed on more than skillDiffHoldThreshold of its
// candidates (R3). A run that never ran skill-diff, or ran it with zero
// candidates, never holds.
func shouldHoldLastRun(ran bool, failed, total int) bool {
	if !ran || total <= 0 {
		return false
	}
	return float64(failed)/float64(total) > skillDiffHoldThreshold
}

// buildDraftHeader builds the skill-diff draft header (R4). Reading only this
// header must be enough to tell whether the file is a complete or partial run:
// it carries the Mode, the Coverage (ok/total) and the explicit list of skills
// that were NOT covered (the failed ones).
func buildDraftHeader(date, modeLabel string, ok, total int, failedNames []string) string {
	var h strings.Builder
	fmt.Fprintf(&h, "# Skill Diff Draft — %s\n\n> Generated by Dream Engine Phase 4  \n> Mode: %s  \n> Coverage: %d/%d\n", date, modeLabel, ok, total)
	if len(failedNames) > 0 {
		fmt.Fprintf(&h, "> Uncovered skills (proposal failed): %s\n", strings.Join(failedNames, ", "))
	} else {
		h.WriteString("> Uncovered skills (proposal failed): none\n")
	}
	h.WriteString("\n---\n\n")
	return h.String()
}

// writeReport writes the dream run report to workspace.
func (e *Engine) writeReport() error {
	wsDir := "/data/armyoftheagent/workspace"
	if err := os.MkdirAll(wsDir, 0755); err != nil {
		return err
	}

	date := time.Now().Format("2006-01-02")
	filename := fmt.Sprintf("dream-run-%s.md", date)
	path := filepath.Join(wsDir, filename)

	mode := "DRY-RUN"
	if !e.cfg.DryRun {
		mode = "LIVE"
	}

	content := fmt.Sprintf("# Dream Run Report — %s (%s)\n\n", date, mode)
	content += fmt.Sprintf("> Started: %s\n\n", e.log.StartedAt)

	for _, p := range e.log.Phases {
		content += fmt.Sprintf("## Phase: %s\n\n", p.Name)
		content += fmt.Sprintf("- Duration: %s\n", p.Duration)
		if p.Error != "" {
			content += fmt.Sprintf("- **Error:** %s\n", p.Error)
		}
		for _, item := range p.Items {
			content += fmt.Sprintf("- %s\n", item)
		}
		content += "\n"
	}

	if e.log.Summary != "" {
		content += fmt.Sprintf("---\n\n%s\n", e.log.Summary)
	}
	if e.log.SkillDiffPath != "" {
		content += fmt.Sprintf("\n**Skill Diff Draft:** `%s`\n", e.log.SkillDiffPath)
	}

	return os.WriteFile(path, []byte(content), 0644)
}

// skillDiffResult carries the observable outcome of the skill-diff sub-step so
// Run()/buildSummary() can surface partial failures instead of swallowing them.
type skillDiffResult struct {
	draftPath   string   // path to the written draft, if any
	ran         bool     // true if the per-skill LLM loop actually executed
	ok          int      // proposals successfully generated
	failed      int      // proposals that failed
	total       int      // candidate skills attempted (ok+failed)
	failedNames []string // names of skills whose proposal failed
}

// skillDiff scans Engram insights for skill-improvement signals, calls the LLM to
// generate a structured diff draft per skill, and writes the draft to workspace.
// Returns (result, logItems, error). result is nil when the sub-step short-circuits
// before running the per-skill loop (no skills / no candidates).
// In dry-run mode it still generates the draft (for review) but marks it as dry-run.
func (e *Engine) skillDiff(ctx context.Context) (*skillDiffResult, []string, error) {
	var items []string

	skillsDir := "/data/armyoftheagent/skills"
	wsDir := "/data/armyoftheagent/workspace"

	// List all skill directories.
	entries, err := os.ReadDir(skillsDir)
	if err != nil {
		return nil, nil, fmt.Errorf("read skills dir: %w", err)
	}

	// Collect skill names.
	var skillNames []string
	for _, entry := range entries {
		if entry.IsDir() {
			skillNames = append(skillNames, entry.Name())
		}
	}
	sort.Strings(skillNames)

	if len(skillNames) == 0 {
		items = append(items, "skill diff: no skills found")
		return nil, items, nil
	}

	// Search Engram for insights and directives. Use scrollAll pagination so
	// long-lived collections (> page limit) don't silently drop candidates.
	allInsights, err := scrollAll(ctx, e.store, memory.ScrollOptions{
		Limit:   200,
		Filters: []memory.Filter{{Field: "type", Op: memory.OpEq, Value: string(memory.TypeInsight)}},
	}, 2000)
	if err != nil {
		return nil, nil, fmt.Errorf("scroll insights for skill diff: %w", err)
	}
	allDirectives, _ := scrollAll(ctx, e.store, memory.ScrollOptions{
		Limit:   200,
		Filters: []memory.Filter{{Field: "type", Op: memory.OpEq, Value: string(memory.TypeDirective)}},
	}, 1000)
	allMemories := append(allInsights, allDirectives...)

	// Map each skill to relevant memory entries.
	type skillEntry struct {
		name     string
		skillMD  string
		memories []memory.Memory
	}

	var candidates []skillEntry
	for _, sn := range skillNames {
		snLower := strings.ToLower(sn)
		snNorm := strings.ReplaceAll(snLower, "-", "")

		// Read SKILL.md content.
		skillMDPath := filepath.Join(skillsDir, sn, "SKILL.md")
		skillMDBytes, _ := os.ReadFile(skillMDPath)
		skillMDContent := string(skillMDBytes)

		// Find memories related to this skill.
		var related []memory.Memory
		for _, m := range allMemories {
			contentLower := strings.ToLower(m.Content)
			contentNorm := strings.ReplaceAll(contentLower, "-", "")
			tagMatch := false
			for _, tag := range m.Tags {
				tLower := strings.ToLower(tag)
				tNorm := strings.ReplaceAll(tLower, "-", "")
				if strings.Contains(tNorm, snNorm) || strings.Contains(snNorm, tNorm) {
					tagMatch = true
					break
				}
			}
			contentMatch := strings.Contains(contentNorm, snNorm)
			if tagMatch || contentMatch {
				related = append(related, m)
			}
		}

		if len(related) >= 2 {
			candidates = append(candidates, skillEntry{
				name:     sn,
				skillMD:  skillMDContent,
				memories: related,
			})
		}
	}

	if len(candidates) == 0 {
		items = append(items, "skill diff: no skills with 2+ relevant insights found")
		return nil, items, nil
	}

	items = append(items, fmt.Sprintf("skill diff: %d skills have improvement signals", len(candidates)))

	// Build the combined draft. Body is buffered separately from the header so the
	// header can be built AFTER the loop with the true coverage counts and the list
	// of uncovered (failed) skills (R4).
	date := time.Now().Format("2006-01-02")
	modeLabel := map[bool]string{true: "DRY-RUN", false: "LIVE"}[e.cfg.DryRun]

	var body strings.Builder
	ok, failed := 0, 0
	var failedNames []string

	for _, c := range candidates {
		items = append(items, fmt.Sprintf("  skill: %s (%d relevant memories)", c.name, len(c.memories)))

		// Build LLM prompt.
		var sb strings.Builder
		fmt.Fprintf(&sb,
			"You are a skill improvement analyst for an AI agent named Siri. "+
				"Below is the current SKILL.md for '%s' and a list of recent insights/directives.\n\n"+
				"Task: generate a structured diff proposal for this skill file.\n"+
				"Format your response EXACTLY as:\n\n"+
				"## %s\n\n"+
				"**Proposed Changes:**\n"+
				"- [ADD/MODIFY/REMOVE] <section or rule>: <specific change>\n\n"+
				"**Reasoning:** <1-2 sentences>\n\n"+
				"Only propose changes clearly supported by the insights. If no changes are needed, write \"No changes needed.\"\n\n",
			c.name, c.name,
		)

		skillMDSnippet := c.skillMD
		if len(skillMDSnippet) > 1500 {
			skillMDSnippet = skillMDSnippet[:1500] + "\n...[truncated]"
		}
		fmt.Fprintf(&sb, "### Current SKILL.md:\n```\n%s\n```\n\n", skillMDSnippet)

		sb.WriteString("### Recent Insights & Directives:\n")
		limit := 8
		if len(c.memories) < limit {
			limit = len(c.memories)
		}
		for i, m := range c.memories[:limit] {
			content := m.Content
			if len(content) > 200 {
				content = content[:200] + "..."
			}
			fmt.Fprintf(&sb, "%d. [%s] %s\n", i+1, m.Type, content)
		}
		sb.WriteString("\nDiff proposal:")

		proposal, hErr := llm.Call(ctx, sb.String())
		if hErr != nil {
			failed++
			failedNames = append(failedNames, c.name)
			items = append(items, fmt.Sprintf("  llm error for %s: %v", c.name, hErr))
			fmt.Fprintf(&body, "## %s\n\nError generating proposal: %v\n\n---\n\n", c.name, hErr)
		} else {
			ok++
			items = append(items, fmt.Sprintf("  generated proposal for %s", c.name))
			body.WriteString(proposal + "\n\n---\n\n")
		}
	}

	total := len(candidates)

	// Build the header now that coverage and uncovered skills are known (R4).
	// Reading only this header must be enough to tell the file is a fragment.
	draftContent := buildDraftHeader(date, modeLabel, ok, total, failedNames) + body.String()

	result := &skillDiffResult{
		ran:         true,
		ok:          ok,
		failed:      failed,
		total:       total,
		failedNames: failedNames,
	}

	// Write the draft file.
	if err := os.MkdirAll(wsDir, 0755); err != nil {
		return result, items, fmt.Errorf("create workspace dir: %w", err)
	}
	draftPath := filepath.Join(wsDir, fmt.Sprintf("skill-diff-draft-%s.md", date))
	if writeErr := os.WriteFile(draftPath, []byte(draftContent), 0644); writeErr != nil {
		return result, items, fmt.Errorf("write skill diff draft: %w", writeErr)
	}
	result.draftPath = draftPath

	items = append(items, fmt.Sprintf("skill diff draft written to: %s", draftPath))
	return result, items, nil
}

// ── W17 v1.1 helpers ────────────────────────────────────────────────────────

// isRecentReflection returns true if the memory carries the "source:reflection"
// tag AND was created less than 7 days ago. Such memories are excluded from
// Dream Engine consolidation to prevent premature compression or deletion of
// young, unvalidated reflection-origin insights (W17 v1.1 decision 2).
func isRecentReflection(m memory.Memory) bool {
	hasTag := false
	for _, t := range m.Tags {
		if t == "source:reflection" {
			hasTag = true
			break
		}
	}
	if !hasTag {
		return false
	}
	sevenDaysAgo := float64(time.Now().Add(-7 * 24 * time.Hour).Unix())
	return m.CreatedAt > sevenDaysAgo
}
