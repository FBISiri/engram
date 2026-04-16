package dream

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/embedding"
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
	StartedAt       string     `json:"started_at"`
	DryRun          bool       `json:"dry_run"`
	Phases          []PhaseLog `json:"phases"`
	Summary         string     `json:"summary,omitempty"`
	SkillDiffPath   string     `json:"skill_diff_path,omitempty"`   // path to skill diff draft file
	HasSkillDiff    bool       `json:"has_skill_diff,omitempty"`    // true if diff was generated
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
		diffPath, diffItems, diffErr := e.skillDiff(ctx)
		if diffErr != nil {
			fmt.Fprintf(os.Stderr, "warning: skill diff failed: %v\n", diffErr)
		} else if diffPath != "" {
			e.log.SkillDiffPath = diffPath
			e.log.HasSkillDiff = true
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
	if e.cfg.Phase == "" && !e.cfg.DryRun {
		if err := UpdateRunTimestamp(); err != nil {
			return fmt.Errorf("update timestamp: %w", err)
		}
	}

	e.log.Summary = e.buildSummary()

	// Write report to workspace.
	if err := e.writeReport(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not write report: %v\n", err)
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

// consolidate merges events and generates insights using Haiku LLM.
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

		// Build prompt for Haiku.
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
	allMemories, _, err := e.store.Scroll(ctx, memory.ScrollOptions{Limit: 500})
	if err != nil {
		items = append(items, fmt.Sprintf("usage-frequency scan: error (%v)", err))
	} else {
		var highAccess, lowAccess int
		for _, m := range allMemories {
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
			fourteenDaysAgo := float64(time.Now().Add(-14 * 24 * time.Hour).Unix())
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

// generateInsight calls Haiku to produce a consolidated insight string from a group of events.
// Returns the insight text and the list of source memory IDs.
func (e *Engine) generateInsight(ctx context.Context, tag string, group []memory.Memory) (string, []string, error) {
	_ = ctx // future: context-aware LLM calls

	// Build compact event summaries for the prompt.
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf(
		"You are a memory consolidation AI. Below are %d related memory events tagged %q, "+
			"recorded over the past 7 days. Synthesize them into a single concise insight (2-4 sentences). "+
			"Focus on patterns, trends, or key conclusions — not just summarizing individual events. "+
			"Write in English, third-person style (e.g. \"Siri has been...\", \"Frank tends to...\").\n\n",
		len(group), tag,
	))
	sb.WriteString("Events:\n")
	for i, m := range group {
		content := m.Content
		if len(content) > 200 {
			content = content[:200] + "..."
		}
		sb.WriteString(fmt.Sprintf("%d. [%s] %s\n", i+1, time.Unix(int64(m.CreatedAt), 0).Format("2006-01-02"), content))
	}
	sb.WriteString("\nInsight:")

	insight, err := callHaiku(sb.String())
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

// prune cleans up low-value memories.
func (e *Engine) prune(ctx context.Context) ([]string, error) {
	var items []string

	// Find memories with importance <= 3 AND access_count = 0.
	candidates, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit: 200,
		Filters: []memory.Filter{
			{Field: "importance", Op: memory.OpLte, Value: 3.0},
			{Field: "access_count", Op: memory.OpEq, Value: int64(0)},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("scroll prune candidates: %w", err)
	}

	items = append(items, fmt.Sprintf("prune candidates (importance<=3, access_count=0): %d", len(candidates)))

	if len(candidates) > 0 {
		ids := make([]string, 0, len(candidates))
		for i, m := range candidates {
			ids = append(ids, m.ID)
			if i < 10 {
				summary := m.Content
				if len(summary) > 60 {
					summary = summary[:60] + "..."
				}
				items = append(items, fmt.Sprintf("  - [%s] %s (type=%s, importance=%.0f)",
					m.ID[:8], summary, m.Type, m.Importance))
			}
		}
		if len(candidates) > 10 {
			items = append(items, fmt.Sprintf("  ... and %d more", len(candidates)-10))
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
	return fmt.Sprintf("Dream run completed (%s): %d phases, %d items logged", mode, len(e.log.Phases), total)
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

// skillDiff scans Engram insights for skill-improvement signals, calls Haiku to
// generate a structured diff draft per skill, and writes the draft to workspace.
// Returns (draftFilePath, logItems, error).
// In dry-run mode it still generates the draft (for review) but marks it as dry-run.
func (e *Engine) skillDiff(ctx context.Context) (string, []string, error) {
	var items []string

	skillsDir := "/data/armyoftheagent/skills"
	wsDir := "/data/armyoftheagent/workspace"

	// List all skill directories.
	entries, err := os.ReadDir(skillsDir)
	if err != nil {
		return "", nil, fmt.Errorf("read skills dir: %w", err)
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
		return "", items, nil
	}

	// Search Engram for insights and directives.
	allInsights, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit:   500,
		Filters: []memory.Filter{{Field: "type", Op: memory.OpEq, Value: string(memory.TypeInsight)}},
	})
	if err != nil {
		return "", nil, fmt.Errorf("scroll insights for skill diff: %w", err)
	}
	allDirectives, _, _ := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit:   200,
		Filters: []memory.Filter{{Field: "type", Op: memory.OpEq, Value: string(memory.TypeDirective)}},
	})
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
		return "", items, nil
	}

	items = append(items, fmt.Sprintf("skill diff: %d skills have improvement signals", len(candidates)))

	// Build the combined draft.
	date := time.Now().Format("2006-01-02")
	modeLabel := map[bool]string{true: "DRY-RUN", false: "LIVE"}[e.cfg.DryRun]
	draftContent := fmt.Sprintf("# Skill Diff Draft — %s\n\n> Generated by Dream Engine Phase 4  \n> Mode: %s\n\n---\n\n", date, modeLabel)

	for _, c := range candidates {
		items = append(items, fmt.Sprintf("  skill: %s (%d relevant memories)", c.name, len(c.memories)))

		// Build Haiku prompt.
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf(
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
		))

		skillMDSnippet := c.skillMD
		if len(skillMDSnippet) > 1500 {
			skillMDSnippet = skillMDSnippet[:1500] + "\n...[truncated]"
		}
		sb.WriteString(fmt.Sprintf("### Current SKILL.md:\n```\n%s\n```\n\n", skillMDSnippet))

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
			sb.WriteString(fmt.Sprintf("%d. [%s] %s\n", i+1, m.Type, content))
		}
		sb.WriteString("\nDiff proposal:")

		proposal, hErr := callHaiku(sb.String())
		if hErr != nil {
			items = append(items, fmt.Sprintf("  haiku error for %s: %v", c.name, hErr))
			draftContent += fmt.Sprintf("## %s\n\nError generating proposal: %v\n\n---\n\n", c.name, hErr)
		} else {
			items = append(items, fmt.Sprintf("  generated proposal for %s", c.name))
			draftContent += proposal + "\n\n---\n\n"
		}
	}

	// Write the draft file.
	if err := os.MkdirAll(wsDir, 0755); err != nil {
		return "", items, fmt.Errorf("create workspace dir: %w", err)
	}
	draftPath := filepath.Join(wsDir, fmt.Sprintf("skill-diff-draft-%s.md", date))
	if writeErr := os.WriteFile(draftPath, []byte(draftContent), 0644); writeErr != nil {
		return "", items, fmt.Errorf("write skill diff draft: %w", writeErr)
	}

	items = append(items, fmt.Sprintf("skill diff draft written to: %s", draftPath))
	return draftPath, items, nil
}
