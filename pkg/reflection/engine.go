// Package reflection implements the Engram Reflection Engine v1.
//
// The Reflection Engine periodically synthesizes high-level insights from
// unreflected memories, inspired by the Generative Agents paper. It runs
// lighter and more frequently than the Dream Engine (1-3x/day vs 1x/day),
// and focuses on cross-domain pattern discovery rather than same-tag dedup.
//
// Trigger: accumulated importance of unreflected memories >= threshold (default: 40).
// Min interval: 2h. Max per day: 3 runs.
//
// Output: TypeInsight memories with source="system", tagged with
// metadata.reflection_source_ids and metadata.reflected=true on source memories.
package reflection

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/embedding"
	"github.com/FBISiri/engram/pkg/llm"
	"github.com/FBISiri/engram/pkg/memory"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

var tracer = otel.Tracer("engram.reflection")

// Config holds Reflection Engine runtime configuration.
type Config struct {
	// Threshold is the cumulative importance of unreflected memories required
	// to trigger a reflection run. Default: 40.
	Threshold float64

	// MaxInputSize is the maximum number of memories to feed into one reflection
	// cycle. Default: 20.
	MaxInputSize int

	// MinIntervalH is the minimum hours between reflection runs. Default: 2.
	MinIntervalH float64

	// DryRun disables all writes (no new insights, no marking as reflected).
	DryRun bool

	// Force bypasses min-interval and daily-limit gates (Gate 1 & 2).
	Force bool

	// Mode selects the reflection algorithm: "v1" (flat, default) or "v2" (focal point).
	// V2 requires 4x LLM calls but produces higher-quality insights.
	Mode string

	// FocalInputSize is the number of unreflected memories fed to focal question
	// generation (Step 1 of V2). Default: 50.
	FocalInputSize int

	// FocalQuestions is the number of focal questions to generate in V2. Default: 3.
	FocalQuestions int

	// EvidencePerFocal is the number of memories retrieved per focal question
	// via semantic search in V2. Default: 10.
	EvidencePerFocal int

	// AllowedProvenances is a whitelist of provenance values for evidence filtering.
	// Only used when RequireProvenance is true.
	//
	// Deprecated: use ProvenanceFilter. Retained for backward compatibility;
	// migrated into ProvenanceFilter via resolveProvenanceFilter() when
	// ProvenanceFilter is not explicitly enabled.
	AllowedProvenances []string

	// RequireProvenance enables provenance-based filtering in evidence retrieval.
	// Default: false (provenance P0 not yet landed).
	//
	// Deprecated: use ProvenanceFilter. See AllowedProvenances.
	RequireProvenance bool

	// ProvenanceFilter configures provenance-based evidence filtering and
	// write-back enforcement (Phase 2 §4.1). When Enabled is false the legacy
	// RequireProvenance/AllowedProvenances fields are migrated in.
	ProvenanceFilter ProvenanceFilterConfig

	// EvidenceSearchTimeout is the per-question store.Search timeout. Default: 3s, hard max: 10s.
	EvidenceSearchTimeout time.Duration

	// DialecticTimeout is the per-question LLM call timeout for Part3. Default: 15s, hard max: 30s.
	DialecticTimeout time.Duration

	// MinInsightConfidence is the minimum confidence for a dialectic insight to
	// be written back to the store. Below this → counted as Skipped. Default: 0.5.
	MinInsightConfidence float64

	// WriteBackTimeout is the total timeout for Stage 4 write-back. Default: 30s, hard max: 120s.
	WriteBackTimeout time.Duration

	// DebugEvidence enables per-question evidence retrieval diagnostics in RunResult.
	DebugEvidence bool

	// MaxInsightImportance caps the importance of reflection-generated insights.
	// Any DialecticInsight with a higher importance is clamped to this value
	// before write-back. Default: 8.
	MaxInsightImportance int

	// InsightDedupThreshold is the similarity threshold for pre-write dedup.
	// Before writing a new reflection insight, the engine searches existing
	// reflections; if a match exceeds this threshold, the new insight is
	// skipped. Default: 0.78 (catches paraphrases that pass the server 0.92 gate).
	InsightDedupThreshold float64
}

// DefaultConfig returns the default Reflection Engine configuration.
func DefaultConfig() Config {
	return Config{
		Threshold:             40.0,
		MaxInputSize:          20,
		MinIntervalH:          2.0,
		DryRun:                false,
		Mode:                  "v1", // V1 by default; V2 requires explicit opt-in
		FocalInputSize:        50,
		FocalQuestions:        3,
		EvidencePerFocal:      10,
		MaxInsightImportance:  8,
		InsightDedupThreshold: 0.78,
		ProvenanceFilter:      ProvenanceFilterConfig{Enabled: false},
	}
}

// resolveProvenanceFilter returns the effective ProvenanceFilterConfig,
// migrating the legacy RequireProvenance/AllowedProvenances fields when the
// richer ProvenanceFilter is not explicitly enabled (Phase 2 §4.4).
func (c Config) resolveProvenanceFilter() ProvenanceFilterConfig {
	if c.ProvenanceFilter.Enabled {
		pf := c.ProvenanceFilter
		if pf.Mode == "" {
			pf.Mode = ProvenanceModeDefault
		}
		return pf
	}
	if c.RequireProvenance && len(c.AllowedProvenances) > 0 {
		return ProvenanceFilterConfig{
			Enabled:            true,
			Mode:               ProvenanceModeDefault,
			AllowedProvenances: c.AllowedProvenances,
		}
	}
	return ProvenanceFilterConfig{Enabled: false}
}

// RunResult holds the output of a single reflection run.
type RunResult struct {
	Triggered            bool     `json:"triggered"`
	SkipReason           string   `json:"skip_reason,omitempty"`
	InputCount           int      `json:"input_count"`
	InsightsCreated      int      `json:"insights_created"`
	InsightsDedupSkipped int      `json:"insights_dedup_skipped"` // pre-write dedup skips (V1 batch + single-event)
	DraftsWritten        int      `json:"drafts_written"`         // W17 v1.1: confidence<0.6 → Obsidian draft
	SourcesMarked        int      `json:"sources_marked"`
	SourcesOrphaned      int      `json:"sources_orphaned,omitempty"` // IDs deleted between fetch and mark (TOCTOU)
	Duration             string   `json:"duration"`
	TriggerImportance    float64  `json:"trigger_importance,omitempty"`
	DryRun               bool     `json:"dry_run"`
	Errors               []string `json:"errors,omitempty"`

	// V2 fields (populated when Mode == "v2").
	Mode            string   `json:"mode"`                      // "v1-flat" | "v2-focal"
	FocalQuestions  []string `json:"focal_questions,omitempty"` // V2 only
	EvidenceCount   int      `json:"evidence_count,omitempty"`  // V2: evidence set size
	LLMCalls        int      `json:"llm_calls"`                 // total LLM calls
	LLMCostEstimate float64  `json:"llm_cost_estimate_usd"`     // estimated USD cost

	// V2 Part2 fields: per-question evidence retrieval observability.
	EvidenceOverlap   int   `json:"evidence_overlap,omitempty"`
	PerQuestionCounts []int `json:"per_question_counts,omitempty"`
	DroppedNoEvidence int   `json:"dropped_no_evidence,omitempty"`
	EvidenceSearchMs  int64 `json:"evidence_search_ms,omitempty"`

	// V2 Part3 fields: dialectic insight observability.
	DialecticOkCount           int   `json:"dialectic_ok_count,omitempty"`
	DialecticFailedCount       int   `json:"dialectic_failed_count,omitempty"`
	DialecticDroppedNoEvidence int   `json:"dialectic_dropped_no_evidence,omitempty"`
	DialecticDroppedLowConf    int   `json:"dialectic_dropped_low_conf,omitempty"`
	DialecticLLMCalls          int   `json:"dialectic_llm_calls,omitempty"`
	DialecticLLMMs             int64 `json:"dialectic_llm_ms,omitempty"`

	// V2 Day5 fields: write-back observability.
	InsightsWritten     int   `json:"insights_written,omitempty"`
	InsightsSkipped     int   `json:"insights_skipped,omitempty"`
	InsightsWriteFailed int   `json:"insights_write_failed,omitempty"`
	WriteBackMs         int64 `json:"write_back_ms,omitempty"`

	// §1.1 v0.3: LLM confidence parsing counters (both V1 and V2 paths).
	// Invariants: Default+ParseFail+Explicit == blocks; Explicit == High+Mid+Low+OutOfBounds.
	LLMConfDefaultCount     int `json:"llm_conf_default_count,omitempty"`
	LLMConfParseFailCount   int `json:"llm_conf_parse_fail_count,omitempty"`
	LLMConfExplicitCount    int `json:"llm_conf_explicit_count,omitempty"`
	LLMConfHighCount        int `json:"llm_conf_high_count,omitempty"`
	LLMConfMidCount         int `json:"llm_conf_mid_count,omitempty"`
	LLMConfLowCount         int `json:"llm_conf_low_count,omitempty"`
	LLMConfOutOfBoundsCount int `json:"llm_conf_out_of_bounds_count,omitempty"`

	// §1.1 v0.3: runs_today tracks how many reflection runs have occurred today (CST day).
	RunsToday int `json:"runs_today,omitempty"`

	// Evidence debug (--debug-evidence).
	EvidenceDebug []EvidenceQuestionDebug `json:"evidence_debug,omitempty"`

	// R-S3: TTL observability fields.
	// ValidUntilSet is true when ALL insights produced in this run have a
	// non-zero ValidUntil — i.e. the entire run output is TTL-reclaimable.
	// Any single missing TTL → false (fail-closed for observability).
	ValidUntilSet bool `json:"valid_until_set"`
	// ValidUntil is the earliest (minimum) expiry across all produced insights,
	// in RFC 3339 UTC. Null when ValidUntilSet is false.
	ValidUntil *string `json:"valid_until"`
}

// Engine orchestrates the reflection cycle.
type Engine struct {
	store    memory.Store
	embedder embedding.Embedder
	cfg      Config
}

// NewEngine creates a new Reflection Engine instance.
func NewEngine(store memory.Store, embedder embedding.Embedder, cfg Config) *Engine {
	if cfg.Threshold == 0 {
		cfg.Threshold = DefaultConfig().Threshold
	}
	if cfg.MaxInputSize == 0 {
		cfg.MaxInputSize = DefaultConfig().MaxInputSize
	}
	if cfg.MinIntervalH == 0 {
		cfg.MinIntervalH = DefaultConfig().MinIntervalH
	}
	return &Engine{
		store:    store,
		embedder: embedder,
		cfg:      cfg,
	}
}

// Check evaluates whether reflection should run now without executing it.
// Returns a CheckResult with should_trigger, skip_reason, and metrics.
func (e *Engine) Check(ctx context.Context) (*CheckResult, error) {
	result, _, err := e.check(ctx)
	return result, err
}

// Run executes one reflection cycle. Respects DryRun mode.
// Returns RunResult with metrics on what was done.
func (e *Engine) Run(ctx context.Context) (*RunResult, error) {
	ctx, span := tracer.Start(ctx, "engram.reflection.run")
	defer span.End()

	mode := e.cfg.Mode
	if mode == "" {
		mode = "v1"
	}

	if mode == "v2" {
		return e.RunV2(ctx)
	}

	start := time.Now()
	result := &RunResult{DryRun: e.cfg.DryRun, Mode: mode + "-flat"}
	defer func() { setRunSpanAttributes(span, result) }()

	// Evaluate trigger conditions.
	checkResult, unreflected, err := e.check(ctx)
	if err != nil {
		return nil, fmt.Errorf("trigger check: %w", err)
	}

	result.RunsToday = checkResult.RunsToday
	if !checkResult.ShouldTrigger {
		result.SkipReason = checkResult.SkipReason
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	result.Triggered = true
	result.TriggerImportance = checkResult.AccumulatedImportance

	// Select input batch: top-N by importance.
	batch := selectInputBatch(unreflected, e.cfg.MaxInputSize)
	result.InputCount = len(batch)

	if len(batch) == 0 {
		result.SkipReason = "no unreflected memories available"
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	// W17 v1.1 batch 2: Evidence grounding enforcement. Reflections require
	// ≥ minEvidenceCount source memories to be written; otherwise reject the
	// entire run. See validateEvidenceGrounding.
	if err := validateEvidenceGrounding(batch); err != nil {
		result.SkipReason = err.Error()
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	// Build prompt and call the LLM.
	prompt := buildPrompt(batch)
	llmResponse, err := callLLM(ctx, prompt)
	result.LLMCalls++ // count the LLM call
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("llm call failed: %v", err))
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	// Parse insights and collect confidence counters (§1.1 single-point hook).
	insights, confCounts := parseLLMResponse(llmResponse)
	result.LLMConfDefaultCount = confCounts.LLMConfDefaultCount
	result.LLMConfParseFailCount = confCounts.LLMConfParseFailCount
	result.LLMConfExplicitCount = confCounts.LLMConfExplicitCount
	result.LLMConfHighCount = confCounts.LLMConfHighCount
	result.LLMConfMidCount = confCounts.LLMConfMidCount
	result.LLMConfLowCount = confCounts.LLMConfLowCount
	result.LLMConfOutOfBoundsCount = confCounts.LLMConfOutOfBoundsCount
	if len(insights) == 0 {
		result.Errors = append(result.Errors, "llm returned no parseable insights")
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	// Collect source IDs.
	sourceIDs := make([]string, len(batch))
	for i, m := range batch {
		sourceIDs[i] = m.ID
	}

	if !e.cfg.DryRun {
		// W17 v1.1: TTL + source:reflection tag for boundary isolation.
		reflectionValidUntil := float64(time.Now().Add(30 * 24 * time.Hour).Unix())

		var storedTTLs []float64 // R-S3: track TTLs for observability

		// Store each parsed insight.
		for _, ins := range insights {
			// W17 v1.1: enforce source:reflection tag on every reflection-origin insight.
			tags := ensureSourceReflectionTag(ins.Tags)

			// W17 v1.1: low-confidence insights are diverted to Obsidian drafts
			// instead of polluting Engram.
			if ins.Confidence < 0.6 {
				if werr := writeReflectionDraft(ins, tags, sourceIDs); werr != nil {
					result.Errors = append(result.Errors,
						fmt.Sprintf("write draft failed: %v", werr))
				} else {
					result.DraftsWritten++
				}
				continue
			}

			// Collect source IDs for metadata (convert to []any for Qdrant compatibility).
			sourceIDsAny := make([]any, len(sourceIDs))
			for i, id := range sourceIDs {
				sourceIDsAny[i] = id
			}

			insightMem := memory.New(ins.Content,
				memory.WithType(memory.TypeInsight),
				memory.WithSource("system"),
				memory.WithImportance(ins.Importance),
				memory.WithTags(tags...),
				memory.WithConfidence(ins.Confidence),
				memory.WithValidUntil(reflectionValidUntil),
				memory.WithMetadata(map[string]any{
					"reflection_source_ids": sourceIDsAny,
					"reflection_count":      len(sourceIDs),
					"trigger_importance":    checkResult.AccumulatedImportance,
					// W20 Day2 Phase 3: tag metadata for Phase 4 physical isolation routing.
					"caller_type":       "reflection",
					"target_collection": "engram_reflection",
					"source_type":       "reflection",
					// §1.1 v0.3: run-level confidence counters written to engram_reflection.
					"llm_conf_default_count":    result.LLMConfDefaultCount,
					"llm_conf_parse_fail_count": result.LLMConfParseFailCount,
					"llm_conf_explicit_count":   result.LLMConfExplicitCount,
					"llm_conf_high_count":       result.LLMConfHighCount,
					"llm_conf_mid_count":        result.LLMConfMidCount,
					"llm_conf_low_count":        result.LLMConfLowCount,
					"llm_conf_oob_count":        result.LLMConfOutOfBoundsCount,
					"runs_today":                result.RunsToday,
				}),
			)
			insightMem.Collection = "engram_reflection"

			// Embed the insight.
			var vec []float32
			if e.embedder != nil {
				vec, err = e.embedder.Embed(ctx, ins.Content)
				if err != nil {
					result.Errors = append(result.Errors,
						fmt.Sprintf("embed insight failed: %v (skipping)", err))
					continue
				}
			} else {
				// No embedder available: skip insertion rather than storing a
				// zero vector. A zero vector with hardcoded dimension would fail
				// if the collection dimension differs (e.g. 1536 != 1024).
				result.Errors = append(result.Errors,
					fmt.Sprintf("skipped insight (no embedder): %s", ins.Content[:min(40, len(ins.Content))]))
				continue
			}

			// Pre-write dedup: skip if a semantically similar reflection insight already exists.
			dedupThreshold := e.cfg.InsightDedupThreshold
			if dedupThreshold == 0 {
				dedupThreshold = 0.78 // default: catch paraphrases in the 0.78-0.92 band
			}
			existing, searchErr := e.store.Search(ctx, vec, memory.SearchOptions{
				Limit: 1,
				Filters: []memory.Filter{
					{Field: "collection", Op: memory.OpIn, Value: []string{"engram_reflection"}},
					{Field: "type", Op: memory.OpEq, Value: string(memory.TypeInsight)},
				},
			})
			if searchErr == nil && len(existing) > 0 && existing[0].Score >= dedupThreshold {
				fmt.Fprintf(os.Stderr, "reflection: dedup skip — existing score %.3f >= %.3f threshold\n", existing[0].Score, dedupThreshold)
				result.InsightsDedupSkipped++
				continue
			}
			// Fail-open: if the search errors, fall through and insert (dedup is best-effort).

			if err := e.store.Insert(ctx, insightMem, vec); err != nil {
				result.Errors = append(result.Errors,
					fmt.Sprintf("store insight failed: %v", err))
				continue
			}
			result.InsightsCreated++
			storedTTLs = append(storedTTLs, reflectionValidUntil)
		}

		setValidUntilFields(result, storedTTLs)

		// Mark source memories as reflected only if at least one insight was
		// produced. If all embedding/storage calls failed, skip marking so the
		// same source memories can be retried on the next run (transaction boundary
		// guard — prevents "reflected with no insights" data loss on transient errors).
		if result.InsightsCreated > 0 || result.DraftsWritten > 0 {
			reflectedTimestamp := float64(time.Now().Unix())

			// TOCTOU guard: verify source IDs still exist before marking. TTL
			// expiry or consolidation may have deleted some memories between
			// fetchUnreflected and now. Orphaned IDs would cause Update to fail
			// (Qdrant NotFound), blocking mark-reflected for the whole batch.
			validIDs, orphanCount, lookupErr := filterExistingIDs(ctx, e.store, sourceIDs)
			if lookupErr != nil {
				result.Errors = append(result.Errors,
					fmt.Sprintf("source-id existence check failed (using all %d ids): %v", len(sourceIDs), lookupErr))
				validIDs = sourceIDs
			} else if orphanCount > 0 {
				result.SourcesOrphaned = orphanCount
				result.Errors = append(result.Errors,
					fmt.Sprintf("skipped %d orphan source IDs (already deleted by TTL or consolidation)", orphanCount))
			}

			for _, id := range validIDs {
				if err := e.store.Update(ctx, id, map[string]any{
					"reflected_at": reflectedTimestamp,
				}); err != nil {
					result.Errors = append(result.Errors,
						fmt.Sprintf("mark reflected failed for %s: %v", id[:8], err))
					continue
				}
				result.SourcesMarked++
			}

			// Update run timestamp and daily count.
			if err := updateLastRun(); err != nil {
				result.Errors = append(result.Errors,
					fmt.Sprintf("update last run failed: %v", err))
			}
		} else {
			result.Errors = append(result.Errors,
				"no insights produced — sources not marked to allow retry")
		}
	} else {
		// Dry run: no writes occurred; keep counters at zero.
		result.InsightsCreated = 0
		result.SourcesMarked = 0
		dryTTL := float64(time.Now().Add(30 * 24 * time.Hour).Unix())
		dryTTLs := make([]float64, len(insights))
		for i := range dryTTLs {
			dryTTLs[i] = dryTTL
		}
		setValidUntilFields(result, dryTTLs)
	}

	result.Duration = formatDuration(time.Since(start))
	return result, nil
}

// setValidUntilFields populates the R-S3 observability fields on RunResult.
// ttls collects the ValidUntil value for each insight that was actually stored
// (not drafts, not skipped). If all are non-zero → set=true, valid_until=min.
func setValidUntilFields(result *RunResult, ttls []float64) {
	if len(ttls) == 0 {
		result.ValidUntilSet = false
		result.ValidUntil = nil
		return
	}
	minTTL := ttls[0]
	for _, v := range ttls {
		if v == 0 {
			result.ValidUntilSet = false
			result.ValidUntil = nil
			return
		}
		if v < minTTL {
			minTTL = v
		}
	}
	result.ValidUntilSet = true
	ts := time.Unix(int64(minTTL), 0).UTC().Format(time.RFC3339)
	result.ValidUntil = &ts
}

// setRunSpanAttributes writes observability attributes onto the OTel span.
func setRunSpanAttributes(span trace.Span, r *RunResult) {
	if r == nil {
		return
	}
	span.SetAttributes(
		attribute.Bool("engram.memory.valid_until_set", r.ValidUntilSet),
	)
	if r.ValidUntil != nil {
		span.SetAttributes(attribute.String("engram.memory.valid_until", *r.ValidUntil))
	} else {
		span.SetAttributes(attribute.String("engram.memory.valid_until", ""))
	}
	// §1.1 v0.3: LLM confidence parsing counters and runs_today.
	span.SetAttributes(
		attribute.Int("reflection.llm.conf.default_count", r.LLMConfDefaultCount),
		attribute.Int("reflection.llm.conf.parse_fail_count", r.LLMConfParseFailCount),
		attribute.Int("reflection.llm.conf.explicit_count", r.LLMConfExplicitCount),
		attribute.Int("reflection.llm.conf.high_count", r.LLMConfHighCount),
		attribute.Int("reflection.llm.conf.mid_count", r.LLMConfMidCount),
		attribute.Int("reflection.llm.conf.low_count", r.LLMConfLowCount),
		attribute.Int("reflection.llm.conf.out_of_bounds_count", r.LLMConfOutOfBoundsCount),
		attribute.Int("reflection.runs_today", r.RunsToday),
	)
}

// ── LLM call ────────────────────────────────────────────────────────────────

// callLLMFunc is the package-level LLM call function. Tests override this
// to inject mock responses without hitting the real API.
var callLLMFunc = llm.Call

// callLLM sends a prompt to the shared LLM client and returns the text response.
func callLLM(ctx context.Context, prompt string) (string, error) {
	return callLLMFunc(ctx, prompt)
}

// ── W17 v1.1 helpers ────────────────────────────────────────────────────────

// minEvidenceCount is the minimum number of source memories required for a
// reflection run to produce insights. Fewer than this → reject the entire
// batch-mode run (accumulator track). Event-driven single-event reflections
// bypass this check via their own TriggerCause-based evidence rule.
//
// W17 v1.1 batch 2 (2026-04-17): Frank approved enforcement of
// reflection_source_ids length >= 2 for batch-mode insights, so a reflection
// cannot be stored as "generalized pattern" when grounded in a single memory.
const minEvidenceCount = 2

// validateEvidenceGrounding checks that a selected input batch has enough
// source memories to justify a cross-memory pattern synthesis. Returns a
// descriptive error suitable for RunResult.SkipReason when rejected.
func validateEvidenceGrounding(batch []memory.Memory) error {
	if len(batch) < minEvidenceCount {
		return fmt.Errorf(
			"evidence grounding: insufficient source memories (%d < %d) — refusing to synthesize pattern from single source",
			len(batch), minEvidenceCount,
		)
	}
	return nil
}

// sourceReflectionTag marks a memory as originating from the Reflection Engine,
// enabling Dream Engine to skip recent reflection insights during consolidation.
const sourceReflectionTag = "source:reflection"

// ensureSourceReflectionTag returns tags with "source:reflection" appended if
// not already present. Reflection-origin insights MUST carry this tag per
// W17 v1.1 boundary isolation (decision 2).
func ensureSourceReflectionTag(tags []string) []string {
	for _, t := range tags {
		if t == sourceReflectionTag {
			return tags
		}
	}
	return append(append([]string{}, tags...), sourceReflectionTag)
}

// writeReflectionDraft writes a low-confidence (conf < 0.6) reflection to an
// Obsidian markdown draft instead of storing it in Engram. This keeps
// Engram clean of speculative / weakly-grounded inferences while preserving
// them for later human review in /data/armyoftheagent/siri-vault/Reflection/drafts/.
func writeReflectionDraft(ins ParsedInsight, tags []string, sourceIDs []string) error {
	dir := "/data/armyoftheagent/siri-vault/Reflection/drafts"
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}

	now := time.Now().UTC()
	ts := now.Format("20060102-150405")
	shortID := fmt.Sprintf("%x", now.UnixNano())
	if len(shortID) > 8 {
		shortID = shortID[len(shortID)-8:]
	}
	path := fmt.Sprintf("%s/%s-%s.md", dir, ts, shortID)

	var sb strings.Builder
	sb.WriteString("# Low-confidence reflection draft\n\n")
	fmt.Fprintf(&sb, "- created: %s\n", now.Format(time.RFC3339))
	fmt.Fprintf(&sb, "- confidence: %.2f\n", ins.Confidence)
	fmt.Fprintf(&sb, "- importance: %.0f\n", ins.Importance)
	fmt.Fprintf(&sb, "- tags: %s\n", strings.Join(tags, ", "))
	fmt.Fprintf(&sb, "- source_ids: %s\n", strings.Join(sourceIDs, ", "))
	sb.WriteString("\n## Insight\n\n")
	sb.WriteString(ins.Content)
	sb.WriteString("\n")

	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		return fmt.Errorf("write draft %s: %w", path, err)
	}
	return nil
}

// ── W17 v1.1 batch 2: Failure-driven (event-driven) reflection track ────────

// TriggerCause categorises what event prompted an immediate single-event
// reflection. Used to tag the resulting insight so downstream analytics can
// group failure-driven insights separately from accumulator-driven ones.
type TriggerCause string

const (
	// TriggerTaskFailure is raised when a task hit its retry budget
	// (e.g. event-loop retry_count >= 3).
	TriggerTaskFailure TriggerCause = "task_failure"
	// TriggerUserCorrection is raised when Frank explicitly corrects Siri
	// ("错了", "不对", "not right", etc.) via email or UI feedback.
	TriggerUserCorrection TriggerCause = "user_correction"
	// TriggerExternalEvent is a catch-all for other event-driven triggers.
	TriggerExternalEvent TriggerCause = "external_event"
)

// SingleEventInput is a compact payload for RunSingleEvent. The caller passes
// the event text, optional evidence memories retrieved via their own channel
// (e.g. the related Gmail thread or calendar event), and a cause tag.
type SingleEventInput struct {
	// Cause classifies the triggering event. Required.
	Cause TriggerCause
	// Summary is a one-line description of the event (e.g. "task
	// sanitize-description retried 3x and failed"). Required.
	Summary string
	// EvidenceIDs are existing Engram memory IDs that ground this reflection.
	// May be empty — event-driven reflections are permitted to synthesize
	// from the Summary alone if no related memory is available.
	EvidenceIDs []string
	// Importance lets the caller override the default importance of the
	// resulting insight. 0 → use default (8, since failures are high-signal).
	Importance float64
	// ExtraTags are appended to the insight's tags.
	ExtraTags []string
}

// RunSingleEvent performs an immediate, lightweight reflection driven by a
// specific triggering event (task failure, user correction, etc.). Unlike
// the batch Run path, it bypasses the accumulator thresholds and the
// batch-mode evidence-grounding floor — a single, specific event is allowed
// to produce one insight by design.
//
// Safety rails still applied:
//   - DryRun respected.
//   - source:reflection tag enforced.
//   - 30-day TTL applied.
//   - Low-confidence insights (<0.6) still diverted to Obsidian draft.
//   - Daily-count + min-interval gates are SKIPPED (event-driven track is
//     supposed to react quickly; failures shouldn't be silenced by quotas).
func (e *Engine) RunSingleEvent(ctx context.Context, in SingleEventInput) (*RunResult, error) {
	if in.Cause == "" {
		return nil, fmt.Errorf("RunSingleEvent: Cause is required")
	}
	if strings.TrimSpace(in.Summary) == "" {
		return nil, fmt.Errorf("RunSingleEvent: Summary is required")
	}

	ctx, span := tracer.Start(ctx, "engram.reflection.run")
	defer span.End()

	start := time.Now()
	result := &RunResult{
		DryRun: e.cfg.DryRun,
		Mode:   "v1-event-" + string(in.Cause),
	}
	defer func() { setRunSpanAttributes(span, result) }()

	result.Triggered = true
	result.InputCount = len(in.EvidenceIDs) // 0 is valid here

	// Build a compact prompt from the event summary and optional evidence IDs.
	prompt := buildSingleEventPrompt(in)
	llmResponse, err := callLLM(ctx, prompt)
	result.LLMCalls++
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("llm call failed: %v", err))
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	insights, _ := parseLLMResponse(llmResponse)
	if len(insights) == 0 {
		result.Errors = append(result.Errors, "llm returned no parseable insights")
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}
	// Single-event track: only keep the first insight to avoid cascading
	// speculation from one failure event.
	insights = insights[:1]

	if e.cfg.DryRun {
		result.InsightsCreated = len(insights)
		dryTTL := float64(time.Now().Add(30 * 24 * time.Hour).Unix())
		setValidUntilFields(result, []float64{dryTTL})
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	reflectionValidUntil := float64(time.Now().Add(30 * 24 * time.Hour).Unix())
	var storedTTLs []float64 // R-S3: track TTLs for observability

	for _, ins := range insights {
		// Merge tags: insight tags + cause + trigger marker + extras.
		tags := ensureSourceReflectionTag(ins.Tags)
		tags = append(tags, "trigger:"+string(in.Cause), "reflection-track:event-driven")
		tags = append(tags, in.ExtraTags...)

		// Caller-provided importance override.
		importance := ins.Importance
		if in.Importance > 0 {
			importance = in.Importance
		}

		// Low-confidence diversion preserved.
		if ins.Confidence < 0.6 {
			if werr := writeReflectionDraft(ins, tags, in.EvidenceIDs); werr != nil {
				result.Errors = append(result.Errors,
					fmt.Sprintf("write draft failed: %v", werr))
			} else {
				result.DraftsWritten++
			}
			continue
		}

		evidenceAny := make([]any, len(in.EvidenceIDs))
		for i, id := range in.EvidenceIDs {
			evidenceAny[i] = id
		}

		insightMem := memory.New(ins.Content,
			memory.WithType(memory.TypeInsight),
			memory.WithSource("system"),
			memory.WithImportance(importance),
			memory.WithTags(tags...),
			memory.WithConfidence(ins.Confidence),
			memory.WithValidUntil(reflectionValidUntil),
			memory.WithMetadata(map[string]any{
				"reflection_source_ids": evidenceAny,
				"reflection_count":      len(in.EvidenceIDs),
				"reflection_trigger":    string(in.Cause),
				"reflection_summary":    in.Summary,
				// W20 Day2 Phase 3: tag metadata for Phase 4 physical isolation routing.
				"caller_type":       "reflection",
				"source_type":       "reflection",
				"target_collection": "engram_reflection",
			}),
		)
		insightMem.Collection = "engram_reflection"

		var vec []float32
		if e.embedder != nil {
			vec, err = e.embedder.Embed(ctx, ins.Content)
			if err != nil {
				result.Errors = append(result.Errors,
					fmt.Sprintf("embed insight failed: %v (skipping)", err))
				continue
			}
		} else {
			result.Errors = append(result.Errors,
				"skipped insight (no embedder)")
			continue
		}

		// Pre-write dedup: skip if a semantically similar reflection insight already exists.
		dedupThreshold := e.cfg.InsightDedupThreshold
		if dedupThreshold == 0 {
			dedupThreshold = 0.78 // default: catch paraphrases in the 0.78-0.92 band
		}
		existing, searchErr := e.store.Search(ctx, vec, memory.SearchOptions{
			Limit: 1,
			Filters: []memory.Filter{
				{Field: "collection", Op: memory.OpIn, Value: []string{"engram_reflection"}},
				{Field: "type", Op: memory.OpEq, Value: string(memory.TypeInsight)},
			},
		})
		if searchErr == nil && len(existing) > 0 && existing[0].Score >= dedupThreshold {
			fmt.Fprintf(os.Stderr, "reflection: dedup skip — existing score %.3f >= %.3f threshold\n", existing[0].Score, dedupThreshold)
			result.InsightsDedupSkipped++
			continue
		}
		// Fail-open: if the search errors, fall through and insert (dedup is best-effort).

		if err := e.store.Insert(ctx, insightMem, vec); err != nil {
			result.Errors = append(result.Errors,
				fmt.Sprintf("store insight failed: %v", err))
			continue
		}
		result.InsightsCreated++
		storedTTLs = append(storedTTLs, reflectionValidUntil)
	}

	setValidUntilFields(result, storedTTLs)

	// Note: do NOT call updateLastRun() here — event-driven runs should not
	// eat into the daily batch-reflection quota.
	result.Duration = formatDuration(time.Since(start))
	return result, nil
}

// buildSingleEventPrompt constructs a compact prompt tailored to a single
// triggering event. It's deliberately shorter than the batch prompt and
// asks for a single focused insight.
func buildSingleEventPrompt(in SingleEventInput) string {
	var sb strings.Builder
	sb.WriteString("You are the reflection engine for an AI agent named Siri. ")
	sb.WriteString("A specific event just occurred that warrants an immediate one-line insight.\n\n")
	fmt.Fprintf(&sb, "Trigger cause: %s\n", in.Cause)
	fmt.Fprintf(&sb, "Event summary: %s\n", in.Summary)
	if len(in.EvidenceIDs) > 0 {
		fmt.Fprintf(&sb, "Related memory IDs (grounding): %s\n", strings.Join(in.EvidenceIDs, ", "))
	}
	sb.WriteString("\nProduce EXACTLY ONE insight in this format (including --- delimiters):\n")
	sb.WriteString("---\n")
	sb.WriteString("INSIGHT: <2-3 sentences in English, third person, focused on what Siri should learn or change>\n")
	sb.WriteString("IMPORTANCE: <integer 1-10>\n")
	sb.WriteString("CONFIDENCE: <float 0.0-1.0; <0.6 means speculative>\n")
	sb.WriteString("TAGS: <comma-separated, max 5>\n")
	sb.WriteString("---\n\n")
	sb.WriteString("Keep it grounded to the event. If the event is too vague to draw a useful lesson, emit CONFIDENCE: 0.4 and state the uncertainty honestly.\n")
	return sb.String()
}
