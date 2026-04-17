// Package reflection implements the Engram Reflection Engine v1.
//
// The Reflection Engine periodically synthesizes high-level insights from
// unreflected memories, inspired by the Generative Agents paper. It runs
// lighter and more frequently than the Dream Engine (1-3x/day vs 1x/day),
// and focuses on cross-domain pattern discovery rather than same-tag dedup.
//
// Trigger: accumulated importance of unreflected memories >= threshold (default: 50).
// Min interval: 2h. Max per day: 3 runs.
//
// Output: TypeInsight memories with source="system", tagged with
// metadata.reflection_source_ids and metadata.reflected=true on source memories.
package reflection

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/embedding"
	"github.com/FBISiri/engram/pkg/memory"
)

// Config holds Reflection Engine runtime configuration.
type Config struct {
	// Threshold is the cumulative importance of unreflected memories required
	// to trigger a reflection run. Default: 50.
	Threshold float64

	// MaxInputSize is the maximum number of memories to feed into one reflection
	// cycle. Default: 20.
	MaxInputSize int

	// MinIntervalH is the minimum hours between reflection runs. Default: 2.
	MinIntervalH float64

	// DryRun disables all writes (no new insights, no marking as reflected).
	DryRun bool

	// Mode selects the reflection algorithm: "v1" (flat, default) or "v2" (focal point).
	// V2 requires 4x Haiku calls but produces higher-quality insights.
	Mode string

	// FocalInputSize is the number of unreflected memories fed to focal question
	// generation (Step 1 of V2). Default: 50.
	FocalInputSize int

	// FocalQuestions is the number of focal questions to generate in V2. Default: 3.
	FocalQuestions int

	// EvidencePerFocal is the number of memories retrieved per focal question
	// via semantic search in V2. Default: 10.
	EvidencePerFocal int
}

// DefaultConfig returns the default Reflection Engine configuration.
func DefaultConfig() Config {
	return Config{
		Threshold:        50.0,
		MaxInputSize:     20,
		MinIntervalH:     2.0,
		DryRun:           false,
		Mode:             "v1",   // V1 by default; V2 requires explicit opt-in
		FocalInputSize:   50,
		FocalQuestions:   3,
		EvidencePerFocal: 10,
	}
}

// RunResult holds the output of a single reflection run.
type RunResult struct {
	Triggered         bool     `json:"triggered"`
	SkipReason        string   `json:"skip_reason,omitempty"`
	InputCount        int      `json:"input_count"`
	InsightsCreated   int      `json:"insights_created"`
	DraftsWritten     int      `json:"drafts_written"`      // W17 v1.1: confidence<0.6 → Obsidian draft
	SourcesMarked     int      `json:"sources_marked"`
	Duration          string   `json:"duration"`
	TriggerImportance float64  `json:"trigger_importance,omitempty"`
	DryRun            bool     `json:"dry_run"`
	Errors            []string `json:"errors,omitempty"`

	// V2 fields (populated when Mode == "v2").
	Mode            string   `json:"mode"`                        // "v1-flat" | "v2-focal"
	FocalQuestions  []string `json:"focal_questions,omitempty"`   // V2 only
	EvidenceCount   int      `json:"evidence_count,omitempty"`    // V2: evidence set size
	LLMCalls        int      `json:"llm_calls"`                   // total Haiku calls
	LLMCostEstimate float64  `json:"llm_cost_estimate_usd"`       // estimated USD cost
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
	start := time.Now()
	mode := e.cfg.Mode
	if mode == "" {
		mode = "v1"
	}
	result := &RunResult{DryRun: e.cfg.DryRun, Mode: mode + "-flat"}

	// Evaluate trigger conditions.
	checkResult, unreflected, err := e.check(ctx)
	if err != nil {
		return nil, fmt.Errorf("trigger check: %w", err)
	}

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

	// Build prompt and call Haiku.
	prompt := buildPrompt(batch)
	haikuResponse, err := callHaiku(ctx, prompt)
	result.LLMCalls++ // count the Haiku call
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("haiku call failed: %v", err))
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	// Parse insights.
	insights := parseHaikuResponse(haikuResponse)
	if len(insights) == 0 {
		result.Errors = append(result.Errors, "haiku returned no parseable insights")
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

		// Store each parsed insight.
		for _, ins := range insights {
			// W17 v1.1: enforce source:reflection tag on every reflection-origin insight.
			tags := ensureSourceReflectionTag(ins.Tags)

			// W17 v1.1: low-confidence insights are diverted to Obsidian drafts
			// instead of polluting Engram.
			if ins.Confidence > 0 && ins.Confidence < 0.6 {
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
				}),
			)

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

			if err := e.store.Insert(ctx, insightMem, vec); err != nil {
				result.Errors = append(result.Errors,
					fmt.Sprintf("store insight failed: %v", err))
				continue
			}
			result.InsightsCreated++
		}

		// Mark source memories as reflected using ReflectedAt timestamp (W16).
		// Legacy metadata["reflected"] is no longer written; isReflected() has fallback.
		reflectedTimestamp := float64(time.Now().Unix())
		for _, id := range sourceIDs {
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
		// Dry run: report what would happen.
		result.InsightsCreated = len(insights) // what would be created
		result.SourcesMarked = len(sourceIDs)  // what would be marked
	}

	result.Duration = formatDuration(time.Since(start))
	return result, nil
}

// ── Haiku LLM call ─────────────────────────────────────────────────────────

// haikuConfig holds credentials for Haiku API calls.
type haikuConfig struct {
	APIKey  string
	BaseURL string
	Model   string
	IsOAuth bool
}

// getHaikuConfig returns the best available Haiku configuration.
// Mirrors the logic in pkg/dream/llm.go.
func getHaikuConfig() *haikuConfig {
	model := os.Getenv("ANTHROPIC_LIGHT_MODEL")
	if model == "" {
		model = "claude-haiku-4-5-20251001"
	}

	// 1. Claude Code OAuth token from env.
	if token := os.Getenv("CLAUDE_CODE_OAUTH_TOKEN"); token != "" {
		return &haikuConfig{
			APIKey:  token,
			BaseURL: "https://api.anthropic.com",
			Model:   model,
			IsOAuth: true,
		}
	}

	// 2. Claude Code credentials file.
	if token := readClaudeOAuthToken(); token != "" {
		return &haikuConfig{
			APIKey:  token,
			BaseURL: "https://api.anthropic.com",
			Model:   model,
			IsOAuth: true,
		}
	}

	// 3. Direct Anthropic API key.
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		return &haikuConfig{
			APIKey:  key,
			BaseURL: "https://api.anthropic.com",
			Model:   model,
			IsOAuth: false,
		}
	}

	return nil
}

// readClaudeOAuthToken reads the OAuth access token from /root/.claude/.credentials.json.
func readClaudeOAuthToken() string {
	data, err := os.ReadFile("/root/.claude/.credentials.json")
	if err != nil {
		return ""
	}
	var creds struct {
		ClaudeAiOauth struct {
			AccessToken string `json:"accessToken"`
		} `json:"claudeAiOauth"`
	}
	if err := json.Unmarshal(data, &creds); err != nil {
		return ""
	}
	return creds.ClaudeAiOauth.AccessToken
}

// callHaiku sends a prompt to Claude Haiku and returns the text response.
func callHaiku(ctx context.Context, prompt string) (string, error) {
	cfg := getHaikuConfig()
	if cfg == nil {
		return "", fmt.Errorf("no Haiku API credentials available")
	}

	reqBody, err := json.Marshal(map[string]any{
		"model":      cfg.Model,
		"max_tokens": haikuMaxTokens,
		"messages": []map[string]any{
			{"role": "user", "content": prompt},
		},
	})
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", cfg.BaseURL+"/v1/messages", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	if cfg.IsOAuth {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		req.Header.Set("Anthropic-Beta", "claude-code-20250219,oauth-2025-04-20")
	} else {
		req.Header.Set("X-Api-Key", cfg.APIKey)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("haiku request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("haiku returned status %d", resp.StatusCode)
	}

	var apiResp struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return "", fmt.Errorf("decode haiku response: %w", err)
	}

	for _, block := range apiResp.Content {
		if block.Type == "text" {
			return strings.TrimSpace(block.Text), nil
		}
	}
	return "", fmt.Errorf("no text content in haiku response")
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
// them for later human review in $HOME/siri-vault/Reflection/drafts/.
func writeReflectionDraft(ins ParsedInsight, tags []string, sourceIDs []string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("home dir: %w", err)
	}
	dir := home + "/siri-vault/Reflection/drafts"
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
	sb.WriteString(fmt.Sprintf("- created: %s\n", now.Format(time.RFC3339)))
	sb.WriteString(fmt.Sprintf("- confidence: %.2f\n", ins.Confidence))
	sb.WriteString(fmt.Sprintf("- importance: %.0f\n", ins.Importance))
	sb.WriteString(fmt.Sprintf("- tags: %s\n", strings.Join(tags, ", ")))
	sb.WriteString(fmt.Sprintf("- source_ids: %s\n", strings.Join(sourceIDs, ", ")))
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
	start := time.Now()
	result := &RunResult{
		DryRun: e.cfg.DryRun,
		Mode:   "v1-event-" + string(in.Cause),
	}

	if in.Cause == "" {
		return nil, fmt.Errorf("RunSingleEvent: Cause is required")
	}
	if strings.TrimSpace(in.Summary) == "" {
		return nil, fmt.Errorf("RunSingleEvent: Summary is required")
	}

	result.Triggered = true
	result.InputCount = len(in.EvidenceIDs) // 0 is valid here

	// Build a compact prompt from the event summary and optional evidence IDs.
	prompt := buildSingleEventPrompt(in)
	haikuResponse, err := callHaiku(ctx, prompt)
	result.LLMCalls++
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("haiku call failed: %v", err))
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	insights := parseHaikuResponse(haikuResponse)
	if len(insights) == 0 {
		result.Errors = append(result.Errors, "haiku returned no parseable insights")
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}
	// Single-event track: only keep the first insight to avoid cascading
	// speculation from one failure event.
	insights = insights[:1]

	if e.cfg.DryRun {
		result.InsightsCreated = len(insights)
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	reflectionValidUntil := float64(time.Now().Add(30 * 24 * time.Hour).Unix())

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
		if ins.Confidence > 0 && ins.Confidence < 0.6 {
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
			}),
		)

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

		if err := e.store.Insert(ctx, insightMem, vec); err != nil {
			result.Errors = append(result.Errors,
				fmt.Sprintf("store insight failed: %v", err))
			continue
		}
		result.InsightsCreated++
	}

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
	sb.WriteString(fmt.Sprintf("Trigger cause: %s\n", in.Cause))
	sb.WriteString(fmt.Sprintf("Event summary: %s\n", in.Summary))
	if len(in.EvidenceIDs) > 0 {
		sb.WriteString(fmt.Sprintf("Related memory IDs (grounding): %s\n", strings.Join(in.EvidenceIDs, ", ")))
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
