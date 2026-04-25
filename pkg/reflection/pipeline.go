package reflection

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// RunV2 executes the V2 focal-point reflection pipeline:
// Stage 1: focal question generation, Stage 2: evidence retrieval,
// Stage 3: dialectic synthesis, Stage 4: write-back, Stage 5: source-marking.
func (e *Engine) RunV2(ctx context.Context) (*RunResult, error) {
	ctx, span := tracer.Start(ctx, "engram.reflection.run_v2")
	defer span.End()

	start := time.Now()
	result := &RunResult{DryRun: e.cfg.DryRun, Mode: "v2-focal"}
	defer func() { setRunSpanAttributes(span, result) }()

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

	focalInputSize := e.cfg.FocalInputSize
	if focalInputSize == 0 {
		focalInputSize = 50
	}
	batch := selectInputBatch(unreflected, focalInputSize)
	result.InputCount = len(batch)

	if len(batch) == 0 {
		result.SkipReason = "no unreflected memories available"
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	if err := validateEvidenceGrounding(batch); err != nil {
		result.SkipReason = err.Error()
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	// Stage 1: Focal question generation.
	numQuestions := e.cfg.FocalQuestions
	if numQuestions == 0 {
		numQuestions = 3
	}
	questions, err := generateFocalQuestions(ctx, batch, numQuestions)
	result.LLMCalls++
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("focal question generation failed: %v", err))
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}
	result.FocalQuestions = questions

	// Stage 2: Evidence retrieval.
	evidenceList := retrieveAllEvidence(ctx, questions, e.store, e.embedder, e.cfg, result)

	// Stage 3: Dialectic synthesis.
	if e.cfg.FocalModelStep3 != "" {
		haikuModelOverride = e.cfg.FocalModelStep3
		defer func() { haikuModelOverride = "" }()
	}
	dialectics, dStats, err := e.generateDialecticInsights(ctx, evidenceList, e.cfg)
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("dialectic synthesis failed: %v", err))
		result.Duration = formatDuration(time.Since(start))
		return result, nil
	}

	result.LLMCalls += dStats.LLMCalls
	result.DialecticOkCount = dStats.OkCount
	result.DialecticFailedCount = dStats.FailedCount
	result.DialecticDroppedNoEvidence = dStats.DroppedNoEvidence
	result.DialecticDroppedLowConf = dStats.DroppedLowConf
	result.DialecticLLMCalls = dStats.LLMCalls
	result.DialecticLLMMs = dStats.LLMMs
	result.Errors = append(result.Errors, dStats.Errors...)

	if !e.cfg.DryRun {
		// Stage 4: Write-back.
		wbStats := e.writeDialecticInsights(ctx, dialectics, evidenceList, e.cfg)
		result.InsightsWritten = wbStats.Written
		result.InsightsSkipped = wbStats.Skipped
		result.InsightsWriteFailed = wbStats.Failed
		result.WriteBackMs = wbStats.Ms
		result.InsightsCreated = wbStats.Written
		result.Errors = append(result.Errors, wbStats.Errors...)

		var storedTTLs []float64
		ttl := float64(time.Now().Add(30 * 24 * time.Hour).Unix())
		for i := 0; i < wbStats.Written; i++ {
			storedTTLs = append(storedTTLs, ttl)
		}
		setValidUntilFields(result, storedTTLs)

		// Stage 5: Source-marking (reuse V1 logic).
		reflectedTimestamp := float64(time.Now().Unix())
		sourceIDs := make([]string, len(batch))
		for i, m := range batch {
			sourceIDs[i] = m.ID
		}
		for _, id := range sourceIDs {
			if err := e.store.Update(ctx, id, map[string]any{
				"reflected_at": reflectedTimestamp,
			}); err != nil {
				result.Errors = append(result.Errors,
					fmt.Sprintf("mark reflected failed for %s: %v", id[:min(8, len(id))], err))
				continue
			}
			result.SourcesMarked++
		}

		if err := updateLastRun(); err != nil {
			result.Errors = append(result.Errors,
				fmt.Sprintf("update last run failed: %v", err))
		}
	} else {
		result.InsightsWritten = 0
		result.InsightsSkipped = 0
		result.InsightsWriteFailed = 0
		result.InsightsCreated = 0
		dryTTL := float64(time.Now().Add(30 * 24 * time.Hour).Unix())
		dryTTLs := make([]float64, len(dialectics))
		for i := range dryTTLs {
			dryTTLs[i] = dryTTL
		}
		setValidUntilFields(result, dryTTLs)
	}

	result.Duration = formatDuration(time.Since(start))
	return result, nil
}

// generateFocalQuestions calls Haiku to generate N focal questions from a batch
// of unreflected memories.
func generateFocalQuestions(ctx context.Context, batch []memory.Memory, n int) ([]string, error) {
	prompt := buildFocalPrompt(batch, n)
	response, err := callHaiku(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("haiku call: %w", err)
	}
	return parseFocalResponse(response, n)
}

func buildFocalPrompt(batch []memory.Memory, n int) string {
	var sb strings.Builder
	sb.WriteString("You are the reflection engine for an AI agent named Siri.\n\n")
	sb.WriteString(fmt.Sprintf("Below are %d recent memories. Generate exactly %d focal questions ", len(batch), n))
	sb.WriteString("that would be most productive for cross-domain reflection and pattern discovery.\n\n")
	sb.WriteString("Memories:\n")

	for i, m := range batch {
		content := m.Content
		if len(content) > 200 {
			content = content[:200] + "..."
		}
		sb.WriteString(fmt.Sprintf("%d. [%s, importance=%.0f] %s\n", i+1, m.Type, m.Importance, content))
	}

	sb.WriteString(fmt.Sprintf("\nRespond with a JSON array of exactly %d question strings. No markdown fences.\n", n))
	sb.WriteString(`Example: ["What patterns emerge...", "How does X relate to Y...", "What tensions exist..."]`)
	return sb.String()
}

func parseFocalResponse(response string, n int) ([]string, error) {
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimPrefix(response, "```")
	response = strings.TrimSuffix(response, "```")
	response = strings.TrimSpace(response)

	var questions []string
	if err := json.Unmarshal([]byte(response), &questions); err != nil {
		return nil, fmt.Errorf("JSON parse focal questions: %w", err)
	}
	if len(questions) == 0 {
		return nil, fmt.Errorf("no focal questions generated")
	}
	if len(questions) > n {
		questions = questions[:n]
	}
	return questions, nil
}
