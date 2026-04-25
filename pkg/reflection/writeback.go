package reflection

import (
	"context"
	"fmt"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

const (
	defaultWriteBackTimeout = 30 * time.Second
	maxWriteBackTimeout     = 120 * time.Second
	defaultMinInsightConf   = 0.5
	provenanceV2            = "reflection-v2"
)

type writeBackStats struct {
	Written int
	Skipped int
	Failed  int
	Ms      int64
	Errors  []string
}

func (e *Engine) writeDialecticInsights(ctx context.Context, dialectics []DialecticInsight, evidenceList []PerQuestionEvidence, cfg Config) writeBackStats {
	timeout := cfg.WriteBackTimeout
	if timeout == 0 {
		timeout = defaultWriteBackTimeout
	}
	if timeout > maxWriteBackTimeout {
		timeout = maxWriteBackTimeout
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	start := time.Now()
	var stats writeBackStats

	minConf := cfg.MinInsightConfidence
	if minConf == 0 {
		minConf = defaultMinInsightConf
	}

	reflectionValidUntil := float64(time.Now().Add(30 * 24 * time.Hour).Unix())

	for _, di := range dialectics {
		if di.Confidence < minConf {
			stats.Skipped++
			continue
		}

		tags := ensureSourceReflectionTag(di.Tags)

		tensionsVal := di.Tensions
		if tensionsVal == nil {
			tensionsVal = []string{}
		}

		sourceIDsAny := make([]any, len(di.SourceIDs))
		for j, sid := range di.SourceIDs {
			sourceIDsAny[j] = sid
		}

		tensionsAny := make([]any, len(tensionsVal))
		for j, t := range tensionsVal {
			tensionsAny[j] = t
		}

		insightMem := memory.New(di.Content,
			memory.WithType(memory.TypeInsight),
			memory.WithSource("system"),
			memory.WithImportance(float64(di.Importance)),
			memory.WithTags(tags...),
			memory.WithConfidence(di.Confidence),
			memory.WithValidUntil(reflectionValidUntil),
			memory.WithMetadata(map[string]any{
				"tensions":       tensionsAny,
				"source_ids":     sourceIDsAny,
				"focal_question": di.Question,
				"provenance":     provenanceV2,
			}),
		)

		var vec []float32
		var err error
		if e.embedder != nil {
			vec, err = e.embedder.Embed(ctx, di.Content)
			if err != nil {
				stats.Failed++
				stats.Errors = append(stats.Errors, fmt.Sprintf("embed insight failed: %v", err))
				continue
			}
		} else {
			stats.Failed++
			stats.Errors = append(stats.Errors, "skipped insight (no embedder)")
			continue
		}

		if err := e.store.Insert(ctx, insightMem, vec); err != nil {
			stats.Failed++
			stats.Errors = append(stats.Errors, fmt.Sprintf("store insert failed: %v", err))
			continue
		}
		stats.Written++
	}

	stats.Ms = time.Since(start).Milliseconds()
	return stats
}
