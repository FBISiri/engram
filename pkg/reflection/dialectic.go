package reflection

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
)

const (
	defaultDialecticTimeout = 15 * time.Second
	maxDialecticTimeout     = 30 * time.Second
)

type dialecticStatus int

const (
	dialecticStatusPending dialecticStatus = iota
	dialecticStatusOk
	dialecticStatusFailed
	dialecticStatusDroppedNoEvidence
	dialecticStatusDroppedLowConf
)

// DialecticInsight is the output of a single per-question dialectic synthesis.
type DialecticInsight struct {
	Question   string   `json:"question"`
	Content    string   `json:"content"`
	Tensions   []string `json:"tensions"`
	SourceIDs  []string `json:"source_ids"`
	Confidence float64  `json:"confidence"`
	Importance int      `json:"importance"`
	Tags       []string `json:"tags"`
}

// DialecticStats tracks Part3 observability counters.
type DialecticStats struct {
	OkCount            int      `json:"ok_count"`
	FailedCount        int      `json:"failed_count"`
	DroppedNoEvidence  int      `json:"dropped_no_evidence"`
	DroppedLowConf     int      `json:"dropped_low_conf"`
	LLMCalls           int      `json:"llm_calls"`
	LLMMs              int64    `json:"llm_ms"`
	Errors             []string `json:"errors,omitempty"`
}

// dialecticLLMResponse is the expected JSON schema from the LLM.
type dialecticLLMResponse struct {
	Content    string   `json:"content"`
	Tensions   []string `json:"tensions"`
	SourceIDs  []string `json:"source_ids"`
	Confidence float64  `json:"confidence"`
	Importance int      `json:"importance"`
	Tags       []string `json:"tags"`
}

// generateDialecticInsights runs dialectic synthesis for each focal question
// using its retrieved evidence. Questions with empty evidence skip LLM calls.
// Single-question failures are isolated via errgroup (goroutines never fail the group).
func (e *Engine) generateDialecticInsights(ctx context.Context, evidenceList []PerQuestionEvidence, cfg Config) ([]DialecticInsight, DialecticStats, error) {
	if len(evidenceList) != 3 {
		return nil, DialecticStats{}, fmt.Errorf("generateDialecticInsights: expected 3 questions, got %d", len(evidenceList))
	}

	timeout := cfg.DialecticTimeout
	if timeout == 0 {
		timeout = defaultDialecticTimeout
	}
	if timeout > maxDialecticTimeout {
		timeout = maxDialecticTimeout
	}

	insights := make([]DialecticInsight, len(evidenceList))
	statuses := make([]dialecticStatus, len(evidenceList))

	g, gctx := errgroup.WithContext(ctx)
	var errMu sync.Mutex
	var stats DialecticStats

	for i, pq := range evidenceList {
		i, pq := i, pq
		g.Go(func() error {
			if len(pq.Evidence) == 0 {
				statuses[i] = dialecticStatusDroppedNoEvidence
				return nil
			}

			qctx, cancel := context.WithTimeout(gctx, timeout)
			defer cancel()

			prompt := buildDialecticPrompt(pq)

			llmStart := time.Now()
			response, err := callHaiku(qctx, prompt)
			llmElapsed := time.Since(llmStart).Milliseconds()

			errMu.Lock()
			stats.LLMCalls++
			stats.LLMMs += llmElapsed
			errMu.Unlock()

			if err != nil {
				errMu.Lock()
				stats.Errors = append(stats.Errors, fmt.Sprintf("dialectic q%d: %v", i+1, err))
				errMu.Unlock()
				statuses[i] = dialecticStatusFailed
				return nil
			}

			insight, err := parseDialecticResponse(response, pq)
			if err != nil {
				errMu.Lock()
				stats.Errors = append(stats.Errors, fmt.Sprintf("dialectic q%d parse: %v", i+1, err))
				errMu.Unlock()
				statuses[i] = dialecticStatusFailed
				return nil
			}

			insight.Question = pq.Question
			insights[i] = *insight
			statuses[i] = dialecticStatusOk
			return nil
		})
	}

	_ = g.Wait()

	for _, s := range statuses {
		switch s {
		case dialecticStatusOk:
			stats.OkCount++
		case dialecticStatusFailed:
			stats.FailedCount++
		case dialecticStatusDroppedNoEvidence:
			stats.DroppedNoEvidence++
		case dialecticStatusDroppedLowConf:
			stats.DroppedLowConf++
		}
	}

	var result []DialecticInsight
	for i, s := range statuses {
		if s == dialecticStatusOk {
			result = append(result, insights[i])
		}
	}

	return result, stats, nil
}

func promptIDForm(id string) string {
	if len(id) > 12 {
		return id[:12]
	}
	return id
}

func buildDialecticPrompt(pq PerQuestionEvidence) string {
	var sb strings.Builder
	sb.WriteString("You are the dialectic reflection engine for an AI agent named Siri.\n\n")
	sb.WriteString(fmt.Sprintf("Focal question: %s\n\n", pq.Question))
	sb.WriteString(fmt.Sprintf("Below are %d evidence memories retrieved for this question. ", len(pq.Evidence)))
	sb.WriteString("Synthesize a dialectic insight that identifies tensions, contradictions, or nuanced patterns across these memories.\n\n")

	sb.WriteString("Evidence:\n")
	for i, m := range pq.Evidence {
		content := m.Content
		if len(content) > 300 {
			content = content[:300] + "..."
		}
		sb.WriteString(fmt.Sprintf("%d. [id=%s] %s\n", i+1, promptIDForm(m.ID), content))
	}

	sb.WriteString("\nRespond with EXACTLY ONE JSON object (no markdown fences, no extra text):\n")
	sb.WriteString(`{
  "content": "2-4 sentences synthesizing the dialectic insight",
  "tensions": ["tension 1", "tension 2"],
  "source_ids": ["id1", "id2"],
  "confidence": 0.8,
  "importance": 7,
  "tags": ["tag1", "tag2"]
}`)
	sb.WriteString("\n\nRules:\n")
	sb.WriteString("- tensions: ALWAYS include this field. If no contradictions exist, use an empty array []. Max 5 entries.\n")
	sb.WriteString("- source_ids: list the evidence IDs (from the [id=...] prefixes above) that ground this insight. Minimum 2.\n")
	sb.WriteString("- confidence: 0.0-1.0, how well-grounded this insight is in the evidence.\n")
	sb.WriteString("- importance: 1-10 integer.\n")
	sb.WriteString("- tags: max 5, lowercase, hyphen-separated.\n")

	return sb.String()
}

// parseDialecticResponse parses the LLM JSON response and validates source_ids
// against the evidence set (prompt injection defense).
func parseDialecticResponse(response string, pq PerQuestionEvidence) (*DialecticInsight, error) {
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimPrefix(response, "```")
	response = strings.TrimSuffix(response, "```")
	response = strings.TrimSpace(response)

	var parsed dialecticLLMResponse
	if err := json.Unmarshal([]byte(response), &parsed); err != nil {
		return nil, fmt.Errorf("JSON parse: %w", err)
	}

	if parsed.Content == "" {
		return nil, fmt.Errorf("missing content field")
	}

	if parsed.Tensions == nil {
		return nil, fmt.Errorf("missing tensions field")
	}
	if len(parsed.Tensions) > 5 {
		parsed.Tensions = parsed.Tensions[:5]
	}

	if len(parsed.SourceIDs) < 2 {
		return nil, fmt.Errorf("source_ids must have >= 2 entries, got %d", len(parsed.SourceIDs))
	}

	evidenceIDs := make(map[string]struct{}, len(pq.Evidence)*2)
	for _, m := range pq.Evidence {
		evidenceIDs[m.ID] = struct{}{}
		evidenceIDs[promptIDForm(m.ID)] = struct{}{}
	}
	for _, sid := range parsed.SourceIDs {
		if _, ok := evidenceIDs[sid]; !ok {
			return nil, fmt.Errorf("source_id %q not in evidence set (prompt injection defense)", sid)
		}
	}

	if parsed.Confidence < 0 {
		parsed.Confidence = 0
	}
	if parsed.Confidence > 1 {
		parsed.Confidence = 1
	}
	if parsed.Importance < 1 {
		parsed.Importance = 1
	}
	if parsed.Importance > 10 {
		parsed.Importance = 10
	}

	if len(parsed.Tags) > 5 {
		parsed.Tags = parsed.Tags[:5]
	}

	return &DialecticInsight{
		Content:    parsed.Content,
		Tensions:   parsed.Tensions,
		SourceIDs:  parsed.SourceIDs,
		Confidence: parsed.Confidence,
		Importance: parsed.Importance,
		Tags:       parsed.Tags,
	}, nil
}
