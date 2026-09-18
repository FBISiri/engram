package reflection

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/FBISiri/engram/pkg/llm"
	"golang.org/x/sync/errgroup"
)

const (
	defaultDialecticTimeout = 45 * time.Second
	maxDialecticTimeout     = 90 * time.Second
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
	LLMConfHighCount   int      `json:"llm_conf_high_count"`
	LLMConfMidCount    int      `json:"llm_conf_mid_count"`
	LLMConfLowCount    int      `json:"llm_conf_low_count"`
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
	if len(evidenceList) == 0 {
		return nil, DialecticStats{}, fmt.Errorf("generateDialecticInsights: empty evidenceList")
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

			stage := fmt.Sprintf("dialectic q%d", i+1)
			llmStart := time.Now()
			response, meta, err := callLLMWithRetry(qctx, prompt, llm.DialecticMaxTokens(), llm.MaxTokensCeiling(), stage)
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

			log.Printf("[reflection] dialectic q%d llm: finish_reason=%q raw_len=%d max_tokens=%d completion_tokens=%d reasoning_tokens=%d prompt_tokens=%d",
				i+1, meta.FinishReason, meta.RawLen, meta.MaxTokens, meta.CompletionTokens, meta.ReasoningTokens, meta.PromptTokens)

			if terr := truncationError(stage, meta); terr != nil {
				errMu.Lock()
				stats.Errors = append(stats.Errors, terr.Error())
				errMu.Unlock()
				statuses[i] = dialecticStatusFailed
				return nil
			}

			insight, warnings, err := parseDialecticResponse(response, pq, meta, fmt.Sprintf("dialectic-q%d", i+1))
			if len(warnings) > 0 {
				errMu.Lock()
				for _, w := range warnings {
					stats.Errors = append(stats.Errors, fmt.Sprintf("dialectic q%d parse: %s", i+1, w))
				}
				errMu.Unlock()
			}
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

	for i, s := range statuses {
		switch s {
		case dialecticStatusOk:
			stats.OkCount++
			conf := insights[i].Confidence
			if conf >= 0.6 {
				stats.LLMConfHighCount++
			} else if conf > 0 {
				stats.LLMConfMidCount++
			} else {
				stats.LLMConfLowCount++
			}
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
	fmt.Fprintf(&sb, "Focal question: %s\n\n", pq.Question)
	fmt.Fprintf(&sb, "Below are %d evidence memories retrieved for this question. ", len(pq.Evidence))
	sb.WriteString("Synthesize a dialectic insight that identifies tensions, contradictions, or nuanced patterns across these memories.\n\n")

	sb.WriteString("Evidence:\n")
	for i, m := range pq.Evidence {
		content := m.Content
		if len(content) > 300 {
			content = content[:300] + "..."
		}
		fmt.Fprintf(&sb, "%d. [id=%s] %s\n", i+1, promptIDForm(m.ID), content)
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

// sanitizeJSONControlChars walks s as a byte-wise state machine tracking
// in-string and backslash-escape state. Literal control chars (< 0x20) that
// appear INSIDE a JSON string literal are escaped (LF->\n, CR->\r, TAB->\t,
// others->\u00XX). Bytes outside string literals are left untouched, so an
// already-clean JSON input passes through byte-identical.
func sanitizeJSONControlChars(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inString {
			if escaped {
				b.WriteByte(c)
				escaped = false
				continue
			}
			switch {
			case c == '\\':
				b.WriteByte(c)
				escaped = true
			case c == '"':
				b.WriteByte(c)
				inString = false
			case c < 0x20:
				switch c {
				case '\n':
					b.WriteString(`\n`)
				case '\r':
					b.WriteString(`\r`)
				case '\t':
					b.WriteString(`\t`)
				default:
					fmt.Fprintf(&b, `\u%04x`, c)
				}
			default:
				b.WriteByte(c)
			}
		} else {
			if c == '"' {
				inString = true
			}
			b.WriteByte(c)
		}
	}
	return b.String()
}

// repairStructuralSemicolons rewrites `;` to `,` ONLY in structural position:
// immediately after a closing quote that terminates a string value, where the
// next non-space token starts a new key (a `"`). It reuses the in-string /
// escape state machine from sanitizeJSONControlChars so a legitimate `;` (or
// `";`) INSIDE a string literal is never rewritten.
func repairStructuralSemicolons(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false
	escaped := false
	afterCloseQuote := false // value-closing quote seen, only whitespace since
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inString {
			b.WriteByte(c)
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
				afterCloseQuote = true
			}
			continue
		}
		switch {
		case c == '"':
			inString = true
			afterCloseQuote = false
			b.WriteByte(c)
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			b.WriteByte(c) // preserve afterCloseQuote across whitespace
		case c == ';' && afterCloseQuote:
			j := i + 1
			for j < len(s) && (s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r') {
				j++
			}
			if j < len(s) && s[j] == '"' {
				b.WriteByte(',')
			} else {
				b.WriteByte(c)
			}
			afterCloseQuote = false
		default:
			b.WriteByte(c)
			afterCloseQuote = false
		}
	}
	return b.String()
}

// resolveEvidenceID maps a source_id from the LLM to a canonical evidence id,
// or reports that it cannot be trusted. It NEVER invents ids: a returned id is
// always already present in the evidence set (prompt injection defense).
//   - direct hit (full id or prompt id form) -> returned verbatim
//   - otherwise, if sid is >= 8 chars and is a prefix of EXACTLY ONE full
//     evidence id, treat it as a truncated/garbled id and repair to that full id
//   - otherwise -> not resolvable (caller drops it)
func resolveEvidenceID(sid string, evidenceIDs map[string]struct{}, fullIDs []string) (string, bool) {
	if _, ok := evidenceIDs[sid]; ok {
		return sid, true
	}
	if len(sid) >= 8 {
		match := ""
		count := 0
		for _, full := range fullIDs {
			if strings.HasPrefix(full, sid) {
				match = full
				count++
				if count > 1 {
					break
				}
			}
		}
		if count == 1 {
			return match, true
		}
	}
	return "", false
}

// parseDialecticResponse parses the LLM JSON response and validates source_ids
// against the evidence set (prompt injection defense). It returns a slice of
// human-readable warnings for fields/entries that were repaired or dropped so
// the caller can record them without voiding the whole question.
func parseDialecticResponse(response string, pq PerQuestionEvidence, meta llm.Meta, stage string) (*DialecticInsight, []string, error) {
	raw := response
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimPrefix(response, "```")
	response = strings.TrimSuffix(response, "```")
	response = strings.TrimSpace(response)

	var parsed dialecticLLMResponse
	stage2, err := parseLenientJSON(response, &parsed)
	if err != nil {
		// No parseable JSON survived the recovery ladder: there is nothing
		// to salvage, so this stays fatal.
		path := dumpRawResponse(stage, raw)
		log.Printf("[reflection] dialectic JSON parse failed: finish_reason=%q raw_len=%d dump=%s: %v",
			meta.FinishReason, meta.RawLen, path, err)
		return nil, nil, fmt.Errorf("JSON parse (raw dump: %s): %w", path, err)
	}
	if stage2 != "strict" {
		log.Printf("[reflection] %s JSON recovered via %s: finish_reason=%q raw_len=%d",
			stage, stage2, meta.FinishReason, meta.RawLen)
	}

	var warnings []string

	// Content is the core of the insight; without it there is nothing to keep.
	if parsed.Content == "" {
		return nil, nil, fmt.Errorf("missing content field")
	}

	// Missing tensions degrades gracefully: an insight can stand without any
	// explicit tensions, so default to an empty (non-nil) slice.
	if parsed.Tensions == nil {
		parsed.Tensions = []string{}
	}
	if len(parsed.Tensions) > 5 {
		parsed.Tensions = parsed.Tensions[:5]
	}

	// source_id validation — DEGRADE GRACEFULLY. A single blemished id used to
	// void the entire question (~33% output wasted per run). Instead we repair
	// what is safely repairable and DROP what is not, keeping the rest.
	//
	// PROMPT-INJECTION DEFENSE IS PRESERVED: an id outside the evidence set is
	// never accepted into the output. Dropping an untrusted id is allowed;
	// trusting it is not. Warnings quote ids with %q so Go escapes any control
	// chars, keeping them safe from log injection.
	evidenceIDs := make(map[string]struct{}, len(pq.Evidence)*2)
	fullIDs := make([]string, 0, len(pq.Evidence))
	for _, m := range pq.Evidence {
		evidenceIDs[m.ID] = struct{}{}
		evidenceIDs[promptIDForm(m.ID)] = struct{}{}
		fullIDs = append(fullIDs, m.ID)
	}
	validIDs := make([]string, 0, len(parsed.SourceIDs))
	for _, sid := range parsed.SourceIDs {
		if resolved, ok := resolveEvidenceID(sid, evidenceIDs, fullIDs); ok {
			validIDs = append(validIDs, resolved)
		} else {
			warnings = append(warnings, fmt.Sprintf("source_id %q dropped: not in evidence set (prompt injection defense)", sid))
		}
	}

	// An insight needs >= 2 corroborating sources by design. If fewer than 2
	// survive filtering, drop the QUESTION with a clear error — but still return
	// the warnings so the dropped id(s) remain observable.
	if len(validIDs) < 2 {
		return nil, warnings, fmt.Errorf("source_ids: only %d valid of %d after filtering (need >= 2); dropped %d", len(validIDs), len(parsed.SourceIDs), len(warnings))
	}
	parsed.SourceIDs = validIDs

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
	}, warnings, nil
}
