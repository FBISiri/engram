package reflection

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/FBISiri/engram/pkg/llm"
)

// reflectionLastFocalFile stores the last SUCCESSFUL focal question set so a
// future degraded run (Stage 1 429) can fall back to real, recently-generated
// questions instead of the generic preset.
const reflectionLastFocalFile = "reflection_last_focal_questions.json"

// presetFocalQuestions is the always-available generic fallback set. Used when
// no persisted question set exists yet. Cross-domain and content-agnostic so a
// degraded run still produces something meaningful while quota is exhausted.
var presetFocalQuestions = []string{
	"What recurring patterns or themes emerge across these memories?",
	"What tensions or contradictions appear between different memories?",
	"What cross-domain connections link otherwise unrelated memories?",
	"What generalizable lessons or principles can be drawn from these experiences?",
	"What open questions or uncertainties remain unresolved?",
}

// persistFocalQuestions saves a successful focal question set to the state dir
// for future fallback. Best-effort: returns error for the caller to log.
func persistFocalQuestions(questions []string) error {
	dir, err := siriDirPath()
	if err != nil {
		return err
	}
	data, err := json.Marshal(questions)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, reflectionLastFocalFile), data, 0644)
}

// loadPersistedFocalQuestions reads the last persisted focal question set.
func loadPersistedFocalQuestions() ([]string, error) {
	dir, err := siriDirPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(dir, reflectionLastFocalFile))
	if err != nil {
		return nil, err
	}
	var questions []string
	if err := json.Unmarshal(data, &questions); err != nil {
		return nil, err
	}
	return questions, nil
}

// focalFallbackQuestions returns a non-empty fallback question set and its
// source. It prefers the last persisted successful set ("persisted") and falls
// back to the generic preset ("preset"). When n>0 and the set is longer than n,
// it is capped to n. NEVER returns empty (preset is always non-empty).
func focalFallbackQuestions(n int) ([]string, string) {
	questions, source := presetFocalQuestions, "preset"
	if persisted, err := loadPersistedFocalQuestions(); err == nil && len(persisted) > 0 {
		questions, source = persisted, "persisted"
	}
	if n > 0 && len(questions) > n {
		questions = questions[:n]
	}
	return questions, source
}

// formatLastRateLimit returns a compact one-line description of the most recent
// 429 rate-limit snapshot observed by the LLM client, or "none" if there has
// been no 429 since process start. Pure read of llm.LastRateLimit().
func formatLastRateLimit() string {
	s := llm.LastRateLimit()
	if s == nil {
		return "none"
	}
	return fmt.Sprintf("status=%d retry_after=%q remaining_req=%s remaining_tok=%s at=%s",
		s.StatusCode, s.RetryAfter, s.RequestsRemaining, s.TokensRemaining, s.Timestamp)
}
