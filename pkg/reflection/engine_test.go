package reflection

import (
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// ── parseLLMResponse tests ───────────────────────────────────────────────

func TestParseLLMResponse_SingleInsight(t *testing.T) {
	input := `---
INSIGHT: Siri has been consistently creating calendar events before committing to tasks, reflecting improved adherence to the task-scheduling discipline that Frank reinforced.
IMPORTANCE: 8
TAGS: siri-behavior, task-scheduling, improvement
---`

	insights, _ := parseLLMResponse(input)
	if len(insights) != 1 {
		t.Fatalf("expected 1 insight, got %d", len(insights))
	}
	if insights[0].Importance != 8 {
		t.Errorf("expected importance 8, got %.0f", insights[0].Importance)
	}
	if len(insights[0].Tags) != 3 {
		t.Errorf("expected 3 tags, got %d: %v", len(insights[0].Tags), insights[0].Tags)
	}
	if !strings.Contains(insights[0].Content, "Siri has been") {
		t.Errorf("unexpected content: %q", insights[0].Content)
	}
}

func TestParseLLMResponse_MultipleInsights(t *testing.T) {
	input := `---
INSIGHT: Frank tends to prioritize cycling goals over other fitness activities, dedicating significant planning energy to route optimization and performance tracking.
IMPORTANCE: 7
TAGS: frank, cycling, fitness
---
---
INSIGHT: Siri has demonstrated a recurring pattern of proposing future task plans without immediately creating corresponding calendar events, despite repeated corrections from Frank.
IMPORTANCE: 9
TAGS: siri-behavior, calendar, recurring-failure, frank-feedback
---`

	insights, _ := parseLLMResponse(input)
	if len(insights) != 2 {
		t.Fatalf("expected 2 insights, got %d", len(insights))
	}
	if insights[0].Importance != 7 {
		t.Errorf("first insight importance: want 7, got %.0f", insights[0].Importance)
	}
	if insights[1].Importance != 9 {
		t.Errorf("second insight importance: want 9, got %.0f", insights[1].Importance)
	}
}

func TestParseLLMResponse_CapsAt3(t *testing.T) {
	input := `---
INSIGHT: Insight one.
IMPORTANCE: 5
TAGS: tag1
---
---
INSIGHT: Insight two.
IMPORTANCE: 6
TAGS: tag2
---
---
INSIGHT: Insight three.
IMPORTANCE: 7
TAGS: tag3
---
---
INSIGHT: Insight four (should be discarded).
IMPORTANCE: 8
TAGS: tag4
---`

	insights, _ := parseLLMResponse(input)
	if len(insights) != 3 {
		t.Fatalf("expected 3 insights (capped), got %d", len(insights))
	}
}

func TestParseLLMResponse_Empty(t *testing.T) {
	insights, _ := parseLLMResponse("")
	if len(insights) != 0 {
		t.Errorf("expected 0 insights for empty input, got %d", len(insights))
	}
}

func TestParseLLMResponse_MissingInsightField(t *testing.T) {
	// Block without INSIGHT: should be skipped.
	input := `---
IMPORTANCE: 5
TAGS: foo
---
---
INSIGHT: Valid insight.
IMPORTANCE: 7
TAGS: bar
---`
	insights, _ := parseLLMResponse(input)
	if len(insights) != 1 {
		t.Fatalf("expected 1 valid insight, got %d", len(insights))
	}
	if insights[0].Importance != 7 {
		t.Errorf("expected importance 7, got %.0f", insights[0].Importance)
	}
}

func TestParseLLMResponse_TagNormalization(t *testing.T) {
	input := `---
INSIGHT: Some insight about Siri.
IMPORTANCE: 5
TAGS: Siri Behavior, Frank Feedback, Task Scheduling, Extra Tag One, Extra Tag Two, This Should Be Cut
---`
	insights, _ := parseLLMResponse(input)
	if len(insights) != 1 {
		t.Fatalf("expected 1 insight, got %d", len(insights))
	}
	// Tags should be normalized and capped at 5.
	if len(insights[0].Tags) != 5 {
		t.Errorf("expected 5 tags (capped), got %d: %v", len(insights[0].Tags), insights[0].Tags)
	}
	// Check normalization: spaces → hyphens, lowercase.
	for _, tag := range insights[0].Tags {
		if strings.Contains(tag, " ") {
			t.Errorf("tag should not contain spaces: %q", tag)
		}
		if tag != strings.ToLower(tag) {
			t.Errorf("tag should be lowercase: %q", tag)
		}
	}
}

// ── parseLLMResponse confidence counting tests (§1.1 v0.3) ──────────────

func TestParseLLMResponse_ConfDefault(t *testing.T) {
	// No CONFIDENCE line → default_count incremented.
	input := `---
INSIGHT: Siri tends to improve over time.
IMPORTANCE: 7
TAGS: growth
---`
	_, counts := parseLLMResponse(input)
	if counts.LLMConfDefaultCount != 1 {
		t.Errorf("expected DefaultCount=1, got %d", counts.LLMConfDefaultCount)
	}
	if counts.LLMConfExplicitCount != 0 {
		t.Errorf("expected ExplicitCount=0, got %d", counts.LLMConfExplicitCount)
	}
}

func TestParseLLMResponse_ConfParseFail(t *testing.T) {
	// CONFIDENCE line present but not a float → parse_fail_count.
	input := `---
INSIGHT: Some insight.
IMPORTANCE: 5
CONFIDENCE: high
TAGS: foo
---`
	_, counts := parseLLMResponse(input)
	if counts.LLMConfParseFailCount != 1 {
		t.Errorf("expected ParseFailCount=1, got %d", counts.LLMConfParseFailCount)
	}
	if counts.LLMConfExplicitCount != 0 {
		t.Errorf("expected ExplicitCount=0, got %d", counts.LLMConfExplicitCount)
	}
}

func TestParseLLMResponse_ConfHigh(t *testing.T) {
	input := `---
INSIGHT: Some insight.
IMPORTANCE: 8
CONFIDENCE: 0.9
TAGS: foo
---`
	_, counts := parseLLMResponse(input)
	if counts.LLMConfExplicitCount != 1 {
		t.Errorf("expected ExplicitCount=1, got %d", counts.LLMConfExplicitCount)
	}
	if counts.LLMConfHighCount != 1 {
		t.Errorf("expected HighCount=1, got %d", counts.LLMConfHighCount)
	}
}

func TestParseLLMResponse_ConfMid(t *testing.T) {
	input := `---
INSIGHT: Some insight.
IMPORTANCE: 5
CONFIDENCE: 0.4
TAGS: foo
---`
	_, counts := parseLLMResponse(input)
	if counts.LLMConfMidCount != 1 {
		t.Errorf("expected MidCount=1, got %d", counts.LLMConfMidCount)
	}
}

func TestParseLLMResponse_ConfLow(t *testing.T) {
	input := `---
INSIGHT: Some insight.
IMPORTANCE: 5
CONFIDENCE: 0.0
TAGS: foo
---`
	_, counts := parseLLMResponse(input)
	if counts.LLMConfLowCount != 1 {
		t.Errorf("expected LowCount=1, got %d", counts.LLMConfLowCount)
	}
}

func TestParseLLMResponse_ConfOOB(t *testing.T) {
	// Raw value > 1 → oob_count (then clamped to 1).
	input := `---
INSIGHT: Some insight.
IMPORTANCE: 5
CONFIDENCE: 1.5
TAGS: foo
---`
	insights, counts := parseLLMResponse(input)
	if counts.LLMConfOutOfBoundsCount != 1 {
		t.Errorf("expected OOBCount=1, got %d", counts.LLMConfOutOfBoundsCount)
	}
	if counts.LLMConfExplicitCount != 1 {
		t.Errorf("expected ExplicitCount=1, got %d", counts.LLMConfExplicitCount)
	}
	// Value should be clamped to 1.0.
	if insights[0].Confidence != 1.0 {
		t.Errorf("expected confidence clamped to 1.0, got %f", insights[0].Confidence)
	}
}

func TestParseLLMResponse_ConfInvariant(t *testing.T) {
	// 3 blocks: one default, one explicit-high, one parse-fail.
	// Invariant: Default+ParseFail+Explicit == 3; Explicit == High+Mid+Low+OOB.
	input := `---
INSIGHT: No confidence line here.
IMPORTANCE: 5
TAGS: a
---
---
INSIGHT: With high confidence.
IMPORTANCE: 8
CONFIDENCE: 0.85
TAGS: b
---
---
INSIGHT: Bad confidence value.
IMPORTANCE: 4
CONFIDENCE: notanumber
TAGS: c
---`
	_, counts := parseLLMResponse(input)
	total := counts.LLMConfDefaultCount + counts.LLMConfParseFailCount + counts.LLMConfExplicitCount
	if total != 3 {
		t.Errorf("invariant Default+ParseFail+Explicit=%d, want 3", total)
	}
	explicitSub := counts.LLMConfHighCount + counts.LLMConfMidCount + counts.LLMConfLowCount + counts.LLMConfOutOfBoundsCount
	if explicitSub != counts.LLMConfExplicitCount {
		t.Errorf("invariant Explicit=%d != High+Mid+Low+OOB=%d", counts.LLMConfExplicitCount, explicitSub)
	}
}

// ── selectInputBatch tests ─────────────────────────────────────────────────

func TestSelectInputBatch_LargerThanMax(t *testing.T) {
	mems := make([]memory.Memory, 30)
	for i := range mems {
		mems[i] = memory.Memory{
			ID:         "mem" + string(rune('a'+i)),
			Importance: float64(i + 1),
		}
	}

	batch := selectInputBatch(mems, 20)
	if len(batch) != 20 {
		t.Fatalf("expected 20, got %d", len(batch))
	}

	// Should be sorted by importance DESC — top 20 should have importance >= 11.
	for _, m := range batch {
		if m.Importance < 11 {
			t.Errorf("expected top-20 by importance, but got importance %.0f", m.Importance)
		}
	}
}

func TestSelectInputBatch_SmallerThanMax(t *testing.T) {
	mems := []memory.Memory{
		{ID: "a", Importance: 5},
		{ID: "b", Importance: 8},
	}
	batch := selectInputBatch(mems, 20)
	if len(batch) != 2 {
		t.Fatalf("expected 2, got %d", len(batch))
	}
}

// ── isReflected tests ──────────────────────────────────────────────────────

func TestIsReflected(t *testing.T) {
	tests := []struct {
		name     string
		mem      memory.Memory
		expected bool
	}{
		{
			name:     "nil metadata",
			mem:      memory.Memory{Metadata: nil},
			expected: false,
		},
		{
			name:     "missing reflected key",
			mem:      memory.Memory{Metadata: map[string]any{"foo": "bar"}},
			expected: false,
		},
		{
			name:     "reflected=true",
			mem:      memory.Memory{Metadata: map[string]any{"reflected": true}},
			expected: true,
		},
		{
			name:     "reflected=false",
			mem:      memory.Memory{Metadata: map[string]any{"reflected": false}},
			expected: false,
		},
		{
			name:     "reflected=string (wrong type)",
			mem:      memory.Memory{Metadata: map[string]any{"reflected": "true"}},
			expected: false,
		},
		// W17 T1 Part 2: top-level ReflectedAt path.
		{
			name:     "ReflectedAt > 0 (V2 field)",
			mem:      memory.Memory{ReflectedAt: 1700000000},
			expected: true,
		},
		{
			name:     "ReflectedAt > 0 wins over metadata=false",
			mem:      memory.Memory{ReflectedAt: 1700000000, Metadata: map[string]any{"reflected": false}},
			expected: true,
		},
		{
			name:     "ReflectedAt == 0, metadata=true — legacy fallback still works",
			mem:      memory.Memory{ReflectedAt: 0, Metadata: map[string]any{"reflected": true}},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isReflected(tt.mem)
			if got != tt.expected {
				t.Errorf("isReflected(%v) = %v, want %v", tt.mem.Metadata, got, tt.expected)
			}
		})
	}
}

// ── readDailyCount / writeDailyCount tests ─────────────────────────────────

func TestDailyCount_RoundTrip(t *testing.T) {
	path := t.TempDir() + "/daily_count"
	
	// Initial count should be 0.
	count, err := readDailyCount(path)
	if err != nil {
		t.Fatalf("readDailyCount: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0, got %d", count)
	}

	// Write count 2.
	if err := writeDailyCount(path, 2); err != nil {
		t.Fatalf("writeDailyCount: %v", err)
	}

	// Read back.
	count, err = readDailyCount(path)
	if err != nil {
		t.Fatalf("readDailyCount after write: %v", err)
	}
	if count != 2 {
		t.Errorf("expected 2, got %d", count)
	}
}

// ── buildPrompt tests ──────────────────────────────────────────────────────

func TestBuildPrompt_ContainsMemories(t *testing.T) {
	mems := []memory.Memory{
		{
			ID:         "abc12345",
			Type:       memory.TypeEvent,
			Content:    "Frank asked Siri to create calendar events immediately.",
			Importance: 8,
			CreatedAt:  float64(time.Now().Unix()),
		},
		{
			ID:         "def67890",
			Type:       memory.TypeInsight,
			Content:    "Siri consistently improves task scheduling discipline.",
			Importance: 7,
			CreatedAt:  float64(time.Now().Unix()),
		},
	}

	prompt := buildPrompt(mems)

	if !strings.Contains(prompt, "reflection engine") {
		t.Error("prompt should mention 'reflection engine'")
	}
	if !strings.Contains(prompt, "abc12345") {
		t.Error("prompt should contain memory ID")
	}
	if !strings.Contains(prompt, "Frank asked Siri") {
		t.Error("prompt should contain memory content")
	}
	if !strings.Contains(prompt, "INSIGHT:") {
		t.Error("prompt should contain INSIGHT: format instruction")
	}
	if !strings.Contains(prompt, "IMPORTANCE:") {
		t.Error("prompt should contain IMPORTANCE: format instruction")
	}
	if !strings.Contains(prompt, "TAGS:") {
		t.Error("prompt should contain TAGS: format instruction")
	}
}

func TestBuildPrompt_TruncatesLongContent(t *testing.T) {
	longContent := strings.Repeat("x", 300)
	mems := []memory.Memory{
		{
			ID:         "test1234",
			Type:       memory.TypeEvent,
			Content:    longContent,
			Importance: 5,
		},
	}

	prompt := buildPrompt(mems)
	// The content should be truncated to 200 chars + "..."
	if strings.Contains(prompt, longContent) {
		t.Error("prompt should truncate long content")
	}
	if !strings.Contains(prompt, "...") {
		t.Error("prompt should contain ... for truncated content")
	}
}

// ── RunResult.RunsToday tests (§1.1 v0.3) ─────────────────────────────────

func TestRunResult_RunsToday_Default(t *testing.T) {
	r := &RunResult{}
	if r.RunsToday != 0 {
		t.Errorf("expected RunsToday=0 by default, got %d", r.RunsToday)
	}
}

func TestRunResult_RunsToday_SetAndRead(t *testing.T) {
	r := &RunResult{RunsToday: 2}
	if r.RunsToday != 2 {
		t.Errorf("expected RunsToday=2, got %d", r.RunsToday)
	}
}
