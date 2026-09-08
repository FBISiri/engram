package reflection

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/llm"
)

func TestTruncationError(t *testing.T) {
	// Non-length finish reason => nil.
	if err := truncationError("focal", llm.Meta{FinishReason: "stop"}); err != nil {
		t.Fatalf("expected nil for finish_reason=stop, got %v", err)
	}
	// Length => wrapped ErrTruncatedResponse, no format/JSON wording.
	err := truncationError("dialectic q1", llm.Meta{
		FinishReason: "length", MaxTokens: 4000, CompletionTokens: 4000, ReasoningTokens: 3900, RawLen: 12,
	})
	if err == nil {
		t.Fatal("expected error for finish_reason=length")
	}
	if !errors.Is(err, ErrTruncatedResponse) {
		t.Errorf("errors.Is(ErrTruncatedResponse) = false, err=%v", err)
	}
	msg := strings.ToLower(err.Error())
	for _, bad := range []string{"json", "parse", "format"} {
		if strings.Contains(msg, bad) {
			t.Errorf("truncation message must not mention %q: %q", bad, err.Error())
		}
	}
}

func TestCallLLMWithRetry_RetriesOnLength(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	var budgets []int
	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, maxTokens int) (string, llm.Meta, error) {
		budgets = append(budgets, maxTokens)
		calls++
		if calls == 1 {
			return "trunc", llm.Meta{FinishReason: "length", MaxTokens: maxTokens}, nil
		}
		return "full", llm.Meta{FinishReason: "stop", MaxTokens: maxTokens}, nil
	}

	resp, meta, err := callLLMWithRetry(context.Background(), "p", 4000, 8000, "dialectic q1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != "full" || meta.FinishReason != "stop" {
		t.Errorf("expected second (retry) result, got resp=%q meta=%+v", resp, meta)
	}
	if calls != 2 {
		t.Fatalf("expected exactly 2 calls, got %d", calls)
	}
	if len(budgets) != 2 || budgets[0] != 4000 || budgets[1] != 8000 {
		t.Errorf("budget escalation = %v, want [4000 8000]", budgets)
	}
}

func TestCallLLMWithRetry_BothLength(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, maxTokens int) (string, llm.Meta, error) {
		calls++
		return "trunc", llm.Meta{FinishReason: "length", MaxTokens: maxTokens}, nil
	}

	_, meta, err := callLLMWithRetry(context.Background(), "p", 4000, 8000, "focal")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 calls, got %d", calls)
	}
	if meta.FinishReason != "length" {
		t.Errorf("expected length meta so caller emits truncation, got %+v", meta)
	}
	if terr := truncationError("focal", meta); terr == nil {
		t.Error("expected truncationError to be non-nil for both-length outcome")
	}
}

func TestDialecticPath_RecordsTruncationNotParseError(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	// Both attempts return a truncated fragment with finish_reason=length.
	callLLMMetaBudget = func(_ context.Context, _ string, maxTokens int) (string, llm.Meta, error) {
		return `{"content":"partial frag`, llm.Meta{
			FinishReason: "length", MaxTokens: maxTokens, CompletionTokens: maxTokens, ReasoningTokens: maxTokens - 100,
		}, nil
	}

	e := dialecticTestEngine()
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2", "e3")},
	}

	_, stats, err := e.generateDialecticInsights(context.Background(), evidenceList, e.cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stats.FailedCount != 1 {
		t.Errorf("expected FailedCount=1, got %d", stats.FailedCount)
	}
	if len(stats.Errors) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(stats.Errors), stats.Errors)
	}
	msg := strings.ToLower(stats.Errors[0])
	if strings.Contains(msg, "json") || strings.Contains(msg, "parse") {
		t.Errorf("expected budget/truncation error, got format error: %q", stats.Errors[0])
	}
	if !strings.Contains(msg, "truncated") {
		t.Errorf("expected truncation wording, got %q", stats.Errors[0])
	}
}
