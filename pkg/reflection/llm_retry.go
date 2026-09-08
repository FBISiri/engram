package reflection

import (
	"context"
	"errors"
	"fmt"
	"log"
	"reflect"

	"github.com/FBISiri/engram/pkg/llm"
)

// ErrTruncatedResponse marks an LLM response that was cut off because the token
// budget was exhausted (finish_reason="length"), as opposed to a malformed
// payload. Callers wrap it so the failure is reported as a BUDGET problem, never
// a FORMAT / JSON parse problem.
var ErrTruncatedResponse = errors.New("llm response truncated: token budget exhausted")

// truncationError returns a wrapped ErrTruncatedResponse with diagnostics when
// the response was cut off by the budget; nil otherwise. The message must NOT
// mention "JSON"/"parse"/"format" — this is a budget signal, not a format one.
func truncationError(stage string, meta llm.Meta) error {
	if meta.FinishReason != "length" {
		return nil
	}
	return fmt.Errorf("%s: %w (finish_reason=%q max_tokens=%d completion_tokens=%d reasoning_tokens=%d raw_len=%d)",
		stage, ErrTruncatedResponse, meta.FinishReason, meta.MaxTokens, meta.CompletionTokens, meta.ReasoningTokens, meta.RawLen)
}

// callLLMMetaBudget is the test seam for the budgeted retry path. It honors the
// callLLMFunc override property: when the seam still points at the real client,
// real metadata (finish_reason, usage) is captured via llm.CallWithBudget; when
// a test overrides callLLMFunc, best-effort meta is returned.
var callLLMMetaBudget = func(ctx context.Context, prompt string, maxTokens int) (string, llm.Meta, error) {
	if reflect.ValueOf(callLLMFunc).Pointer() == reflect.ValueOf(llm.Call).Pointer() {
		return llm.CallWithBudget(ctx, prompt, maxTokens)
	}
	content, err := callLLMFunc(ctx, prompt)
	return content, llm.Meta{RawLen: len(content), MaxTokens: maxTokens,
		PromptTokens: -1, CompletionTokens: -1, TotalTokens: -1, ReasoningTokens: -1}, err
}

// callLLMWithRetry runs prompt at budget; on finish_reason=="length" it retries
// ONCE at min(2*budget, ceiling). At most one retry. Logs the retry (attempt,
// old/new budget, outcome).
func callLLMWithRetry(ctx context.Context, prompt string, budget, ceiling int, stage string) (string, llm.Meta, error) {
	resp, meta, err := callLLMMetaBudget(ctx, prompt, budget)
	if err != nil || meta.FinishReason != "length" {
		return resp, meta, err
	}
	newBudget := budget * 2
	if ceiling > 0 && newBudget > ceiling {
		newBudget = ceiling
	}
	if newBudget <= budget {
		return resp, meta, err
	}
	log.Printf("[reflection] %s truncated at max_tokens=%d (completion_tokens=%d reasoning_tokens=%d) — retry attempt 2 old_budget=%d new_budget=%d",
		stage, budget, meta.CompletionTokens, meta.ReasoningTokens, budget, newBudget)
	resp2, meta2, err2 := callLLMMetaBudget(ctx, prompt, newBudget)
	if err2 != nil {
		log.Printf("[reflection] %s retry attempt 2 error: %v", stage, err2)
		return resp2, meta2, err2
	}
	log.Printf("[reflection] %s retry attempt 2 outcome: finish_reason=%q raw_len=%d max_tokens=%d completion_tokens=%d reasoning_tokens=%d",
		stage, meta2.FinishReason, meta2.RawLen, meta2.MaxTokens, meta2.CompletionTokens, meta2.ReasoningTokens)
	return resp2, meta2, err2
}
