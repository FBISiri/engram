package reflection

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestRunStage0Success(t *testing.T) {
	err := runStage0(context.Background(), stage0Config{
		command: "true",
		timeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatalf("expected nil error for `true`, got %v", err)
	}
}

func TestRunStage0NonZeroExitSurfaces(t *testing.T) {
	err := runStage0(context.Background(), stage0Config{
		command: "false",
		timeout: 5 * time.Second,
	})
	if err == nil {
		t.Fatal("expected non-nil error for `false` (non-zero exit must not be swallowed)")
	}
}

func TestRunStage0Timeout(t *testing.T) {
	err := runStage0(context.Background(), stage0Config{
		command: "sleep",
		args:    []string{"5"},
		timeout: 50 * time.Millisecond,
	})
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("expected timeout error, got %v", err)
	}
}

func TestStage0DisabledByDefault(t *testing.T) {
	t.Setenv("ENGRAM_CONSOLIDATION_STAGE0_ENABLED", "")
	if stage0ConfigFromEnv().enabled {
		t.Fatal("stage0 must be disabled when the env var is unset/empty")
	}
	// maybeRunConsolidationStage0 must be a no-op (no spawn) when disabled.
	(&Engine{}).maybeRunConsolidationStage0(context.Background())
}
