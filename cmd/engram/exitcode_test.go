package main

import (
	"errors"
	"fmt"
	"testing"

	"github.com/FBISiri/engram/pkg/dream"
)

func TestExitCodeFor(t *testing.T) {
	joined := errors.Join(
		fmt.Errorf("%w: x", dream.ErrSkillDiffPartial),
		fmt.Errorf("%w: y", dream.ErrConsolidateDegraded),
	)
	cases := []struct {
		name string
		err  error
		want int
	}{
		{"nil", nil, 0},
		{"help", errHelp, 0},
		{"unknown-flag", fmt.Errorf("%w: --bogus", errUnknownFlag), 2},
		{"skill-diff", fmt.Errorf("%w: 1/2", dream.ErrSkillDiffPartial), 3},
		{"llm-unavailable", fmt.Errorf("%w: no key", dream.ErrLLMUnavailable), 4},
		{"consolidate-degraded", fmt.Errorf("%w: 2/3", dream.ErrConsolidateDegraded), 4},
		{"first-run-guard", fmt.Errorf("%w: blocked", dream.ErrFirstRunGuard), 4},
		{"joined-skilldiff+consolidate", joined, 3}, // skill-diff precedence
		{"generic", errors.New("boom"), 1},
	}
	for _, c := range cases {
		if got := exitCodeFor(c.err); got != c.want {
			t.Errorf("%s: exitCodeFor=%d, want %d", c.name, got, c.want)
		}
	}
}
