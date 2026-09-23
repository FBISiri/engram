package server

import (
	"errors"
	"testing"

	"github.com/FBISiri/engram/pkg/reflection"
)

// TestReflectionRunLogError_FailedRunNeverEmpty is the regression guard for the
// observability poison pill (B3): a reflection run that returns err==nil but
// populates result.Errors (the dominant Stage 1 429 mode, where RunV2 returns
// result,nil) MUST NOT be logged with error="". Before the fix the
// "reflection run finished" log derived its error field from the transport
// error alone, so journalctl rendered failed runs as successful runs.
func TestReflectionRunLogError_FailedRunNeverEmpty(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		result  *reflection.RunResult
		want    string
		wantNon bool // want a NON-empty string
	}{
		{
			name:    "nil err with result.Errors (Stage 1 429) → non-empty",
			err:     nil,
			result:  &reflection.RunResult{Errors: []string{"focal question generation failed: llm rate limited (429)"}},
			wantNon: true,
		},
		{
			name:    "nil err with multiple result.Errors → summarized non-empty",
			err:     nil,
			result:  &reflection.RunResult{Errors: []string{"first error", "second error"}},
			want:    "first error (+1 more)",
			wantNon: true,
		},
		{
			name:   "transport error wins",
			err:    errors.New("context deadline exceeded"),
			result: &reflection.RunResult{Errors: []string{"other"}},
			want:   "context deadline exceeded",
		},
		{
			name:   "genuine success → empty",
			err:    nil,
			result: &reflection.RunResult{InsightsCreated: 3},
			want:   "",
		},
		{
			name:   "nil result, nil err → empty",
			err:    nil,
			result: nil,
			want:   "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := reflectionRunLogError(tc.err, tc.result)
			if tc.wantNon {
				if got == "" {
					t.Fatalf("expected non-empty log error for a failed run, got empty")
				}
				if tc.want != "" && got != tc.want {
					t.Fatalf("got %q, want %q", got, tc.want)
				}
				return
			}
			if got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}
