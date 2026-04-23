package reflection

import (
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// TestHumanAge_Buckets exercises the four bucket boundaries
// (<1h / <24h / <7d / >=7d) and the future-timestamp clamp.
// W17 v1.2 T2: humanAge must accept an explicit "now" so callers can
// inject a fixed clock; otherwise we can't write deterministic tests.
func TestHumanAge_Buckets(t *testing.T) {
	now := time.Date(2026, 4, 23, 12, 0, 0, 0, time.UTC)
	nowSec := float64(now.Unix())

	cases := []struct {
		name      string
		createdAt float64
		want      string
	}{
		{"just now (<1min)", nowSec - 30, "0min ago"},
		{"45 minutes ago", nowSec - 45*60, "45min ago"},
		{"on the hour boundary", nowSec - 60*60, "1h ago"},
		{"5 hours ago", nowSec - 5*3600, "5h ago"},
		{"23h59m ago", nowSec - (24*3600 - 60), "23h ago"},
		{"24h ago bumps to days", nowSec - 24*3600, "1d ago"},
		{"3 days ago", nowSec - 3*24*3600, "3d ago"},
		{"6d23h ago", nowSec - (7*24*3600 - 3600), "6d ago"},
		{"7 days ago bumps to weeks", nowSec - 7*24*3600, "1w ago"},
		{"3 weeks ago", nowSec - 21*24*3600, "3w ago"},
		// future timestamps shouldn't yield negatives in the prompt.
		{"future timestamp clamps to 0min", nowSec + 3600, "0min ago"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := humanAge(tc.createdAt, now)
			if got != tc.want {
				t.Fatalf("humanAge(%.0f, now) = %q, want %q", tc.createdAt, got, tc.want)
			}
		})
	}
}

// TestBuildPromptAt_InjectsAgeField verifies that the prompt template
// emits an `age=...` field for every memory line, deterministically,
// when given a fixed reference time.
func TestBuildPromptAt_InjectsAgeField(t *testing.T) {
	now := time.Date(2026, 4, 23, 12, 0, 0, 0, time.UTC)
	mems := []memory.Memory{
		{
			ID:         "abcdef1234",
			Type:       "insight",
			Content:    "Frank prefers concise answers.",
			Importance: 7,
			CreatedAt:  float64(now.Add(-2 * time.Hour).Unix()),
		},
		{
			ID:         "0987654321",
			Type:       "event",
			Content:    "W17 Day3 T2 kickoff.",
			Importance: 5,
			CreatedAt:  float64(now.Add(-3 * 24 * time.Hour).Unix()),
		},
	}

	out := buildPromptAt(mems, now)

	// Every memory line must carry an age=... field.
	for _, want := range []string{"age=2h ago", "age=3d ago"} {
		if !strings.Contains(out, want) {
			t.Errorf("expected prompt to contain %q, got:\n%s", want, out)
		}
	}

	// Sanity: id prefix + type + importance still rendered.
	for _, want := range []string{"id=abcdef12", "id=09876543", "type=insight", "type=event"} {
		if !strings.Contains(out, want) {
			t.Errorf("expected prompt to contain %q", want)
		}
	}
}

// TestBuildPrompt_DefaultClock just confirms buildPrompt() (which uses
// time.Now() internally) still wires through to the same template — guards
// against accidental signature drift on the public entry point.
func TestBuildPrompt_DefaultClock(t *testing.T) {
	mems := []memory.Memory{{
		ID:         "xx",
		Type:       "insight",
		Content:    "hello",
		Importance: 5,
		CreatedAt:  float64(time.Now().Add(-30 * time.Minute).Unix()),
	}}
	out := buildPrompt(mems)
	if !strings.Contains(out, "age=") {
		t.Fatalf("buildPrompt output missing age= field:\n%s", out)
	}
}
