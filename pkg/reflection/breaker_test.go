package reflection

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCircuitState(t *testing.T) {
	base := time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)
	cooldown := 24 * time.Hour
	tests := []struct {
		name        string
		failCount   int
		threshold   int
		lastAttempt time.Time
		now         time.Time
		wantOpen    bool
		wantProbe   bool
	}{
		{"below threshold", 3, 4, base, base, false, false},
		{"at threshold, cooldown not elapsed", 4, 4, base, base.Add(1 * time.Hour), true, false},
		{"at threshold, cooldown elapsed -> probe", 4, 4, base, base.Add(25 * time.Hour), true, true},
		{"above threshold, cooldown not elapsed", 9, 4, base, base.Add(time.Hour), true, false},
		{"above threshold, exactly cooldown -> probe", 5, 4, base, base.Add(24 * time.Hour), true, true},
		{"open but zero lastAttempt -> probe (never latch)", 4, 4, time.Time{}, base, true, true},
		{"threshold clamp <1 treated as 1", 1, 0, base, base.Add(time.Hour), true, false},
		{"zero failCount never open", 0, 4, base, base.Add(100 * time.Hour), false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			open, probe := circuitState(tt.failCount, tt.threshold, tt.lastAttempt, cooldown, tt.now)
			if open != tt.wantOpen || probe != tt.wantProbe {
				t.Fatalf("circuitState = (open=%v, probe=%v), want (open=%v, probe=%v)",
					open, probe, tt.wantOpen, tt.wantProbe)
			}
		})
	}
}

func TestBreakerThreshold(t *testing.T) {
	tests := []struct {
		env  string
		set  bool
		want int
	}{
		{"", false, defaultBreakerThreshold},
		{"", true, defaultBreakerThreshold},
		{"6", true, 6},
		{"1", true, 1},
		{"0", true, defaultBreakerThreshold},
		{"-3", true, defaultBreakerThreshold},
		{"abc", true, defaultBreakerThreshold},
		{" 7 ", true, 7},
	}
	for _, tt := range tests {
		t.Run(tt.env, func(t *testing.T) {
			if tt.set {
				t.Setenv(envBreakerThreshold, tt.env)
			} else {
				t.Setenv(envBreakerThreshold, "")
			}
			if got := breakerThreshold(); got != tt.want {
				t.Fatalf("breakerThreshold(%q) = %d, want %d", tt.env, got, tt.want)
			}
		})
	}
}

func TestBreakerCooldown(t *testing.T) {
	def := time.Duration(defaultBreakerCooldownH * float64(time.Hour))
	tests := []struct {
		env  string
		want time.Duration
	}{
		{"", def},
		{"24", 24 * time.Hour},
		{"0.5", 30 * time.Minute},
		{"0", 0},
		{"-1", def},
		{"nope", def},
	}
	for _, tt := range tests {
		t.Run(tt.env, func(t *testing.T) {
			t.Setenv(envBreakerCooldownH, tt.env)
			if got := breakerCooldown(); got != tt.want {
				t.Fatalf("breakerCooldown(%q) = %s, want %s", tt.env, got, tt.want)
			}
		})
	}
}

func TestCreditErrorStatus(t *testing.T) {
	tests := []struct {
		name string
		errs []string
		want int
	}{
		{"402 billing", []string{"llm call failed: llm returned status 402: no credits"}, 402},
		{"401 auth", []string{"llm returned status 401: bad key"}, 401},
		{"403 forbidden", []string{"llm returned status 403: region blocked"}, 403},
		{"429 transient not fast-trip", []string{"llm returned status 429: rate limited"}, 0},
		{"500 transient not fast-trip", []string{"llm returned status 500: overloaded"}, 0},
		{"no llm error", []string{"llm returned no parseable insights"}, 0},
		{"empty", nil, 0},
		{"lowest credit code wins", []string{"llm returned status 403: x", "llm returned status 402: y"}, 402},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := creditErrorStatus(tt.errs); got != tt.want {
				t.Fatalf("creditErrorStatus(%v) = %d, want %d", tt.errs, got, tt.want)
			}
		})
	}
}

func TestFinalizeFailureAccounting(t *testing.T) {
	readCount := func(dir string) int {
		n, _ := readFailureCount(filepath.Join(dir, reflectionFailureCountFile))
		return n
	}
	tests := []struct {
		name      string
		result    *RunResult
		succeeded bool
		wantCount int
	}{
		{
			name:      "402 fast-trips to threshold",
			result:    &RunResult{Triggered: true, Errors: []string{"llm call failed: llm returned status 402: no credits"}},
			wantCount: defaultBreakerThreshold,
		},
		{
			name:      "429 bumps by exactly 1",
			result:    &RunResult{Triggered: true, Errors: []string{"llm call failed: llm returned status 429: rate limited"}},
			wantCount: 1,
		},
		{
			name:      "DryRun is a no-op",
			result:    &RunResult{Triggered: true, DryRun: true, Errors: []string{"llm returned status 402: no credits"}},
			wantCount: 0,
		},
		{
			name:      "succeeded is a no-op",
			result:    &RunResult{Triggered: true, Errors: []string{"llm returned status 402: no credits"}},
			succeeded: true,
			wantCount: 0,
		},
		{
			name:      "not triggered is a no-op",
			result:    &RunResult{Triggered: false, Errors: []string{"llm returned status 402: x"}},
			wantCount: 0,
		},
		{
			name:      "no errors is a no-op",
			result:    &RunResult{Triggered: true},
			wantCount: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("ENGRAM_STATE_DIR", dir)
			t.Setenv("SIRI_HOME", "")
			t.Setenv(envBreakerThreshold, "") // default 4
			// Sanity: siriDirPath must honour ENGRAM_STATE_DIR for this test to be valid.
			if got, _ := siriDirPath(); got != dir {
				t.Fatalf("siriDirPath()=%q, want ENGRAM_STATE_DIR %q", got, dir)
			}
			finalizeFailureAccounting(tt.result, tt.succeeded)
			if got := readCount(dir); got != tt.wantCount {
				t.Fatalf("failure count = %d, want %d", got, tt.wantCount)
			}
		})
	}
}

func TestSchedulerPaused(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir)
	t.Setenv("SIRI_HOME", "")

	// env pause variants
	for _, v := range []string{"1", "true", "TRUE"} {
		t.Setenv("ENGRAM_REFLECTION_PAUSED", v)
		if p, _ := SchedulerPaused(); !p {
			t.Fatalf("ENGRAM_REFLECTION_PAUSED=%q: want paused", v)
		}
	}
	// not paused when env off and no file
	t.Setenv("ENGRAM_REFLECTION_PAUSED", "0")
	if p, _ := SchedulerPaused(); p {
		t.Fatalf("env=0, no file: want not paused")
	}
	// file pause
	if err := os.WriteFile(filepath.Join(dir, reflectionPausedFile), []byte(""), 0644); err != nil {
		t.Fatal(err)
	}
	if p, reason := SchedulerPaused(); !p || reason == "" {
		t.Fatalf("pause file present: want paused with reason, got (%v, %q)", p, reason)
	}
}
