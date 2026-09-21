package dream

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// newTestEngine builds an Engine with an empty log for summary/decision tests.
func newTestEngine(cfg Config) *Engine {
	return &Engine{
		cfg: cfg,
		log: &Log{},
	}
}

// R1: buildSummary must carry ok / total / failed when skill-diff ran.
func TestBuildSummary_SkillDiffCounts(t *testing.T) {
	e := newTestEngine(Config{})
	e.log.SkillDiffRan = true
	e.log.SkillDiffOK = 10
	e.log.SkillDiffTotal = 23
	e.log.SkillDiffFailed = 13
	e.log.LastRunAdvanced = false

	got := e.buildSummary()
	for _, want := range []string{"skill-diff:", "10", "23", "13", "ok", "failed"} {
		if !strings.Contains(got, want) {
			t.Errorf("summary %q missing %q", got, want)
		}
	}
	if !strings.Contains(got, "10/23 ok, 13 failed") {
		t.Errorf("summary should read counts cleanly, got %q", got)
	}
	if !strings.Contains(got, "HELD") {
		t.Errorf("summary should note last_run HELD, got %q", got)
	}
}

// R1: a clean run still reads cleanly (zero failures).
func TestBuildSummary_SkillDiffCleanRun(t *testing.T) {
	e := newTestEngine(Config{})
	e.log.SkillDiffRan = true
	e.log.SkillDiffOK = 23
	e.log.SkillDiffTotal = 23
	e.log.SkillDiffFailed = 0
	e.log.LastRunAdvanced = true

	got := e.buildSummary()
	if !strings.Contains(got, "23/23 ok, 0 failed") {
		t.Errorf("clean run summary wrong, got %q", got)
	}
	if !strings.Contains(got, "last_run advanced") {
		t.Errorf("clean run should advance last_run, got %q", got)
	}
}

// A single-phase or dry-run invocation that never ran skill-diff must not add
// skill-diff/last_run noise to the summary (protects R2's --phase/--dry-run rule).
func TestBuildSummary_NoSkillDiff(t *testing.T) {
	e := newTestEngine(Config{DryRun: true})
	got := e.buildSummary()
	if strings.Contains(got, "skill-diff:") {
		t.Errorf("summary should not mention skill-diff when it did not run, got %q", got)
	}
	if strings.Contains(got, "last_run") {
		t.Errorf("dry-run summary should not mention last_run, got %q", got)
	}
}

// R3: hold/advance decision AT and AROUND the 50% threshold.
func TestShouldHoldLastRun_Threshold(t *testing.T) {
	cases := []struct {
		name        string
		ran         bool
		failed, tot int
		wantHold    bool
	}{
		{"not run", false, 0, 0, false},
		{"zero candidates", true, 0, 0, false},
		{"no failures", true, 0, 23, false},
		{"below threshold 10/23", true, 10, 23, false}, // 43% <= 50%
		{"exactly at threshold", true, 5, 10, false},   // 50% is not > 50%
		{"just above threshold 6/10", true, 6, 10, true},
		{"above threshold 13/23", true, 13, 23, true}, // 56%
		{"all failed", true, 23, 23, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := shouldHoldLastRun(c.ran, c.failed, c.tot); got != c.wantHold {
				t.Errorf("shouldHoldLastRun(%v,%d,%d)=%v want %v", c.ran, c.failed, c.tot, got, c.wantHold)
			}
		})
	}
}

// R3: a fully successful run must still advance last_run (write the file).
// UpdateRunTimestamp resolves through statedir.Dir(), which honours
// ENGRAM_STATE_DIR FIRST (pkg/statedir/statedir.go), so isolating via that env
// var is airtight and NEVER falls back to /root — no root guard needed, and this
// advance regression runs everywhere (including non-root CI).
func TestUpdateRunTimestamp_Isolated(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", tmp)

	if err := UpdateRunTimestamp(); err != nil {
		t.Fatalf("UpdateRunTimestamp: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tmp, lastRunFile)); err != nil {
		t.Fatalf("last_run not written to isolated dir: %v", err)
	}
}

// R4: draft header must carry Coverage: N/M and the explicit uncovered list.
func TestBuildDraftHeader_Partial(t *testing.T) {
	h := buildDraftHeader("2026-09-21", "LIVE", 10, 23, []string{"alpha", "beta"})
	if !strings.Contains(h, "Coverage: 10/23") {
		t.Errorf("header missing Coverage line: %q", h)
	}
	if !strings.Contains(h, "alpha") || !strings.Contains(h, "beta") {
		t.Errorf("header missing uncovered skill names: %q", h)
	}
	if !strings.Contains(h, "Mode: LIVE") {
		t.Errorf("header missing Mode line: %q", h)
	}
	if !strings.Contains(h, "Uncovered skills") {
		t.Errorf("header missing uncovered label: %q", h)
	}
}

// R4: a fully-covered header states none uncovered.
func TestBuildDraftHeader_Complete(t *testing.T) {
	h := buildDraftHeader("2026-09-21", "DRY-RUN", 23, 23, nil)
	if !strings.Contains(h, "Coverage: 23/23") {
		t.Errorf("header missing Coverage line: %q", h)
	}
	if !strings.Contains(h, "Uncovered skills (proposal failed): none") {
		t.Errorf("complete header should say none uncovered: %q", h)
	}
}
