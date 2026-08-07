// Command replay is the Engram Replay Harness MVP CLI. It loads recorded
// retrieve trajectories, replays them against a running Engram server (with an
// optional config override for A/B testing), compares recorded vs live
// results, and emits JSON + Markdown reports.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/FBISiri/engram/pkg/replay"
)

type args struct {
	trace       string
	baseline    string
	candidate   string
	configJSON  string
	baselineCfg string
	snapshot    string
	ackSnapshot bool
	collection  string
	outputDir   string
	ci          bool
	meanRecall  float64
	regressPct  float64
	topK        int
}

func parseFlags() args {
	var a args
	flag.StringVar(&a.trace, "trace", "", "trajectory JSONL file to replay")
	flag.StringVar(&a.baseline, "baseline", "", "baseline trajectory JSONL (multi-day compare; requires --candidate)")
	flag.StringVar(&a.candidate, "candidate", "", "candidate trajectory JSONL (multi-day compare; requires --baseline)")
	flag.StringVar(&a.configJSON, "config", "", "config override as JSON (e.g. '{\"retrieve_config\":{\"recency_weight\":0.25}}'); requires --baseline-config")
	flag.StringVar(&a.baselineCfg, "baseline-config", "", "REQUIRED with --config: operator's current config (JSON). memory_apply_config is set-only and cannot clear fields, so a concrete non-zero baseline is mandatory to restore state after the run")
	flag.StringVar(&a.snapshot, "snapshot", "", "snapshot id to restore before replay. DANGER: server memory_reset is GLOBAL — it wipes and re-embeds the ENTIRE store (not just the eval collection), so --snapshot REWRITES the whole production store and has a data-loss window if killed mid-run. Requires --ack-snapshot-global-rewrite.")
	flag.BoolVar(&a.ackSnapshot, "ack-snapshot-global-rewrite", false, "explicit acknowledgement that --snapshot rewrites the ENTIRE global store (see --snapshot). Required to use --snapshot.")
	flag.StringVar(&a.collection, "collection", replay.DefaultCollection, "eval collection (must have engram_eval_ prefix)")
	flag.StringVar(&a.outputDir, "output-dir", "eval/reports/", "directory for report output")
	flag.BoolVar(&a.ci, "ci", false, "exit non-zero on FAIL verdict")
	flag.Float64Var(&a.meanRecall, "mean-recall", 0.80, "mean recall required to PASS")
	flag.Float64Var(&a.regressPct, "regression-pct", 0.05, "max regression fraction to PASS")
	flag.IntVar(&a.topK, "top-k", 10, "floor on results requested per query (per-case uses max(recorded,top-k))")
	flag.Usage = usage
	flag.Parse()
	return a
}

func usage() {
	fmt.Fprintf(os.Stderr, `engram replay — replay recorded retrieve trajectories against a live Engram server.

USAGE:
  replay --trace <file> [--config <json> --baseline-config <json>] [--ci]
  replay --baseline <file> --candidate <file> [--ci]   # multi-day compare

KNOWN SERVER-SIDE LIMITATIONS (cannot be fixed client-side):
  - memory_apply_config is SET-ONLY: it only honors recency_weight, top_k and
    dedupe_threshold, and cannot CLEAR a field. Other keys are ignored; --config
    is validated against the supported set and a non-zero --baseline-config is
    mandatory so the override can be reverted.
  - memory_reset (--snapshot) is GLOBAL: it rewrites the ENTIRE store, not just
    the eval collection; requires --ack-snapshot-global-rewrite.
  - /memories/search fires an async server goroutine that bumps access_count /
    last_accessed_at on every matched memory, so replay perturbs the very
    recency signal a recency-weight A/B is measuring.
  - Caller isolation is NOT reproduced: recorded results from an isolated
    (collection-scoped) caller are replayed as a non-isolated global fan-out.

FLAGS:
`)
	flag.PrintDefaults()
}

// validateArgs enforces mutual exclusion between --trace and the
// --baseline/--candidate pair. Returns "single" or "multi" mode, or an error.
func validateArgs(a args) (string, error) {
	multi := a.baseline != "" || a.candidate != ""
	if a.trace != "" && multi {
		return "", fmt.Errorf("--trace is mutually exclusive with --baseline/--candidate")
	}
	// Fail closed: a --config override cannot be safely reverted without an
	// explicit baseline (memory_apply_config is set-only, cannot clear fields).
	if a.configJSON != "" && a.baselineCfg == "" {
		return "", fmt.Errorf("refusing: cannot safely restore config; supply --baseline-config with current values")
	}
	// Fail closed: --snapshot triggers a GLOBAL store rewrite (memory_reset is
	// not collection-scoped), so require explicit operator acknowledgement.
	if a.snapshot != "" && !a.ackSnapshot {
		return "", fmt.Errorf("refusing: --snapshot rewrites the ENTIRE global store (memory_reset is global, not eval-scoped); pass --ack-snapshot-global-rewrite to proceed")
	}
	// Reject config keys the server does not honor (false-A/B guard).
	if err := replay.ValidateSupportedConfig(a.configJSON); err != nil {
		return "", err
	}
	if multi {
		if a.baseline == "" || a.candidate == "" {
			return "", fmt.Errorf("multi-day compare requires BOTH --baseline and --candidate")
		}
		// Multi-day replays both days under the LIVE config; a --config override
		// would be silently dropped, yielding a false A/B. Reject it.
		if a.configJSON != "" {
			return "", fmt.Errorf("--config is not supported with --baseline/--candidate (multi-day runs under live config only)")
		}
		return "multi", nil
	}
	if a.trace == "" {
		return "", fmt.Errorf("one of --trace or (--baseline AND --candidate) is required")
	}
	return "single", nil
}

func main() {
	a := parseFlags()
	mode, err := validateArgs(a)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		flag.Usage()
		os.Exit(2)
	}

	th := replay.DefaultThresholds()
	th.MeanRecallPass = a.meanRecall
	th.RegressionPct = a.regressPct
	th = th.Normalize()

	cfg, err := parseConfig(a.configJSON)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: invalid --config: %v\n", err)
		os.Exit(2)
	}
	baselineCfg, err := parseConfig(a.baselineCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: invalid --baseline-config: %v\n", err)
		os.Exit(2)
	}

	eng := replay.NewEngineFromEnv(a.collection)
	var override, baseCfgPtr *replay.MemoryConfig
	if a.configJSON != "" {
		override = &cfg
	}
	if a.baselineCfg != "" {
		if (baselineCfg == replay.MemoryConfig{}) {
			fmt.Fprintln(os.Stderr, "error: refusing: --baseline-config parsed to all-zero values; supply the current non-zero config so it can be restored")
			os.Exit(2)
		}
		baseCfgPtr = &baselineCfg
	}

	if mode == "multi" {
		runMultiDay(a, eng, th)
		return
	}

	report, err := runSingle(a, a.trace, eng, override, baseCfgPtr, cfg, th)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	if err := writeReport(a.outputDir, "replay", report); err != nil {
		fmt.Fprintf(os.Stderr, "error: writing reports: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("verdict=%s mean_recall=%.4f regressions=%d/%d\n",
		report.Aggregate.Verdict, report.Aggregate.MeanRecall,
		report.Aggregate.RegressionCount, report.Aggregate.TotalCases)
	if a.ci && report.Aggregate.Verdict == replay.VerdictFail {
		os.Exit(1)
	}
}

func runSingle(a args, trace string, eng *replay.Engine, override, baseCfg *replay.MemoryConfig, cfg replay.MemoryConfig, th replay.Thresholds) (replay.Report, error) {
	cases, err := replay.LoadTrace(trace)
	if err != nil {
		return replay.Report{}, err
	}
	if len(cases) == 0 {
		// A run with nothing to replay must NOT silently PASS (would mask a
		// wrong/empty --trace path, especially under --ci).
		return replay.Report{}, fmt.Errorf("no replayable retrieve cases loaded from %s (wrong path or non-retrieve trace?)", trace)
	}
	fmt.Fprintf(os.Stderr, "loaded %d retrieve cases from %s\n", len(cases), trace)
	results, err := eng.Run(context.Background(), cases, replay.RunOptions{
		SnapshotID:     a.snapshot,
		ConfigOverride: override,
		BaselineConfig: baseCfg,
		TopK:           a.topK,
	})
	if err != nil {
		return replay.Report{}, fmt.Errorf("replay failed: %w", err)
	}
	// Load raw records (with task_id/task_result) for the Memory Worth join. A
	// read failure here is non-fatal: MW is simply omitted from the report.
	records, _ := replay.LoadRecords(trace)
	return replay.BuildReport(trace, a.collection, cfg, results, th, records...), nil
}

func runMultiDay(a args, eng *replay.Engine, th replay.Thresholds) {
	base, err := runSingle(a, a.baseline, eng, nil, nil, replay.MemoryConfig{}, th)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: baseline: %v\n", err)
		os.Exit(1)
	}
	cand, err := runSingle(a, a.candidate, eng, nil, nil, replay.MemoryConfig{}, th)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: candidate: %v\n", err)
		os.Exit(1)
	}
	report := replay.BuildMultiDayReport(base, cand)
	if err := writeMultiDay(a.outputDir, report); err != nil {
		fmt.Fprintf(os.Stderr, "error: writing reports: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("baseline_recall=%.4f candidate_recall=%.4f delta=%+.4f\n",
		base.Aggregate.MeanRecall, cand.Aggregate.MeanRecall, report.Delta.MeanRecallDelta)
	if a.ci && cand.Aggregate.Verdict == replay.VerdictFail {
		os.Exit(1)
	}
}

func parseConfig(s string) (replay.MemoryConfig, error) {
	var cfg replay.MemoryConfig
	if s == "" {
		return cfg, nil
	}
	err := json.Unmarshal([]byte(s), &cfg)
	return cfg, err
}

func writeReport(dir, prefix string, r replay.Report) error {
	jsonBytes, err := replay.RenderJSON(r)
	if err != nil {
		return err
	}
	return writeStamped(dir, prefix, jsonBytes, []byte(replay.RenderMarkdown(r)))
}

func writeMultiDay(dir string, r replay.MultiDayReport) error {
	jsonBytes, err := replay.RenderMultiDayJSON(r)
	if err != nil {
		return err
	}
	return writeStamped(dir, "replay_multiday", jsonBytes, []byte(replay.RenderMultiDayMarkdown(r)))
}

func writeStamped(dir, prefix string, jsonBytes, mdBytes []byte) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	stamp := time.Now().UTC().Format("20060102-150405")
	jsonPath := filepath.Join(dir, prefix+"_"+stamp+".json")
	mdPath := filepath.Join(dir, prefix+"_"+stamp+".md")
	if err := os.WriteFile(jsonPath, jsonBytes, 0644); err != nil {
		return err
	}
	if err := os.WriteFile(mdPath, mdBytes, 0644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s and %s\n", jsonPath, mdPath)
	return nil
}
