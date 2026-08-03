package main

import (
	"os"
	"testing"

	"github.com/FBISiri/engram/pkg/replay"
)

func TestValidateArgs(t *testing.T) {
	cases := []struct {
		name    string
		a       args
		want    string
		wantErr bool
	}{
		{"single trace", args{trace: "a.jsonl"}, "single", false},
		{"multi day", args{baseline: "a.jsonl", candidate: "b.jsonl"}, "multi", false},
		{"trace + baseline conflict", args{trace: "a.jsonl", baseline: "b.jsonl"}, "", true},
		{"trace + candidate conflict", args{trace: "a.jsonl", candidate: "b.jsonl"}, "", true},
		{"baseline without candidate", args{baseline: "a.jsonl"}, "", true},
		{"candidate without baseline", args{candidate: "b.jsonl"}, "", true},
		{"nothing", args{}, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mode, err := validateArgs(tc.a)
			if tc.wantErr && err == nil {
				t.Fatalf("expected error, got mode=%q", mode)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if mode != tc.want {
				t.Errorf("mode = %q, want %q", mode, tc.want)
			}
		})
	}
}

func TestValidateArgs_ConfigRequiresBaseline(t *testing.T) {
	// --config without --baseline-config must be rejected (fail closed).
	if _, err := validateArgs(args{trace: "a.jsonl", configJSON: `{"retrieve_config":{"recency_weight":0.25}}`}); err == nil {
		t.Fatal("expected error: --config without --baseline-config")
	}
	// with a baseline it is accepted.
	mode, err := validateArgs(args{trace: "a.jsonl", configJSON: `{"retrieve_config":{"recency_weight":0.25}}`, baselineCfg: `{"retrieve_config":{"recency_weight":0.5}}`})
	if err != nil || mode != "single" {
		t.Fatalf("mode=%q err=%v, want single/nil", mode, err)
	}
}

func TestValidateArgs_SnapshotNeedsAck(t *testing.T) {
	if _, err := validateArgs(args{trace: "a.jsonl", snapshot: "20260802-120000"}); err == nil {
		t.Fatal("expected refusal: --snapshot without --ack-snapshot-global-rewrite")
	}
	mode, err := validateArgs(args{trace: "a.jsonl", snapshot: "20260802-120000", ackSnapshot: true})
	if err != nil || mode != "single" {
		t.Fatalf("mode=%q err=%v, want single/nil with ack", mode, err)
	}
}

func TestValidateArgs_UnsupportedConfigKeyRejected(t *testing.T) {
	a := args{trace: "a.jsonl", configJSON: `{"retrieve_config":{"score_threshold":0.5}}`, baselineCfg: `{"retrieve_config":{"recency_weight":0.5}}`}
	if _, err := validateArgs(a); err == nil {
		t.Fatal("expected rejection of unsupported config key score_threshold")
	}
}

func TestRunSingle_EmptyTraceErrors(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/updates-only.jsonl"
	// Only update ops → LoadTrace yields 0 replayable cases.
	content := `{"timestamp":"2026-08-02T00:00:00Z","operation":"update","content":"x"}` + "\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	// engine is never reached because the empty-case guard returns first.
	_, err := runSingle(args{topK: 10}, path, nil, nil, nil, replay.MemoryConfig{}, replay.DefaultThresholds())
	if err == nil {
		t.Fatal("expected error for a trace with 0 replayable cases")
	}
}

func TestValidateArgs_ConfigRejectedInMultiDay(t *testing.T) {
	a := args{baseline: "a.jsonl", candidate: "b.jsonl", configJSON: `{"retrieve_config":{"recency_weight":0.25}}`, baselineCfg: `{"retrieve_config":{"recency_weight":0.5}}`}
	if _, err := validateArgs(a); err == nil {
		t.Fatal("expected rejection of --config with --baseline/--candidate (would be silently dropped)")
	}
}
