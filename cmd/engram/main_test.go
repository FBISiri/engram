package main

import (
	"errors"
	"testing"
)

func TestResolveReflectionMode(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		envMode string
		want    string
	}{
		{"flag only", []string{"--dry-run", "--mode", "v2"}, "", "v2"},
		{"env only", []string{"--dry-run"}, "v2", "v2"},
		{"flag beats env", []string{"--mode", "v1"}, "v2", "v1"},
		{"neither defaults v1", []string{"--dry-run"}, "", "v1"},
		{"empty flag value falls through to env", []string{"--mode", ""}, "v2", "v2"},
		{"empty flag value falls through to default", []string{"--mode", ""}, "", "v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveReflectionMode(tt.args, tt.envMode); got != tt.want {
				t.Fatalf("resolveReflectionMode(%v, %q) = %q, want %q", tt.args, tt.envMode, got, tt.want)
			}
		})
	}
}

func TestParseReflectionFlags(t *testing.T) {
	// help
	if _, wantHelp, err := parseReflectionFlags([]string{"--help"}, ""); err != nil || !wantHelp {
		t.Fatalf("--help: wantHelp=%v err=%v", wantHelp, err)
	}
	if _, wantHelp, err := parseReflectionFlags([]string{"-h"}, ""); err != nil || !wantHelp {
		t.Fatalf("-h: wantHelp=%v err=%v", wantHelp, err)
	}

	// unknown flag
	_, _, err := parseReflectionFlags([]string{"--bogus-flag"}, "")
	if !errors.Is(err, errUnknownFlag) {
		t.Fatalf("--bogus-flag: err=%v, want errUnknownFlag", err)
	}

	// --mode with no value
	if _, _, err := parseReflectionFlags([]string{"--dry-run", "--mode"}, ""); err == nil {
		t.Fatalf("--mode with no value: expected error")
	}

	// normal flags parse, mode resolved (never empty)
	opts, wantHelp, err := parseReflectionFlags([]string{"--dry-run", "--force", "--debug-evidence", "--mode", "v2"}, "")
	if err != nil || wantHelp {
		t.Fatalf("normal flags: wantHelp=%v err=%v", wantHelp, err)
	}
	if !opts.DryRun || !opts.Force || !opts.DebugEvidence {
		t.Fatalf("normal flags: opts=%+v", opts)
	}
	if opts.Mode != "v2" {
		t.Fatalf("normal flags: Mode=%q, want v2", opts.Mode)
	}

	// no --mode, env supplies value
	opts, _, err = parseReflectionFlags([]string{"--dry-run"}, "v2")
	if err != nil {
		t.Fatalf("env mode: err=%v", err)
	}
	if opts.Mode != "v2" {
		t.Fatalf("env mode: Mode=%q, want v2", opts.Mode)
	}

	// no --mode, no env -> default v1
	opts, _, err = parseReflectionFlags([]string{"--dry-run"}, "")
	if err != nil {
		t.Fatalf("default mode: err=%v", err)
	}
	if opts.Mode != "v1" {
		t.Fatalf("default mode: Mode=%q, want v1", opts.Mode)
	}
}

func TestParseDreamFlags(t *testing.T) {
	if _, wantHelp, err := parseDreamFlags([]string{"--help"}); err != nil || !wantHelp {
		t.Fatalf("--help: wantHelp=%v err=%v", wantHelp, err)
	}
	if _, _, err := parseDreamFlags([]string{"--bogus"}); !errors.Is(err, errUnknownFlag) {
		t.Fatalf("--bogus: err=%v, want errUnknownFlag", err)
	}
	if _, _, err := parseDreamFlags([]string{"--phase"}); err == nil {
		t.Fatalf("--phase with no value: expected error")
	}
	opts, _, err := parseDreamFlags([]string{"--dry-run", "--phase", "orient"})
	if err != nil || !opts.DryRun || opts.Phase != "orient" {
		t.Fatalf("normal: opts=%+v err=%v", opts, err)
	}
}

func TestParseHelpOnly(t *testing.T) {
	if wantHelp, err := parseHelpOnly([]string{"--help"}); err != nil || !wantHelp {
		t.Fatalf("--help: wantHelp=%v err=%v", wantHelp, err)
	}
	if wantHelp, err := parseHelpOnly(nil); err != nil || wantHelp {
		t.Fatalf("no args: wantHelp=%v err=%v", wantHelp, err)
	}
	if _, err := parseHelpOnly([]string{"--bogus"}); !errors.Is(err, errUnknownFlag) {
		t.Fatalf("--bogus: err=%v, want errUnknownFlag", err)
	}
}
