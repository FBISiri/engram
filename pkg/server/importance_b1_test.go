package server

import (
	"testing"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
)

// TestB1_DirectiveImportanceSix is an EMPIRICAL reproduction (asserts CURRENT
// behavior; changes nothing) documenting why real source_type=user_input
// directive memories store importance 6.
//
// Root cause — under the DEFAULT config AMACEnabled=false (config.go:160):
//   - handleAdd / REST create take the amac-OFF else-branch (server.go:626-634,
//     crud.go:83-94): importance is the caller-supplied value, defaulting to 5.0
//     when omitted, clamped ONLY to the global range [1,10]. There is NO
//     per-type floor on this path, and the directive+user_input +1 governance
//     boost is SKIPPED because it is gated behind amacEnabled (server.go:617,
//     crud.go:74). So a genuine user_input directive is stored at exactly the
//     importance the writer supplied -> importance 6 arises when the caller
//     passes 6.
//   - The per-type MIN bound ENGRAM_IMPORTANCE_MIN_DIRECTIVE=6 (config.go:267,
//     DefaultImportanceBounds) is enforced ONLY on the amac-ON path
//     (amacImportance, server.go:1680-1699). On that path the +1 user_input
//     boost then lifts the value to >=7, so amac-ON can NEVER store 6 for a
//     user_input directive.
//
// Verdict: CORRECT-AS-DESIGNED. Under the default (amac-off) config a genuine
// user_input directive stores exactly the caller's importance (6 here); the
// premise "user_input directive at importance 6" is satisfiable, not a defect.
// (See DONE report for the full branch/config trace.)
func TestB1_DirectiveImportanceSix(t *testing.T) {
	// (a) REPRODUCE: DEFAULT (amac-off) config, genuine source_type=user_input
	//     directive with importance 6 -> stored 6, with NO +1 boost applied.
	srv, _ := newTestServer() // AMACEnabled=false (default)
	if srv.amacEnabled() {
		t.Fatal("precondition: newTestServer must be amac-OFF (default config)")
	}
	res, err := callTool(srv, "memory_add", map[string]any{
		"content":     "always run migrations inside a transaction",
		"type":        "directive",
		"importance":  float64(6),
		"source_type": "user_input",
	})
	if err != nil {
		t.Fatalf("add user_input directive: %v", err)
	}
	mem := parseAddMemory(t, res)
	if mem.Importance != 6 {
		t.Fatalf("amac-off user_input directive importance = %v, want 6 (caller value, no boost)", mem.Importance)
	}

	// (b) Evidence there is NO per-type floor on the amac-off path: a sub-6
	//     importance is stored verbatim (3, not clamped up to the directive
	//     MIN bound of 6). Proves the 6 above is the caller value, not a floor.
	srvB, _ := newTestServer()
	resB, err := callTool(srvB, "memory_add", map[string]any{
		"content":     "prefer feature flags over long-lived branches",
		"type":        "directive",
		"importance":  float64(3),
		"source_type": "user_input",
	})
	if err != nil {
		t.Fatalf("add sub-floor directive: %v", err)
	}
	if got := parseAddMemory(t, resB).Importance; got != 3 {
		t.Fatalf("amac-off directive importance = %v, want 3 (no per-type floor when amac off)", got)
	}

	// (c) Evidence the +1 boost is amac-gated: on the amac-ON path the SAME
	//     user_input directive with importance 3 is floored to 6 then boosted to
	//     7 (never 6), confirming amac-ON cannot produce 6 for user_input.
	srvC, _ := newAMACServer()
	resC, err := callTool(srvC, "memory_add", map[string]any{
		"content":     "never delete production data without a backup",
		"type":        "directive",
		"importance":  float64(3),
		"source_type": "user_input",
	})
	if err != nil {
		t.Fatalf("add amac-on user_input directive: %v", err)
	}
	if got := parseAddMemory(t, resC).Importance; got != 7 {
		t.Fatalf("amac-on user_input directive importance = %v, want 7 (floor 6 +1 boost)", got)
	}

	// (d) Confirm the floor value's source is the config bound (used on the
	//     amac-ON path), not a magic literal: directive MIN bound = 6.
	if bounds := config.DefaultImportanceBounds()[memory.TypeDirective]; bounds[0] != 6 {
		t.Fatalf("directive min bound = %v, want 6 (ENGRAM_IMPORTANCE_MIN_DIRECTIVE)", bounds[0])
	}
}
