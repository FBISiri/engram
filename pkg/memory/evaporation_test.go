package memory

import (
	"math"
	"testing"
	"time"
)

// base is a fixed reference time for deterministic evaporation tests.
var base = time.Unix(1_700_000_000, 0)

// daysAgo returns a CreatedAt unix-seconds float `days` before base.
func daysAgo(days float64) float64 {
	return float64(base.Unix()) - days*86400.0
}

func enabledEvapConfig() EvaporationConfig {
	c := DefaultEvaporationConfig()
	c.Enabled = true
	return c
}

func TestEffectiveImportance_NoDecay(t *testing.T) {
	// Disabled config → effective == base.
	m := New("x", WithType(TypeEvent), WithImportance(6))
	m.CreatedAt = daysAgo(100)
	if got := EffectiveImportance(m, DefaultEvaporationConfig(), base); got != 6 {
		t.Errorf("disabled: got %f, want 6", got)
	}

	// Identity (half-life 0) → no decay even when enabled.
	id := New("x", WithType(TypeIdentity), WithImportance(6))
	id.CreatedAt = daysAgo(1000)
	if got := EffectiveImportance(id, enabledEvapConfig(), base); got != 6 {
		t.Errorf("identity: got %f, want 6", got)
	}
}

func TestEffectiveImportance_HalfLife(t *testing.T) {
	// event, imp=6, age=30d (== half-life), access=0 → ≈ 3.0.
	m := New("x", WithType(TypeEvent), WithImportance(6))
	m.CreatedAt = daysAgo(30)
	got := EffectiveImportance(m, enabledEvapConfig(), base)
	if math.Abs(got-3.0) > 0.05 {
		t.Errorf("half-life: got %f, want ≈3.0", got)
	}
}

func TestEffectiveImportance_AccessBoost(t *testing.T) {
	// access_count=20, α=0.15 → boost ≈ 1.456. Verify via age=0 (decay=1).
	m := New("x", WithType(TypeEvent), WithImportance(1))
	m.CreatedAt = daysAgo(0)
	m.AccessCount = 20
	cfg := enabledEvapConfig()
	cfg.AccessBoostMax = 0 // this test verifies the uncapped α·ln boost formula
	got := EffectiveImportance(m, cfg, base)
	wantBoost := 1.0 + 0.15*math.Log(1.0+20.0)
	if math.Abs(got-wantBoost) > 1e-6 {
		t.Errorf("access boost: got %f, want %f", got, wantBoost)
	}
	if math.Abs(wantBoost-1.456) > 0.01 {
		t.Errorf("boost sanity: got %f, want ≈1.456", wantBoost)
	}
}

func TestEffectiveImportance_BelowThreshold(t *testing.T) {
	// event, imp=3, age=60d, access=0 → effective < 1.0.
	m := New("x", WithType(TypeEvent), WithImportance(3))
	m.CreatedAt = daysAgo(60)
	got := EffectiveImportance(m, enabledEvapConfig(), base)
	if got >= 1.0 {
		t.Errorf("below threshold: got %f, want < 1.0", got)
	}
}

func TestEffectiveImportance_Deterministic(t *testing.T) {
	// Same (m, cfg, now) → identical result across calls.
	m := New("x", WithType(TypeEvent), WithImportance(6))
	m.CreatedAt = daysAgo(15)
	cfg := enabledEvapConfig()
	a := EffectiveImportance(m, cfg, base)
	b := EffectiveImportance(m, cfg, base)
	if a != b {
		t.Errorf("determinism: %f != %f", a, b)
	}
}

func TestEffectiveImportance_DecaysOverTime(t *testing.T) {
	// Same memory, later `now` → strictly lower effective importance.
	m := New("x", WithType(TypeEvent), WithImportance(6))
	m.CreatedAt = float64(base.Unix())
	cfg := enabledEvapConfig()
	early := EffectiveImportance(m, cfg, base.Add(10*24*time.Hour))
	late := EffectiveImportance(m, cfg, base.Add(40*24*time.Hour))
	if !(late < early) {
		t.Errorf("decay over time: late=%f not < early=%f", late, early)
	}
}

// TestEffectiveImportance_DecayBasis proves the reinforcement clock (spec §4.2):
// under last_access, decay runs from max(created, last_accessed), so a recently
// read memory decays less than the same memory under the legacy created basis.
// TestEffectiveImportance_InvalidBasisFallsBack: an invalid DecayBasis behaves
// like last_access (spec §4.2 / §6.7). The one-line stderr warning is emitted
// once by referenceTime; here we assert behavioural equivalence.
func TestEffectiveImportance_InvalidBasisFallsBack(t *testing.T) {
	m := New("x", WithType(TypeEvent), WithImportance(6))
	m.CreatedAt = daysAgo(60)
	m.LastAccessedAt = daysAgo(5)

	bogus := enabledEvapConfig()
	bogus.DecayBasis = "nonsense"
	lastAccess := enabledEvapConfig()
	lastAccess.DecayBasis = "last_access"

	if a, b := EffectiveImportance(m, bogus, base), EffectiveImportance(m, lastAccess, base); a != b {
		t.Errorf("invalid basis (%f) should equal last_access (%f)", a, b)
	}
}

func TestEffectiveImportance_DecayBasis(t *testing.T) {
	m := New("x", WithType(TypeEvent), WithImportance(6))
	m.CreatedAt = daysAgo(60)
	m.LastAccessedAt = daysAgo(5) // read recently

	created := enabledEvapConfig()
	created.DecayBasis = "created"
	lastAccess := enabledEvapConfig()
	lastAccess.DecayBasis = "last_access"

	gotCreated := EffectiveImportance(m, created, base)
	gotLast := EffectiveImportance(m, lastAccess, base)
	if !(gotLast > gotCreated) {
		t.Errorf("last_access (%f) should exceed created (%f) for a recently-read memory", gotLast, gotCreated)
	}

	// Never accessed (last_accessed 0): both bases decay from created_at.
	never := New("y", WithType(TypeEvent), WithImportance(6))
	never.CreatedAt = daysAgo(60)
	if a, b := EffectiveImportance(never, created, base), EffectiveImportance(never, lastAccess, base); a != b {
		t.Errorf("never-accessed: created=%f last_access=%f should be equal", a, b)
	}
}

// TestEvaporationExempt_EachRuleAlone asserts every protection rule P1..P8 is,
// on its own, sufficient to exempt a memory (spec §4.1), and that a plain aged
// low-value memory is NOT exempt.
func TestEvaporationExempt_EachRuleAlone(t *testing.T) {
	cfg := DefaultEvaporationConfig()

	mk := func(mut func(*Memory)) *Memory {
		m := New("x", WithType(TypeEvent), WithImportance(3))
		m.CreatedAt = daysAgo(100)
		mut(m)
		return m
	}

	cases := []struct {
		name string
		m    *Memory
		want string
	}{
		{"P1 type", mk(func(m *Memory) { m.Type = TypeIdentity }), "P1"},
		{"P2 importance", mk(func(m *Memory) { m.Importance = 8 }), "P2"},
		{"P3 access_count", mk(func(m *Memory) { m.AccessCount = 5 }), "P3"},
		{"P4 recent access", mk(func(m *Memory) { m.LastAccessedAt = daysAgo(10) }), "P4"},
		{"P5 tag", mk(func(m *Memory) { m.Tags = []string{"permanent"} }), "P5"},
		{"P6 superseded", mk(func(m *Memory) { m.SupersededBy = "other-id" }), "P6"},
		{"P7 corroborated", mk(func(m *Memory) { m.Metadata = map[string]any{"provenance_history": []any{"src"}} }), "P7"},
		{"P8 young", mk(func(m *Memory) { m.CreatedAt = daysAgo(5) }), "P8"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, rule := EvaporationExempt(tc.m, cfg, base)
			if !got || rule != tc.want {
				t.Errorf("EvaporationExempt = (%v, %q), want (true, %q)", got, rule, tc.want)
			}
		})
	}

	// Plain aged low-value event: no rule fires.
	plain := mk(func(m *Memory) {})
	if got, rule := EvaporationExempt(plain, cfg, base); got {
		t.Errorf("plain aged event should not be exempt, got rule %q", rule)
	}
}

// TestEvaporationExempt_P1Structural is the critical regression: an identity or
// directive memory is exempt via P1 even under an adversarial config that sets
// its half-life > 0. P1 must be structural, not configurational (spec §4.1).
func TestEvaporationExempt_P1Structural(t *testing.T) {
	cfg := DefaultEvaporationConfig()
	cfg.Enabled = true
	cfg.HalfLifeDays[TypeIdentity] = 1  // adversarial: 1-day half-life
	cfg.HalfLifeDays[TypeDirective] = 1 // adversarial: 1-day half-life

	for _, ty := range []MemoryType{TypeIdentity, TypeDirective} {
		m := New("x", WithType(ty), WithImportance(3))
		m.CreatedAt = daysAgo(3650) // 10 years old
		if exempt, rule := EvaporationExempt(m, cfg, base); !exempt || rule != "P1" {
			t.Errorf("%s: exempt=(%v,%q), want (true, P1) despite half-life=1", ty, exempt, rule)
		}
	}
}

// TestEvaporationExempt_CorroboratedToggle: P7 is gated by ProtectCorroborated.
func TestEvaporationExempt_CorroboratedToggle(t *testing.T) {
	m := New("x", WithType(TypeEvent), WithImportance(3))
	m.CreatedAt = daysAgo(100)
	m.Metadata = map[string]any{"provenance_history": []any{"src"}}

	on := DefaultEvaporationConfig()
	if exempt, rule := EvaporationExempt(m, on, base); !exempt || rule != "P7" {
		t.Errorf("ProtectCorroborated on: got (%v,%q), want (true, P7)", exempt, rule)
	}
	off := DefaultEvaporationConfig()
	off.ProtectCorroborated = false
	if exempt, _ := EvaporationExempt(m, off, base); exempt {
		t.Error("ProtectCorroborated off: memory should not be exempt via P7")
	}
}

// TestAccessBoostBounded verifies the ln(1+access_count) boost term is bounded
// above by AccessBoostMax for extreme access counts.
func TestAccessBoostBounded(t *testing.T) {
	cfg := DefaultEvaporationConfig()
	cfg.Enabled = true
	now := time.Now()
	m := &Memory{
		Type:           TypeInsight, // HalfLifeDays[insight]=180
		Importance:     10,
		AccessCount:    1_000_000_000, // 1e9
		CreatedAt:      float64(now.Unix()),
		LastAccessedAt: float64(now.Unix()),
	}
	eff := EffectiveImportance(m, cfg, now)

	bounded := 10*cfg.AccessBoostMax + 1e-9
	if eff > bounded {
		t.Errorf("effImp=%v exceeds bounded=%v", eff, bounded)
	}
	uncapped := 10 * (1 + 0.15*math.Log(1+1e9))
	if !(eff < uncapped) {
		t.Errorf("effImp=%v not strictly less than uncapped=%v", eff, uncapped)
	}
}

// TestP12OrderingRegression reproduces the P12 fact at EffectiveImportance
// level: a hot insight can no longer runaway past a directive.
func TestP12OrderingRegression(t *testing.T) {
	now := time.Now()
	directive := &Memory{Type: TypeDirective, Importance: 8} // returns early → eff 8.0
	insight := &Memory{
		Type:           TypeInsight,
		Importance:     10,
		AccessCount:    625,
		CreatedAt:      float64(now.Unix()),
		LastAccessedAt: float64(now.Unix()),
	}
	cfg := DefaultEvaporationConfig()
	cfg.Enabled = true

	effDir := EffectiveImportance(directive, cfg, now)
	effIns := EffectiveImportance(insight, cfg, now)

	if math.Abs(effDir-8.0) > 1e-9 {
		t.Errorf("effDir=%v, want 8.0", effDir)
	}
	if math.Abs(effIns-13.0) > 1e-6 {
		t.Errorf("effIns=%v, want ≈13.0", effIns)
	}
	if !(effIns < 19.0) {
		t.Errorf("effIns=%v not < 19.0 (pre-fix runaway ≈19.66)", effIns)
	}
	// Pre-fix margin (effIns-effDir) was ≈11.66; the cap must shrink it well below that.
	if !(effIns-effDir < 6.0) {
		t.Errorf("margin effIns-effDir=%v not << pre-fix ≈11.66", effIns-effDir)
	}
}
