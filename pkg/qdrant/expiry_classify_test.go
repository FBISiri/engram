package qdrant

import (
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"

	"github.com/FBISiri/engram/pkg/memory"
)

// TestClassifyExpired exercises the pure P17 expiry decision: self-heal (the
// recomputed TTL is permanent) must WIN over the importance guard, which in
// turn precedes the normal dwell stamp/delete logic. A point whose current
// type+importance maps to a finite TTL and is below the importance floor falls
// through to the ordinary dwell behaviour.
func TestClassifyExpired(t *testing.T) {
	cfg := memory.DefaultTTLConfig()
	dwell := 72 * time.Hour
	now := time.Unix(1_700_000_000, 0)
	// A stamp recorded a full dwell window ago, matching the current episode, so
	// the dwell decision alone would be "delete".
	oldStamp := float64(now.Add(-dwell - time.Hour).Unix())
	validUntil := float64(now.Add(-dwell - 2*time.Hour).Unix())

	cases := []struct {
		name       string
		typ        string
		importance float64
		tags       []string
		eligibleAt float64
		want       expiryVerdict
	}{
		{
			name:       "insight imp>=8 recomputes permanent -> heal (wins over protect)",
			typ:        "insight",
			importance: 10,
			eligibleAt: oldStamp,
			want:       verdictHeal,
		},
		{
			name:       "permanent tag -> heal even at low importance",
			typ:        "insight",
			importance: 2,
			tags:       []string{"permanent"},
			eligibleAt: oldStamp,
			want:       verdictHeal,
		},
		{
			name:       "event imp>=8 recomputes finite 90d -> protect (never delete)",
			typ:        "event",
			importance: 9,
			eligibleAt: oldStamp,
			want:       verdictProtect,
		},
		{
			name:       "insight imp<8 finite TTL, dwell cleared -> delete",
			typ:        "insight",
			importance: 5,
			eligibleAt: oldStamp,
			want:       verdictDelete,
		},
		{
			name:       "insight imp<8 finite TTL, never stamped -> stamp",
			typ:        "insight",
			importance: 5,
			eligibleAt: 0,
			want:       verdictStamp,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := &expiredPointDetail{
				typ:        c.typ,
				importance: c.importance,
				tags:       c.tags,
				eligibleAt: c.eligibleAt,
				validUntil: validUntil,
			}
			if got := classifyExpired(p, cfg, dwell, now); got != c.want {
				t.Fatalf("classifyExpired = %v, want %v", got, c.want)
			}
		})
	}
}

// TestExpiryContentPrefix verifies the audit prefix is single-line and capped
// at 120 runes.
func TestExpiryContentPrefix(t *testing.T) {
	if got := expiryContentPrefix("line one\nline two"); got != "line one line two" {
		t.Fatalf("newline not flattened: %q", got)
	}
	long := make([]rune, 200)
	for i := range long {
		long[i] = 'x'
	}
	if got := expiryContentPrefix(string(long)); len([]rune(got)) != 120 {
		t.Fatalf("prefix len = %d, want 120", len([]rune(got)))
	}
}

// TestPayloadToExpiredPointDetail exercises the payload decoding seam, in
// particular that importance tolerates BOTH integer and double encodings (a
// silently-zeroed importance would bypass the imp>=8 guard) and that missing
// fields default to zero / empty.
func TestPayloadToExpiredPointDetail(t *testing.T) {
	// importance + created_at as INTEGERs, valid_until as DOUBLE, tags as list,
	// expiry_eligible_at absent.
	payload := qdrant.NewValueMap(map[string]any{
		"type":        "insight",
		"content":     "hello",
		"tags":        []string{"a", "b"},
		"importance":  10,
		"valid_until": 1700.5,
		"created_at":  1600,
	})
	det := payloadToExpiredPointDetail("id-1", payload)
	if det.id != "id-1" || det.typ != "insight" || det.content != "hello" {
		t.Fatalf("string fields wrong: %+v", det)
	}
	if det.importance != 10 {
		t.Fatalf("importance (integer encoding) = %v, want 10", det.importance)
	}
	if det.validUntil != 1700.5 {
		t.Fatalf("validUntil (double) = %v, want 1700.5", det.validUntil)
	}
	if det.createdAt != 1600 {
		t.Fatalf("createdAt (integer) = %v, want 1600", det.createdAt)
	}
	if det.eligibleAt != 0 {
		t.Fatalf("eligibleAt (missing) = %v, want 0", det.eligibleAt)
	}
	if len(det.tags) != 2 || det.tags[0] != "a" || det.tags[1] != "b" {
		t.Fatalf("tags = %v, want [a b]", det.tags)
	}

	// An integer importance of 10 must trip the guard (classify -> protect for
	// an event, whose matrix TTL stays finite), proving it is not zeroed.
	det.typ = "event"
	if got := classifyExpired(&det, memory.DefaultTTLConfig(), 72*time.Hour, time.Unix(1_700_000_000, 0)); got != verdictProtect {
		t.Fatalf("integer importance=10 event classify=%v, want verdictProtect", got)
	}
}
