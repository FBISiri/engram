package server

import (
	"errors"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// TestRateLimiter verifies the sliding-window limiter: the 6th identity write
// within the hour is rejected with a RateLimitError, and the window slides so a
// write succeeds again once the oldest timestamp ages out.
func TestRateLimiter(t *testing.T) {
	rl := NewRateLimiter(map[memory.MemoryType]int{
		memory.TypeIdentity: 5,
		memory.TypeEvent:    50,
	})

	base := time.Now()
	rl.now = func() time.Time { return base }

	// First 5 identity writes succeed.
	for i := 0; i < 5; i++ {
		if err := rl.Allow("engram_user", memory.TypeIdentity); err != nil {
			t.Fatalf("write %d should succeed, got %v", i+1, err)
		}
	}

	// 6th is rate limited.
	err := rl.Allow("engram_user", memory.TypeIdentity)
	if err == nil {
		t.Fatal("6th identity write should be rate_limited")
	}
	var rle *RateLimitError
	if !errors.As(err, &rle) {
		t.Fatalf("expected *RateLimitError, got %T", err)
	}
	if rle.RetryAfterSeconds <= 0 {
		t.Errorf("retry_after_seconds should be > 0, got %d", rle.RetryAfterSeconds)
	}

	// A different type is unaffected.
	if err := rl.Allow("engram_user", memory.TypeEvent); err != nil {
		t.Errorf("event write should succeed independently, got %v", err)
	}
	// A different collection has its own window.
	if err := rl.Allow("engram_reflection", memory.TypeIdentity); err != nil {
		t.Errorf("identity write to a different collection should succeed, got %v", err)
	}

	// Advance past the window: the oldest write ages out, so a write succeeds.
	rl.now = func() time.Time { return base.Add(time.Hour + time.Second) }
	if err := rl.Allow("engram_user", memory.TypeIdentity); err != nil {
		t.Errorf("after the window slides, a write should succeed, got %v", err)
	}

	// A type with no configured limit is unlimited.
	rl2 := NewRateLimiter(map[memory.MemoryType]int{})
	for i := 0; i < 100; i++ {
		if err := rl2.Allow("c", memory.TypeIdentity); err != nil {
			t.Fatalf("unlimited type should never be limited, got %v at %d", err, i)
		}
	}
}

// TestUtilization verifies the CP3 utilization calculation, the 80% threshold
// boundary, and that unlimited types return (0,0,0).
func TestUtilization(t *testing.T) {
	rl := NewRateLimiter(map[memory.MemoryType]int{
		memory.TypeEvent: 10,
	})
	base := time.Now()
	rl.now = func() time.Time { return base }

	// No events yet → 0/10, fraction 0.
	count, limit, frac := rl.Utilization("engram_user", memory.TypeEvent)
	if count != 0 || limit != 10 || frac != 0 {
		t.Fatalf("empty: want (0,10,0), got (%d,%d,%v)", count, limit, frac)
	}

	// 8 writes → 8/10 = 0.8 (>= threshold boundary).
	for i := 0; i < 8; i++ {
		if err := rl.Allow("engram_user", memory.TypeEvent); err != nil {
			t.Fatalf("write %d should succeed, got %v", i, err)
		}
	}
	count, limit, frac = rl.Utilization("engram_user", memory.TypeEvent)
	if count != 8 || limit != 10 || frac != 0.8 {
		t.Fatalf("after 8: want (8,10,0.8), got (%d,%d,%v)", count, limit, frac)
	}

	// Unlimited type returns (0,0,0).
	c2, l2, f2 := rl.Utilization("engram_user", memory.TypeIdentity)
	if c2 != 0 || l2 != 0 || f2 != 0 {
		t.Errorf("unlimited: want (0,0,0), got (%d,%d,%v)", c2, l2, f2)
	}

	// Aged-out events are not counted.
	rl.now = func() time.Time { return base.Add(time.Hour + time.Second) }
	count, _, frac = rl.Utilization("engram_user", memory.TypeEvent)
	if count != 0 || frac != 0 {
		t.Errorf("after window slides: want count 0, got %d (frac %v)", count, frac)
	}
}
