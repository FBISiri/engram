// ratelimit.go — A-MAC per-collection, per-type sliding-window write limiter.
//
// In-memory only (spec v1 §3.3): counters are keyed by collection+type and hold
// the write timestamps within the trailing window. Resets on restart are
// acceptable. Enforcement is non-destructive: a rejected write returns a
// RateLimitError with retry_after_seconds; the caller may retry.
package server

import (
	"fmt"
	"sync"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// RateLimiter enforces per-(collection,type) sliding-window write limits.
type RateLimiter struct {
	mu     sync.Mutex
	window time.Duration
	limits map[memory.MemoryType]int
	events map[string][]time.Time
	now    func() time.Time // injectable for tests
}

// NewRateLimiter builds a limiter with a 1-hour window from the given per-type
// limits. A type with no entry (or a non-positive limit) is unlimited.
func NewRateLimiter(limits map[memory.MemoryType]int) *RateLimiter {
	return &RateLimiter{
		window: time.Hour,
		limits: limits,
		events: map[string][]time.Time{},
		now:    time.Now,
	}
}

// RateLimitError is returned when a write exceeds its type's window limit.
type RateLimitError struct {
	Type              memory.MemoryType
	Collection        string
	RetryAfterSeconds int
}

func (e *RateLimitError) Error() string {
	return fmt.Sprintf("rate_limited: %s writes to %q exceeded limit; retry after %ds",
		e.Type, e.Collection, e.RetryAfterSeconds)
}

// Allow records a write attempt for (collection, memType). If it is within the
// limit, the timestamp is recorded and nil is returned. Otherwise nothing is
// recorded and a *RateLimitError with retry_after_seconds (until the oldest
// timestamp in the window ages out) is returned.
func (rl *RateLimiter) Allow(collection string, memType memory.MemoryType) error {
	limit, ok := rl.limits[memType]
	if !ok || limit <= 0 {
		return nil // unlimited for this type
	}

	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := rl.now()
	cutoff := now.Add(-rl.window)
	key := collection + "\x00" + string(memType)

	// Drop timestamps that have aged out of the window.
	kept := rl.events[key][:0]
	for _, ts := range rl.events[key] {
		if ts.After(cutoff) {
			kept = append(kept, ts)
		}
	}

	if len(kept) >= limit {
		oldest := kept[0]
		retry := int(oldest.Add(rl.window).Sub(now).Seconds()) + 1
		if retry < 1 {
			retry = 1
		}
		rl.events[key] = kept
		return &RateLimitError{Type: memType, Collection: collection, RetryAfterSeconds: retry}
	}

	kept = append(kept, now)
	rl.events[key] = kept
	return nil
}

// Utilization returns the current usage for (collection, memType): the number of
// writes within the trailing window, the limit, and their fraction. A type with
// no configured (or non-positive) limit is unlimited and returns (0, 0, 0).
func (rl *RateLimiter) Utilization(collection string, memType memory.MemoryType) (count int, limit int, fraction float64) {
	limit, ok := rl.limits[memType]
	if !ok || limit <= 0 {
		return 0, 0, 0
	}

	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := rl.now()
	cutoff := now.Add(-rl.window)
	key := collection + "\x00" + string(memType)

	for _, ts := range rl.events[key] {
		if ts.After(cutoff) {
			count++
		}
	}
	return count, limit, float64(count) / float64(limit)
}
