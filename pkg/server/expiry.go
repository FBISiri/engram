package server

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	engrammetrics "github.com/FBISiri/engram/pkg/metrics"
)

// DefaultExpiryInterval is the default period between expiry cleanup runs.
const DefaultExpiryInterval = 10 * time.Minute

// expiryReasonUnset is the byReason map key used for expired memories that have
// no deprecated_reason metadata (TTL expiry or unset).
const expiryReasonUnset = "ttl_or_unset"

// StartExpiryCleanup launches a background goroutine that periodically deletes
// expired memories (valid_until > 0 && valid_until < now) from the store.
//
// The goroutine respects ctx cancellation and stops cleanly when ctx is done.
// interval controls how often cleanup runs (default: 10 minutes).
// Pass 0 to use DefaultExpiryInterval.
//
// The first run happens after one full interval (not immediately at startup)
// to avoid competing with the startup sequence.
//
// An OPTIONAL trailing *engrammetrics.Metrics may be passed; when non-nil, the
// EvaporationHardDeletedTotal counter is fed for evaporation-reason expired
// memories (by type). The variadic signature keeps the existing main.go call
// site (StartExpiryCleanup(ctx, store, 0)) compiling unchanged.
func StartExpiryCleanup(ctx context.Context, store memory.Store, interval time.Duration, m ...*engrammetrics.Metrics) {
	if interval <= 0 {
		interval = DefaultExpiryInterval
	}
	var metrics *engrammetrics.Metrics
	if len(m) > 0 {
		metrics = m[0]
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				fmt.Fprintf(os.Stderr, "[expiry] cleanup goroutine stopped\n")
				return
			case t := <-ticker.C:
				runExpiryTick(ctx, store, metrics, t)
			}
		}
	}()

	fmt.Fprintf(os.Stderr, "[expiry] cleanup goroutine started (interval: %s)\n", interval)
}

// runExpiryTick performs one expiry cleanup pass: it pre-scans the store to
// build a best-effort attributable breakdown of what is expiring (by
// collection, by deprecated_reason, and by type for evaporation-reason
// deletions), then calls DeleteExpired and enriches the log line.
//
// The breakdown is a best-effort snapshot at scan time: a memory could expire
// in the tiny window between the scan and the delete. This is observability,
// not a guarantee.
func runExpiryTick(ctx context.Context, store memory.Store, metrics *engrammetrics.Metrics, t time.Time) {
	byCollection, byReason, evapByType := scanExpiring(ctx, store, t)

	n, err := store.DeleteExpired(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[expiry] cleanup error at %s: %v\n", t.Format(time.RFC3339), err)
		return
	}
	if n > 0 {
		fmt.Fprintf(os.Stderr,
			"[expiry] deleted %d expired memories at %s collection=%v reason=%v\n",
			n, t.Format(time.RFC3339), byCollection, byReason)
		if metrics != nil {
			for typ, count := range evapByType {
				metrics.EvaporationHardDeletedTotal.WithLabelValues(typ).Add(float64(count))
			}
		}
	}
}

// expiredScroller is an OPTIONAL interface implemented by stores that can
// return ONLY expired memories directly (the logical complement of Scroll).
// scanExpiring type-asserts the store to this interface and, when available,
// uses it instead of Scroll (which excludes expired memories and would yield
// empty breakdowns).
type expiredScroller interface {
	ScrollExpired(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error)
}

// scanExpiring pages through the store (same Scroll(Limit:100, Offset)
// pagination pattern as handleEvaporationStatus) and tallies memories that are
// expired at time t. Returns per-collection and per-reason counts, plus a
// per-type count restricted to evaporation-reason expirations.
func scanExpiring(ctx context.Context, store memory.Store, t time.Time) (byCollection, byReason, evapByType map[string]int) {
	byCollection = map[string]int{}
	byReason = map[string]int{}
	evapByType = map[string]int{}

	nowUnix := float64(t.Unix())
	var offset string
	es, useExpired := store.(expiredScroller)
	for {
		var mems []memory.Memory
		var next string
		var err error
		if useExpired {
			mems, next, err = es.ScrollExpired(ctx, memory.ScrollOptions{Limit: 100, Offset: offset})
		} else {
			mems, next, err = store.Scroll(ctx, memory.ScrollOptions{Limit: 100, Offset: offset})
		}
		if err != nil {
			// Best-effort: on scan error, return whatever we have so far.
			break
		}
		for i := range mems {
			mem := &mems[i]
			if !(mem.ValidUntil > 0 && mem.ValidUntil < nowUnix) {
				continue
			}
			byCollection[mem.Collection]++
			reason, _ := mem.Metadata["deprecated_reason"].(string)
			if reason == "" {
				reason = expiryReasonUnset
			}
			byReason[reason]++
			if reason == "evaporation" {
				evapByType[string(mem.Type)]++
			}
		}
		if next == "" {
			break
		}
		offset = next
	}
	return byCollection, byReason, evapByType
}
