package server

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// DefaultExpiryInterval is the default period between expiry cleanup runs.
const DefaultExpiryInterval = 10 * time.Minute

// StartExpiryCleanup launches a background goroutine that periodically deletes
// expired memories (valid_until > 0 && valid_until < now) from the store.
//
// The goroutine respects ctx cancellation and stops cleanly when ctx is done.
// interval controls how often cleanup runs (default: 10 minutes).
// Pass 0 to use DefaultExpiryInterval.
//
// The first run happens after one full interval (not immediately at startup)
// to avoid competing with the startup sequence.
func StartExpiryCleanup(ctx context.Context, store memory.Store, interval time.Duration) {
	if interval <= 0 {
		interval = DefaultExpiryInterval
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
				n, err := store.DeleteExpired(ctx)
				if err != nil {
					fmt.Fprintf(os.Stderr, "[expiry] cleanup error at %s: %v\n", t.Format(time.RFC3339), err)
					continue
				}
				if n > 0 {
					fmt.Fprintf(os.Stderr, "[expiry] deleted %d expired memories at %s\n", n, t.Format(time.RFC3339))
				}
			}
		}
	}()

	fmt.Fprintf(os.Stderr, "[expiry] cleanup goroutine started (interval: %s)\n", interval)
}
