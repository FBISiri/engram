package server

import (
	"fmt"
	"io"
	"os"
	"sync"
)

// Package-level diagnostic stderr sink. Diagnostic/heartbeat writes from
// background goroutines (expiry, evaporation sweep, reflection scheduler) and
// one-off helpers go through stderrf so tests can redirect them race-free via
// setStderrWriter instead of reassigning the global os.Stderr (which races with
// goroutines still reading os.Stderr across tests in the same binary).
var (
	stderrMu sync.RWMutex
	stderrW  io.Writer = os.Stderr
)

// stderrf writes a diagnostic line to the current sink under a read lock.
func stderrf(format string, args ...any) {
	stderrMu.RLock()
	w := stderrW
	stderrMu.RUnlock()
	_, _ = fmt.Fprintf(w, format, args...)
}

// setStderrWriter swaps the diagnostic sink and returns the previous writer so
// callers (tests) can restore it. Race-free under the write lock.
func setStderrWriter(w io.Writer) io.Writer {
	stderrMu.Lock()
	prev := stderrW
	stderrW = w
	stderrMu.Unlock()
	return prev
}
