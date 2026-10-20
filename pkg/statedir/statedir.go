// Package statedir resolves the Siri state directory (~/.siri) tolerantly,
// so callers work even when the process has no $HOME (e.g. systemd root unit).
package statedir

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// Dir returns the Siri state directory, creating it if needed.
//
// Resolution order:
//  1. $ENGRAM_STATE_DIR (used directly as the state dir)
//  2. $SIRI_HOME        -> filepath.Join(SIRI_HOME, ".siri")
//  3. os.UserHomeDir()  -> filepath.Join(home, ".siri")
//  4. fallback          -> /root/.siri (with a WARN log)
//
// Never hard-fails solely because HOME is missing; only returns an error if
// MkdirAll fails.
//
// TEST SAFETY (P17 followup): when running under `go test`, Dir REFUSES to
// resolve the PRODUCTION state dir /root/.siri (reached when HOME is unset or
// HOME=/root with no override) so a test can never clobber the production
// audit/state files. Tests that point HOME/SIRI_HOME/ENGRAM_STATE_DIR at a
// temp dir (e.g. t.TempDir()) are unaffected.
func Dir() (string, error) {
	var dir string
	switch {
	case os.Getenv("ENGRAM_STATE_DIR") != "":
		dir = os.Getenv("ENGRAM_STATE_DIR")
	case os.Getenv("SIRI_HOME") != "":
		dir = filepath.Join(os.Getenv("SIRI_HOME"), ".siri")
	default:
		home, err := os.UserHomeDir()
		if err != nil {
			if underTest() {
				return "", fmt.Errorf("statedir: refusing %s fallback under go test ($HOME unset); set ENGRAM_STATE_DIR, SIRI_HOME, or HOME to a temp dir (e.g. t.TempDir())", prodStateDir)
			}
			log.Printf("[WARN] statedir: $HOME not set (%v); falling back to /root/.siri", err)
			dir = prodStateDir
		} else {
			dir = filepath.Join(home, ".siri")
		}
	}
	// TEST SAFETY: never let a test resolve the production state dir, regardless
	// of how it was reached (HOME=/root, SIRI_HOME=/root, or an explicit env).
	if underTest() && dir == prodStateDir {
		return "", fmt.Errorf("statedir: refusing production state dir %s under go test; set ENGRAM_STATE_DIR/SIRI_HOME/HOME to a temp dir (e.g. t.TempDir())", prodStateDir)
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create state dir %s: %w", dir, err)
	}
	return dir, nil
}

// prodStateDir is the production Siri state dir used as the last-resort
// fallback; tests are never allowed to resolve it (see Dir).
const prodStateDir = "/root/.siri"

// underTest reports whether the current process is a `go test` binary, used to
// refuse the production-state-dir fallback so tests cannot write to /root/.siri.
func underTest() bool {
	if flag.Lookup("test.v") != nil {
		return true
	}
	return strings.HasSuffix(os.Args[0], ".test")
}
