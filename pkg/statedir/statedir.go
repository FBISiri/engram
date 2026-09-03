// Package statedir resolves the Siri state directory (~/.siri) tolerantly,
// so callers work even when the process has no $HOME (e.g. systemd root unit).
package statedir

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
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
			log.Printf("[WARN] statedir: $HOME not set (%v); falling back to /root/.siri", err)
			dir = "/root/.siri"
		} else {
			dir = filepath.Join(home, ".siri")
		}
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create state dir %s: %w", dir, err)
	}
	return dir, nil
}
