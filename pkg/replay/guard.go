package replay

import (
	"fmt"
	"strings"
)

// EvalPrefix is the mandatory prefix for any collection the replay engine may
// touch. This replicates eval/harness/guard.py in Go (R10).
const EvalPrefix = "engram_eval_"

// productionCollections is a defense-in-depth deny list; the prefix check
// already excludes all of these, mirroring guard.py.
var productionCollections = map[string]bool{
	"engram_user":       true,
	"engram_agent_self": true,
	"engram_reflection": true,
	"engram":            true,
	"siri":              true,
	"bmo":               true,
}

// GuardViolation is returned when a collection name violates the eval prefix.
type GuardViolation struct{ Name string }

func (e *GuardViolation) Error() string {
	return fmt.Sprintf("SAFETY VIOLATION: collection %q does not have required %q prefix; "+
		"refusing to operate on non-eval collection", e.Name, EvalPrefix)
}

// guardCollection validates that name carries the engram_eval_ prefix. It is
// the single line of defense protecting production collections from the
// replay harness. Returns name unchanged on success.
func guardCollection(name string) (string, error) {
	if name == "" {
		return "", &GuardViolation{Name: name}
	}
	if !strings.HasPrefix(name, EvalPrefix) {
		return "", &GuardViolation{Name: name}
	}
	if productionCollections[name] {
		return "", &GuardViolation{Name: name} // unreachable given prefix, but defense in depth
	}
	return name, nil
}
