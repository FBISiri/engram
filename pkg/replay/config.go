package replay

import (
	"encoding/json"
	"fmt"
	"sort"
)

// SupportedConfigKeys are the ONLY config-override keys the server's
// memory_apply_config actually honors (see pkg/server/eval.go handleApplyConfig:
// it applies recency_weight, top_k, and dedupe_threshold only). Any other key
// in the mirror struct (score_threshold, query_rewrite_prompt, max_entries,
// eviction_policy) is silently ignored by the server, so a --config touching
// ONLY such a key would pass the fail-closed check yet have ZERO server effect —
// a false A/B conclusion. We reject them up front.
var SupportedConfigKeys = map[string]map[string]bool{
	"retrieve_config": {"recency_weight": true, "top_k": true},
	"update_config":   {"dedupe_threshold": true},
}

// ValidateSupportedConfig parses the raw --config JSON and rejects any key that
// the server does not honor, and any override that sets NO supported key.
// Returns an error naming the unsupported keys.
func ValidateSupportedConfig(raw string) error {
	if raw == "" {
		return nil
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &top); err != nil {
		return fmt.Errorf("invalid config JSON: %w", err)
	}
	var unsupported []string
	supportedSet := 0
	for section, body := range top {
		allowed, known := SupportedConfigKeys[section]
		if !known {
			unsupported = append(unsupported, section+".*")
			continue
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(body, &fields); err != nil {
			return fmt.Errorf("invalid config section %q: %w", section, err)
		}
		for k := range fields {
			if allowed[k] {
				supportedSet++
			} else {
				unsupported = append(unsupported, section+"."+k)
			}
		}
	}
	if len(unsupported) > 0 {
		sort.Strings(unsupported)
		return fmt.Errorf("unsupported config keys (server memory_apply_config ignores these): %v; "+
			"supported keys: retrieve_config.recency_weight, retrieve_config.top_k, update_config.dedupe_threshold", unsupported)
	}
	if supportedSet == 0 {
		return fmt.Errorf("config override sets no server-supported key")
	}
	return nil
}
