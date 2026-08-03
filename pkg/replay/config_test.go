package replay

import "testing"

func TestValidateSupportedConfig(t *testing.T) {
	cases := []struct {
		name    string
		raw     string
		wantErr bool
	}{
		{"empty ok", "", false},
		{"supported recency", `{"retrieve_config":{"recency_weight":0.25}}`, false},
		{"supported topk", `{"retrieve_config":{"top_k":5}}`, false},
		{"supported dedupe", `{"update_config":{"dedupe_threshold":0.9}}`, false},
		{"unsupported score_threshold", `{"retrieve_config":{"score_threshold":0.5}}`, true},
		{"unsupported query_rewrite", `{"retrieve_config":{"query_rewrite_prompt":"x"}}`, true},
		{"unsupported max_entries", `{"update_config":{"max_entries":10}}`, true},
		{"unsupported eviction", `{"update_config":{"eviction_policy":"lru"}}`, true},
		{"unknown section", `{"weird":{"k":1}}`, true},
		{"only unsupported field mixed section", `{"retrieve_config":{"recency_weight":0.25,"score_threshold":0.5}}`, true},
		{"no supported key set", `{"retrieve_config":{}}`, true},
		{"bad json", `{`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateSupportedConfig(tc.raw)
			if tc.wantErr && err == nil {
				t.Errorf("expected error for %q", tc.raw)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error for %q: %v", tc.raw, err)
			}
		})
	}
}
