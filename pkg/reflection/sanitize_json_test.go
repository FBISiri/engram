package reflection

import (
	"encoding/json"
	"testing"
)

func TestSanitizeJSONControlChars(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "bare LF inside string",
			in:   "{\"a\":\"line1\nline2\"}",
			want: `{"a":"line1\nline2"}`,
		},
		{
			name: "bare LF outside strings preserved",
			in:   "{\n\"a\":\"x\"\n}",
			want: "{\n\"a\":\"x\"\n}",
		},
		{
			name: "escaped quote then bare LF",
			in:   "{\"a\":\"say \\\"hi\\\"\nbye\"}",
			want: "{\"a\":\"say \\\"hi\\\"\\nbye\"}",
		},
		{
			name: "CR and TAB inside string",
			in:   "{\"a\":\"x\ry\tz\"}",
			want: `{"a":"x\ry\tz"}`,
		},
		{
			name: "other control char to unicode escape",
			in:   "{\"a\":\"x\x07y\"}",
			want: `{"a":"x\u0007y"}`,
		},
		{
			name: "already valid passes through byte-identical",
			in:   `{"a":"hello","b":[1,2,3],"c":"tab\tin escaped"}`,
			want: `{"a":"hello","b":[1,2,3],"c":"tab\tin escaped"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeJSONControlChars(tt.in)
			if got != tt.want {
				t.Fatalf("sanitizeJSONControlChars(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestSanitizeJSONControlChars_DialecticRegression(t *testing.T) {
	// Inline fixture resembling the real failure: a bare LF inside the
	// "content" string value.
	raw := "{" +
		"\"content\":\"First point.\nSecond point about tension.\"," +
		"\"tensions\":[\"a vs b\",\"c vs d\"]," +
		"\"source_ids\":[\"id1\",\"id2\"]," +
		"\"confidence\":0.8," +
		"\"importance\":7," +
		"\"tags\":[\"t1\"]}"

	var parsed dialecticLLMResponse
	if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
		t.Fatalf("expected raw fixture to fail unmarshal (bare LF), but it succeeded")
	}

	if err := json.Unmarshal([]byte(sanitizeJSONControlChars(raw)), &parsed); err != nil {
		t.Fatalf("expected sanitized fixture to unmarshal, got err: %v", err)
	}
	if parsed.Content != "First point.\nSecond point about tension." {
		t.Fatalf("content mismatch: %q", parsed.Content)
	}
	if len(parsed.Tensions) != 2 || parsed.Tensions[0] != "a vs b" {
		t.Fatalf("tensions mismatch: %v", parsed.Tensions)
	}
	if parsed.Importance != 7 {
		t.Fatalf("importance mismatch: %d", parsed.Importance)
	}
	if parsed.Confidence != 0.8 {
		t.Fatalf("confidence mismatch: %v", parsed.Confidence)
	}
}

func TestSanitizeJSONControlChars_UnrecoverableStillFails(t *testing.T) {
	// Truncated payload: missing closing brace. Sanitization can't fix structure.
	raw := "{\"content\":\"text with bare\nLF\",\"tensions\":[\"x\"]"

	var parsed dialecticLLMResponse
	if err := json.Unmarshal([]byte(sanitizeJSONControlChars(raw)), &parsed); err == nil {
		t.Fatalf("expected sanitized truncated payload to still fail unmarshal")
	}
}
