package reflection

import (
	"encoding/json"
	"strings"
)

// parseLenientJSON attempts to unmarshal s into *out, tolerating the malformed
// JSON that LLMs sporadically emit. It runs a repair LADDER: a strict parse
// first, then progressively repaired retries. Each attempt unmarshals into a
// freshly zeroed T so a partially-applied failed attempt can never contaminate
// the winning one. The returned stage names the attempt that succeeded
// ("strict" when no repair was needed); on total failure it returns "" and a
// non-nil error (the strict-parse error, the most faithful diagnostic).
//
// It is shared by BOTH LLM-JSON call sites (focal question arrays and dialectic
// insight objects) so every defect in this family is handled in one place.
func parseLenientJSON[T any](s string, out *T) (stage string, err error) {
	s = strings.TrimSpace(s)

	var strictErr error
	try := func(name, candidate string) bool {
		var v T
		if e := json.Unmarshal([]byte(candidate), &v); e == nil {
			*out = v
			stage = name
			return true
		} else if strictErr == nil {
			strictErr = e
		}
		return false
	}

	// 1. strict — s as-is.
	if try("strict", s) {
		return stage, nil
	}
	// 2. control-char sanitization — variant (2): bare control chars (e.g. a raw
	//    newline) inside a string value.
	if try("control-char sanitization", sanitizeJSONControlChars(s)) {
		return stage, nil
	}
	// 3. bare-quote escaping — variant (1): an unescaped `"` inside a string
	//    value that would otherwise close the string early.
	if try("bare-quote escaping", escapeBareQuotes(s)) {
		return stage, nil
	}
	// 4. delimiter repair — variant (3): the structural `";` separator typo.
	if try(`delimiter repair (";"->",")`, repairStructuralSemicolons(s)) {
		return stage, nil
	}
	// 5. combined repair — all three repairs stacked.
	combined := repairStructuralSemicolons(escapeBareQuotes(sanitizeJSONControlChars(s)))
	if try("combined repair", combined) {
		return stage, nil
	}
	// 6. first-valid-block extraction — tolerate prose prologue/epilogue and a
	//    leading malformed block by pulling the first balanced {..} object OR
	//    [..] array (from the combined-repaired text) that parses cleanly.
	for _, cand := range firstBalancedJSONValues(combined) {
		if try("first-valid-block extraction", cand) {
			return stage, nil
		}
	}
	// 7. structural salvage — strip junk trailing a completed value and close
	//    truncated strings/containers, then re-extract. Last resort.
	salvaged := salvageJSON(sanitizeJSONControlChars(s))
	if try("structural salvage", salvaged) {
		return stage, nil
	}
	for _, cand := range firstBalancedJSONValues(salvaged) {
		if try("structural salvage", cand) {
			return stage, nil
		}
	}

	return "", strictErr
}

// escapeBareQuotes escapes unescaped double quotes that appear INSIDE a JSON
// string value (variant 1 of the defect family). It mirrors the in-string /
// backslash-escape tracking of sanitizeJSONControlChars. While inside a string,
// on an unescaped `"` it looks ahead past ASCII whitespace: if the next byte is
// a structural token (`, ] } :`) or end-of-input, the quote legitimately closes
// the string and is emitted verbatim; otherwise it is a bare quote inside the
// value and is emitted as `\"` while staying in the string. On already-valid
// JSON every closing quote is followed by a structural token, so this is a
// no-op (and the strict parse wins first regardless).
func escapeBareQuotes(s string) string {
	var b strings.Builder
	b.Grow(len(s) + 16)
	inString := false
	escaped := false
	structuralAhead := func(i int) bool {
		j := i + 1
		for j < len(s) && (s[j] == ' ' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r') {
			j++
		}
		if j >= len(s) {
			return true
		}
		switch s[j] {
		case ',', ']', '}', ':':
			return true
		}
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inString {
			if escaped {
				b.WriteByte(c)
				escaped = false
				continue
			}
			switch c {
			case '\\':
				b.WriteByte(c)
				escaped = true
			case '"':
				if structuralAhead(i) {
					b.WriteByte(c)
					inString = false
				} else {
					b.WriteString(`\"`)
				}
			default:
				b.WriteByte(c)
			}
		} else {
			if c == '"' {
				inString = true
			}
			b.WriteByte(c)
		}
	}
	return b.String()
}

// firstBalancedJSONValues returns, in source order, every top-level balanced
// JSON value in s — both `{..}` objects and `[..]` arrays — using string- and
// escape-aware brace/bracket counting. Callers try each candidate in turn and
// keep the first that unmarshals into the desired type. This is the generic,
// type-agnostic replacement for the old object-only firstValidJSONObject.
func firstBalancedJSONValues(s string) []string {
	var out []string
	inString := false
	escaped := false
	var stack []byte
	start := -1
	for i := 0; i < len(s); i++ {
		c := s[i]
		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{', '[':
			if len(stack) == 0 {
				start = i
			}
			stack = append(stack, c)
		case '}', ']':
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
				if len(stack) == 0 && start >= 0 {
					out = append(out, s[start:i+1])
					start = -1
				}
			}
		}
	}
	return out
}

// salvageJSON is the last-resort repair for junk-and-truncation damage that the
// earlier stages cannot fix: e.g. a spurious method-call chain appended after a
// completed string value (`"...".trim().slice(0,900),`) and a response cut off
// mid-string / mid-array. It walks s as a tolerant scanner that (a) skips junk
// bytes between a completed value and the next `,`/`}`/`]` delimiter (paren-aware
// so commas inside `(...)` are ignored) and (b) at end-of-input closes any open
// string and unclosed `{`/`[` containers, dropping a dangling trailing comma.
// The result is best-effort valid JSON preserving every complete value seen.
func salvageJSON(s string) string {
	type frame struct {
		closer    byte // '}' or ']'
		expectVal bool // object only: true right after ':'
	}
	var b strings.Builder
	b.Grow(len(s) + 16)
	var stack []frame
	inString := false
	escaped := false
	valueJustClosed := false
	isWS := func(c byte) bool { return c == ' ' || c == '\t' || c == '\n' || c == '\r' }

	i := 0
	for i < len(s) {
		c := s[i]
		if inString {
			b.WriteByte(c)
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
				isValue := true
				if len(stack) > 0 && stack[len(stack)-1].closer == '}' {
					isValue = stack[len(stack)-1].expectVal
				}
				valueJustClosed = isValue
			}
			i++
			continue
		}
		if valueJustClosed {
			if isWS(c) {
				b.WriteByte(c)
				i++
				continue
			}
			if c == ',' || c == '}' || c == ']' {
				valueJustClosed = false
			} else {
				// Junk after a completed value: skip to the next delimiter,
				// ignoring commas nested inside parentheses.
				paren := 0
				for i < len(s) {
					d := s[i]
					switch {
					case d == '(':
						paren++
					case d == ')':
						if paren > 0 {
							paren--
						}
					case paren == 0 && (d == ',' || d == '}' || d == ']'):
						valueJustClosed = false
						goto skipped
					}
					i++
				}
			skipped:
				valueJustClosed = false
				continue
			}
		}
		switch c {
		case '"':
			inString = true
			b.WriteByte(c)
		case '{':
			stack = append(stack, frame{closer: '}'})
			b.WriteByte(c)
		case '[':
			stack = append(stack, frame{closer: ']'})
			b.WriteByte(c)
		case '}', ']':
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
			b.WriteByte(c)
			valueJustClosed = true
		case ':':
			if len(stack) > 0 && stack[len(stack)-1].closer == '}' {
				stack[len(stack)-1].expectVal = true
			}
			b.WriteByte(c)
		case ',':
			if len(stack) > 0 && stack[len(stack)-1].closer == '}' {
				stack[len(stack)-1].expectVal = false
			}
			b.WriteByte(c)
		default:
			b.WriteByte(c)
		}
		i++
	}

	if inString {
		b.WriteByte('"')
	}
	out := strings.TrimRight(b.String(), " \t\r\n")
	out = strings.TrimSuffix(out, ",")
	var cb strings.Builder
	cb.WriteString(out)
	for j := len(stack) - 1; j >= 0; j-- {
		cb.WriteByte(stack[j].closer)
	}
	return cb.String()
}
