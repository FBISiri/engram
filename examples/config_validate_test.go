// Package examples contains no runnable Go code; this test guards the example
// env files shipped alongside the example apps. It fails `go test ./...`
// whenever an example ENGRAM_* key drifts away from either docs/configuration.md
// or the Go env-parsing code, so the three sources can never silently diverge.
//
// Source is read relative to this test file's directory (examples/):
//   - examples/ itself is "."   (env files live here and in ./*/)
//   - repo root  is        ".." (docs/ and the Go sources live under it)
package examples

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// envLineRe matches a `ENGRAM_KEY=value` assignment line in an env file.
var envLineRe = regexp.MustCompile(`^(ENGRAM_[A-Z0-9_]+)=(.*)$`)

// docsTokenRe matches ENGRAM_* tokens in docs, including brace placeholders
// such as {TYPE} or {IDENTITY,DIRECTIVE,INSIGHT,EVENT}.
var docsTokenRe = regexp.MustCompile(`ENGRAM_[A-Z0-9_{},]+`)

// codeTokenRe matches ENGRAM_* tokens in Go source, including single-name brace
// placeholders like {TYPE} used in templated comments.
var codeTokenRe = regexp.MustCompile(`ENGRAM_[A-Z0-9_{}]+`)

// helperRe captures config.go env-helper calls, e.g. envBool("ENGRAM_X", ...),
// mapping a concrete key to the helper that reads it (drives the type check).
var helperRe = regexp.MustCompile(`env(Str|Bool|Int|Float|Duration)\("(ENGRAM_[A-Z0-9_]+)"`)

// braceGroupRe extracts the contents of a single {...} group.
var braceGroupRe = regexp.MustCompile(`\{([^}]*)\}`)

// memTypes is the concrete expansion of the {TYPE} placeholder.
var memTypes = []string{"IDENTITY", "DIRECTIVE", "INSIGHT", "EVENT"}

// docsAllowlist: keys legitimately present in example env files but absent from
// docs/configuration.md because they are NOT Go-binary config keys.
//   - ENGRAM_HOST / ENGRAM_PORT: example-app-only, read by the example
//     bootstrap.py scripts (not the Go server), so they are not documented in
//     the server's configuration reference.
var docsAllowlist = map[string]bool{
	"ENGRAM_HOST": true,
	"ENGRAM_PORT": true,
}

// codeAllowlist: keys legitimately present in example env files but read by no
// Go env parser.
//   - ENGRAM_HOST / ENGRAM_PORT: example-app-only, read by bootstrap.py not Go.
//   - ENGRAM_COLLECTION_NAME: removed from the Go binary in the multi-collection
//     migration (see docs/configuration.md "Deprecated Variables"); still read
//     by the example bootstrap.py / docker-compose for backward compatibility.
var codeAllowlist = map[string]bool{
	"ENGRAM_HOST":            true,
	"ENGRAM_PORT":            true,
	"ENGRAM_COLLECTION_NAME": true,
}

// codeFiles are the non-test Go sources that read ENGRAM_* env vars, relative
// to this test file's directory.
var codeFiles = []string{
	"../pkg/config/config.go",
	"../internal/otel/config.go",
	"../pkg/statedir/statedir.go",
	"../pkg/llm/llm.go",
	"../cmd/engram/main.go",
}

// expand resolves brace placeholders in a token to concrete keys. {TYPE} maps
// to the four memory types; any other {a,b,c} group is a comma list. Multiple
// groups expand as a cartesian product.
func expand(tok string) []string {
	out := []string{tok}
	for {
		loc := braceGroupRe.FindStringSubmatchIndex(out[0])
		if loc == nil {
			// No braces left in the first element => none in any (all share shape).
			break
		}
		var next []string
		for _, s := range out {
			m := braceGroupRe.FindStringSubmatch(s)
			if m == nil {
				next = append(next, s)
				continue
			}
			var opts []string
			if m[1] == "TYPE" {
				opts = memTypes
			} else {
				opts = strings.Split(m[1], ",")
			}
			for _, o := range opts {
				next = append(next, strings.Replace(s, m[0], strings.TrimSpace(o), 1))
			}
		}
		out = next
	}
	return out
}

// tokenSet reads a file and returns the brace-expanded ENGRAM_* key set found
// by re.
func tokenSet(t *testing.T, path string, re *regexp.Regexp) map[string]bool {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	set := map[string]bool{}
	for _, tok := range re.FindAllString(string(data), -1) {
		for _, k := range expand(tok) {
			set[k] = true
		}
	}
	return set
}

// helperType maps a concrete key to the config.go helper that reads it, giving
// the type used for value validation. Keys read via raw os.Getenv (e.g. the
// OTEL vars) are not in the map and skip the type check.
func helperType(t *testing.T) map[string]string {
	t.Helper()
	m := map[string]string{}
	for _, f := range codeFiles {
		data, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for _, mm := range helperRe.FindAllStringSubmatch(string(data), -1) {
			m[mm[2]] = mm[1] // key -> helper suffix (Str/Bool/Int/Float/Duration)
		}
	}
	return m
}

// checkValue validates raw against the type implied by helper. Returns an empty
// string on success or the type name that failed to parse.
func checkValue(helper, raw string) string {
	switch helper {
	case "Bool":
		if _, err := strconv.ParseBool(raw); err != nil {
			return "bool"
		}
	case "Int":
		if _, err := strconv.Atoi(raw); err != nil {
			return "int"
		}
	case "Float":
		if _, err := strconv.ParseFloat(raw, 64); err != nil {
			return "float"
		}
	case "Duration":
		if _, err := time.ParseDuration(raw); err != nil {
			return "duration"
		}
	case "Str":
		// envStr accepts any string (placeholders, empty). No check.
	}
	return ""
}

// envKeys parses a single env file into an ordered list of key/value pairs.
func envKeys(t *testing.T, path string) [][2]string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var pairs [][2]string
	for _, line := range strings.Split(string(data), "\n") {
		m := envLineRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		val := strings.TrimSpace(m[2])
		// Strip a single pair of surrounding quotes if present.
		if len(val) >= 2 && (val[0] == '"' || val[0] == '\'') && val[len(val)-1] == val[0] {
			val = val[1 : len(val)-1]
		}
		pairs = append(pairs, [2]string{m[1], val})
	}
	return pairs
}

func TestExampleConfigs(t *testing.T) {
	docsSet := tokenSet(t, "../docs/configuration.md", docsTokenRe)

	codeSet := map[string]bool{}
	for _, f := range codeFiles {
		for k := range tokenSet(t, f, codeTokenRe) {
			codeSet[k] = true
		}
	}

	types := helperType(t)

	subExamples, err := filepath.Glob("*/.env.example")
	if err != nil {
		t.Fatalf("glob env examples: %v", err)
	}
	sort.Strings(subExamples)
	files := append([]string{"lifecycle.env"}, subExamples...)

	for _, file := range files {
		file := file
		t.Run(file, func(t *testing.T) {
			for _, kv := range envKeys(t, file) {
				key, val := kv[0], kv[1]

				if !docsSet[key] && !docsAllowlist[key] {
					t.Errorf("%s: key %s not documented in docs/configuration.md", file, key)
				}
				if !codeSet[key] && !codeAllowlist[key] {
					t.Errorf("%s: key %s not read by any Go env parser", file, key)
				}
				if helper, ok := types[key]; ok {
					if bad := checkValue(helper, val); bad != "" {
						t.Errorf("%s: key %s value %q not a valid %s", file, key, val, bad)
					}
				}
			}
		})
	}
}
