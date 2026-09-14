package replay

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/trajectory"
)

// -----------------------------------------------------------------------------
// Candidate replay (write-side external anchor). See S29 item 4.
//
// The candidate replay is a pure, zero-LLM, zero-Qdrant, zero-network function
// over trajectory JSONL. It loads the D4 "candidate" records (one per
// memory_add) and reports how the admission decision set WOULD change under
// operator-overridden dedup thresholds / importance bounds, using ONLY the data
// recorded at write time.
//
// Honesty constraints (why "unresolvable" exists):
//   - dedup re-decision is LAYERED by data vintage. Record.DedupTopScore (the
//     top dedup-search score) landed in 5c1434a and is written on BOTH admitted
//     and dedup_rejected records on/after 2026-09-12T19:07Z:
//       * records written before that carry NO dedup_top_score, so under a
//         changed threshold they cannot be re-decided → UNRESOLVABLE (legacy
//         data vintage; this share dilutes as new logs accrue).
//       * records written on/after it carry dedup_top_score and ARE re-decidable:
//         an admitted record whose top_score >= a raised threshold flips
//         admitted → newly_rejected.
//   - dedup_rejected records also carry the score in gate_details
//     ("dedup score=%.4f against id=%s") → they can flip to admitted under a
//     lower threshold.
//   - Record.Importance is POST-clamp. A change to importance bounds is only
//     resolvable when the recorded value falls OUTSIDE the new bounds (it would
//     then be re-clamped to the new bound); otherwise it is UNRESOLVABLE because
//     the original pre-clamp value is not recorded. This gap is separate from
//     the dedup vintage layering above and is still real.
// -----------------------------------------------------------------------------

var validMemTypes = map[string]bool{
	"identity": true, "event": true, "insight": true, "directive": true,
}

// dedupScoreRe extracts the recorded dedup score and rival id from a
// dedup_rejected candidate record's gate_details string. Format written by
// server.go / crud.go: `dedup score=0.9234 against id=<uuid>`.
var dedupScoreRe = regexp.MustCompile(`dedup score=([0-9]*\.?[0-9]+) against id=(\S*)`)

// CandidateRecord is one D4 "candidate" trajectory entry, decoded for the
// candidate replay. It mirrors the write-side fields of trajectory.Record plus
// the dedup score parsed out of gate_details.
type CandidateRecord struct {
	Timestamp         time.Time `json:"timestamp"`
	Content           string    `json:"content"`
	Type              string    `json:"type"`
	Importance        float64   `json:"importance"` // POST-clamp
	SourceType        string    `json:"source_type"`
	Tags              []string  `json:"tags"`
	AdmissionDecision string    `json:"admission_decision"` // admitted|dedup_rejected|rate_limited|error
	GateDetails       string    `json:"gate_details"`
	Caller            string    `json:"caller"`
	TaskID            string    `json:"task_id"`

	// Parsed from GateDetails (only populated for dedup_rejected records).
	DedupScore     float64 `json:"dedup_score,omitempty"`
	DedupAgainstID string  `json:"dedup_against_id,omitempty"`
	HasDedupScore  bool    `json:"has_dedup_score"`

	// Top dedup-search score recorded at write time (Record.DedupTopScore,
	// landed in 5c1434a). Present on BOTH admitted and dedup_rejected records
	// written on/after 2026-09-12T19:07Z; absent (zero) on older records.
	DedupTopScore    float64 `json:"dedup_top_score,omitempty"`
	HasDedupTopScore bool    `json:"has_dedup_top_score"`
}

// LoadCandidates reads one trajectory JSONL file and returns ONLY the
// Operation=="candidate" records (the write-side admission flow). It does NOT
// touch LoadTrace's retrieve-only contract. Malformed lines and non-candidate
// records are skipped silently (production logs interleave both operations).
func LoadCandidates(path string) ([]CandidateRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open trace %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	var recs []CandidateRecord
	sc := bufio.NewScanner(f)
	// Candidate records embed full memory content (can be multi-KB); grow the
	// scanner buffer well past the 64KB default.
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var rec trajectory.Record
		if err := json.Unmarshal(line, &rec); err != nil {
			continue // skip malformed line
		}
		if rec.Operation != "candidate" {
			continue
		}
		recs = append(recs, recordToCandidate(rec))
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("scan trace %s: %w", path, err)
	}
	return recs, nil
}

func recordToCandidate(rec trajectory.Record) CandidateRecord {
	ts, _ := time.Parse(time.RFC3339, rec.Timestamp)
	c := CandidateRecord{
		Timestamp:         ts,
		Content:           rec.Content,
		Type:              rec.Type,
		Importance:        rec.Importance,
		SourceType:        rec.SourceType,
		Tags:              rec.Tags,
		AdmissionDecision: rec.AdmissionDecision,
		GateDetails:       rec.GateDetails,
		Caller:            rec.Caller,
		TaskID:            rec.TaskID,
	}
	if rec.DedupTopScore != 0 {
		c.DedupTopScore = rec.DedupTopScore
		c.HasDedupTopScore = true
	}
	if rec.AdmissionDecision == "dedup_rejected" {
		if m := dedupScoreRe.FindStringSubmatch(rec.GateDetails); m != nil {
			if v, err := strconv.ParseFloat(m[1], 64); err == nil {
				c.DedupScore = v
				c.DedupAgainstID = m[2]
				c.HasDedupScore = true
			}
		}
	}
	return c
}

// CandidateOptions carries the operator-overridden parameters for a candidate
// replay. A zero-value CandidateOptions applies no override (report is a pure
// census of the recorded decisions).
type CandidateOptions struct {
	// Dedup threshold override. DedupPerType wins over DedupGlobal for a type.
	DedupGlobal    float64
	HasDedupGlobal bool
	DedupPerType   map[string]float64

	// Importance bounds override, per memory type: [lo, hi].
	ImportanceBounds map[string][2]float64

	// ContentPrefixLen bounds the content preview in deltas (default 60).
	ContentPrefixLen int
}

func (o CandidateOptions) hasDedupOverride() bool {
	return o.HasDedupGlobal || len(o.DedupPerType) > 0
}

// resolveDedupThreshold returns the overridden dedup threshold for a memory
// type, if any override applies to it.
func (o CandidateOptions) resolveDedupThreshold(memType string) (float64, bool) {
	if v, ok := o.DedupPerType[memType]; ok {
		return v, true
	}
	if o.HasDedupGlobal {
		return o.DedupGlobal, true
	}
	return 0, false
}

// CandidateDelta is one record whose admission decision would flip under the
// override.
type CandidateDelta struct {
	AgainstID     string  `json:"against_id,omitempty"`
	ContentPrefix string  `json:"content_prefix"`
	Type          string  `json:"type"`
	Score         float64 `json:"score"`
	OldDecision   string  `json:"old_decision"`
	NewThreshold  float64 `json:"new_threshold"`
}

// ImportanceDelta is one record whose resolved importance would change under
// the overridden bounds (a re-clamp).
type ImportanceDelta struct {
	ContentPrefix string  `json:"content_prefix"`
	Type          string  `json:"type"`
	OldImportance float64 `json:"old_importance"`
	NewImportance float64 `json:"new_importance"`
	NewLo         float64 `json:"new_lo"`
	NewHi         float64 `json:"new_hi"`
}

// CandidateReport is the deterministic output of a candidate replay.
type CandidateReport struct {
	GeneratedAt        time.Time `json:"generated_at"`
	Traces             []string  `json:"traces"`
	DedupOverride      string    `json:"dedup_override,omitempty"`
	ImportanceOverride string    `json:"importance_override,omitempty"`

	// Census of the recorded decisions.
	TotalCandidates int `json:"total_candidates"`
	Admitted        int `json:"admitted"`
	DedupRejected   int `json:"dedup_rejected"`
	RateLimited     int `json:"rate_limited"`
	Errored         int `json:"errored"`
	OtherDecision   int `json:"other_decision,omitempty"`

	// Dedup-threshold diff.
	NewlyAdmitted []CandidateDelta `json:"newly_admitted"`
	NewlyRejected []CandidateDelta `json:"newly_rejected"`

	// Importance-bounds diff.
	ImportanceReclamped []ImportanceDelta `json:"importance_reclamped,omitempty"`

	// Records the override could not be applied to for lack of recorded data.
	UnresolvableDedupLegacyNoScore int `json:"unresolvable_legacy_no_score"`
	UnresolvableDedupHasScore      int `json:"unresolvable_has_score"`
	UnresolvableDedup              int `json:"unresolvable_dedup"` // == legacy_no_score + has_score
	UnresolvableImportance         int `json:"unresolvable_importance,omitempty"`
	Unresolvable                   int `json:"unresolvable"`

	Notes []string `json:"notes,omitempty"`
}

// BuildCandidateReport computes the deterministic decision-diff report. It is a
// pure function: no I/O, no network, no LLM. traces is the list of source files
// (for provenance in the report only).
func BuildCandidateReport(traces []string, recs []CandidateRecord, opts CandidateOptions) CandidateReport {
	prefixLen := opts.ContentPrefixLen
	if prefixLen <= 0 {
		prefixLen = 60
	}
	rep := CandidateReport{
		GeneratedAt:     time.Now().UTC(),
		Traces:          traces,
		TotalCandidates: len(recs),
		NewlyAdmitted:   []CandidateDelta{},
		NewlyRejected:   []CandidateDelta{},
	}

	for i := range recs {
		r := &recs[i]
		switch r.AdmissionDecision {
		case "admitted":
			rep.Admitted++
		case "dedup_rejected":
			rep.DedupRejected++
		case "rate_limited":
			rep.RateLimited++
		case "error":
			rep.Errored++
		default:
			rep.OtherDecision++
		}

		// unresolvableRec tracks whether THIS record is unresolvable in any
		// requested dimension, so the Unresolvable total counts distinct records
		// (never double-counting one record across dedup+importance).
		unresolvableRec := false

		// --- dedup-threshold recompute ---
		if opts.hasDedupOverride() {
			if newT, ok := opts.resolveDedupThreshold(r.Type); ok {
				switch r.AdmissionDecision {
				case "dedup_rejected":
					if !r.HasDedupScore {
						// rejected but no parseable score → cannot re-decide
						// (a rejected record with no score is legacy-no-score).
						rep.UnresolvableDedupLegacyNoScore++
						unresolvableRec = true
					} else if r.DedupScore < newT {
						// was rejected because score >= old threshold; under the
						// lower override it now clears the gate → admitted.
						rep.NewlyAdmitted = append(rep.NewlyAdmitted, CandidateDelta{
							AgainstID:     r.DedupAgainstID,
							ContentPrefix: truncate(r.Content, prefixLen),
							Type:          r.Type,
							Score:         r.DedupScore,
							OldDecision:   r.AdmissionDecision,
							NewThreshold:  newT,
						})
					}
					// score >= newT → stays rejected (no change).
				case "admitted":
					if r.HasDedupTopScore {
						// records written on/after 2026-09-12T19:07Z carry the
						// top dedup score → re-decidable. If it meets/exceeds the
						// override it would now trip the dedup gate → rejected.
						if r.DedupTopScore >= newT {
							rep.NewlyRejected = append(rep.NewlyRejected, CandidateDelta{
								ContentPrefix: truncate(r.Content, prefixLen),
								Type:          r.Type,
								Score:         r.DedupTopScore,
								OldDecision:   r.AdmissionDecision,
								NewThreshold:  newT,
								// AgainstID left empty: admits record no rival id.
							})
						}
						// score < newT → stays admitted (no change, not unresolvable).
					} else {
						// legacy record (written before top_score landed) → cannot
						// tell whether a lower threshold would trip it → unresolvable.
						rep.UnresolvableDedupLegacyNoScore++
						unresolvableRec = true
					}
					// rate_limited / error / other: terminal before/around the
					// dedup gate, unaffected by a threshold override.
				}
			}
		}

		// --- importance-bounds recompute ---
		if b, ok := opts.ImportanceBounds[r.Type]; ok {
			lo, hi := b[0], b[1]
			switch {
			case hi > 0 && r.Importance > hi:
				rep.ImportanceReclamped = append(rep.ImportanceReclamped, ImportanceDelta{
					ContentPrefix: truncate(r.Content, prefixLen),
					Type:          r.Type,
					OldImportance: r.Importance,
					NewImportance: hi,
					NewLo:         lo,
					NewHi:         hi,
				})
			case lo > 0 && r.Importance < lo:
				rep.ImportanceReclamped = append(rep.ImportanceReclamped, ImportanceDelta{
					ContentPrefix: truncate(r.Content, prefixLen),
					Type:          r.Type,
					OldImportance: r.Importance,
					NewImportance: lo,
					NewLo:         lo,
					NewHi:         hi,
				})
			default:
				// recorded value is inside the new bounds. The recorded value is
				// post-clamp, so we cannot know if the ORIGINAL provided value
				// was outside the new bounds → unresolvable.
				rep.UnresolvableImportance++
				unresolvableRec = true
			}
		}

		if unresolvableRec {
			rep.Unresolvable++
		}
	}

	// Deterministic ordering (highest score first) for stable output.
	rep.UnresolvableDedup = rep.UnresolvableDedupLegacyNoScore + rep.UnresolvableDedupHasScore
	sort.SliceStable(rep.NewlyAdmitted, func(i, j int) bool {
		return rep.NewlyAdmitted[i].Score > rep.NewlyAdmitted[j].Score
	})
	sort.SliceStable(rep.NewlyRejected, func(i, j int) bool {
		return rep.NewlyRejected[i].Score > rep.NewlyRejected[j].Score
	})

	if opts.hasDedupOverride() {
		rep.Notes = append(rep.Notes,
			"dedup re-decision is layered: records written before 2026-09-12T19:07Z carry no dedup_top_score and are permanently unresolvable (data vintage — this share dilutes as new logs accrue).")
		rep.Notes = append(rep.Notes,
			"records written on/after 2026-09-12T19:07Z carry dedup_top_score and are re-decidable: an admitted record whose top_score >= the override flips admitted → newly_rejected.")
	}
	if len(opts.ImportanceBounds) > 0 {
		rep.Notes = append(rep.Notes,
			"importance is post-clamp: a bound change is only resolvable when the recorded value falls OUTSIDE the new bounds; in-bounds records are unresolvable (pre-clamp value not recorded).")
	}
	return rep
}

// ParseDedupOverride parses the --dedup-threshold flag value. It accepts either
// a single global float ("0.88") or a per-type list ("insight=0.90,directive=0.92").
func ParseDedupOverride(s string) (global float64, hasGlobal bool, perType map[string]float64, err error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false, nil, nil
	}
	if !strings.Contains(s, "=") {
		v, perr := strconv.ParseFloat(s, 64)
		if perr != nil {
			return 0, false, nil, fmt.Errorf("invalid --dedup-threshold %q: %w", s, perr)
		}
		if v <= 0 || v > 1 {
			return 0, false, nil, fmt.Errorf("invalid --dedup-threshold %q: must be in (0,1]", s)
		}
		return v, true, nil, nil
	}
	perType = map[string]float64{}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		kv := strings.SplitN(part, "=", 2)
		if len(kv) != 2 {
			return 0, false, nil, fmt.Errorf("invalid --dedup-threshold entry %q: want type=value", part)
		}
		t := strings.TrimSpace(kv[0])
		if !validMemTypes[t] {
			return 0, false, nil, fmt.Errorf("invalid --dedup-threshold type %q: want one of identity,event,insight,directive", t)
		}
		v, perr := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64)
		if perr != nil {
			return 0, false, nil, fmt.Errorf("invalid --dedup-threshold value for %q: %w", t, perr)
		}
		if v <= 0 || v > 1 {
			return 0, false, nil, fmt.Errorf("invalid --dedup-threshold value for %q: must be in (0,1]", t)
		}
		perType[t] = v
	}
	if len(perType) == 0 {
		return 0, false, nil, fmt.Errorf("invalid --dedup-threshold %q: no entries parsed", s)
	}
	return 0, false, perType, nil
}

// ParseImportanceBounds parses the --importance-bounds flag value, a per-type
// list of lo:hi bounds ("insight=5:8,directive=6:10").
func ParseImportanceBounds(s string) (map[string][2]float64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	out := map[string][2]float64{}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		kv := strings.SplitN(part, "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid --importance-bounds entry %q: want type=lo:hi", part)
		}
		t := strings.TrimSpace(kv[0])
		if !validMemTypes[t] {
			return nil, fmt.Errorf("invalid --importance-bounds type %q: want one of identity,event,insight,directive", t)
		}
		lohi := strings.SplitN(strings.TrimSpace(kv[1]), ":", 2)
		if len(lohi) != 2 {
			return nil, fmt.Errorf("invalid --importance-bounds value for %q: want lo:hi", t)
		}
		lo, err := strconv.ParseFloat(strings.TrimSpace(lohi[0]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid --importance-bounds lo for %q: %w", t, err)
		}
		hi, err := strconv.ParseFloat(strings.TrimSpace(lohi[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid --importance-bounds hi for %q: %w", t, err)
		}
		if lo > hi {
			return nil, fmt.Errorf("invalid --importance-bounds for %q: lo %.2f > hi %.2f", t, lo, hi)
		}
		out[t] = [2]float64{lo, hi}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("invalid --importance-bounds %q: no entries parsed", s)
	}
	return out, nil
}

// RenderCandidateJSON returns the machine-readable candidate report.
func RenderCandidateJSON(r CandidateReport) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// RenderCandidateMarkdown returns a human-readable candidate report.
func RenderCandidateMarkdown(r CandidateReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Candidate Replay Report\n\n")
	fmt.Fprintf(&b, "> Generated: %s\n\n", r.GeneratedAt.Format(time.RFC3339))
	fmt.Fprintf(&b, "Traces: ")
	for i, t := range r.Traces {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "`%s`", t)
	}
	b.WriteString("\n\n")
	if r.DedupOverride != "" {
		fmt.Fprintf(&b, "Dedup override: `%s`\n\n", r.DedupOverride)
	}
	if r.ImportanceOverride != "" {
		fmt.Fprintf(&b, "Importance bounds override: `%s`\n\n", r.ImportanceOverride)
	}

	fmt.Fprintf(&b, "## Census\n\n")
	fmt.Fprintf(&b, "| Decision | Count |\n|---|---|\n")
	fmt.Fprintf(&b, "| total candidates | %d |\n", r.TotalCandidates)
	fmt.Fprintf(&b, "| admitted | %d |\n", r.Admitted)
	fmt.Fprintf(&b, "| dedup_rejected | %d |\n", r.DedupRejected)
	fmt.Fprintf(&b, "| rate_limited | %d |\n", r.RateLimited)
	fmt.Fprintf(&b, "| error | %d |\n", r.Errored)
	if r.OtherDecision > 0 {
		fmt.Fprintf(&b, "| other/unknown | %d |\n", r.OtherDecision)
	}
	b.WriteString("\n")

	fmt.Fprintf(&b, "## Diff under override\n\n")
	fmt.Fprintf(&b, "| Metric | Count |\n|---|---|\n")
	fmt.Fprintf(&b, "| newly_admitted | %d |\n", len(r.NewlyAdmitted))
	fmt.Fprintf(&b, "| newly_rejected | %d |\n", len(r.NewlyRejected))
	if len(r.ImportanceReclamped) > 0 {
		fmt.Fprintf(&b, "| importance_reclamped | %d |\n", len(r.ImportanceReclamped))
	}
	fmt.Fprintf(&b, "| unresolvable | %d |\n", r.Unresolvable)
	fmt.Fprintf(&b, "| unresolvable_legacy_no_score | %d |\n", r.UnresolvableDedupLegacyNoScore)
	fmt.Fprintf(&b, "| unresolvable_has_score | %d |\n", r.UnresolvableDedupHasScore)
	b.WriteString("\n")

	if len(r.NewlyAdmitted) > 0 {
		fmt.Fprintf(&b, "### Newly admitted (dedup_rejected → admitted)\n\n")
		fmt.Fprintf(&b, "| Type | Score | New threshold | Against id | Content |\n|---|---|---|---|---|\n")
		for _, d := range r.NewlyAdmitted {
			fmt.Fprintf(&b, "| %s | %.4f | %.4f | %s | %s |\n",
				d.Type, d.Score, d.NewThreshold, d.AgainstID, truncate(d.ContentPrefix, 60))
		}
		b.WriteString("\n")
	}
	if len(r.NewlyRejected) > 0 {
		fmt.Fprintf(&b, "### Newly rejected (admitted → dedup_rejected)\n\n")
		fmt.Fprintf(&b, "| Type | Score | New threshold | Content |\n|---|---|---|---|\n")
		for _, d := range r.NewlyRejected {
			fmt.Fprintf(&b, "| %s | %.4f | %.4f | %s |\n",
				d.Type, d.Score, d.NewThreshold, truncate(d.ContentPrefix, 60))
		}
		b.WriteString("\n")
	}
	if len(r.ImportanceReclamped) > 0 {
		fmt.Fprintf(&b, "### Importance re-clamped\n\n")
		fmt.Fprintf(&b, "| Type | Old | New | New bounds | Content |\n|---|---|---|---|---|\n")
		for _, d := range r.ImportanceReclamped {
			fmt.Fprintf(&b, "| %s | %.1f | %.1f | [%.1f,%.1f] | %s |\n",
				d.Type, d.OldImportance, d.NewImportance, d.NewLo, d.NewHi, truncate(d.ContentPrefix, 60))
		}
		b.WriteString("\n")
	}

	if len(r.Notes) > 0 {
		fmt.Fprintf(&b, "## Notes\n\n")
		for _, n := range r.Notes {
			fmt.Fprintf(&b, "- %s\n", n)
		}
		b.WriteString("\n")
	}
	return b.String()
}
