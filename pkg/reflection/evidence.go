package reflection

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/FBISiri/engram/pkg/embedding"
	"github.com/FBISiri/engram/pkg/memory"
	"golang.org/x/sync/errgroup"
)

const (
	defaultEvidenceSearchTimeout = 3 * time.Second
	maxEvidenceSearchTimeout     = 10 * time.Second
	evidenceConfidenceThreshold  = 0.6
	reflectionOriginMaxAge       = 7 * 24 * time.Hour
)

// PerQuestionEvidence pairs a focal question with its retrieved evidence.
type PerQuestionEvidence struct {
	Question string
	Evidence []memory.Memory
}

// EvidenceQuestionDebug holds per-question debug info for --debug-evidence.
type EvidenceQuestionDebug struct {
	Question    string    `json:"question"`
	QuerySource string    `json:"query_source"`
	TopK        int       `json:"top_k"`
	RawHits     int       `json:"raw_hits"`
	Scores      []float64 `json:"scores"`
	Kept        int       `json:"kept"`
	FilteredConf int      `json:"filtered_confidence"`
	FilteredAge  int      `json:"filtered_reflection_age"`
}

// retrieveEvidence searches the store for evidence relevant to a single focal
// question. Applies confidence, reflection-origin age, and provenance filters.
// The caller is responsible for concurrency; this function handles one question.
func retrieveEvidence(ctx context.Context, question string, store memory.Store, embedder embedding.Embedder, cfg Config) ([]memory.Memory, *EvidenceQuestionDebug, error) {
	vec, err := embedder.Embed(ctx, question)
	if err != nil {
		return nil, nil, fmt.Errorf("embed question: %w", err)
	}

	topK := cfg.EvidencePerFocal
	if topK == 0 {
		topK = 10
	}

	// Confidence filter is applied post-search to correctly handle absent
	// values (omitempty). Per Memory.Confidence doc: 0 means "not set" and
	// is treated as 1.0 for backward compatibility.
	var filters []memory.Filter

	if cfg.RequireProvenance && len(cfg.AllowedProvenances) > 0 {
		filters = append(filters, memory.Filter{
			Field: "provenance", Op: memory.OpIn, Value: cfg.AllowedProvenances,
		})
	}

	scored, err := store.Search(ctx, vec, memory.SearchOptions{
		Limit:   topK,
		Filters: filters,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("store search: %w", err)
	}

	var dbg *EvidenceQuestionDebug
	if cfg.DebugEvidence {
		dbg = &EvidenceQuestionDebug{
			Question:    question,
			QuerySource: "direct_embed",
			TopK:        topK,
			RawHits:     len(scored),
			Scores:      make([]float64, len(scored)),
		}
		for i, sm := range scored {
			dbg.Scores[i] = sm.Score
		}
	}

	cutoff := float64(time.Now().Add(-reflectionOriginMaxAge).Unix())
	result := make([]memory.Memory, 0, len(scored))
	for _, sm := range scored {
		if hasTag(sm.Tags, sourceReflectionTag) && sm.CreatedAt > cutoff {
			if dbg != nil {
				dbg.FilteredAge++
			}
			continue
		}
		conf := sm.Confidence
		if conf == 0 {
			conf = 1.0
		}
		if conf < evidenceConfidenceThreshold {
			if dbg != nil {
				dbg.FilteredConf++
			}
			continue
		}
		result = append(result, sm.Memory)
	}
	if dbg != nil {
		dbg.Kept = len(result)
	}

	return result, dbg, nil
}

func hasTag(tags []string, target string) bool {
	for _, t := range tags {
		if t == target {
			return true
		}
	}
	return false
}

// retrieveAllEvidence runs retrieveEvidence for each focal question in parallel
// using errgroup. Single-question failures are isolated: the failing question
// gets empty evidence. Populates RunResult observability fields.
func retrieveAllEvidence(ctx context.Context, questions []string, store memory.Store, embedder embedding.Embedder, cfg Config, result *RunResult) []PerQuestionEvidence {
	timeout := cfg.EvidenceSearchTimeout
	if timeout == 0 {
		timeout = defaultEvidenceSearchTimeout
	}
	if timeout > maxEvidenceSearchTimeout {
		timeout = maxEvidenceSearchTimeout
	}

	start := time.Now()
	perQ := make([]PerQuestionEvidence, len(questions))
	debugInfos := make([]*EvidenceQuestionDebug, len(questions))

	g, gctx := errgroup.WithContext(ctx)
	var errMu sync.Mutex // protects result.Errors and debugInfos

	for i, q := range questions {
		i, q := i, q
		g.Go(func() error {
			qctx, cancel := context.WithTimeout(gctx, timeout)
			defer cancel()

			evidence, dbg, err := retrieveEvidence(qctx, q, store, embedder, cfg)
			if err != nil {
				errMu.Lock()
				result.Errors = append(result.Errors, fmt.Sprintf("evidence q%d: %v", i+1, err))
				errMu.Unlock()
				evidence = nil
			}
			if dbg != nil {
				errMu.Lock()
				debugInfos[i] = dbg
				errMu.Unlock()
			}

			perQ[i] = PerQuestionEvidence{Question: q, Evidence: evidence}
			return nil // never fail the group
		})
	}

	_ = g.Wait()

	result.EvidenceSearchMs = time.Since(start).Milliseconds()

	// Compute per-question counts, dedup, overlap.
	seen := make(map[string]struct{})
	totalPerQ := 0
	result.PerQuestionCounts = make([]int, len(perQ))
	for i, pq := range perQ {
		result.PerQuestionCounts[i] = len(pq.Evidence)
		totalPerQ += len(pq.Evidence)
		if len(pq.Evidence) == 0 {
			result.DroppedNoEvidence++
		}
		for _, m := range pq.Evidence {
			seen[m.ID] = struct{}{}
		}
	}
	result.EvidenceCount = len(seen)
	result.EvidenceOverlap = totalPerQ - len(seen)

	if cfg.DebugEvidence {
		result.EvidenceDebug = make([]EvidenceQuestionDebug, 0, len(debugInfos))
		for _, d := range debugInfos {
			if d != nil {
				result.EvidenceDebug = append(result.EvidenceDebug, *d)
			}
		}
	}

	return perQ
}
