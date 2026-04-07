package memory

import "math"

// MMR performs Maximal Marginal Relevance reranking to balance relevance and diversity.
// lambda controls the tradeoff: 1.0 = pure relevance, 0.0 = pure diversity.
func MMR(candidates []ScoredMemory, vectors [][]float32, topK int, lambda float64) []ScoredMemory {
	if len(candidates) <= topK {
		return candidates
	}

	// Normalize scores to [0, 1]
	maxScore := candidates[0].Score
	minScore := candidates[len(candidates)-1].Score
	scoreRange := maxScore - minScore
	if scoreRange == 0 {
		scoreRange = 1.0 // avoid division by zero
	}

	type candidate struct {
		idx    int
		score  float64
		vector []float32
	}

	remaining := make([]candidate, len(candidates))
	for i, c := range candidates {
		remaining[i] = candidate{
			idx:    i,
			score:  (c.Score - minScore) / scoreRange,
			vector: vectors[i],
		}
	}

	selected := make([]ScoredMemory, 0, topK)
	selectedVecs := make([][]float32, 0, topK)
	used := make(map[int]bool)

	// Seed with the highest scoring candidate
	bestIdx := 0
	selected = append(selected, candidates[bestIdx])
	selectedVecs = append(selectedVecs, vectors[bestIdx])
	used[bestIdx] = true

	for len(selected) < topK {
		bestMMR := math.Inf(-1)
		bestCandidate := -1

		for _, c := range remaining {
			if used[c.idx] {
				continue
			}

			// Max similarity to any already-selected item
			maxSim := 0.0
			for _, sv := range selectedVecs {
				sim := cosineSim(c.vector, sv)
				if sim > maxSim {
					maxSim = sim
				}
			}

			mmrScore := lambda*c.score - (1-lambda)*maxSim
			if mmrScore > bestMMR {
				bestMMR = mmrScore
				bestCandidate = c.idx
			}
		}

		if bestCandidate < 0 {
			break
		}

		selected = append(selected, candidates[bestCandidate])
		selectedVecs = append(selectedVecs, vectors[bestCandidate])
		used[bestCandidate] = true
	}

	return selected
}

// cosineSim computes cosine similarity between two vectors.
func cosineSim(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
