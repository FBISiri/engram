package reflection

import (
	"fmt"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

const haikuMaxTokens = 1500

// buildPrompt constructs the Haiku reflection prompt from a batch of memories.
func buildPrompt(memories []memory.Memory) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf(
		"You are a reflection engine for an AI agent named Siri. "+
			"Below are %d recent memories that have not yet been reflected upon.\n\n"+
			"Your task: synthesize these memories into 1-3 high-level insights about "+
			"patterns, behaviors, or trends observed. Focus on:\n"+
			"- What themes are recurring for Siri or Frank?\n"+
			"- What behavioral patterns or preferences are emerging?\n"+
			"- What notable shifts in focus or priorities have occurred?\n\n"+
			"For EACH insight, respond in this EXACT format (including the --- delimiters):\n"+
			"---\n"+
			"INSIGHT: <2-4 sentences in English, third person (e.g. \"Siri has been...\", \"Frank tends to...\")>\n"+
			"IMPORTANCE: <integer 1-10, where 10 = directly impacts agent behavior>\n"+
			"TAGS: <comma-separated topic tags, max 5>\n"+
			"---\n\n"+
			"Generate only insights clearly supported by the memories below. "+
			"If no meaningful pattern can be identified, output a single insight with IMPORTANCE: 3.\n\n"+
			"Recent memories:\n",
		len(memories),
	))

	for i, m := range memories {
		content := m.Content
		if len(content) > 200 {
			content = content[:200] + "..."
		}
		idShort := m.ID
		if len(idShort) > 8 {
			idShort = idShort[:8]
		}
		sb.WriteString(fmt.Sprintf(
			"%d. [id=%s type=%s importance=%.0f] %s\n",
			i+1, idShort, m.Type, m.Importance, content,
		))
	}

	sb.WriteString("\nGenerate insights now:\n")
	return sb.String()
}

// ParsedInsight holds a single parsed insight from Haiku's response.
type ParsedInsight struct {
	Content    string
	Importance float64
	Tags       []string
}

// parseHaikuResponse parses the structured Haiku response into insights.
// Format per block:
//
//	---
//	INSIGHT: <text>
//	IMPORTANCE: <int>
//	TAGS: <comma-separated>
//	---
func parseHaikuResponse(response string) []ParsedInsight {
	var insights []ParsedInsight

	// Split on --- delimiter.
	blocks := strings.Split(response, "---")
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}

		var insight ParsedInsight
		insight.Importance = 5.0 // default

		lines := strings.Split(block, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			if strings.HasPrefix(line, "INSIGHT:") {
				insight.Content = strings.TrimSpace(strings.TrimPrefix(line, "INSIGHT:"))
			} else if strings.HasPrefix(line, "IMPORTANCE:") {
				raw := strings.TrimSpace(strings.TrimPrefix(line, "IMPORTANCE:"))
				var imp float64
				if _, err := fmt.Sscanf(raw, "%f", &imp); err == nil {
					if imp >= 1 && imp <= 10 {
						insight.Importance = imp
					}
				}
			} else if strings.HasPrefix(line, "TAGS:") {
				rawTags := strings.TrimSpace(strings.TrimPrefix(line, "TAGS:"))
				for _, tag := range strings.Split(rawTags, ",") {
					tag = strings.TrimSpace(tag)
					// Normalize: lowercase, replace spaces with hyphens.
					tag = strings.ToLower(tag)
					tag = strings.ReplaceAll(tag, " ", "-")
					if tag != "" {
						insight.Tags = append(insight.Tags, tag)
					}
				}
				// Cap at 5 tags.
				if len(insight.Tags) > 5 {
					insight.Tags = insight.Tags[:5]
				}
			}
		}

		// Only include blocks with both INSIGHT and at least some content.
		if insight.Content != "" {
			insights = append(insights, insight)
		}
	}

	// Cap at 3 insights.
	if len(insights) > 3 {
		insights = insights[:3]
	}

	return insights
}

// selectInputBatch picks up to maxSize memories from unreflected, sorted by importance DESC.
func selectInputBatch(unreflected []memory.Memory, maxSize int) []memory.Memory {
	if len(unreflected) <= maxSize {
		return unreflected
	}

	// Sort by importance descending (simple insertion sort for small N).
	sorted := make([]memory.Memory, len(unreflected))
	copy(sorted, unreflected)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j].Importance > sorted[j-1].Importance; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	return sorted[:maxSize]
}

// formatDuration formats a time.Duration as a human-readable string.
func formatDuration(d time.Duration) string {
	d = d.Truncate(time.Millisecond)
	return d.String()
}
