// content_check.go — CP4: lightweight content-quality signals (spec v1 §2.4).
//
// ContentChecker emits advisories for content that is too short/long or that is
// missing tags/source_type. These are non-blocking signals; the source_type
// advisory does not overlap provenance strict mode (which hard-rejects a missing
// source_type before the success-path check runs).
package server

import (
	"fmt"
	"unicode/utf8"
)

// ContentChecker holds the tunable content-quality thresholds.
type ContentChecker struct {
	MinContentLength  int  // default 20
	MaxContentLength  int  // default 2000
	RequireTags       bool // default true (advisory, not enforcement)
	RequireSourceType bool // default true (advisory; strict enforcement lives in provenance mode)
}

// Check runs the content-quality checks and returns any advisories.
func (c *ContentChecker) Check(content string, tags []string, sourceType string) []Advisory {
	var advisories []Advisory
	n := utf8.RuneCountInString(content)

	if n < c.MinContentLength {
		advisories = append(advisories, Advisory{
			Type:     "content_too_short",
			Severity: "info",
			Message:  fmt.Sprintf("Content is very short (%d chars). Short memories may lack sufficient context for future retrieval.", n),
			Data:     map[string]any{"content_length": n, "min_recommended": c.MinContentLength},
		})
	}

	if n > c.MaxContentLength {
		advisories = append(advisories, Advisory{
			Type:     "content_too_long",
			Severity: "warning",
			Message:  fmt.Sprintf("Content is very long (%d chars). Consider summarizing — long execution logs should not be stored as memories.", n),
			Data:     map[string]any{"content_length": n, "max_recommended": c.MaxContentLength},
		})
	}

	if c.RequireTags && len(tags) == 0 {
		advisories = append(advisories, Advisory{
			Type:     "tags_missing",
			Severity: "info",
			Message:  "No tags provided. Tags improve retrieval accuracy and enable thread-based recall.",
		})
	}

	if c.RequireSourceType && sourceType == "" {
		advisories = append(advisories, Advisory{
			Type:     "source_type_missing",
			Severity: "info",
			Message:  "No source_type provided. Provenance tracking requires source_type for EU AI Act compliance.",
		})
	}

	return advisories
}
