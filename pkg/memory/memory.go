// Package memory defines the core types and scoring algorithms for Engram.
package memory

import (
	"fmt"
	"math"
	"time"

	"github.com/google/uuid"
)

// MemoryType represents the category of a memory.
type MemoryType string

const (
	// TypeIdentity represents stable facts about the user (name, job, preferences, relationships).
	TypeIdentity MemoryType = "identity"
	// TypeEvent represents something that happened or was observed.
	TypeEvent MemoryType = "event"
	// TypeInsight represents inferred patterns, reflections, or conclusions.
	TypeInsight MemoryType = "insight"
	// TypeDirective represents explicit instructions from the user.
	TypeDirective MemoryType = "directive"
)

// ValidTypes is the set of all valid memory types.
var ValidTypes = map[MemoryType]bool{
	TypeIdentity:  true,
	TypeEvent:     true,
	TypeInsight:   true,
	TypeDirective: true,
}

// Memory is a single unit of stored knowledge.
type Memory struct {
	ID         string         `json:"id"`
	Type       MemoryType     `json:"type"`
	Content    string         `json:"content"`
	Source     string         `json:"source"`     // "user", "agent", "system"
	Importance float64        `json:"importance"`  // 1-10
	Tags       []string       `json:"tags"`
	CreatedAt  float64        `json:"created_at"`  // UTC Unix timestamp
	UpdatedAt  float64        `json:"updated_at"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// New creates a new Memory with defaults.
func New(content string, opts ...Option) *Memory {
	now := float64(time.Now().Unix())
	m := &Memory{
		ID:         uuid.New().String(),
		Type:       TypeEvent,
		Content:    content,
		Source:     "agent",
		Importance: 5.0,
		Tags:       []string{},
		CreatedAt:  now,
		UpdatedAt:  now,
		Metadata:   map[string]any{},
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Option configures a Memory during creation.
type Option func(*Memory)

func WithType(t MemoryType) Option      { return func(m *Memory) { m.Type = t } }
func WithSource(s string) Option        { return func(m *Memory) { m.Source = s } }
func WithImportance(i float64) Option   { return func(m *Memory) { m.Importance = i } }
func WithTags(tags ...string) Option    { return func(m *Memory) { m.Tags = tags } }
func WithMetadata(md map[string]any) Option { return func(m *Memory) { m.Metadata = md } }

// Validate checks that a Memory has valid fields.
func (m *Memory) Validate() error {
	if m.Content == "" {
		return fmt.Errorf("memory content cannot be empty")
	}
	if !ValidTypes[m.Type] {
		return fmt.Errorf("invalid memory type: %s", m.Type)
	}
	if m.Importance < 1 || m.Importance > 10 {
		return fmt.Errorf("importance must be between 1 and 10, got %f", m.Importance)
	}
	return nil
}

// ScoredMemory is a Memory with a computed relevance score.
type ScoredMemory struct {
	Memory
	Score float64 `json:"score"` // Final weighted score
}

// DecayConfig holds per-type decay factors.
type DecayConfig struct {
	Identity  float64 // default: 1.0 (no decay)
	Event     float64 // default: 0.99 (~69h half-life)
	Insight   float64 // default: 0.9998 (~144d half-life)
	Directive float64 // default: 1.0 (no decay)
}

// DefaultDecayConfig returns the default decay configuration.
func DefaultDecayConfig() DecayConfig {
	return DecayConfig{
		Identity:  1.0,
		Event:     0.99,
		Insight:   0.9998,
		Directive: 1.0,
	}
}

// DecayFactor returns the decay factor for a given memory type.
func (dc DecayConfig) DecayFactor(t MemoryType) float64 {
	switch t {
	case TypeIdentity:
		return dc.Identity
	case TypeEvent:
		return dc.Event
	case TypeInsight:
		return dc.Insight
	case TypeDirective:
		return dc.Directive
	default:
		return 0.99 // safe default
	}
}

// ScoringWeights holds the weights for the three scoring components.
type ScoringWeights struct {
	Relevance  float64 // default: 1.0
	Recency    float64 // default: 0.5
	Importance float64 // default: 0.3
}

// DefaultScoringWeights returns the default scoring weights.
func DefaultScoringWeights() ScoringWeights {
	return ScoringWeights{
		Relevance:  1.0,
		Recency:    0.5,
		Importance: 0.3,
	}
}

// Score computes the final score for a memory given its raw cosine similarity.
func Score(m *Memory, cosineSim float64, weights ScoringWeights, decay DecayConfig) float64 {
	// S_relevance: clamp cosine similarity to [0, 1]
	relevance := math.Max(0, math.Min(1, cosineSim))

	// S_recency: exponential decay based on hours since creation
	hoursPassed := math.Max(0, (float64(time.Now().Unix())-m.CreatedAt)/3600.0)
	decayFactor := decay.DecayFactor(m.Type)
	recency := math.Pow(decayFactor, hoursPassed)

	// S_importance: normalized to [0, 1]
	importance := m.Importance / 10.0

	return weights.Relevance*relevance + weights.Recency*recency + weights.Importance*importance
}

// clamp restricts v to [lo, hi].
func clamp(v, lo, hi float64) float64 {
	return math.Max(lo, math.Min(hi, v))
}
