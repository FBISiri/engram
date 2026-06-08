// Package trajectory provides async JSONL logging of memory search and update operations.
// Records are buffered in a channel and flushed by a background goroutine without blocking callers.
package trajectory

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// ResultItem is one entry in a retrieve record's results list.
type ResultItem struct {
	ID      string  `json:"id"`
	Content string  `json:"content"`
	Score   float64 `json:"score"`
}

// Record is one trajectory entry written to the JSONL file.
type Record struct {
	Timestamp string `json:"timestamp"`
	Operation string `json:"operation"` // "retrieve" | "update"

	// retrieve-side fields
	Query    string       `json:"query,omitempty"`
	Results  []ResultItem `json:"results,omitempty"`
	Strategy string       `json:"strategy,omitempty"`

	// update-side fields
	Content  string   `json:"content,omitempty"`
	Type     string   `json:"type,omitempty"`
	Tags     []string `json:"tags,omitempty"`
	DedupHit bool     `json:"dedup_hit,omitempty"`

	// common
	LatencyMs int64  `json:"latency_ms"`
	Caller    string `json:"caller,omitempty"`
}

// Logger writes trajectory records to per-day JSONL files under Dir.
// All writes are asynchronous; callers are never blocked.
type Logger struct {
	dir string
	ch  chan Record
}

// New creates a Logger that writes to dir/YYYY-MM-DD.jsonl and starts its background goroutine.
func New(dir string) *Logger {
	l := &Logger{dir: dir, ch: make(chan Record, 512)}
	go l.run()
	return l
}

// Log queues a record for writing. Drops silently if the buffer is full.
func (l *Logger) Log(r Record) {
	select {
	case l.ch <- r:
	default:
	}
}

// Close stops the background goroutine after draining queued records.
func (l *Logger) Close() { close(l.ch) }

func (l *Logger) run() {
	var (
		currentDate string
		f           *os.File
	)
	defer func() {
		if f != nil {
			f.Close()
		}
	}()
	for r := range l.ch {
		date := r.Timestamp[:10] // YYYY-MM-DD
		if date != currentDate {
			if f != nil {
				f.Close()
				f = nil
			}
			if err := os.MkdirAll(l.dir, 0755); err == nil {
				path := filepath.Join(l.dir, date+".jsonl")
				f, _ = os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			}
			currentDate = date
		}
		if f == nil {
			continue
		}
		data, err := json.Marshal(r)
		if err == nil {
			_, _ = f.Write(append(data, '\n'))
		}
	}
}
