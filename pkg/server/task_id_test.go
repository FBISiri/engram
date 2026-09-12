package server

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/trajectory"
	"github.com/mark3labs/mcp-go/mcp"
)

// readCandidateWithTaskID polls the trajectory dir for a candidate record whose
// content matches and returns its task_id. The trajectory logger flushes
// asynchronously, so we poll with a short deadline rather than sleep blindly.
func readCandidateWithTaskID(t *testing.T, dir, wantContent string) (string, bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		matches, _ := filepath.Glob(filepath.Join(dir, "*.jsonl"))
		for _, path := range matches {
			f, err := os.Open(path)
			if err != nil {
				continue
			}
			sc := bufio.NewScanner(f)
			sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
			for sc.Scan() {
				var rec trajectory.Record
				if json.Unmarshal(sc.Bytes(), &rec) != nil {
					continue
				}
				if rec.Operation == "candidate" && rec.Content == wantContent {
					_ = f.Close()
					return rec.TaskID, true
				}
			}
			_ = f.Close()
		}
		time.Sleep(20 * time.Millisecond)
	}
	return "", false
}

func TestMemoryAdd_RecordsTaskID(t *testing.T) {
	srv, _ := newTestServer()
	dir := t.TempDir()
	logger := trajectory.New(dir)
	srv.SetTrajectoryLogger(logger)

	const content = "mcp add carries task id"
	_, err := srv.handleAdd(context.Background(), mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name: "memory_add",
			Arguments: map[string]any{
				"content":     content,
				"type":        "event",
				"source_type": "tool_output",
				"task_id":     "loop-mcp-1",
			},
		},
	})
	if err != nil {
		t.Fatalf("handleAdd: %v", err)
	}
	logger.Close()

	got, ok := readCandidateWithTaskID(t, dir, content)
	if !ok {
		t.Fatal("candidate record for content not found")
	}
	if got != "loop-mcp-1" {
		t.Errorf("candidate task_id = %q, want loop-mcp-1", got)
	}
}

func TestMemoryAdd_NoTaskIDLeavesEmpty(t *testing.T) {
	srv, _ := newTestServer()
	dir := t.TempDir()
	logger := trajectory.New(dir)
	srv.SetTrajectoryLogger(logger)

	const content = "mcp add without task id"
	if _, err := srv.handleAdd(context.Background(), mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      "memory_add",
			Arguments: map[string]any{"content": content, "type": "event", "source_type": "tool_output"},
		},
	}); err != nil {
		t.Fatalf("handleAdd: %v", err)
	}
	logger.Close()

	got, ok := readCandidateWithTaskID(t, dir, content)
	if !ok {
		t.Fatal("candidate record not found")
	}
	if got != "" {
		t.Errorf("candidate task_id = %q, want empty", got)
	}
}

func TestRESTCreateMemory_RecordsTaskID_Body(t *testing.T) {
	collection.DefaultRegistry.Init()
	srv, _ := newTestServer()
	dir := t.TempDir()
	logger := trajectory.New(dir)
	srv.SetTrajectoryLogger(logger)
	h := &HTTPServer{srv: srv}

	const content = "rest add carries task id via body"
	body := `{"content":"` + content + `","type":"event","metadata":{"source_type":"tool_output"},"task_id":"loop-rest-body"}`
	req := httptest.NewRequest(http.MethodPost, "/memories", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.handleCreateMemory(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	logger.Close()

	got, ok := readCandidateWithTaskID(t, dir, content)
	if !ok {
		t.Fatal("candidate record not found")
	}
	if got != "loop-rest-body" {
		t.Errorf("candidate task_id = %q, want loop-rest-body", got)
	}
}

func TestRESTCreateMemory_RecordsTaskID_Header(t *testing.T) {
	collection.DefaultRegistry.Init()
	srv, _ := newTestServer()
	dir := t.TempDir()
	logger := trajectory.New(dir)
	srv.SetTrajectoryLogger(logger)
	h := &HTTPServer{srv: srv}

	const content = "rest add carries task id via header"
	body := `{"content":"` + content + `","type":"event","metadata":{"source_type":"tool_output"}}`
	req := httptest.NewRequest(http.MethodPost, "/memories", strings.NewReader(body))
	req.Header.Set("X-Task-ID", "loop-rest-header")
	rec := httptest.NewRecorder()
	h.handleCreateMemory(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	logger.Close()

	got, ok := readCandidateWithTaskID(t, dir, content)
	if !ok {
		t.Fatal("candidate record not found")
	}
	if got != "loop-rest-header" {
		t.Errorf("candidate task_id = %q, want loop-rest-header", got)
	}
}
