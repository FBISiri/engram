package server

import (
	"testing"

	"github.com/FBISiri/engram/pkg/metrics"
	"github.com/FBISiri/engram/pkg/trajectory"
)

// TestAdmissionTotalIncrements verifies engram_admission_total{type,decision}
// increments with the right label values on the memory_add admission path.
func TestAdmissionTotalIncrements(t *testing.T) {
	srv, _ := newTestServer()
	m := metrics.New(nil, nil)
	srv.SetMetrics(m)
	// The admission counter fires inside the deferred candidate-record closure,
	// which is only registered when a trajectory logger is present.
	traj := trajectory.New(t.TempDir())
	t.Cleanup(traj.Close)
	srv.SetTrajectoryLogger(traj)

	if _, err := callTool(srv, "memory_add", map[string]any{
		"content":     "the deploy pipeline runs nightly at 2am utc",
		"type":        "event",
		"source_type": "user_input",
	}); err != nil {
		t.Fatalf("add: %v", err)
	}

	if got := counterValue(t, m.AdmissionTotal.WithLabelValues("event", "admitted")); got != 1 {
		t.Errorf("AdmissionTotal{event,admitted}=%v, want 1", got)
	}

	// A near-identical add is rejected by dedup (mock embedder collides similar
	// sentences) -> decision=dedup_rejected.
	if _, err := callTool(srv, "memory_add", map[string]any{
		"content":     "the deploy pipeline runs nightly at 2am utc",
		"type":        "event",
		"source_type": "user_input",
	}); err != nil {
		t.Fatalf("add dup: %v", err)
	}
	if got := counterValue(t, m.AdmissionTotal.WithLabelValues("event", "dedup_rejected")); got != 1 {
		t.Errorf("AdmissionTotal{event,dedup_rejected}=%v, want 1", got)
	}
}

// TestCheckpointTotalIncrements verifies engram_checkpoint_total{kind,bucket}
// increments for CP1 (dedup advisory) and CP2 (importance monitor). A single
// fresh add produces the "clean" bucket for both.
func TestCheckpointTotalIncrements(t *testing.T) {
	srv := NewServer(newMockStore(), newMockEmbedder(), checkpointConfig())
	m := metrics.New(nil, nil)
	srv.SetMetrics(m)

	if _, err := callTool(srv, "memory_add", map[string]any{
		"content":     "remember to rotate the tls certificates each quarter",
		"type":        "event",
		"source_type": "user_input",
		"tags":        []string{"ops"},
		"importance":  float64(5),
	}); err != nil {
		t.Fatalf("add: %v", err)
	}

	if got := counterValue(t, m.CheckpointTotal.WithLabelValues("cp1", "clean")); got != 1 {
		t.Errorf("CheckpointTotal{cp1,clean}=%v, want 1", got)
	}
	if got := counterValue(t, m.CheckpointTotal.WithLabelValues("cp2", "clean")); got != 1 {
		t.Errorf("CheckpointTotal{cp2,clean}=%v, want 1", got)
	}
}
