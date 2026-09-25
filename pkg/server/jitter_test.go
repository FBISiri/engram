package server

import (
	"context"
	"testing"
	"time"
)

func TestJitterMax(t *testing.T) {
	iv := 20 * time.Minute
	tests := []struct {
		name   string
		envVal string
		iv     time.Duration
		want   time.Duration
	}{
		{"default when empty", "", iv, defaultReflectionJitter},
		{"custom under cap", "3m", iv, 3 * time.Minute},
		{"capped at interval/2", "30m", iv, 10 * time.Minute},
		{"negative -> default", "-5m", iv, defaultReflectionJitter},
		{"unparseable -> default", "abc", iv, defaultReflectionJitter},
		{"zero allowed", "0s", iv, 0},
		{"default capped by tiny interval", "", 4 * time.Minute, 2 * time.Minute},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := jitterMax(tt.envVal, tt.iv); got != tt.want {
				t.Fatalf("jitterMax(%q, %s) = %s, want %s", tt.envVal, tt.iv, got, tt.want)
			}
		})
	}
}

func TestJitterWaitCtxCancel(t *testing.T) {
	t.Setenv("ENGRAM_REFLECTION_JITTER", "1h") // large so timer won't fire
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled
	s := &Server{}
	done := make(chan bool, 1)
	go func() { done <- s.jitterWait(ctx, 20*time.Minute) }()
	select {
	case ok := <-done:
		if ok {
			t.Fatalf("jitterWait returned true on cancelled ctx, want false")
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("jitterWait did not return promptly on cancelled ctx")
	}
}
