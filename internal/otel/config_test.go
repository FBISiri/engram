package otel

import "testing"

func TestLoadConfigFromEnv_Defaults(t *testing.T) {
	for _, key := range []string{
		"ENGRAM_OTEL_ENABLED", "ENGRAM_OTEL_EXPORTER",
		"ENGRAM_OTEL_FILE_DIR", "ENGRAM_OTEL_FILE_ROTATION",
		"ENGRAM_OTEL_SAMPLE_RATIO",
	} {
		t.Setenv(key, "")
	}

	cfg := LoadConfigFromEnv()

	if !cfg.Enabled {
		t.Error("expected Enabled=true by default")
	}
	if cfg.Exporter != "file" {
		t.Errorf("expected Exporter=file, got %s", cfg.Exporter)
	}
	if cfg.FileDir != "/tmp/siri-state/engram-traces" {
		t.Errorf("expected default FileDir, got %s", cfg.FileDir)
	}
	if cfg.FileRotation != "daily" {
		t.Errorf("expected FileRotation=daily, got %s", cfg.FileRotation)
	}
	if cfg.SampleRatio != 1.0 {
		t.Errorf("expected SampleRatio=1.0, got %f", cfg.SampleRatio)
	}
}

func TestLoadConfigFromEnv_Overrides(t *testing.T) {
	t.Setenv("ENGRAM_OTEL_ENABLED", "false")
	t.Setenv("ENGRAM_OTEL_EXPORTER", "stdout")
	t.Setenv("ENGRAM_OTEL_FILE_DIR", "/custom/dir")
	t.Setenv("ENGRAM_OTEL_FILE_ROTATION", "size")
	t.Setenv("ENGRAM_OTEL_SAMPLE_RATIO", "0.5")

	cfg := LoadConfigFromEnv()

	if cfg.Enabled {
		t.Error("expected Enabled=false")
	}
	if cfg.Exporter != "stdout" {
		t.Errorf("expected Exporter=stdout, got %s", cfg.Exporter)
	}
	if cfg.FileDir != "/custom/dir" {
		t.Errorf("expected FileDir=/custom/dir, got %s", cfg.FileDir)
	}
	if cfg.FileRotation != "size" {
		t.Errorf("expected FileRotation=size, got %s", cfg.FileRotation)
	}
	if cfg.SampleRatio != 0.5 {
		t.Errorf("expected SampleRatio=0.5, got %f", cfg.SampleRatio)
	}
}

func TestLoadConfigFromEnv_InvalidFallback(t *testing.T) {
	t.Setenv("ENGRAM_OTEL_EXPORTER", "jaeger")
	t.Setenv("ENGRAM_OTEL_SAMPLE_RATIO", "notanumber")
	t.Setenv("ENGRAM_OTEL_FILE_ROTATION", "hourly")

	cfg := LoadConfigFromEnv()

	if cfg.Exporter != "file" {
		t.Errorf("invalid exporter should fall back to 'file', got %s", cfg.Exporter)
	}
	if cfg.SampleRatio != 1.0 {
		t.Errorf("invalid sample ratio should fall back to 1.0, got %f", cfg.SampleRatio)
	}
	if cfg.FileRotation != "daily" {
		t.Errorf("invalid rotation should fall back to 'daily', got %s", cfg.FileRotation)
	}
}

func TestLoadConfigFromEnv_SampleRatioOutOfRange(t *testing.T) {
	t.Setenv("ENGRAM_OTEL_SAMPLE_RATIO", "1.5")
	cfg := LoadConfigFromEnv()
	if cfg.SampleRatio != 1.0 {
		t.Errorf("out-of-range ratio should fall back to 1.0, got %f", cfg.SampleRatio)
	}

	t.Setenv("ENGRAM_OTEL_SAMPLE_RATIO", "-0.1")
	cfg = LoadConfigFromEnv()
	if cfg.SampleRatio != 1.0 {
		t.Errorf("negative ratio should fall back to 1.0, got %f", cfg.SampleRatio)
	}
}
