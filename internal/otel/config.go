package otel

import (
	"os"
	"strconv"
)

type Config struct {
	Enabled      bool
	Exporter     string // "file", "stdout", "none"
	FileDir      string
	FileRotation string // "daily", "size"
	SampleRatio  float64
}

func LoadConfigFromEnv() Config {
	c := Config{
		Enabled:      true,
		Exporter:     "file",
		FileDir:      "/tmp/siri-state/engram-traces",
		FileRotation: "daily",
		SampleRatio:  1.0,
	}

	if os.Getenv("ENGRAM_OTEL_ENABLED") == "false" {
		c.Enabled = false
	}

	if v := os.Getenv("ENGRAM_OTEL_EXPORTER"); v != "" {
		switch v {
		case "file", "stdout", "none":
			c.Exporter = v
		}
	}

	if v := os.Getenv("ENGRAM_OTEL_FILE_DIR"); v != "" {
		c.FileDir = v
	}

	if v := os.Getenv("ENGRAM_OTEL_FILE_ROTATION"); v != "" {
		switch v {
		case "daily", "size":
			c.FileRotation = v
		}
	}

	if v := os.Getenv("ENGRAM_OTEL_SAMPLE_RATIO"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f >= 0 && f <= 1 {
			c.SampleRatio = f
		}
	}

	return c
}
