package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewVoyage_Defaults(t *testing.T) {
	v := NewVoyage(VoyageConfig{APIKey: "test-key"})
	if v.model != "voyage-3" {
		t.Errorf("model = %q, want voyage-3", v.model)
	}
	if v.baseURL != "https://api.voyageai.com/v1" {
		t.Errorf("baseURL = %q, want https://api.voyageai.com/v1", v.baseURL)
	}
	if v.dimension != 1024 {
		t.Errorf("dimension = %d, want 1024", v.dimension)
	}
	if v.Dimension() != 1024 {
		t.Errorf("Dimension() = %d, want 1024", v.Dimension())
	}
}

func TestNewVoyage_CustomConfig(t *testing.T) {
	v := NewVoyage(VoyageConfig{
		APIKey:    "pa-test",
		Model:     "voyage-3-lite",
		BaseURL:   "https://custom.voyage.com/v1",
		Dimension: 512,
	})
	if v.model != "voyage-3-lite" {
		t.Errorf("model = %q, want voyage-3-lite", v.model)
	}
	if v.baseURL != "https://custom.voyage.com/v1" {
		t.Errorf("baseURL = %q, want https://custom.voyage.com/v1", v.baseURL)
	}
	if v.dimension != 512 {
		t.Errorf("dimension = %d, want 512", v.dimension)
	}
}

func TestVoyage_EmbedBatch_Empty(t *testing.T) {
	v := NewVoyage(VoyageConfig{APIKey: "test"})
	vecs, err := v.EmbedBatch(context.Background(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vecs != nil {
		t.Errorf("expected nil for empty input, got %v", vecs)
	}
}

func TestVoyage_EmbedBatch_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %s, want /embeddings", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer pa-test" {
			t.Errorf("Authorization = %q, want Bearer pa-test", auth)
		}

		var req voyageEmbeddingRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Input) != 2 {
			t.Errorf("input length = %d, want 2", len(req.Input))
		}

		resp := voyageEmbeddingResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Object: "embedding", Index: 0, Embedding: []float32{0.1, 0.2}},
				{Object: "embedding", Index: 1, Embedding: []float32{0.3, 0.4}},
			},
			Model: "voyage-3",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	v := NewVoyage(VoyageConfig{APIKey: "pa-test", BaseURL: server.URL, Dimension: 2})
	vecs, err := v.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vecs))
	}
	if vecs[0][0] != 0.1 {
		t.Errorf("vecs[0][0] = %f, want 0.1", vecs[0][0])
	}
	if vecs[1][0] != 0.3 {
		t.Errorf("vecs[1][0] = %f, want 0.3", vecs[1][0])
	}
}

func TestVoyage_EmbedBatch_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(voyageError{Detail: "Invalid API key"})
	}))
	defer server.Close()

	v := NewVoyage(VoyageConfig{APIKey: "bad-key", BaseURL: server.URL})
	_, err := v.EmbedBatch(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error for bad API key")
	}
	if got := err.Error(); !contains(got, "Invalid API key") {
		t.Errorf("error = %q, want to contain 'Invalid API key'", got)
	}
}

func TestVoyage_Embed_Single(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := voyageEmbeddingResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Object: "embedding", Index: 0, Embedding: []float32{1.0, 2.0}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	v := NewVoyage(VoyageConfig{APIKey: "k", BaseURL: server.URL, Dimension: 2})
	vec, err := v.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vec) != 2 {
		t.Errorf("vector length = %d, want 2", len(vec))
	}
	if vec[0] != 1.0 {
		t.Errorf("vec[0] = %f, want 1.0", vec[0])
	}
}
