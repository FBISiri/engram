package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewOpenAI_Defaults(t *testing.T) {
	o := NewOpenAI(OpenAIConfig{APIKey: "test-key"})
	if o.model != "text-embedding-3-small" {
		t.Errorf("model = %q, want text-embedding-3-small", o.model)
	}
	if o.baseURL != "https://api.openai.com/v1" {
		t.Errorf("baseURL = %q, want https://api.openai.com/v1", o.baseURL)
	}
	if o.dimension != 1536 {
		t.Errorf("dimension = %d, want 1536", o.dimension)
	}
	if o.Dimension() != 1536 {
		t.Errorf("Dimension() = %d, want 1536", o.Dimension())
	}
}

func TestNewOpenAI_CustomConfig(t *testing.T) {
	o := NewOpenAI(OpenAIConfig{
		APIKey:    "sk-test",
		Model:     "text-embedding-3-large",
		BaseURL:   "https://custom.api.com/v1",
		Dimension: 3072,
	})
	if o.model != "text-embedding-3-large" {
		t.Errorf("model = %q, want text-embedding-3-large", o.model)
	}
	if o.baseURL != "https://custom.api.com/v1" {
		t.Errorf("baseURL = %q, want https://custom.api.com/v1", o.baseURL)
	}
	if o.dimension != 3072 {
		t.Errorf("dimension = %d, want 3072", o.dimension)
	}
}

func TestOpenAI_EmbedBatch_Empty(t *testing.T) {
	o := NewOpenAI(OpenAIConfig{APIKey: "test"})
	vecs, err := o.EmbedBatch(context.Background(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vecs != nil {
		t.Errorf("expected nil for empty input, got %v", vecs)
	}
}

func TestOpenAI_EmbedBatch_Success(t *testing.T) {
	// Mock server returning valid embeddings.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request.
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %s, want /embeddings", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", auth)
		}

		var req embeddingRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Input) != 2 {
			t.Errorf("input length = %d, want 2", len(req.Input))
		}

		resp := embeddingResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Object: "embedding", Index: 0, Embedding: []float32{0.1, 0.2, 0.3}},
				{Object: "embedding", Index: 1, Embedding: []float32{0.4, 0.5, 0.6}},
			},
			Model: "text-embedding-3-small",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	o := NewOpenAI(OpenAIConfig{
		APIKey:    "test-key",
		Model:     "text-embedding-3-small",
		BaseURL:   server.URL,
		Dimension: 3,
	})

	vecs, err := o.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vecs))
	}
	if vecs[0][0] != 0.1 || vecs[0][1] != 0.2 || vecs[0][2] != 0.3 {
		t.Errorf("vecs[0] = %v, want [0.1, 0.2, 0.3]", vecs[0])
	}
	if vecs[1][0] != 0.4 || vecs[1][1] != 0.5 || vecs[1][2] != 0.6 {
		t.Errorf("vecs[1] = %v, want [0.4, 0.5, 0.6]", vecs[1])
	}
}

func TestOpenAI_EmbedBatch_OutOfOrder(t *testing.T) {
	// API returns embeddings in reverse order.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := embeddingResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Object: "embedding", Index: 1, Embedding: []float32{0.4, 0.5}},
				{Object: "embedding", Index: 0, Embedding: []float32{0.1, 0.2}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: server.URL, Dimension: 2})
	vecs, err := o.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if vecs[0][0] != 0.1 {
		t.Errorf("vecs[0][0] = %f, want 0.1", vecs[0][0])
	}
	if vecs[1][0] != 0.4 {
		t.Errorf("vecs[1][0] = %f, want 0.4", vecs[1][0])
	}
}

func TestOpenAI_EmbedBatch_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(embeddingError{
			Error: struct {
				Message string `json:"message"`
				Type    string `json:"type"`
				Code    string `json:"code"`
			}{
				Message: "Incorrect API key provided",
				Type:    "invalid_request_error",
				Code:    "invalid_api_key",
			},
		})
	}))
	defer server.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "bad-key", BaseURL: server.URL})
	_, err := o.EmbedBatch(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error for bad API key")
	}
	if got := err.Error(); !contains(got, "Incorrect API key") {
		t.Errorf("error = %q, want to contain 'Incorrect API key'", got)
	}
}

func TestOpenAI_EmbedBatch_MismatchCount(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := embeddingResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Object: "embedding", Index: 0, Embedding: []float32{0.1}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: server.URL})
	_, err := o.EmbedBatch(context.Background(), []string{"a", "b"})
	if err == nil {
		t.Fatal("expected error for mismatched count")
	}
}

func TestOpenAI_Embed_Single(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := embeddingResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Object: "embedding", Index: 0, Embedding: []float32{1.0, 2.0, 3.0}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: server.URL, Dimension: 3})
	vec, err := o.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vec) != 3 {
		t.Errorf("vector length = %d, want 3", len(vec))
	}
	if vec[0] != 1.0 {
		t.Errorf("vec[0] = %f, want 1.0", vec[0])
	}
}

func TestOpenAI_EmbedBatch_InvalidIndex(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := embeddingResponse{
			Object: "list",
			Data: []struct {
				Object    string    `json:"object"`
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			}{
				{Object: "embedding", Index: 5, Embedding: []float32{0.1}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: server.URL})
	_, err := o.EmbedBatch(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("expected error for invalid index")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsSubstring(s, sub))
}

func containsSubstring(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
