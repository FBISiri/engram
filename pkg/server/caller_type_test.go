package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/FBISiri/engram/pkg/collection"
)

func TestCallerTypeMiddleware(t *testing.T) {
	collection.DefaultRegistry.Init() // safe — idempotent

	cases := []struct {
		header   string
		wantCT   string
		wantColl string
	}{
		{"user", "user", collection.CollectionUser},
		{"agent-self", "agent-self", collection.CollectionAgentSelf},
		{"reflection", "reflection", collection.CollectionReflection},
		{"pigo", "pigo", collection.CollectionPigo},
		{"", "user", collection.CollectionUser},
		{"bogus", "user", collection.CollectionUser},
	}

	for _, tc := range cases {
		var gotCT, gotColl string
		h := CallerTypeMiddleware(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
			gotCT = CallerTypeFromContext(r.Context())
			gotColl = CollectionFromContext(r.Context())
		}))

		req := httptest.NewRequest("GET", "/anything", nil)
		if tc.header != "" {
			req.Header.Set("X-Caller-Type", tc.header)
		}
		h.ServeHTTP(httptest.NewRecorder(), req)

		if gotCT != tc.wantCT {
			t.Errorf("header=%q: got CT %q, want %q", tc.header, gotCT, tc.wantCT)
		}
		if gotColl != tc.wantColl {
			t.Errorf("header=%q: got coll %q, want %q", tc.header, gotColl, tc.wantColl)
		}
	}
}

func TestCallerTypeFromContext_Default(t *testing.T) {
	if got := CallerTypeFromContext(context.Background()); got != "user" {
		t.Errorf("default got %q, want user", got)
	}
}

func TestWithCallerType_Override(t *testing.T) {
	ctx := WithCallerType(context.Background(), "pigo")
	if got := CallerTypeFromContext(ctx); got != "pigo" {
		t.Errorf("got %q, want pigo", got)
	}
	if got := CollectionFromContext(ctx); got != collection.CollectionPigo {
		t.Errorf("got %q, want %q", got, collection.CollectionPigo)
	}
}
