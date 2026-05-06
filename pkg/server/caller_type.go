// caller_type.go — X-Caller-Type middleware (W20 Day2 Phase 1).
//
// The middleware reads the X-Caller-Type request header and stashes a
// canonicalised value in the request context. Downstream handlers can call
// CallerTypeFromContext(r.Context()) to resolve the target collection name
// without re-parsing headers.
//
// Valid header values: "user" | "agent-self" | "reflection".
// Unknown / missing → defaults to "user" (safe default — user collection is
// the most-restricted recall surface, never the privileged ones).
package server

import (
	"context"
	"net/http"

	"github.com/FBISiri/engram/pkg/collection"
)

type callerTypeKey struct{}

var validCallerTypes = map[string]struct{}{
	"user":       {},
	"agent-self": {},
	"reflection": {},
}

// CallerTypeMiddleware injects the canonicalised X-Caller-Type into ctx.
func CallerTypeMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ct := r.Header.Get("X-Caller-Type")
		if _, ok := validCallerTypes[ct]; !ok {
			ct = "user"
		}
		ctx := context.WithValue(r.Context(), callerTypeKey{}, ct)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// CallerTypeFromContext returns the caller type stashed by the middleware.
// Defaults to "user" when missing.
func CallerTypeFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(callerTypeKey{}).(string); ok && v != "" {
		return v
	}
	return "user"
}

// CollectionFromContext is the convenience wrapper that maps the caller type
// directly to a collection name via the default registry.
func CollectionFromContext(ctx context.Context) string {
	return collection.DefaultRegistry.Resolve(CallerTypeFromContext(ctx))
}
