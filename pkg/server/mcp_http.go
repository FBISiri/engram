// mcp_http.go — MCP streamable-HTTP transport (/mcp).
//
// Exposes the same MCP server that ServeStdio drives over HTTP at /mcp on the
// existing HTTP mux. Authentication is per-principal: identity comes from the
// API key, never from a self-declared header. This lets the pigo caller be
// HARD-isolated to engram_pigo (see registry.IsIsolatedCallerType + the tool
// handlers). stdio behaviour is unaffected — this file only adds an HTTP mount.
package server

import (
	"net/http"
	"strings"

	mcpserver "github.com/mark3labs/mcp-go/server"
)

// registerMCPRoute mounts an mcp-go StreamableHTTPServer (wrapping the shared
// MCP server) at /mcp on h.mux, behind principal-key auth. Called from
// registerRoutes so both the "http" and "both" transports get it.
func (h *HTTPServer) registerMCPRoute() {
	streamable := mcpserver.NewStreamableHTTPServer(
		h.srv.GetMCPServer(),
		mcpserver.WithEndpointPath("/mcp"),
	)
	h.mux.Handle("/mcp", h.mcpAuth(streamable))
}

// mcpAuth authenticates /mcp requests with per-principal API keys only.
//
// The Authorization: Bearer <key> token must EQUAL a configured principal key
// (h.principalKeys). On match the caller type is forced from the key (the
// self-declared X-Caller-Type header is ignored) and the request is delegated.
// The legacy shared apiKey is NOT honoured here. If no principal keys are
// configured the endpoint fails closed (always 401).
func (h *HTTPServer) mcpAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(h.principalKeys) == 0 {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
			return
		}
		token, isBearer := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if isBearer {
			for ct, key := range h.principalKeys {
				if key != "" && token == key {
					next.ServeHTTP(w, r.WithContext(WithCallerType(r.Context(), ct)))
					return
				}
			}
		}
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
	})
}
