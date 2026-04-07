// Package server implements MCP and REST API transports for Engram.
package server

// TODO: Implement MCP server using github.com/mark3labs/mcp-go
//
// MCP Tools:
// - memory.search: Semantic search with type/tag/time filters
// - memory.add: Store memories (single or batch)
// - memory.update: Semantic search old → delete → add new
// - memory.delete: Semantic search → delete matching
//
// REST API:
// - POST   /v1/memory/search
// - POST   /v1/memory
// - PUT    /v1/memory
// - DELETE /v1/memory
// - GET    /v1/health
// - GET    /v1/stats
