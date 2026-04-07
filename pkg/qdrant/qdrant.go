// Package qdrant implements the memory.Store interface using Qdrant vector database.
package qdrant

// TODO: Implement Qdrant store using github.com/qdrant/go-client (gRPC)
//
// Key responsibilities:
// - EnsureCollection: Create collection with cosine distance, create payload indexes
// - Insert: Upsert point with vector + payload
// - Search: Vector search with Qdrant filters mapped from memory.Filter
// - Delete: Delete points by ID
// - Update: SetPayload for in-place field updates
// - Stats: Get collection info
