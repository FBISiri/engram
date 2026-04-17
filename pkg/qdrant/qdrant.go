// Package qdrant implements the memory.Store interface using Qdrant vector database.
package qdrant

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"time"

	"github.com/qdrant/go-client/qdrant"

	"github.com/FBISiri/engram/pkg/memory"
)

// Store implements memory.Store backed by Qdrant via gRPC.
// It is safe for concurrent use.
type Store struct {
	client     *qdrant.Client
	collection string
	dimension  uint64
}

// Config holds connection and collection settings for the Qdrant store.
type Config struct {
	// URL is the Qdrant gRPC address in "host:port" format (default "localhost:6334").
	URL string
	// APIKey for Qdrant Cloud or authenticated deployments. Empty means no auth.
	APIKey string
	// CollectionName is the Qdrant collection to use (default "engram").
	CollectionName string
	// Dimension is the embedding vector size (default 1536).
	Dimension uint64
}

// New creates a new Qdrant-backed Store. The caller must call EnsureCollection
// before performing any read/write operations.
func New(cfg Config) (*Store, error) {
	if cfg.URL == "" {
		cfg.URL = "localhost:6334"
	}
	if cfg.CollectionName == "" {
		cfg.CollectionName = "engram"
	}
	if cfg.Dimension == 0 {
		cfg.Dimension = 1536
	}

	host, portStr, err := net.SplitHostPort(cfg.URL)
	if err != nil {
		// If no port was specified, assume default gRPC port.
		host = cfg.URL
		portStr = "6334"
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return nil, fmt.Errorf("qdrant: invalid port %q: %w", portStr, err)
	}

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   host,
		Port:   port,
		APIKey: cfg.APIKey,
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant: connect: %w", err)
	}

	return &Store{
		client:     client,
		collection: cfg.CollectionName,
		dimension:  cfg.Dimension,
	}, nil
}

// Close closes the underlying gRPC connection.
func (s *Store) Close() error {
	return s.client.Close()
}

// payloadFields defines the payload field names stored alongside each point.
const (
	fieldID            = "id"
	fieldType          = "type"
	fieldContent       = "content"
	fieldSource        = "source"
	fieldImportance    = "importance"
	fieldTags          = "tags"
	fieldCreatedAt     = "created_at"
	fieldUpdatedAt     = "updated_at"
	fieldMetadata      = "metadata"
	fieldValidUntil    = "valid_until"
	fieldSupersededBy  = "superseded_by"
	fieldAccessCount   = "access_count"
	fieldLastAccessedAt = "last_accessed_at"
	fieldReflectedAt   = "reflected_at" // W16: replaces metadata["reflected"]
	fieldConfidence    = "confidence"   // W17 v1.1: reflection-origin grounding score (0-1)
)

// EnsureCollection creates the collection if it doesn't exist, and idempotently
// creates all payload indexes (safe to call on existing collections).
func (s *Store) EnsureCollection(ctx context.Context) error {
	exists, err := s.client.CollectionExists(ctx, s.collection)
	if err != nil {
		return fmt.Errorf("qdrant: check collection: %w", err)
	}
	if !exists {
		err = s.client.CreateCollection(ctx, &qdrant.CreateCollection{
			CollectionName: s.collection,
			VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
				Size:     s.dimension,
				Distance: qdrant.Distance_Cosine,
			}),
		})
		if err != nil {
			return fmt.Errorf("qdrant: create collection: %w", err)
		}
	}

	// Idempotently create all payload indexes; ignore errors for already-existing ones.
	indexes := []struct {
		field     string
		fieldType qdrant.FieldType
	}{
		{fieldType, qdrant.FieldType_FieldTypeKeyword},
		{fieldSource, qdrant.FieldType_FieldTypeKeyword},
		{fieldTags, qdrant.FieldType_FieldTypeKeyword},
		{fieldCreatedAt, qdrant.FieldType_FieldTypeFloat},
		{fieldImportance, qdrant.FieldType_FieldTypeFloat},
		{fieldValidUntil, qdrant.FieldType_FieldTypeFloat},
		{fieldAccessCount, qdrant.FieldType_FieldTypeInteger},
		{fieldLastAccessedAt, qdrant.FieldType_FieldTypeFloat},
		{fieldReflectedAt, qdrant.FieldType_FieldTypeFloat}, // W16: enables O(K) unreflected query
	}

	for _, idx := range indexes {
		_, _ = s.client.CreateFieldIndex(ctx, &qdrant.CreateFieldIndexCollection{
			CollectionName: s.collection,
			FieldName:      idx.field,
			FieldType:      qdrant.PtrOf(idx.fieldType),
		})
	}

	return nil
}

// Insert stores a memory with its embedding vector.
func (s *Store) Insert(ctx context.Context, mem *memory.Memory, vector []float32) error {
	wait := true
	_, err := s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.collection,
		Wait:           &wait,
		Points:         []*qdrant.PointStruct{memoryToPoint(mem, vector)},
	})
	if err != nil {
		return fmt.Errorf("qdrant: insert: %w", err)
	}
	return nil
}

// Search returns scored memories matching the query vector.
// Results are ordered by raw cosine similarity (descending).
// Expired (valid_until > 0 && valid_until < now) and superseded memories are always excluded.
func (s *Store) Search(ctx context.Context, vector []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
	limit := uint64(opts.Limit)
	if limit == 0 {
		limit = 10
	}

	// Combine user filters (Must) with system filters.
	userFilter := buildFilter(opts.Filters)
	var mustConds []*qdrant.Condition
	if userFilter != nil {
		mustConds = append(mustConds, userFilter.Must...)
	}
	// Exclude superseded memories (superseded_by field is absent for valid memories).
	mustConds = append(mustConds, qdrant.NewIsEmpty(fieldSupersededBy))
	// Exclude expired memories: MustNot (valid_until > 0 AND valid_until < now).
	now := float64(time.Now().Unix())
	mustNotConds := []*qdrant.Condition{
		qdrant.NewFilterAsCondition(&qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewRange(fieldValidUntil, &qdrant.Range{Gt: qdrant.PtrOf(0.0)}),
				qdrant.NewRange(fieldValidUntil, &qdrant.Range{Lt: qdrant.PtrOf(now)}),
			},
		}),
	}
	filter := &qdrant.Filter{
		Must:    mustConds,
		MustNot: mustNotConds,
	}

	results, err := s.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: s.collection,
		Query:          qdrant.NewQueryDense(vector),
		Filter:         filter,
		Limit:          qdrant.PtrOf(limit),
		WithPayload:    qdrant.NewWithPayload(true),
		WithVectors:    qdrant.NewWithVectors(true),
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant: search: %w", err)
	}

	scored := make([]memory.ScoredMemory, 0, len(results))
	for _, pt := range results {
		mem := pointToMemory(pt.Id, pt.Payload)
		sm := memory.ScoredMemory{
			Memory: *mem,
			Score:  float64(pt.Score),
		}
		// Extract the dense vector from the result for MMR reranking.
		if pt.Vectors != nil {
			if dense := pt.Vectors.GetVector(); dense != nil {
				sm.Vector = dense.GetData()
			}
		}
		scored = append(scored, sm)
	}
	return scored, nil
}

// Delete removes memories by IDs. Returns the number of successfully deleted items.
func (s *Store) Delete(ctx context.Context, ids []string) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}

	pointIDs := make([]*qdrant.PointId, len(ids))
	for i, id := range ids {
		pointIDs[i] = qdrant.NewID(id)
	}

	wait := true
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collection,
		Wait:           &wait,
		Points: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Points{
				Points: &qdrant.PointsIdsList{
					Ids: pointIDs,
				},
			},
		},
	})
	if err != nil {
		return 0, fmt.Errorf("qdrant: delete: %w", err)
	}
	// Qdrant delete is idempotent; we report len(ids) as deleted since
	// the operation succeeded without error.
	return len(ids), nil
}

// Update modifies payload fields of an existing memory without re-embedding.
func (s *Store) Update(ctx context.Context, id string, fields map[string]any) error {
	if len(fields) == 0 {
		return nil
	}

	payload := qdrant.NewValueMap(fields)
	wait := true
	_, err := s.client.SetPayload(ctx, &qdrant.SetPayloadPoints{
		CollectionName: s.collection,
		Wait:           &wait,
		Payload:        payload,
		PointsSelector: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Points{
				Points: &qdrant.PointsIdsList{
					Ids: []*qdrant.PointId{qdrant.NewID(id)},
				},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("qdrant: update: %w", err)
	}
	return nil
}

// SearchByIDs retrieves specific memories by their IDs.
func (s *Store) SearchByIDs(ctx context.Context, ids []string) ([]memory.Memory, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	pointIDs := make([]*qdrant.PointId, len(ids))
	for i, id := range ids {
		pointIDs[i] = qdrant.NewID(id)
	}

	points, err := s.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: s.collection,
		Ids:            pointIDs,
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant: get by ids: %w", err)
	}

	memories := make([]memory.Memory, 0, len(points))
	for _, pt := range points {
		mem := pointToMemory(pt.Id, pt.Payload)
		memories = append(memories, *mem)
	}
	return memories, nil
}

// Scroll returns memories matching filters without requiring a query vector.
func (s *Store) Scroll(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	limit := uint32(opts.Limit)
	if limit == 0 {
		limit = 50
	}

	// Build filter from user options.
	filter := buildFilter(opts.Filters)

	// Exclude superseded and expired memories (same logic as Search).
	var mustConds []*qdrant.Condition
	if filter != nil {
		mustConds = append(mustConds, filter.Must...)
	}
	mustConds = append(mustConds, qdrant.NewIsEmpty(fieldSupersededBy))
	now := float64(time.Now().Unix())
	mustNotConds := []*qdrant.Condition{
		qdrant.NewFilterAsCondition(&qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewRange(fieldValidUntil, &qdrant.Range{Gt: qdrant.PtrOf(0.0)}),
				qdrant.NewRange(fieldValidUntil, &qdrant.Range{Lt: qdrant.PtrOf(now)}),
			},
		}),
	}

	req := &qdrant.ScrollPoints{
		CollectionName: s.collection,
		Filter: &qdrant.Filter{
			Must:    mustConds,
			MustNot: mustNotConds,
		},
		Limit:       qdrant.PtrOf(limit),
		WithPayload: qdrant.NewWithPayload(true),
	}
	if opts.Offset != "" {
		req.Offset = qdrant.NewID(opts.Offset)
	}

	results, err := s.client.Scroll(ctx, req)
	if err != nil {
		return nil, "", fmt.Errorf("qdrant: scroll: %w", err)
	}

	memories := make([]memory.Memory, 0, len(results))
	for _, pt := range results {
		mem := pointToMemory(pt.Id, pt.Payload)
		memories = append(memories, *mem)
	}

	// Qdrant Scroll returns the next offset via the ScrollPoints response.
	// The go-client returns just the points; we use the last point's ID as offset.
	var nextOffset string
	if len(results) == int(limit) && len(results) > 0 {
		nextOffset = extractString(results[len(results)-1].Id)
	}

	return memories, nextOffset, nil
}

// Stats returns collection statistics.
func (s *Store) Stats(ctx context.Context) (*memory.CollectionStats, error) {
	info, err := s.client.GetCollectionInfo(ctx, s.collection)
	if err != nil {
		return nil, fmt.Errorf("qdrant: stats: %w", err)
	}

	return &memory.CollectionStats{
		PointCount:   info.GetPointsCount(),
		VectorCount:  info.GetPointsCount(), // Qdrant reports points, not vectors separately
		IndexedCount: info.GetIndexedVectorsCount(),
		SegmentCount: info.GetSegmentsCount(),
		Status:       info.GetStatus().String(),
	}, nil
}

// DeleteExpired removes all memories whose valid_until > 0 AND valid_until < now.
// This is the physical cleanup counterpart to the soft-filter applied in Search/Scroll.
// Returns the number of deleted points (always == points matched, since Qdrant delete is
// idempotent and does not report per-point success).
func (s *Store) DeleteExpired(ctx context.Context) (int, error) {
	now := float64(time.Now().Unix())

	// First, scroll to collect the IDs of expired points so we can report a count.
	expiredFilter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			qdrant.NewRange(fieldValidUntil, &qdrant.Range{Gt: qdrant.PtrOf(0.0)}),
			qdrant.NewRange(fieldValidUntil, &qdrant.Range{Lt: qdrant.PtrOf(now)}),
		},
	}

	// Page through expired points in batches of 100.
	const batchSize = uint32(100)
	var allIDs []*qdrant.PointId
	var offset *qdrant.PointId

	for {
		req := &qdrant.ScrollPoints{
			CollectionName: s.collection,
			Filter:         expiredFilter,
			Limit:          qdrant.PtrOf(batchSize),
			WithPayload:    qdrant.NewWithPayload(false),
		}
		if offset != nil {
			req.Offset = offset
		}

		results, err := s.client.Scroll(ctx, req)
		if err != nil {
			return 0, fmt.Errorf("qdrant: delete_expired scroll: %w", err)
		}
		if len(results) == 0 {
			break
		}
		for _, pt := range results {
			allIDs = append(allIDs, pt.Id)
		}
		// If we got fewer than batchSize, we've reached the end.
		if len(results) < int(batchSize) {
			break
		}
		// Advance offset to the last seen ID.
		offset = results[len(results)-1].Id
	}

	if len(allIDs) == 0 {
		return 0, nil
	}

	// Delete all expired points.
	wait := true
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collection,
		Wait:           &wait,
		Points: &qdrant.PointsSelector{
			PointsSelectorOneOf: &qdrant.PointsSelector_Points{
				Points: &qdrant.PointsIdsList{
					Ids: allIDs,
				},
			},
		},
	})
	if err != nil {
		return 0, fmt.Errorf("qdrant: delete_expired delete: %w", err)
	}

	return len(allIDs), nil
}

// buildFilter converts memory.Filter slice into a Qdrant filter.
// All filters are combined with Must (AND) semantics.
func buildFilter(filters []memory.Filter) *qdrant.Filter {
	if len(filters) == 0 {
		return nil
	}

	conditions := make([]*qdrant.Condition, 0, len(filters))
	for _, f := range filters {
		cond := filterToCondition(f)
		if cond != nil {
			conditions = append(conditions, cond)
		}
	}

	if len(conditions) == 0 {
		return nil
	}

	return &qdrant.Filter{
		Must: conditions,
	}
}

// filterToCondition converts a single memory.Filter into a Qdrant condition.
func filterToCondition(f memory.Filter) *qdrant.Condition {
	switch f.Op {
	case memory.OpEq:
		switch v := f.Value.(type) {
		case string:
			return qdrant.NewMatchKeyword(f.Field, v)
		case int64:
			return qdrant.NewMatchInt(f.Field, v)
		case int:
			return qdrant.NewMatchInt(f.Field, int64(v))
		case bool:
			return qdrant.NewMatchBool(f.Field, v)
		case float64:
			// For float equality, use a tight range.
			return qdrant.NewRange(f.Field, &qdrant.Range{
				Gte: qdrant.PtrOf(v),
				Lte: qdrant.PtrOf(v),
			})
		}

	case memory.OpIn:
		switch v := f.Value.(type) {
		case []string:
			return qdrant.NewMatchKeywords(f.Field, v...)
		case []int64:
			return qdrant.NewMatchInts(f.Field, v...)
		}

	case memory.OpGte:
		switch v := f.Value.(type) {
		case float64:
			return qdrant.NewRange(f.Field, &qdrant.Range{
				Gte: qdrant.PtrOf(v),
			})
		case int64:
			return qdrant.NewRange(f.Field, &qdrant.Range{
				Gte: qdrant.PtrOf(float64(v)),
			})
		case int:
			return qdrant.NewRange(f.Field, &qdrant.Range{
				Gte: qdrant.PtrOf(float64(v)),
			})
		}

	case memory.OpLte:
		switch v := f.Value.(type) {
		case float64:
			return qdrant.NewRange(f.Field, &qdrant.Range{
				Lte: qdrant.PtrOf(v),
			})
		case int64:
			return qdrant.NewRange(f.Field, &qdrant.Range{
				Lte: qdrant.PtrOf(float64(v)),
			})
		case int:
			return qdrant.NewRange(f.Field, &qdrant.Range{
				Lte: qdrant.PtrOf(float64(v)),
			})
		}

	case memory.OpRange:
		if r, ok := f.Value.([2]float64); ok {
			return qdrant.NewRange(f.Field, &qdrant.Range{
				Gte: qdrant.PtrOf(r[0]),
				Lte: qdrant.PtrOf(r[1]),
			})
		}

	case memory.OpIsEmpty:
		return qdrant.NewIsEmpty(f.Field)
	}

	return nil
}

// memoryToPoint converts a Memory + vector into a Qdrant PointStruct.
func memoryToPoint(mem *memory.Memory, vector []float32) *qdrant.PointStruct {
	// Build tags as []any for NewValueMap compatibility.
	tags := make([]any, len(mem.Tags))
	for i, t := range mem.Tags {
		tags[i] = t
	}

	payload := map[string]any{
		fieldID:          mem.ID,
		fieldType:        string(mem.Type),
		fieldContent:     mem.Content,
		fieldSource:      mem.Source,
		fieldImportance:  mem.Importance,
		fieldTags:        tags,
		fieldCreatedAt:   mem.CreatedAt,
		fieldUpdatedAt:   mem.UpdatedAt,
		fieldAccessCount: mem.AccessCount,
	}
	if len(mem.Metadata) > 0 {
		payload[fieldMetadata] = mem.Metadata
	}
	if mem.ValidUntil > 0 {
		payload[fieldValidUntil] = mem.ValidUntil
	}
	if mem.SupersededBy != "" {
		payload[fieldSupersededBy] = mem.SupersededBy
	}
	if mem.LastAccessedAt > 0 {
		payload[fieldLastAccessedAt] = mem.LastAccessedAt
	}
	if mem.ReflectedAt > 0 {
		payload[fieldReflectedAt] = mem.ReflectedAt
	}
	if mem.Confidence > 0 {
		payload[fieldConfidence] = mem.Confidence
	}

	return &qdrant.PointStruct{
		Id:      qdrant.NewID(mem.ID),
		Vectors: qdrant.NewVectors(vector...),
		Payload: qdrant.NewValueMap(payload),
	}
}

// pointToMemory converts a Qdrant point (ID + payload) into a Memory.
func pointToMemory(id *qdrant.PointId, payload map[string]*qdrant.Value) *memory.Memory {
	mem := &memory.Memory{
		ID:       extractString(id),
		Type:     memory.MemoryType(getString(payload, fieldType)),
		Content:  getString(payload, fieldContent),
		Source:   getString(payload, fieldSource),
		Tags:     getStringSlice(payload, fieldTags),
		Metadata: getMap(payload, fieldMetadata),
	}

	if v, ok := payload[fieldImportance]; ok {
		mem.Importance = v.GetDoubleValue()
	}
	if v, ok := payload[fieldCreatedAt]; ok {
		mem.CreatedAt = v.GetDoubleValue()
	}
	if v, ok := payload[fieldUpdatedAt]; ok {
		mem.UpdatedAt = v.GetDoubleValue()
	}
	if v, ok := payload[fieldValidUntil]; ok {
		mem.ValidUntil = v.GetDoubleValue()
	}
	if v, ok := payload[fieldSupersededBy]; ok {
		mem.SupersededBy = v.GetStringValue()
	}
	if v, ok := payload[fieldAccessCount]; ok {
		mem.AccessCount = v.GetIntegerValue()
	}
	if v, ok := payload[fieldLastAccessedAt]; ok {
		mem.LastAccessedAt = v.GetDoubleValue()
	}
	if v, ok := payload[fieldReflectedAt]; ok {
		mem.ReflectedAt = v.GetDoubleValue()
	}
	if v, ok := payload[fieldConfidence]; ok {
		mem.Confidence = v.GetDoubleValue()
	}

	return mem
}

// Helper functions for extracting typed values from Qdrant payload.

func extractString(id *qdrant.PointId) string {
	if id == nil {
		return ""
	}
	if uuid := id.GetUuid(); uuid != "" {
		return uuid
	}
	return fmt.Sprintf("%d", id.GetNum())
}

func getString(payload map[string]*qdrant.Value, key string) string {
	if v, ok := payload[key]; ok {
		return v.GetStringValue()
	}
	return ""
}

func getStringSlice(payload map[string]*qdrant.Value, key string) []string {
	v, ok := payload[key]
	if !ok {
		return nil
	}
	list := v.GetListValue()
	if list == nil {
		return nil
	}
	result := make([]string, 0, len(list.Values))
	for _, item := range list.Values {
		result = append(result, item.GetStringValue())
	}
	return result
}

func getMap(payload map[string]*qdrant.Value, key string) map[string]any {
	v, ok := payload[key]
	if !ok {
		return nil
	}
	structVal := v.GetStructValue()
	if structVal == nil {
		return nil
	}
	result := make(map[string]any, len(structVal.Fields))
	for k, val := range structVal.Fields {
		result[k] = valueToInterface(val)
	}
	return result
}

// valueToInterface converts a Qdrant Value to a Go interface{}.
func valueToInterface(v *qdrant.Value) any {
	if v == nil {
		return nil
	}
	switch v.Kind.(type) {
	case *qdrant.Value_StringValue:
		return v.GetStringValue()
	case *qdrant.Value_IntegerValue:
		return v.GetIntegerValue()
	case *qdrant.Value_DoubleValue:
		return v.GetDoubleValue()
	case *qdrant.Value_BoolValue:
		return v.GetBoolValue()
	case *qdrant.Value_ListValue:
		list := v.GetListValue()
		if list == nil {
			return nil
		}
		items := make([]any, len(list.Values))
		for i, item := range list.Values {
			items[i] = valueToInterface(item)
		}
		return items
	case *qdrant.Value_StructValue:
		sv := v.GetStructValue()
		if sv == nil {
			return nil
		}
		m := make(map[string]any, len(sv.Fields))
		for k, val := range sv.Fields {
			m[k] = valueToInterface(val)
		}
		return m
	default:
		return nil
	}
}
