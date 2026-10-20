// Package qdrant implements the memory.Store interface using Qdrant vector database.
package qdrant

import (
	"context"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
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
	// dwellWindow is the expiry/GC delete-path DECISION-AGE grace: how long a
	// point must remain observed as delete-eligible before DeleteExpired may
	// physically remove it. This is ORTHOGONAL to any object-age threshold
	// (created_at / MinAgeDays / maxAgeDays / pruneMinAge, which measure how OLD
	// the memory is). 0 disables the window (legacy immediate delete).
	dwellWindow time.Duration
}

// defaultExpiryDwellWindow is the store-expiry-path dwell grace applied when
// neither Config.ExpiryDwellWindow nor ENGRAM_EXPIRY_DWELL_HOURS is set.
const defaultExpiryDwellWindow = 72 * time.Hour

// Config holds connection and collection settings for the Qdrant store.
type Config struct {
	// URL is the Qdrant gRPC address in "host:port" format (default "localhost:6334").
	URL string
	// APIKey for Qdrant Cloud or authenticated deployments. Empty means no auth.
	APIKey string
	// UseTLS enables TLS for the gRPC connection. Required for Qdrant Cloud.
	UseTLS bool
	// CollectionName is the Qdrant collection to use (default "engram").
	CollectionName string
	// Dimension is the embedding vector size (default 1536).
	Dimension uint64
	// ExpiryDwellWindow is the DECISION-AGE grace on the hard-delete path (see
	// Store.dwellWindow). When 0, New() resolves it from ENGRAM_EXPIRY_DWELL_HOURS
	// (default defaultExpiryDwellWindow). Set to a negative value to force-disable.
	ExpiryDwellWindow time.Duration
}

// resolveExpiryDwellWindow reads ENGRAM_EXPIRY_DWELL_HOURS (a float number of
// hours) and returns the configured dwell window, defaulting to
// defaultExpiryDwellWindow when unset/unparseable. An explicit "0" disables the
// window (legacy immediate delete). This is a decision-age grace, NOT an
// object-age threshold — see Store.dwellWindow.
func resolveExpiryDwellWindow() time.Duration {
	if raw := os.Getenv("ENGRAM_EXPIRY_DWELL_HOURS"); raw != "" {
		if h, err := strconv.ParseFloat(raw, 64); err == nil {
			return time.Duration(h * float64(time.Hour))
		}
	}
	return defaultExpiryDwellWindow
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
		UseTLS: cfg.UseTLS,
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant: connect: %w", err)
	}

	dwell := cfg.ExpiryDwellWindow
	if dwell == 0 {
		dwell = resolveExpiryDwellWindow()
	}

	return &Store{
		client:      client,
		collection:  cfg.CollectionName,
		dimension:   cfg.Dimension,
		dwellWindow: dwell,
	}, nil
}

// Close closes the underlying gRPC connection.
func (s *Store) Close() error {
	return s.client.Close()
}

// CollectionName returns the Qdrant collection name this store is bound to.
func (s *Store) CollectionName() string { return s.collection }

// DropCollection deletes the Qdrant collection and all its data permanently.
func (s *Store) DropCollection(ctx context.Context) error {
	return s.client.DeleteCollection(ctx, s.collection)
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
	fieldConfidence    = "confidence"      // W17 v1.1: reflection-origin grounding score (0-1)
	fieldArchivedAt         = "archived_at"          // W17: memory-expiry schema
	fieldArchiveReason      = "archive_reason"       // W17: memory-expiry schema
	fieldLifecycleStatus    = "lifecycle_status"     // v0.2: FSM state
	fieldLastAccessedSource = "last_accessed_source" // v0.2: caller type on last search hit
	fieldCollection         = "collection"            // logical collection label set at write time
	fieldExpiryEligibleAt   = "expiry_eligible_at"      // G4: decision-age stamp for the expiry/GC dwell window
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
		{fieldReflectedAt, qdrant.FieldType_FieldTypeFloat},     // W16: enables O(K) unreflected query
		{fieldArchivedAt, qdrant.FieldType_FieldTypeFloat},         // W17: memory-expiry
		{fieldArchiveReason, qdrant.FieldType_FieldTypeKeyword},  // W17: memory-expiry
		{fieldSupersededBy, qdrant.FieldType_FieldTypeKeyword},   // required by Qdrant Cloud (>=1.x) for IsEmpty filter in Search
		{fieldLifecycleStatus, qdrant.FieldType_FieldTypeKeyword},    // v0.2: FSM state filter
		{fieldLastAccessedSource, qdrant.FieldType_FieldTypeKeyword}, // v0.2: caller-type tracking
		{fieldCollection, qdrant.FieldType_FieldTypeKeyword},
		{fieldExpiryEligibleAt, qdrant.FieldType_FieldTypeFloat}, // G4: expiry dwell-window stamp
		{fieldMetadata + ".source_type", qdrant.FieldType_FieldTypeKeyword}, // C1: provenance source_type filter
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
	point, err := memoryToPoint(mem, vector)
	if err != nil {
		return err
	}
	wait := true
	_, err = s.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: s.collection,
		Wait:           &wait,
		Points:         []*qdrant.PointStruct{point},
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
	if !opts.IncludeSuperseded {
		mustConds = append(mustConds, qdrant.NewIsEmpty(fieldSupersededBy))
	}
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
	if opts.ExcludeArchived {
		mustNotConds = append(mustNotConds, qdrant.NewMatchKeyword(fieldLifecycleStatus, "archived"))
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
				sm.Vector = dense.GetDense().GetData()
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

// normalizePayload recursively converts a payload map into forms accepted by
// qdrant.NewValueMap (which panics on unsupported types). Struct slices such as
// []memory.ProvenanceEntry and typed primitive slices are converted to
// []interface{}; nested maps are recursed. Supported leaf primitives are left
// as-is.
func normalizePayload(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = normalizeValue(v)
	}
	return out
}

func normalizeValue(v any) any {
	switch t := v.(type) {
	case []memory.ProvenanceEntry:
		return memory.ProvenanceHistoryToAny(t)
	case map[string]any:
		return normalizePayload(t)
	case []any:
		res := make([]any, len(t))
		for i, e := range t {
			res[i] = normalizeValue(e)
		}
		return res
	case []string:
		res := make([]any, len(t))
		for i, e := range t {
			res[i] = e
		}
		return res
	case []int64:
		res := make([]any, len(t))
		for i, e := range t {
			res[i] = e
		}
		return res
	case []int:
		res := make([]any, len(t))
		for i, e := range t {
			res[i] = e
		}
		return res
	case []float64:
		res := make([]any, len(t))
		for i, e := range t {
			res[i] = e
		}
		return res
	case []float32:
		res := make([]any, len(t))
		for i, e := range t {
			res[i] = e
		}
		return res
	case []bool:
		res := make([]any, len(t))
		for i, e := range t {
			res[i] = e
		}
		return res
	default:
		return v
	}
}

// safeNewValueMap wraps qdrant.NewValueMap and recovers any panic (which the
// client raises on unsupported value types) into a returned error.
func safeNewValueMap(fields map[string]any) (payload map[string]*qdrant.Value, err error) {
	defer func() {
		if r := recover(); r != nil {
			payload = nil
			err = fmt.Errorf("qdrant: payload: %v", r)
		}
	}()
	payload = qdrant.NewValueMap(fields)
	return payload, nil
}

// Update modifies payload fields of an existing memory without re-embedding.
func (s *Store) Update(ctx context.Context, id string, fields map[string]any) error {
	if len(fields) == 0 {
		return nil
	}

	payload, err := safeNewValueMap(normalizePayload(fields))
	if err != nil {
		return err
	}
	wait := true
	_, err = s.client.SetPayload(ctx, &qdrant.SetPayloadPoints{
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
	if !opts.IncludeSuperseded {
		mustConds = append(mustConds, qdrant.NewIsEmpty(fieldSupersededBy))
	}
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

// ScrollExpired returns ONLY expired memories (valid_until > 0 && valid_until < now),
// the logical complement of Scroll (which excludes them). It uses the same
// expired filter as DeleteExpired and is used for expiry attribution.
func (s *Store) ScrollExpired(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	limit := uint32(opts.Limit)
	if limit == 0 {
		limit = 50
	}

	now := float64(time.Now().Unix())
	expiredFilter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			qdrant.NewRange(fieldValidUntil, &qdrant.Range{Gt: qdrant.PtrOf(0.0)}),
			qdrant.NewRange(fieldValidUntil, &qdrant.Range{Lt: qdrant.PtrOf(now)}),
		},
	}

	req := &qdrant.ScrollPoints{
		CollectionName: s.collection,
		Filter:         expiredFilter,
		Limit:          qdrant.PtrOf(limit),
		WithPayload:    qdrant.NewWithPayload(true),
	}
	if opts.Offset != "" {
		req.Offset = qdrant.NewID(opts.Offset)
	}

	results, err := s.client.Scroll(ctx, req)
	if err != nil {
		return nil, "", fmt.Errorf("qdrant: scroll_expired: %w", err)
	}

	memories := make([]memory.Memory, 0, len(results))
	for _, pt := range results {
		mem := pointToMemory(pt.Id, pt.Payload)
		memories = append(memories, *mem)
	}

	var nextOffset string
	if len(results) == int(limit) && len(results) > 0 {
		nextOffset = extractString(results[len(results)-1].Id)
	}

	return memories, nextOffset, nil
}

// ScrollWithVectors is like Scroll but includes the embedding vector for each
// returned memory (in ScoredMemory.Vector, Score=0). Used by migration tooling
// to copy points across physical collections without re-embedding.
func (s *Store) ScrollWithVectors(ctx context.Context, opts memory.ScrollOptions) ([]memory.ScoredMemory, string, error) {
	limit := uint32(opts.Limit)
	if limit == 0 {
		limit = 50
	}

	filter := buildFilter(opts.Filters)
	var mustConds []*qdrant.Condition
	if filter != nil {
		mustConds = append(mustConds, filter.Must...)
	}
	if !opts.IncludeSuperseded {
		mustConds = append(mustConds, qdrant.NewIsEmpty(fieldSupersededBy))
	}
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
		WithVectors: qdrant.NewWithVectors(true),
	}
	if opts.Offset != "" {
		req.Offset = qdrant.NewID(opts.Offset)
	}

	results, err := s.client.Scroll(ctx, req)
	if err != nil {
		return nil, "", fmt.Errorf("qdrant: scroll_with_vectors: %w", err)
	}

	out := make([]memory.ScoredMemory, 0, len(results))
	for _, pt := range results {
		sm := memory.ScoredMemory{Memory: *pointToMemory(pt.Id, pt.Payload)}
		if pt.Vectors != nil {
			if dense := pt.Vectors.GetVector(); dense != nil {
				sm.Vector = dense.GetDense().GetData()
			}
		}
		out = append(out, sm)
	}

	var nextOffset string
	if len(results) == int(limit) && len(results) > 0 {
		nextOffset = extractString(results[len(results)-1].Id)
	}
	return out, nextOffset, nil
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

// expiredPoint is a minimal view of an already-expired point used by the pure
// dwell planner: its id, its persisted expiry_eligible_at stamp (0 when the
// stamp is absent, i.e. the point has not yet been observed delete-eligible),
// and its valid_until (the moment it entered its CURRENT expired episode).
type expiredPoint struct {
	id         string
	eligibleAt float64 // unix seconds; 0 = not yet stamped
	validUntil float64 // unix seconds; expiry boundary of the current episode
}

// Expiry action values for ExpiryRecord.Action.
const (
	ExpiryActionDeleted   = "deleted"   // point physically removed this tick
	ExpiryActionProtected = "protected" // importance>=8 guard: left in place, TTL untouched
	ExpiryActionHealed    = "healed"    // stale TTL cleared (recomputed permanent): valid_until/eligible_at zeroed
)

// expiryImportanceFloor mirrors the human expiry path's permanent protection
// for high-importance memories (pkg/server/expiry_policy.go expiryCandidateDecision:
// "importance >= 8 is always protected"). The unattended expiry tick MUST apply
// the same floor so the two "expired" delete paths have identical protection
// semantics (P17 R1). Keep this in sync with that path; it is deliberately
// NOT silently duplicating a bare magic number.
const expiryImportanceFloor = 8.0

// ExpiryRecord is the per-point outcome of one DeleteExpiredDetailed pass.
// Action is one of ExpiryActionDeleted / ExpiryActionProtected / ExpiryActionHealed.
type ExpiryRecord struct {
	ID            string
	Collection    string
	Type          string
	Action        string
	Importance    float64
	ValidUntil    float64
	CreatedAt     float64
	ContentPrefix string
}

// expiredPointDetail is the richer expired-point view used by
// DeleteExpiredDetailed: the dwell inputs plus the fields needed to recompute
// the TTL (self-heal) and to build an audit ExpiryRecord.
type expiredPointDetail struct {
	id         string
	typ        string
	content    string
	tags       []string
	importance float64
	validUntil float64
	eligibleAt float64
	createdAt  float64
}

// payloadNumber reads a Qdrant numeric payload value tolerating BOTH
// DoubleValue and IntegerValue encodings. Qdrant may persist a whole number
// (e.g. importance=10) as an integer; reading it with GetDoubleValue alone
// would silently yield 0 and bypass the imp>=8 guard, so we fall back to the
// integer encoding.
func payloadNumber(payload map[string]*qdrant.Value, key string) float64 {
	v, ok := payload[key]
	if !ok {
		return 0
	}
	if d := v.GetDoubleValue(); d != 0 {
		return d
	}
	return float64(v.GetIntegerValue())
}

// payloadToExpiredPointDetail maps a Qdrant point id + payload into the detail
// view consumed by classifyExpired. Extracted as a pure seam so the payload
// decoding (double/integer/string/missing-field) is unit-testable.
func payloadToExpiredPointDetail(id string, payload map[string]*qdrant.Value) expiredPointDetail {
	return expiredPointDetail{
		id:         id,
		typ:        getString(payload, fieldType),
		content:    getString(payload, fieldContent),
		tags:       getStringSlice(payload, fieldTags),
		importance: payloadNumber(payload, fieldImportance),
		validUntil: payloadNumber(payload, fieldValidUntil),
		eligibleAt: payloadNumber(payload, fieldExpiryEligibleAt),
		createdAt:  payloadNumber(payload, fieldCreatedAt),
	}
}

// expiryContentPrefix returns a single-line, <=120-rune prefix of content for
// audit records.
func expiryContentPrefix(content string) string {
	s := strings.ReplaceAll(content, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	r := []rune(s)
	if len(r) > 120 {
		r = r[:120]
	}
	return strings.TrimSpace(string(r))
}

// expiryVerdict is the per-point outcome of classifyExpired.
type expiryVerdict int

const (
	verdictHeal    expiryVerdict = iota // recomputed TTL is permanent → clear stale stamp
	verdictProtect                      // importance >= floor → never delete, leave TTL
	verdictStamp                        // dwell: first observation → stamp, skip
	verdictDelete                       // dwell cleared → delete now
	verdictWait                         // inside dwell window → skip this tick
)

// classifyExpired is the pure P17 decision for one already-expired point, in
// the documented order: self-heal (recomputed permanent) WINS over the
// importance guard, which in turn precedes the normal dwell logic. Pure and
// deterministic so the guard/heal/dwell interplay is unit-testable without a
// live Qdrant.
func classifyExpired(p *expiredPointDetail, cfg memory.TTLConfig, dwell time.Duration, now time.Time) expiryVerdict {
	recomputed := memory.ComputeValidUntil(cfg, memory.MemoryType(p.typ), p.importance, p.tags, 0)
	if recomputed == 0 {
		return verdictHeal
	}
	if p.importance >= expiryImportanceFloor {
		return verdictProtect
	}
	switch decideExpiryDwell(p.eligibleAt, p.validUntil, dwell, now) {
	case dwellStamp:
		return verdictStamp
	case dwellDelete:
		return verdictDelete
	default:
		return verdictWait
	}
}

func (s *Store) expiryRecord(p *expiredPointDetail, action string) ExpiryRecord {
	return ExpiryRecord{
		ID:            p.id,
		Collection:    s.collection,
		Type:          p.typ,
		Action:        action,
		Importance:    p.importance,
		ValidUntil:    p.validUntil,
		CreatedAt:     p.createdAt,
		ContentPrefix: expiryContentPrefix(p.content),
	}
}

// expiryDwellDecision is the per-point verdict of the dwell planner.
type expiryDwellDecision int

const (
	dwellWait   expiryDwellDecision = iota // inside the dwell window → skip this tick
	dwellStamp                             // never observed eligible (or stale stamp) → stamp now, skip
	dwellDelete                            // past the dwell window (or dwell disabled) → delete now
)

// decideExpiryDwell is the pure per-point dwell decision. eligibleAt is the
// persisted expiry_eligible_at stamp (0/absent = never stamped); validUntil is
// the point's current expiry boundary. dwell is the DECISION-AGE grace measured
// from the stamp — NOT from created_at and NOT reusing any object-age threshold
// (MinAgeDays/maxAgeDays). When dwell <= 0 the window is disabled and expired
// points delete immediately (legacy behaviour).
//
// Stale-stamp guard: a stamp recorded BEFORE the point's current valid_until
// belongs to a previous expiry episode (the point was revived — valid_until
// pushed into the future — and later expired again). Such a stamp is treated as
// absent so the dwell clock re-arms, preventing a revived memory from being
// evaporated on its first re-expiry tick.
func decideExpiryDwell(eligibleAt, validUntil float64, dwell time.Duration, now time.Time) expiryDwellDecision {
	if dwell <= 0 {
		return dwellDelete
	}
	if eligibleAt <= 0 || eligibleAt < validUntil {
		return dwellStamp
	}
	if now.Sub(time.Unix(int64(eligibleAt), 0)) >= dwell {
		return dwellDelete
	}
	return dwellWait
}

// expiryDwellPlan is the outcome of planning one DeleteExpired pass over the set
// of already-expired points. Points inside the window are simply omitted.
type expiryDwellPlan struct {
	toStamp  []string // expiry_eligible_at absent/stale → stamp now, skip deletion this tick
	toDelete []string // past the dwell window → delete now
}

// planExpiryDwell partitions already-expired points by the dwell decision. Pure
// and deterministic (no I/O) so the mark-then-delete-later behaviour is unit
// testable without a live Qdrant.
func planExpiryDwell(points []expiredPoint, dwell time.Duration, now time.Time) expiryDwellPlan {
	var plan expiryDwellPlan
	for _, p := range points {
		switch decideExpiryDwell(p.eligibleAt, p.validUntil, dwell, now) {
		case dwellStamp:
			plan.toStamp = append(plan.toStamp, p.id)
		case dwellDelete:
			plan.toDelete = append(plan.toDelete, p.id)
		}
	}
	return plan
}

// DeleteExpired removes memories whose valid_until > 0 AND valid_until < now,
// but ONLY after they have cleared the expiry DWELL window (G4) and ONLY when
// they are not protected/healed by the P17 guards. It is a thin wrapper over
// DeleteExpiredDetailed that returns the number of points ACTUALLY deleted this
// tick (Action==ExpiryActionDeleted), preserving the historical contract so the
// memory.Store interface signature and every existing caller/mock are untouched.
func (s *Store) DeleteExpired(ctx context.Context) (int, error) {
	records, err := s.DeleteExpiredDetailed(ctx)
	if err != nil {
		return 0, err
	}
	n := 0
	for i := range records {
		if records[i].Action == ExpiryActionDeleted {
			n++
		}
	}
	return n, nil
}

// DeleteExpiredDetailed is the richer expiry pass behind DeleteExpired. For each
// already-expired point it decides, in this order:
//
//  1. SELF-HEAL (P17 R2/P2): recompute the TTL from the current
//     type+importance+tags via ComputeValidUntil(..., explicit=0). If the matrix
//     says permanent (0), the point carries a STALE finite valid_until that
//     nobody recomputed after its importance rose — CANCEL the delete, clear the
//     stamp (valid_until=0 + expiry_eligible_at=0) and record Action="healed".
//     Healed WINS over protected when both apply (a high-importance insight is
//     both permanent-by-matrix and >=8), so the stale stamp is actively cleared
//     rather than merely skipped.
//  2. IMPORTANCE GUARD (P17 R1/P0): a point with importance >= expiryImportanceFloor
//     is NEVER physically deleted by this unattended path (mirrors the human
//     path in pkg/server/expiry_policy.go). Record Action="protected", leave the
//     valid_until untouched, do not stamp.
//  3. DWELL (G4): otherwise apply the existing decision-age grace — stamp
//     expiry_eligible_at and skip on first observation, delete only after the
//     dwell window clears.
//
// The returned records cover deleted/protected/healed outcomes (not points that
// were merely stamped or are still inside the dwell window). The dwell semantics,
// stamping and reflected_at guard are unchanged from before.
func (s *Store) DeleteExpiredDetailed(ctx context.Context) ([]ExpiryRecord, error) {
	now := float64(time.Now().Unix())

	// First, scroll to collect the expired points (with payload so we can read
	// the persisted expiry_eligible_at dwell stamp and the TTL inputs).
	expiredFilter := &qdrant.Filter{
		Must: []*qdrant.Condition{
			qdrant.NewRange(fieldValidUntil, &qdrant.Range{Gt: qdrant.PtrOf(0.0)}),
			qdrant.NewRange(fieldValidUntil, &qdrant.Range{Lt: qdrant.PtrOf(now)}),
		},
	}

	// Page through expired points in batches of 100.
	const batchSize = uint32(100)
	var expired []expiredPointDetail
	idToPoint := map[string]*qdrant.PointId{}
	var offset *qdrant.PointId

	for {
		req := &qdrant.ScrollPoints{
			CollectionName: s.collection,
			Filter:         expiredFilter,
			Limit:          qdrant.PtrOf(batchSize),
			WithPayload:    qdrant.NewWithPayload(true),
		}
		if offset != nil {
			req.Offset = offset
		}

		results, err := s.client.Scroll(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("qdrant: delete_expired scroll: %w", err)
		}
		if len(results) == 0 {
			break
		}
		for _, pt := range results {
			id := extractString(pt.Id)
			expired = append(expired, payloadToExpiredPointDetail(id, pt.Payload))
			idToPoint[id] = pt.Id
		}
		// If we got fewer than batchSize, we've reached the end.
		if len(results) < int(batchSize) {
			break
		}
		// Advance offset to the last seen ID.
		offset = results[len(results)-1].Id
	}

	if len(expired) == 0 {
		return nil, nil
	}

	nowT := time.Now()
	ttlCfg := memory.DefaultTTLConfig()
	var toHeal, toStamp, toDelete []string
	var healPts, deletePts []*expiredPointDetail
	var records []ExpiryRecord

	for i := range expired {
		p := &expired[i]
		switch classifyExpired(p, ttlCfg, s.dwellWindow, nowT) {
		case verdictHeal:
			toHeal = append(toHeal, p.id)
			healPts = append(healPts, p)
		case verdictProtect:
			records = append(records, s.expiryRecord(p, ExpiryActionProtected))
		case verdictStamp:
			toStamp = append(toStamp, p.id)
		case verdictDelete:
			toDelete = append(toDelete, p.id)
			deletePts = append(deletePts, p)
		case verdictWait:
			// inside dwell window → skip silently this tick.
		}
	}

	wait := true

	// Self-heal: clear the stale finite TTL (valid_until=0 → permanent) and zero
	// the dwell stamp so the point leaves the expired set. Only emit the "healed"
	// audit record when the write actually succeeded, so a persistent write
	// failure does not spam the audit log with the same ids every tick.
	if len(toHeal) > 0 {
		healPayload := qdrant.NewValueMap(map[string]any{fieldValidUntil: 0.0, fieldExpiryEligibleAt: 0.0})
		_, healErr := s.client.SetPayload(ctx, &qdrant.SetPayloadPoints{
			CollectionName: s.collection,
			Wait:           &wait,
			Payload:        healPayload,
			PointsSelector: pointsSelector(idsToPoints(toHeal, idToPoint)),
		})
		if healErr == nil {
			for _, p := range healPts {
				records = append(records, s.expiryRecord(p, ExpiryActionHealed))
			}
		}
	}

	// Mark-then-delete-later: stamp expiry_eligible_at=now on points first
	// observed delete-eligible, then SKIP them this tick. Persisted so the dwell
	// clock survives restarts. Best-effort — a failed stamp just retries next tick.
	if len(toStamp) > 0 {
		stampNow := qdrant.NewValueMap(map[string]any{fieldExpiryEligibleAt: now})
		_, _ = s.client.SetPayload(ctx, &qdrant.SetPayloadPoints{
			CollectionName: s.collection,
			Wait:           &wait,
			Payload:        stampNow,
			PointsSelector: pointsSelector(idsToPoints(toStamp, idToPoint)),
		})
	}

	if len(toDelete) == 0 {
		return records, nil
	}
	deleteIDs := idsToPoints(toDelete, idToPoint)

	// Guard: stamp reflected_at on the points we are about to delete so the
	// reflection engine's fetchUnreflected (IsEmpty filter) won't return them
	// after this moment. Best-effort — errors are ignored since we're deleting.
	reflectedNow := qdrant.NewValueMap(map[string]any{"reflected_at": now})
	_, _ = s.client.SetPayload(ctx, &qdrant.SetPayloadPoints{
		CollectionName: s.collection,
		Wait:           &wait,
		Payload:        reflectedNow,
		PointsSelector: pointsSelector(deleteIDs),
	})

	// Delete the points that have cleared the dwell window.
	_, err := s.client.Delete(ctx, &qdrant.DeletePoints{
		CollectionName: s.collection,
		Wait:           &wait,
		Points:         pointsSelector(deleteIDs),
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant: delete_expired delete: %w", err)
	}

	for _, p := range deletePts {
		records = append(records, s.expiryRecord(p, ExpiryActionDeleted))
	}
	return records, nil
}

// idsToPoints maps a slice of string ids back to their *qdrant.PointId via the
// lookup built during the expired scroll.
func idsToPoints(ids []string, idToPoint map[string]*qdrant.PointId) []*qdrant.PointId {
	out := make([]*qdrant.PointId, 0, len(ids))
	for _, id := range ids {
		if pid, ok := idToPoint[id]; ok {
			out = append(out, pid)
		}
	}
	return out
}

// pointsSelector wraps a slice of point ids in a Qdrant PointsSelector.
func pointsSelector(ids []*qdrant.PointId) *qdrant.PointsSelector {
	return &qdrant.PointsSelector{
		PointsSelectorOneOf: &qdrant.PointsSelector_Points{
			Points: &qdrant.PointsIdsList{Ids: ids},
		},
	}
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

	case memory.OpIsNull:
		// Provenance block mode uses OpIsNull to EXCLUDE records lacking the
		// field. buildFilter places all conditions in Must, so wrap the IsNull
		// check in a negated sub-filter ("must NOT be null"); otherwise Must(OpIn)
		// AND Must(IsNull) is contradictory and matches nothing.
		return qdrant.NewFilterAsCondition(&qdrant.Filter{
			MustNot: []*qdrant.Condition{qdrant.NewIsNull(f.Field)},
		})
	}

	return nil
}

// memoryToPoint converts a Memory + vector into a Qdrant PointStruct. It uses
// qdrant.TryValueMap so an unsupported payload value yields an error instead of
// panicking the process (R2: Insert path must be panic-safe).
func memoryToPoint(mem *memory.Memory, vector []float32) (*qdrant.PointStruct, error) {
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
		payload[fieldMetadata] = normalizePayload(mem.Metadata)
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
	if mem.ArchivedAt > 0 {
		payload[fieldArchivedAt] = mem.ArchivedAt
	}
	if mem.ArchiveReason != "" {
		payload[fieldArchiveReason] = mem.ArchiveReason
	}
	if mem.LifecycleStatus != "" {
		payload[fieldLifecycleStatus] = mem.LifecycleStatus
	}
	if mem.LastAccessedSource != "" {
		payload[fieldLastAccessedSource] = mem.LastAccessedSource
	}
	if mem.Collection != "" {
		payload[fieldCollection] = mem.Collection
	}

	values, err := qdrant.TryValueMap(normalizePayload(payload))
	if err != nil {
		return nil, fmt.Errorf("qdrant: payload: %w", err)
	}
	return &qdrant.PointStruct{
		Id:      qdrant.NewID(mem.ID),
		Vectors: qdrant.NewVectors(vector...),
		Payload: values,
	}, nil
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
	if v, ok := payload[fieldArchivedAt]; ok {
		mem.ArchivedAt = v.GetDoubleValue()
	}
	if v, ok := payload[fieldArchiveReason]; ok {
		mem.ArchiveReason = v.GetStringValue()
	}
	if v, ok := payload[fieldLifecycleStatus]; ok {
		mem.LifecycleStatus = v.GetStringValue()
	}
	if v, ok := payload[fieldLastAccessedSource]; ok {
		mem.LastAccessedSource = v.GetStringValue()
	}
	if v, ok := payload[fieldCollection]; ok {
		mem.Collection = v.GetStringValue()
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
