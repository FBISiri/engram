// Package semconv centralizes OpenTelemetry GenAI semantic-convention string
// constants used to annotate engram's memory spans. Its ONLY job is to provide
// gen_ai.* attribute keys + operation-name values (plus tiny helpers) so that
// call sites never embed magic strings.
//
// Reference: OpenTelemetry Semantic Conventions for GenAI.
// Memory-operation names follow the GenAI "operation name" registry values for
// vector/memory stores: search_memory, upsert_memory, delete_memory,
// list_memory. See DONE report for any deviation notes.
package semconv

import "go.opentelemetry.io/otel/attribute"

// Attribute keys (gen_ai.*). Only keys that map to data engram actually has are
// defined here.
const (
	// AttrOperationName is the gen_ai.operation.name key: the high-level GenAI
	// operation being performed on the memory store.
	AttrOperationName = "gen_ai.operation.name"
	// AttrSystem is the gen_ai.system key: the GenAI system / provider. engram
	// is the memory system, so we set it to SystemEngram.
	AttrSystem = "gen_ai.system"
)

// SystemEngram is the gen_ai.system value for the engram memory store.
const SystemEngram = "engram"

// Operation-name values (gen_ai.operation.name). One const per value so no
// call site repeats a literal. These correspond to memory-store operations.
const (
	// OpSearchMemory maps to engram's semantic search.
	OpSearchMemory = "search_memory"
	// OpUpsertMemory maps to engram's add/insert path (incl. dedup +
	// provenance-merge sub-steps, which are part of the upsert flow).
	OpUpsertMemory = "upsert_memory"
	// OpDeleteMemory maps to engram's delete path.
	OpDeleteMemory = "delete_memory"
	// OpListMemory maps to engram's list path.
	OpListMemory = "list_memory"
)

// GenAIAttrs returns the standard gen_ai.* attributes for a memory span given
// its operation-name value. These are emitted IN ADDITION to existing custom
// span attributes.
func GenAIAttrs(operationName string) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(AttrOperationName, operationName),
		attribute.String(AttrSystem, SystemEngram),
	}
}
