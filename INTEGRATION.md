# Engram MCP Server Integration Guide
# ====================================
# How to replace chat2mem with Engram in Army of the Agent config.yaml
#
# CURRENT (chat2mem via remote SSE):
#
#   - name: "chat2mem"
#     transport: sse
#     url: "https://mem.postbook.xyz/sse"
#     access_token: "ThIZ5f5GfXGk4lzYi0XQuxrVAftESL6k-JPUGAXaqcM"
#
# REPLACEMENT (Engram via local stdio):
#
#   - name: "engram"
#     transport: stdio
#     command: /data/armyoftheagent/engram/engram
#     args: ["serve"]
#     env:
#       ENGRAM_QDRANT_URL: "localhost:6334"
#       ENGRAM_COLLECTION_NAME: "engram"
#       ENGRAM_OPENAI_API_KEY: "<your-openai-api-key>"
#       ENGRAM_OPENAI_BASE_URL: "https://api.openai.com/v1"
#       ENGRAM_EMBEDDING_MODEL: "text-embedding-3-small"
#       ENGRAM_EMBEDDING_DIMENSION: "1536"
#       ENGRAM_TRANSPORT: "stdio"
#
# TOOL NAME MAPPING:
# ==================
#   chat2mem                    →  Engram
#   ─────────────────────────────────────────
#   retrieve_memory             →  memory.search
#   add_memory                  →  memory.add
#   update_memory               →  memory.update
#   delete_memory               →  memory.delete
#   update_relationship         →  (use memory.update with tags)
#   list_due_followups          →  (use memory.search with tag filter)
#
# PARAMETER MAPPING:
# ==================
#   chat2mem retrieve_memory    →  Engram memory.search
#     query                     →  query
#     memory_type               →  types (array, mapped: fact→identity, topic/observation→event, reflection→insight)
#     limit                     →  limit
#     time_start                →  time_start
#     time_end                  →  time_end
#     event_date_from           →  (use tags or metadata)
#     event_date_to             →  (use tags or metadata)
#
#   chat2mem add_memory         →  Engram memory.add
#     content                   →  content
#     type                      →  type (mapped: fact/preference→identity, topic/observation/plan/study→event, reflection→insight)
#     importance                →  importance
#     event_date                →  (store in tags or metadata)
#
# MEMORY TYPE MAPPING:
# ====================
#   chat2mem                    →  Engram
#   ─────────────────────────────────────────
#   fact                        →  identity (tag: fact)
#   preference                  →  identity (tag: preference)
#   topic                       →  event (tag: topic)
#   observation                 →  event (tag: observation)
#   plan                        →  event (tag: plan)
#   study                       →  event (tag: study)
#   reflection                  →  insight
#   relationship                →  identity (tag: relationship)
#
# PREREQUISITES:
# ==============
#   1. Qdrant running on localhost:6334
#      docker run -d --name engram-qdrant --security-opt seccomp=unconfined \
#        -p 6333:6333 -p 6334:6334 \
#        -v engram_qdrant_data:/qdrant/storage \
#        qdrant/qdrant:v1.9.7
#
#   2. Engram binary built:
#      cd /data/armyoftheagent/engram && go build -o engram ./cmd/engram/
#
#   3. OpenAI API key for embeddings
#
# DATA MIGRATION:
# ===============
#   Migration tool (`engram migrate`) is planned for M5.
#   It will read from chat2mem's Qdrant collection (memory_stream)
#   and write to Engram's collection (engram) with type mapping.
