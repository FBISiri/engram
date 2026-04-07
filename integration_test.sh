#!/bin/bash
# integration_test.sh — Engram M4 Integration Test
# Tests the MCP server via stdio against a live Qdrant instance.
#
# Prerequisites:
#   - Qdrant running on localhost:6334 (gRPC)
#   - ENGRAM_OPENAI_API_KEY set
#
# Usage:
#   ENGRAM_OPENAI_API_KEY=sk-... ./integration_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/engram"
COLLECTION="engram_test_$(date +%s)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${YELLOW}→ $1${NC}"; }

FAILURES=0
TESTS=0

# Check prerequisites
if [ ! -f "$BINARY" ]; then
    echo "Binary not found. Building..."
    cd "$SCRIPT_DIR" && go build -o engram ./cmd/engram/
fi

if [ -z "${ENGRAM_OPENAI_API_KEY:-}" ]; then
    echo "ERROR: ENGRAM_OPENAI_API_KEY not set"
    echo "Usage: ENGRAM_OPENAI_API_KEY=sk-... $0"
    exit 1
fi

# Check Qdrant
if ! curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "ERROR: Qdrant not running on localhost:6333"
    echo "Start it: docker run -d --name engram-qdrant --security-opt seccomp=unconfined -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.9.7"
    exit 1
fi

info "Using test collection: $COLLECTION"

# Helper: send a JSON-RPC request to the MCP server via stdin/stdout
# Uses a co-process for persistent connection
export ENGRAM_QDRANT_URL=localhost:6334
export ENGRAM_COLLECTION_NAME="$COLLECTION"
export ENGRAM_OPENAI_API_KEY
export ENGRAM_OPENAI_BASE_URL="${ENGRAM_OPENAI_BASE_URL:-https://api.openai.com/v1}"
export ENGRAM_EMBEDDING_MODEL="${ENGRAM_EMBEDDING_MODEL:-text-embedding-3-small}"
export ENGRAM_TRANSPORT=stdio

info "Embedding endpoint: $ENGRAM_OPENAI_BASE_URL (model: $ENGRAM_EMBEDDING_MODEL)"

cleanup() {
    info "Cleaning up test collection: $COLLECTION"
    # Kill the co-process if running
    if [ -n "${COPROC_PID:-}" ]; then
        kill "$COPROC_PID" 2>/dev/null || true
        wait "$COPROC_PID" 2>/dev/null || true
    fi
    # Delete test collection
    curl -sf -X DELETE "http://localhost:6333/collections/$COLLECTION" > /dev/null 2>&1 || true
}
trap cleanup EXIT

# Start engram as a co-process
coproc ENGRAM { "$BINARY" serve 2>/dev/null; }
sleep 1

# Helper to send JSON-RPC and read response
send_rpc() {
    local method="$1"
    local params="$2"
    local id="$3"
    local request="{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":$params,\"id\":$id}"
    echo "$request" >&"${ENGRAM[1]}"
    # Read response line
    local response
    read -r -t 30 response <&"${ENGRAM[0]}" || { fail "Timeout waiting for response to $method"; return 1; }
    echo "$response"
}

# ============================================================================
# Test 1: Initialize MCP session
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 1: MCP Initialize"
INIT_RESP=$(send_rpc "initialize" '{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}' 1)
if echo "$INIT_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('result',{}).get('serverInfo',{}).get('name')=='Engram'" 2>/dev/null; then
    pass "Initialize: server identified as Engram"
else
    fail "Initialize: unexpected response: $INIT_RESP"
fi

# Send initialized notification
echo '{"jsonrpc":"2.0","method":"notifications/initialized"}' >&"${ENGRAM[1]}"
sleep 0.5

# ============================================================================
# Test 2: List Tools
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 2: List Tools"
TOOLS_RESP=$(send_rpc "tools/list" '{}' 2)
TOOL_COUNT=$(echo "$TOOLS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('result',{}).get('tools',[])))" 2>/dev/null || echo "0")
if [ "$TOOL_COUNT" -eq 4 ]; then
    pass "List Tools: found 4 tools (memory.search, memory.add, memory.update, memory.delete)"
else
    fail "List Tools: expected 4 tools, got $TOOL_COUNT. Response: $TOOLS_RESP"
fi

# ============================================================================
# Test 3: memory.add — store a memory
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 3: memory.add"
ADD_RESP=$(send_rpc "tools/call" '{"name":"memory.add","arguments":{"content":"User name is Frank, a software engineer in Beijing","type":"identity","importance":9,"tags":["name","occupation"]}}' 3)
ADD_STATUS=$(echo "$ADD_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); c=json.loads(d['result']['content'][0]['text']); print(c['status'])" 2>/dev/null || echo "error")
if [ "$ADD_STATUS" = "created" ]; then
    pass "memory.add: created identity memory"
else
    fail "memory.add: status=$ADD_STATUS. Response: $ADD_RESP"
fi

# ============================================================================
# Test 4: memory.add — store another memory of different type
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 4: memory.add (event type)"
sleep 1  # Rate limit
ADD2_RESP=$(send_rpc "tools/call" '{"name":"memory.add","arguments":{"content":"Discussed Engram project: Go rewrite of chat2mem, designed for open-source release","type":"event","importance":6,"tags":["topic","engram"]}}' 4)
ADD2_STATUS=$(echo "$ADD2_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); c=json.loads(d['result']['content'][0]['text']); print(c['status'])" 2>/dev/null || echo "error")
if [ "$ADD2_STATUS" = "created" ]; then
    pass "memory.add (event): created event memory"
else
    fail "memory.add (event): status=$ADD2_STATUS. Response: $ADD2_RESP"
fi

# ============================================================================
# Test 5: memory.add — dedup check
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 5: memory.add (duplicate check)"
sleep 1
DUP_RESP=$(send_rpc "tools/call" '{"name":"memory.add","arguments":{"content":"User name is Frank, a software engineer in Beijing","type":"identity","importance":9}}' 5)
DUP_STATUS=$(echo "$DUP_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); c=json.loads(d['result']['content'][0]['text']); print(c['status'])" 2>/dev/null || echo "error")
if [ "$DUP_STATUS" = "duplicate" ]; then
    pass "memory.add (dedup): correctly detected duplicate"
else
    fail "memory.add (dedup): expected 'duplicate', got '$DUP_STATUS'. Response: $DUP_RESP"
fi

# ============================================================================
# Test 6: memory.search — basic search
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 6: memory.search"
sleep 1
SEARCH_RESP=$(send_rpc "tools/call" '{"name":"memory.search","arguments":{"query":"What is the user name and job","limit":5}}' 6)
SEARCH_COUNT=$(echo "$SEARCH_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); c=json.loads(d['result']['content'][0]['text']); print(len(c))" 2>/dev/null || echo "0")
if [ "$SEARCH_COUNT" -ge 1 ]; then
    pass "memory.search: returned $SEARCH_COUNT results"
else
    fail "memory.search: expected >= 1 results, got $SEARCH_COUNT. Response: $SEARCH_RESP"
fi

# ============================================================================
# Test 7: memory.search — with type filter
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 7: memory.search (type filter)"
sleep 1
FILTERED_RESP=$(send_rpc "tools/call" '{"name":"memory.search","arguments":{"query":"user information","limit":5,"types":["identity"]}}' 7)
FILTERED_TYPE=$(echo "$FILTERED_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); c=json.loads(d['result']['content'][0]['text']); print(c[0]['type'] if c else 'none')" 2>/dev/null || echo "error")
if [ "$FILTERED_TYPE" = "identity" ]; then
    pass "memory.search (type filter): correctly filtered to identity type"
else
    fail "memory.search (type filter): expected 'identity', got '$FILTERED_TYPE'. Response: $FILTERED_RESP"
fi

# ============================================================================
# Test 8: memory.update — update existing memory
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 8: memory.update"
sleep 1
UPDATE_RESP=$(send_rpc "tools/call" '{"name":"memory.update","arguments":{"old_content":"User name is Frank software engineer Beijing","new_content":"User name is Frank, a senior software engineer in Shanghai","type":"identity","importance":9,"tags":["name","occupation","location"]}}' 8)
UPDATE_STATUS=$(echo "$UPDATE_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); c=json.loads(d['result']['content'][0]['text']); print(c['status'])" 2>/dev/null || echo "error")
if [ "$UPDATE_STATUS" = "updated" ]; then
    pass "memory.update: successfully updated memory"
else
    fail "memory.update: status=$UPDATE_STATUS. Response: $UPDATE_RESP"
fi

# ============================================================================
# Test 9: memory.search — verify update took effect
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 9: memory.search (verify update)"
sleep 1
VERIFY_RESP=$(send_rpc "tools/call" '{"name":"memory.search","arguments":{"query":"Frank Shanghai senior engineer","limit":3}}' 9)
VERIFY_CONTENT=$(echo "$VERIFY_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); c=json.loads(d['result']['content'][0]['text']); print(c[0]['content'] if c else '')" 2>/dev/null || echo "")
if echo "$VERIFY_CONTENT" | grep -q "Shanghai"; then
    pass "memory.search (verify update): found updated content with 'Shanghai'"
else
    fail "memory.search (verify update): expected 'Shanghai' in content. Got: $VERIFY_CONTENT"
fi

# ============================================================================
# Test 10: memory.delete — delete by semantic search
# ============================================================================
TESTS=$((TESTS + 1))
info "Test 10: memory.delete"
sleep 1
DELETE_RESP=$(send_rpc "tools/call" '{"name":"memory.delete","arguments":{"query":"Engram project Go rewrite chat2mem","similarity_threshold":0.5}}' 10)
DELETE_STATUS=$(echo "$DELETE_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); c=json.loads(d['result']['content'][0]['text']); print(c['status'])" 2>/dev/null || echo "error")
if [ "$DELETE_STATUS" = "deleted" ]; then
    pass "memory.delete: successfully deleted matching memories"
else
    fail "memory.delete: status=$DELETE_STATUS. Response: $DELETE_RESP"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================"
echo "Integration Test Results"
echo "================================"
echo "Total: $TESTS"
echo -e "Passed: ${GREEN}$((TESTS - FAILURES))${NC}"
if [ "$FAILURES" -gt 0 ]; then
    echo -e "Failed: ${RED}$FAILURES${NC}"
    exit 1
else
    echo -e "Failed: ${GREEN}0${NC}"
    echo ""
    echo "All integration tests passed! ✅"
    echo "Engram MCP server is ready for production use."
fi
