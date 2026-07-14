# Engram Config 7 — Chatbot Session Memory

**When to use this config**: You're building a chatbot that needs in-session context (dietary preferences, conversation history, temporary user state) but *not* permanent long-term memory. Memories expire automatically; the system is tuned for low latency over rich recall.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | `text-embedding-3-small` | Lowest latency; session context doesn't need deep semantic richness |
| Recency weight | `1.2` (above default `0.5`) | Recent turns matter more than old sessions for chatbots |
| Importance weight | `0.1` (below default `0.3`) | Chatbot context is uniformly low-importance; importance barely factors in |
| MMR lambda | `0.7` (above default `0.5`) | Favor relevance over diversity — session facts are often related |
| Dedup threshold | `0.88` (below default `0.92`) | Aggressively block near-paraphrases ("no meat" ≈ "I'm vegetarian") |
| Reflection | Disabled | No insight synthesis needed between chat turns |
| OTel sampling | `0.1` | 10% trace sampling reduces I/O at scale |

### TTL Auto-Calculator at Work

With no explicit `valid_until` set, Engram's TTL matrix controls expiration:

| Memory type | Importance | TTL |
|-------------|------------|-----|
| `event` (session facts) | 3–4 | **3 days** |
| `event` | 5–7 | 7 days |
| `insight` (rare, cross-session) | 5–7 | 90 days |
| `directive` (bot rules) | ≥8 | Permanent |

Session context (`event`, importance 3–4) auto-expires in 3 days — no manual cleanup needed.

---

## Architecture

```
User ──► Chatbot Server (your app)
               │
               │ HTTP REST (ENGRAM_TRANSPORT=http)
               │ Authorization: Bearer <ENGRAM_API_KEY>
               ▼
         Engram HTTP API :8080
               │
               ▼
         Qdrant (local) :6334
               └─► engram_user  (session context + user prefs)
```

Per-user isolation: tag every memory with `user:<user_id>` and filter on retrieval:
```
GET /memories/search?q=dietary+preferences&tags=user:abc123&limit=5
```

---

## Setup

### Step 1 — Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Step 2 — Configure

```bash
cp .env.example .env
# Edit: ENGRAM_OPENAI_API_KEY, ENGRAM_API_KEY
```

### Step 3 — Start Engram

```bash
set -a && source .env && set +a
./engram serve
```

Verify:
```bash
curl http://localhost:8080/health
# → {"status":"ok","qdrant":"ok"}
```

### Step 4 — Seed example session memories

```bash
pip install requests python-dotenv
python bootstrap.py
```

---

## Usage Pattern

### Write session context (on each conversation turn)

```python
import requests

ENGRAM_URL = "http://localhost:8080"
HEADERS = {"Authorization": "Bearer YOUR_KEY", "Content-Type": "application/json"}
USER_ID = "user:abc123"

def store_context(text: str, importance: int = 3):
    """Store a session fact. Auto-expires in 3 days at importance=3."""
    requests.post(f"{ENGRAM_URL}/memories", headers=HEADERS, json={
        "content": text,
        "type": "event",
        "importance": importance,
        "tags": [USER_ID, "session-context"],
    })
```

### Retrieve context (before generating a response)

```python
def get_context(query: str, user_id: str, limit: int = 5) -> list[str]:
    """Fetch the most relevant recent context for this user."""
    resp = requests.get(f"{ENGRAM_URL}/memories/search", headers=HEADERS, params={
        "q": query,
        "tags": user_id,
        "limit": limit,
    })
    return [m["content"] for m in resp.json().get("memories", [])]
```

### Example conversation flow

```python
# User says: "I want to order a pizza"
context = get_context("food order preferences", "user:abc123")
# → ["User is vegetarian (mentioned 2 turns ago)",
#    "User prefers thin crust",
#    "User allergic to mushrooms"]

# Store new preference mentioned this turn
store_context("User prefers extra cheese on their pizza", importance=4)
```

---

## Per-User Collection Isolation (Optional)

For strict data isolation between users (compliance / privacy), run one Engram instance per user or use separate Qdrant collections. See [multi-agent-shared-memory](../multi-agent-shared-memory/) for the multi-collection pattern.

For most chatbots, tag-based filtering is sufficient and simpler.

---

## Files

| File | Description |
|------|-------------|
| `.env.example` | Environment variable template — copy to `.env` |
| `bootstrap.py` | Seeds 12 example session memories across 3 simulated users |
| `README.md` | This file |

---

## Related Examples

| Config | Use Case |
|--------|----------|
| [single-agent-personal-memory](../single-agent-personal-memory/) | Single agent, persistent personal memory (Config 1) |
| [qdrant-cloud-production](../qdrant-cloud-production/) | Production deployment with Qdrant Cloud + TLS (Config 6) |
| [multi-agent-shared-memory](../multi-agent-shared-memory/) | Multiple agents with shared collection layer (Config 4) |
