# The Problem Engram Solves

The most intuitive way to give an agent memory is: embed every message, fact, and log line,
push them all into a vector database, and semantic-search top-k when you need something. This
runs great for ten minutes — and starts to rot around week three.

## Four failure modes of "just a vector DB"

**1. Context rot.** A vector store only appends. Three months in, it holds tens of thousands
of vectors, and a single retrieval pulls a top-k laced with tangential noise — a 0.6-similar
stale fragment competes for the same slot as today's genuinely relevant fact. Recall didn't
drop, but the signal-to-noise ratio fell through the floor.

**2. No forgetting.** Human memory decays naturally with time and importance; a vector DB does
not. A throwaway greeting from six months ago and a hard constraint like *"always ask before
changing production config"* carry identical weight in the index. When what should be forgotten
never fades, memory becomes a junkyard.

**3. No importance.** Retrieval only knows semantic similarity. It cannot express *"this one
matters more than that one."* So a critical directive can be buried under ten semantically
closer but practically irrelevant chit-chat entries.

**4. Duplicate fragments.** The same fact, mentioned across different conversations, gets
written repeatedly. Ten near-identical vectors describing one thing waste retrieval budget and
let paraphrases crowd out the top-k, pushing out information that should have surfaced.

Add these up and the conclusion is clear: **a vector database is a retrieval primitive, not a
memory system.** It solves "find semantically similar things in a pile." An agent's long-term
memory needs to answer a different set of questions.

## What an agent's memory actually needs to answer

1. **Is this worth remembering?** — writes need a threshold, not open-door admission.
2. **Have I already remembered this?** — writes need dedup, not piles of synonymous fragments.
3. **Which memories matter more, and which should fade?** — retrieval must rank by
   importance × recency × relevance, not similarity alone.
4. **What kind of memory is this** — a permanent fact about "who I am," or a fleeting event?
   Different kinds of memory deserve different decay rates and recall weights.

Engram is built directly at these four questions. It is not another wrapper over a vector DB;
it is a system for the *lifecycle* of agent memory — write gate, semantic dedup, importance
scoring, type-based time decay, retrieval fusion, and reflection. It still uses vector search
(Qdrant) underneath, but the vector DB is one part, not the whole.

## Positioning: vs vector DB, RAG, and KV memory

**vs a vector database (Qdrant / pgvector / Pinecone…).** The vector DB is Engram's *storage
backend*, not a substitute. The difference is the governance layer on top: the vector DB gives
you "find similar by semantics" but doesn't gate writes, dedup, weigh importance, forget, or
reflect. Think of Engram as "a vector DB with memory discipline."

**vs RAG.** RAG is a *read-time* behavior: at query time it fetches a context chunk from an
external document store, splices it into the prompt, and discards it. Engram is *persistent
memory* that governs quality at *write time*: information is deduped, scored, and typed the
moment it's written, then decays over time and gets synthesized by reflection. RAG solves "my
knowledge lives in external documents"; Engram solves "my agent accumulates and must maintain
its own memory over time." The two are orthogonal and can be used together.

**vs plain KV / session memory.** KV memory relies on exact key hits — no semantic recall, no
importance, no dedup; it's essentially keyed storage. Every Engram memory is semantically
retrievable, importance-weighted, type-forgotten, and deduped on write.

In one line: **a vector DB helps you "find similar things"; Engram helps you "remember what
should be remembered, forget what should be forgotten, and surface the right memory at the
right moment."** That's the difference.
