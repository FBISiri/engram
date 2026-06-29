# Engram Single-Agent Personal Memory — Quickstart

**3 steps to a working personal memory store.**

## Step 1: Start Qdrant
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Step 2: Configure
```bash
cp .env.example .env
# edit ENGRAM_COLLECTION_NAME if you want a custom name
```

## Step 3: Seed example memories
```bash
pip install requests python-dotenv
python bootstrap.py
```

That's it. Your Qdrant collection now has 9 example memories (placeholder vectors).
Connect your agent and start building real memories on top.

## Next steps
- Replace placeholder vectors with real embeddings from your embedding model
- Read `/Engram/example-configs.md` in Obsidian for the full design rationale
- See `../` for other example configs (multi-agent, episodic, RAG-augmented)
