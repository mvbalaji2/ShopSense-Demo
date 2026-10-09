# ShopSense AI

Submission materials, including the architecture diagram, 371-word CTO narrative, trade-off log, JSON triage examples, and three RAG examples are in [DELIVERABLES.md](./DELIVERABLES.md).

A local customer support intelligence demo for a fictional D2C brand. The inbox is an end-to-end pipeline: ticket text is classified, matched against a persisted local TF-IDF vector index, checked against a mock order dataset, routed through a safety decision, and used to draft a grounded response.

## Run locally

Run the local backend 
:

```bash
python3 server.py
```

Then visit `http://127.0.0.1:8000`. The server prefers `GEMINI_API_KEY` from the ignored local `.env` file and uses Gemini Flash Latest for optional generation.
## Walkthrough

1. Select a ticket in the inbox.
2. Review the four completed steps: triage and entities, RAG vector retrieval, order lookup, and decision with a suggested reply.
3. Use **Send & resolve** for an eligible response or **Assign to agent** when the policy or risk requires a human. **Edit reply** lets the agent change the draft first.
4. Add a ticket to run the same pipeline against new text. The architecture link opens the Day 30 client narrative.

## Pipeline and demo data

`rag.py` builds and persists a dependency-free TF-IDF vector index at `data/knowledge_index.json`, then returns cosine-ranked policy passages with scores. `server.py` owns the canonical pipeline: classification, vector retrieval, mock order lookup, deterministic safety decision, and optional Gemini Flash Latest drafting. The browser never receives the API key. If the model is unavailable or billing is not configured, the same server response falls back to a grounded local draft while preserving retrieval citations. Sensitive billing, damage, refund, return, and address requests stay with an agent; model output cannot bypass those rules.

This is a local demo with a production-shaped boundary, not a production deployment. When configured, each ticket analysis sends its message and the retrieved policy context to the configured model provider.
