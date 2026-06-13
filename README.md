## Phase 4 — Production Hardening ✅

### What Was Built

1. **Action visibility** — per-node latency instrumentation + live status updates ("retrieving…", "grading…", "rewriting…") so users see the agent thinking instead of a frozen screen
2. **Mock LLM mode** — zero-token debugging; one env variable flips between real Gemini and instant canned replies for testing graph logic
3. **Company-specific scope** — documents ingested and router taught exactly what knowledge base contains
4. **Detailed router prompts** — clear descriptions of document topics dramatically improved routing accuracy (related questions properly identify whether info is in docs or requires general knowledge)
5. **Model tiering** — split LLM calls: gemini-2.5-flash for generation (needs to be smart), gemini-2.5-flash-lite for routing/grading/rewriting (small tasks, lower cost, faster)
6. **Input guardrails** — safety screening before routing; catches prompt injection attempts and refuses off-topic requests
7. **Semantic caching** — Chroma-backed answer cache; identical or near-duplicate questions return cached responses in <0.5s instead of re-running the full pipeline
8. **Safe invoke with retries** — every LLM call wrapped with automatic retry + fallback, so a single API hiccup doesn't crash the user's turn
9. **Over-retrieve philosophy** — if a question is about company topics but info isn't in docs, route to retrieval anyway; grader + fallback path tells the user "not found in documents, but here's what I know…" — more transparent than guessing upstream
10. **Evaluation & measurement** — golden dataset with routing accuracy baseline; RAGAS metrics (where feasible given token constraints)

### Results

| Metric | Value |
|--------|-------|
| **Routing accuracy** | 84% (16/19 questions) |
| **Latency (first retrieval question)** | 6–8 seconds |
| **Latency (cached repeat)** | <0.5 seconds |
| **Latency improvement** | 95% faster on cache hits |
| **Model calls per question** | Reduced from 5–7 to 3–4 via tiering |

### Key Lessons

- **Detailed prompts beat generic ones:** Routing jumped from 42% → 84% just by describing what documents actually contain. Prompts are architecture.
- **Over-retrieve beats under-retrieve:** Better to send borderline questions to retrieval and let grading reject them than to have the router guess and skip retrieval. Provides better UX and more honest answers.
- **Latency isn't just speed — it's UX:** When users see "retrieving… grading… rewriting…", the same 8-second wait feels productive instead of frozen. Transparency is a feature.
- **Caching semantically, not literally:** Token-matching cache is weak; semantic similarity (0.90 threshold) catches rephrasings and reduces API calls on follow-ups.
- **Mock dependencies let you debug for free:** Separating graph testing from model testing preserved quota while iterating on routing, grading, and retry logic.