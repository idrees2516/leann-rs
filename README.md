# leann-rs

**Storage-efficient vector search + full-stack agentic RAG in Rust — with TypeScript bindings and IBM Granite Guardian hallucination elimination.**

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-83%20passing-brightgreen)]()

`leann-rs` is a from-scratch Rust implementation of the **LEANN** paradigm
([*LEANN: A Low-Storage Overhead Vector Index*, arXiv:2506.08276](https://arxiv.org/abs/2506.08276))
raised to a complete agentic-RAG platform with **memvid-level feature
coverage**, plus a deep implementation of **IBM's Granite Guardian**
([arXiv:2412.07724](https://arxiv.org/abs/2412.07724)) for hallucination
elimination.

```
   documents (pdf/txt/md/html/docx/epub/json/code/audio/video)
        │  leann-ingest      format detection → parsing → token-aware chunking
        ▼
   chunks + metadata
        │  leann-embedder    OpenAI-compatible APIs or zero-dep local embedder
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  leann-core — the LEANN index                                 │
   │  • HNSW graph + high-degree-preserving pruning (β% hubs, M vs M/5) │
   │  • PQ codes (~96× compression), NO stored embeddings          │
   │  • two-level search: PQ approximate queue ∥ exact recompute   │
   │  • dynamic batching across graph hops (adaptive target size)  │
   │  • BM25 + reciprocal-rank fusion hybrid search                │
   │  • metadata filters, context expansion, add/delete, mmap file │
   └──────────────────────────────────────────────────────────────┘
        │  leann-guard (IBM Granite Guardian)
        │  context relevance → groundedness → answer relevance → harm
        │  sentence-level attribution → grounded rewrite → repair loop
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  leann-chat — RAG layer: LLM providers (OpenAI/Anthropic),     │
   │  cited context assembly, rolling memory, guarded answers       │
   └──────────────────────────────────────────────────────────────┘
        │
        ├── leann-node  → napi-rs TypeScript bindings (Node 16+)
        └── leann-cli   → `leann add | search | chat | guard | stats`
```

## Why

Conventional vector indexes store high-dimensional embeddings — several times
larger than the original data. LEANN flips the trade: **store nothing but a
pruned graph + PQ codes, recompute embeddings on the fly** for the handful of
candidates the search actually touches. On top of that core, this repo adds
everything an agentic RAG system needs: every popular document format, hybrid
lexical+semantic retrieval, LLM chat with citations, and IBM-guardian-grade
hallucination elimination.

| | conventional index | **leann-rs** |
|---|---|---|
| Stored vectors | `N × dim × 4` bytes (e.g. 1536-dim → 6KB/chunk) | **0** (PQ codes: `m` bytes ≈ 16–32/chunk) |
| Graph metadata | full HNSW adjacency | hub-pruned (~80% fewer edges) |
| Query cost | pure distance math | recompute embeddings for ~α fraction of visited nodes (batched, cached) |
| Lexical search | usually separate | BM25 built-in + RRF fusion |
| Hallucination guard | none | IBM Granite Guardian pipeline |

## Crates

| Crate | What it does |
|---|---|
| **`leann-core`** | The LEANN index: metrics (L2/cosine/IP), PQ codec with parallel training, HNSW build with RNG pruning, **Algorithm 3** hub-preserving pruning, **Algorithm 2** two-level search (AQ/EQ + promotion ratio α), dynamic batching, sharded merge build, BM25, RRF/weighted fusion, JSON metadata filters, neighbor context expansion, dynamic add/delete, atomic single-file mmap format |
| **`leann-embedder`** | `OpenAiEmbedder` (any `/embeddings` endpoint: OpenAI, Ollama, vLLM, SiliconFlow, …; batching, parallel requests, 429/5xx retries) + `LocalHashEmbedder` (deterministic word/bigram/trigram feature hashing — zero deps, zero network) |
| **`leann-ingest`** | PDF (`pdf-extract`), TXT/LOG, Markdown (heading-aware), HTML (script/style-stripping extractor), DOCX (OOXML), EPUB (spine-ordered), JSON/JSONL, **30+ code languages** (definition-boundary chunking with line ranges), **audio** (ffmpeg → 16k wav → Whisper-compatible transcription, timestamped chunks), **video** (audio track + optional VLM keyframe descriptions). Token-aware chunking with overlap, page/heading/timestamp metadata everywhere |
| **`leann-chat`** | OpenAI-compatible + Anthropic LLM providers, RAG prompt assembly with `[n]` citations, context budgeting, rolling conversation memory, vision (frame description) |
| **`leann-guard`** | **IBM Granite Guardian (arXiv:2412.07724) in depth**: the full risk catalog (groundedness, context relevance, answer relevance, social bias, profanity, violence, sexual content, unethical behavior, jailbreak, function-call hallucination, BYOC), the guardian-block judging protocol (think/no-think modes, `<score>` parsing, verbalized confidence), OpenAI-compatible judge client that talks to **served granite-guardian models** or any judge LLM, deterministic local pre-filters (number consistency, term coverage, citation validation), **sentence-level attribution**, **grounded rewriting** (elimination), composite `RagGuard` gate (Abstain/Repair/Annotate/Log policies) and a generate→guard→repair loop. Fully standalone — usable from any app |
| **`leann-node`** | napi-rs bindings: `Leann` class (addFiles/addTexts/search/hybridSearch/chat with guard/delete/save), `Collections` (multi-user isolation), standalone `Guardian` class, `verifyAnswer` audits |
| **`leann-cli`** | `leann add/search/chat/guard/stats/delete` |

## Quick start

### Rust

```sh
cargo build --release
cargo test --workspace          # 83 tests
cargo run --release -p leann-core --example quickstart
```

```rust
use leann_core::index::{LeannIndex, IndexConfig, HybridOptions, SearchOptions};
use leann_embedder::LocalHashEmbedder;
use std::sync::Arc;

let idx = LeannIndex::build(
    &texts,                       // Vec<String> chunks
    Some(metadatas),              // Vec<serde_json::Value>
    Arc::new(LocalHashEmbedder::new(384)),
    IndexConfig::default(),
)?;

let hits = idx.search("hub pruning", &SearchOptions { k: 5, ..Default::default() })?;
let fused = idx.hybrid_search("M edges graph", &HybridOptions::default())?; // + BM25 + RRF
idx.save("my.leann")?;            // single portable file, mmap-loaded
```

### CLI

```sh
# offline, zero-config (local hash embedder)
export LEANN_EMBEDDER=local

leann add docs/ src/ paper.pdf notes.md recording.mp4   # ingest any formats
leann search "high-degree preserving pruning" --hybrid --k 5
leann search "two level search" --expand                 # neighbor context
leann stats                                              # storage saved vs conventional

# audit any answer for hallucinations (local deterministic mode)
leann guard answer.txt paper.pdf --local --policy repair

# guarded RAG chat (needs an LLM; optional guardian server for full checks)
export OPENAI_API_KEY=sk-...
leann chat "How does LEANN reduce storage?" --guard \
  --guard-base-url http://localhost:11434/v1 \
  --guard-model ibm-granite/granite-guardian-3.3-8b
```

### TypeScript / Node

```sh
cd npm && npm install && npm run build   # napi build --platform --release
```

```ts
import { Leann, Guardian, localGrounding, localVerify, localRewrite } from "leann-node";

const leann = new Leann({
  embedder: { kind: "local-hash", dim: 384 },
  // llm: { model: "gpt-4o-mini" },                       // enables chat()
  // guardian: { baseUrl: "http://localhost:11434/v1",
  //             model: "ibm-granite/granite-guardian-3.3-8b" },
});

await leann.addFiles(["paper.pdf", "notes.md", "src/main.rs", "meeting.mp4"]);

const res = await leann.hybridSearch("hub preserving pruning", {
  k: 5, expandBefore: 1, expandAfter: 1,
});

// RAG chat with built-in hallucination elimination:
const answer = await leann.chat("How does LEANN reduce storage?", {
  guard: { policy: "repair", repairAttempts: 2 },
});
console.log(answer.answer, answer.guard);
// { riskTriggered: false, action: "rewritten", repaired: true, ... }

// Standalone guardian (any app, no index needed):
const guardian = new Guardian({
  baseUrl: "http://localhost:11434/v1",
  model: "ibm-granite/granite-guardian-3.3-8b",
});
const verdict = await guardian.checkGroundedness(
  "The paper was published in 2031.",   // ← fabricated
  ["LEANN ... arXiv:2506.08276 (2025)"],
);
// verdict.triggered === true (ungrounded)

// Zero-dependency offline layer (deterministic, no judge server):
const audit = localGrounding(
  "The budget is 9.7 million dollars.",
  ["LEANN keeps quantized codes in RAM."],
);
// audit.grounded === false, audit.missingNumbers === ["9.7", "million"]

const sentences = localVerify(                       // per-sentence verdicts
  "Raw vectors stay on disk. The budget is 9.7 million dollars.",
  ["LEANN keeps quantized codes in RAM; raw vectors stay on disk."],
);
// [ { grounded: true, ... }, { grounded: false, missingTerms: ["budget", "9.7", ...] } ]

const clean = localRewrite(                          // ungrounded sentences removed
  "Raw vectors stay on disk. The budget is 9.7 million dollars.",
  ["LEANN keeps quantized codes in RAM; raw vectors stay on disk."],
);
// "Raw vectors stay on disk."
```

## The LEANN core (paper → code)

| Paper concept | Where | Notes |
|---|---|---|
| No stored embeddings; recompute at query time | `leann-core::cache` | `EmbeddingProvider` trait + sharded LRU; misses batch into single provider calls |
| PQ ~100× compression | `leann-core::pq` | parallel per-subspace k-means, ADC tables, auto-`m` for dim |
| RNG neighbor shrinking (Alg. 1) | `graph.rs::select_rng` | prune `x` when a kept `a` is closer to `x` than `v` |
| High-degree preserving pruning (Alg. 3) | `graph.rs::hub_prune` | top β% hubs keep `M`; others `m = M/5`; hub targets always kept |
| Two-level search (Alg. 2) | `search.rs::two_level_search` | AQ (PQ) + EQ (exact); per-hop promotion of top α fraction |
| Dynamic batching (§4.2) | `batch.rs::DynamicBatcher` | adaptive target size from measured vectors/sec |
| Sharded merge build (§6) | `index.rs::build_sharded` | k-means soft assignment (2 shards/doc) → per-shard graphs → union |
| ADD with distance caching (Alg. 4 / App. B) | `index.rs::add` | cache pre-warm via batched searches before insertion |

Measured on the paper's own PDF (38 chunks, local embedder): **96× PQ
compression, 608 bytes of codes vs 58 KB of conventional vectors**, hybrid
search in ~15 ms cold.

## The IBM Granite Guardian implementation (leann-guard)

Implements the paper's judging protocol faithfully and pushes it from
*detection* to *elimination*:

1. **Guardian blocks** — exact prompt protocol: `IMD`/`<no-think>` mode
   selector, `### Criteria:`, `### Scoring Schema:` with `<score>yes/no</score>`
   outputs and verbalized confidence.
2. **Full risk catalog** — all RAG risks (context relevance, groundedness,
   answer relevance), all harm risks, jailbreak, function-call hallucination,
   and BYOC.
3. **Pluggable judges** — point `OpenAiJudge` at a served
   `ibm-granite/granite-guardian-3.3-8b` (vLLM/Ollama) or use any capable LLM
   as judge. `MockJudge` ships deterministic heuristics for offline/tests.
4. **Deterministic pre-filters** — number consistency (fabricated dates,
   quantities), term-coverage grounding, citation validation — catch the
   obvious hallucinations before spending judge calls.
5. **Sentence-level attribution & grounded rewrite** — decompose answers,
   verify each sentence, drop the ungrounded ones: the answer that ships is
   the answer that's supported.
6. **Composite gate + repair loop** — `RagGuard` (context relevance →
   groundedness → answer relevance → harm) under Abstain / Repair / Annotate /
   Log policies; `GuardedChat` regenerates with judge feedback when checks
   fail.

```rust
use leann_guard::{OpenAiJudge, Guardian, RagGuard, GuardPolicy};

let judge = OpenAiJudge::new("http://localhost:11434/v1", "",
                             "ibm-granite/granite-guardian-3.3-8b");
let guardian = Guardian::with_judge(judge);
let guard = RagGuard::new(guardian).with_policy(GuardPolicy::Repair);

let decision = guard.check_rag_answer(&query, &candidate_answer, &retrieved_docs)?;
println!("{}", decision.answer);                  // grounded subset or refusal
println!("{:?}", decision.report.action);          // Passed | Rewritten | Rejected
```

## Storage format

One file, atomic writes (tmp + rename), mmap-loaded with zero-copy texts:

```
LEANNIDX | version | metric | dim | pq_m | ksub | meta-json
metadatas-json | text offsets + blob
pq codebook (f32) | pq codes (m bytes/chunk)
graph: levels + upper lists + CSR adjacency | tombstones bitmap
```

## Configuration

Embedder / LLM / transcriber / guardian all resolve from env when not given
explicitly: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LEANN_EMBED_MODEL`,
`LEANN_LLM_BASE_URL`, `LEANN_LLM_API_KEY`, `LEANN_LLM_MODEL`,
`ANTHROPIC_API_KEY`, `LEANN_EMBEDDER=local` (offline mode).

## Benchmarks

```sh
cargo bench -p leann-core   # PQ train/encode/ADC, graph build/prune/search,
                            # two-level search @ 2k/10k, end-to-end index ops
```

## Testing

83 unit + integration tests across the workspace, including:
- recall@k of two-level search vs brute force (≥0.8 asserted)
- hub-prune recall preservation, sharded ≈ single-build parity
- save/load byte-identical results, add/delete dynamics
- parsers for every format (real DOCX/EPUB zips built in-test)
- BM25 ranking, RRF fusion, filters, context expansion
- guardian block structure, score parsing, sentence attribution,
  grounded rewriting, all four guard policies, repair loops

## License

Apache-2.0. Based on the public research of the LEANN authors (UC Berkeley et
al.) and the IBM Granite Guardian team; all code here is an independent
implementation.
