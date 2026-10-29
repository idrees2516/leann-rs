# leann-rs

**Storage-efficient vector search + full-stack agentic RAG in Rust — with TypeScript bindings and IBM Granite Guardian hallucination elimination.**

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-131%20passing-brightgreen)]()

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
        ├── leann-visual → QR-code video memory: the corpus as a portable MP4/GIF
        │                  (memvid-style) with random-access frame retrieval
        ├── leann-server → REST API + web playground (auth, rate limit, CORS)
        ├── leann-node   → napi-rs TypeScript bindings (Node 16+)
        └── leann-cli    → `leann add | search | chat | guard | stats | export |
                            visual build | visual query | serve | mcp`
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
| **`leann-core`** | The LEANN index: **SIMD distance kernels** (AVX-512/AVX2/SSE runtime-detected, 0.05 ns/dim), PQ codec with parallel training, HNSW build with RNG pruning, **Algorithm 3** hub-preserving pruning, **Algorithm 2** two-level search (AQ/EQ + promotion ratio α, allocation-free scratch), dynamic batching, **parallel batch search**, sharded merge build, BM25, RRF/weighted fusion, JSON metadata filters (depth/size-limited), neighbor context expansion, dynamic add/delete, **compaction (tombstone GC)**, **exact + SimHash near-dedup**, **ChaCha20-Poly1305 encryption at rest**, SSRF-hardened URL validation, atomic single-file mmap format, **export/import (JSON / CSV / Markdown)** |
| **`leann-embedder`** | `OpenAiEmbedder` (any `/embeddings` endpoint: OpenAI, Ollama, vLLM, SiliconFlow, …; batching, parallel requests, 429/5xx retries) + `LocalHashEmbedder` (deterministic word/bigram/trigram feature hashing — zero deps, zero network) |
| **`leann-ingest`** | PDF (`pdf-extract`), TXT/LOG, Markdown (heading-aware), HTML (script/style-stripping extractor), DOCX (OOXML), EPUB (spine-ordered), JSON/JSONL, **55+ source/config formats** (Rust, Python, TS/JS, Go, Java, C/C++, C#, Ruby, PHP, Swift, Kotlin, Scala, Shell, SQL, Lua, R, Perl, Elixir, Haskell, Clojure, Dart, Zig, Nim, Vue, Svelte, TOML, YAML, XML, CSV, CSS, INI, Protobuf, GraphQL, LaTeX, Solidity, Assembly, Batch, PowerShell…; definition-boundary chunking with line ranges), **audio** (ffmpeg → 16k wav → Whisper-compatible transcription, timestamped chunks), **video** (audio track + optional VLM keyframe descriptions). Token-aware chunking with overlap, page/heading/timestamp metadata everywhere |
| **`leann-chat`** | OpenAI-compatible + Anthropic LLM providers, RAG prompt assembly with `[n]` citations, context budgeting, rolling conversation memory, vision (frame description) |
| **`leann-guard`** | **IBM Granite Guardian (arXiv:2412.07724) in depth**: the full risk catalog (groundedness, context relevance, answer relevance, social bias, profanity, violence, sexual content, unethical behavior, jailbreak, function-call hallucination, BYOC), the guardian-block judging protocol (think/no-think modes, `<score>` parsing, verbalized confidence), OpenAI-compatible judge client that talks to **served granite-guardian models** or any judge LLM, deterministic local pre-filters (number consistency, term coverage, citation validation), **sentence-level attribution**, **grounded rewriting** (elimination), composite `RagGuard` gate (Abstain/Repair/Annotate/Log policies) and a generate→guard→repair loop. Fully standalone — usable from any app |
| **`leann-visual`** | **QR-code video memory** (memvid parity, in Rust): encode the corpus as one portable **MP4** (lossless H.264 via ffmpeg, all-intra frames for instant seeks) or **pure-Rust GIF** (zero system deps, auto-fallback) — one QR per frame, chunks auto-split across frames with UTF-8-safe framing, sidecar maps ids → frames (text lives *only* in the video), sub-second random-access retrieval, semantic search that decodes hit text straight out of the video |
| **`leann-server`** | **Production REST API** (axum): multi-collection store, search/hybrid/chat/guard, text+file ingestion, delete-by-filter, **export/import (JSON/CSV/Markdown)**, visual-memory build + search, stats — with bearer/X-API-Key auth (constant-time compare), per-client token-bucket rate limiting, strict name/path validation, configurable body limit + CORS, `spawn_blocking` around every index/LLM call, and an embedded **web playground** at `GET /` (zero build step) |
| **`leann-node`** | napi-rs bindings: `Leann` class (addFiles/addTexts/search/**searchBatch** (parallel)/hybridSearch/chat with guard/delete/**compact**/**saveEncrypted**/**openEncrypted**/**addTextsDedup**/**exportCorpus**/**importJson**), `Collections` (multi-user isolation, traversal-safe), standalone `Guardian` class, `verifyAnswer` audits, offline `localGrounding`/`localVerify`/`localRewrite`, `simdLevel()` |
| **`leann-cli`** | `leann add/search/chat/guard/stats/delete/compact/export/visual/serve` + **`leann mcp`**: a Model Context Protocol stdio server exposing 8 tools (search, hybrid search, ingest, RAG chat, hallucination audit, stats, delete-where, compact) — drop LEANN into Claude Desktop or any MCP client as a retrieval tool |

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

# maintenance: garbage-collect deleted chunks
leann compact

# export the live corpus (round-trips through importJson / the REST API)
leann export json --out corpus.json
leann export csv  --out corpus.csv
leann export md   --out corpus.md

# QR-code video memory: the whole corpus as one portable video (memvid, in Rust)
leann visual build --output memory.mp4          # lossless H.264 via ffmpeg
leann visual query "hub pruning" --video memory.mp4   # text decoded from the video

# real-world exposure: REST API + web playground at http://localhost:8080/
leann serve --port 8080 --data-dir ./leann-data --api-key my-secret

# agent-native: run LEANN as an MCP tool server (Claude Desktop, Cursor, ...)
leann mcp --index ./knowledge.leann
# exposes: leann_search, leann_add_files, leann_add_texts, leann_chat,
#          leann_guard, leann_stats, leann_delete_where, leann_compact
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

## QR-code video memory — `leann-visual`

The memvid signature feature, ported to Rust with lossless retrieval: every
chunk becomes a QR code on one video frame; a JSON sidecar maps chunk ids →
frame ranges. **The chunk text exists only in the video** — the file you share,
archive, or play is the database.

```rust
use leann_visual::{build_visual_from_index, VisualRetriever, VisualOptions};

// 1) encode an index's live corpus into memory.mp4 (+ memory.mp4.leannvid.json)
let stats = build_visual_from_index(&idx, "memory.mp4".as_ref(),
                                    VisualOptions::default())?;

// 2) random access + semantic search, text decoded out of the video
let retriever = VisualRetriever::open("memory.mp4".as_ref())?;
let chunk = retriever.get_chunk(0)?;                    // frame-accurate seek
for hit in retriever.search(&idx, "hub pruning", 5)? {  // LEANN search + QR decode
    println!("frame {}: {}", hit.frame, hit.text);
}
```

Why it's safe to put text in pixels:

* **Containers** — MP4 via ffmpeg (`-qp 0` mathematically-lossless H.264,
  `-g 1` all-intra so `-ss` seeks are instant) when available; pure-Rust GIF
  otherwise (auto-selected by `VideoFormat::Auto`, so it works everywhere).
* **Framing** — each QR frame carries a 5-byte ASCII-safe header
  (`last/seq/len`, 7-bit fields); oversized chunks auto-split across
  consecutive frames with a shrinking-size ladder that guarantees every
  frame is QR-encodable *and* valid UTF-8 (string decoders like rqrr and
  phone cameras depend on this).
* **Verification** — decoded chunks are byte-compared to the source in tests,
  including multi-frame CJK/emoji payloads and a 356-chunk self-ingest of
  this repository's source code (see `scripts/e2e_self.sh`).

## REST API server — `leann-server`

One binary exposes the whole stack to real-world systems (any language, any
platform): multi-collections, retrieval, guarded RAG chat, ingestion,
export/import and QR-video memory, plus a built-in web playground.

```sh
export LEANN_EMBEDDER=local            # offline embedder; OPENAI_API_KEY for quality
leann serve --port 8080 --data-dir ./leann-data --api-key my-secret
# → http://localhost:8080/            web playground (search / chat / stats)
```

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | liveness + version + auth mode |
| `GET` / `POST` | `/collections` | list / create collections |
| `GET` / `DELETE` | `/collections/{name}` | stats / drop a collection |
| `POST` | `/collections/{name}/texts` | add texts (+ optional metadata, dedup) |
| `POST` | `/collections/{name}/files` | ingest files (path-scoped: `LEANN_INGEST_ROOT`) |
| `POST` | `/collections/{name}/search` | vector search (k, JSON filter) |
| `POST` | `/collections/{name}/hybrid` | vector + BM25 + RRF fusion |
| `POST` | `/collections/{name}/chat` | RAG chat with citations (optional `guard: true`) |
| `POST` | `/collections/{name}/guard` | offline groundedness audit of any answer |
| `POST` | `/collections/{name}/delete` | delete-by-filter |
| `GET` | `/collections/{name}/export?format=json\|csv\|md` | export the live corpus |
| `POST` | `/collections/{name}/import` | import a `leann-export/v1` document |
| `POST` | `/collections/{name}/visual` | build the QR-video memory (mp4/gif/auto) |
| `POST` | `/collections/{name}/visual-search` | search; hit text decoded from the video |

```sh
curl -H "x-api-key: my-secret" -H 'content-type: application/json' \
     -X POST localhost:8080/collections/self/search \
     -d '{"query": "hub preserving pruning", "k": 3}'
```

Security posture: optional bearer/X-API-Key auth (constant-time compare),
per-client token-bucket rate limiting (default 240 req/min, tunable), strict
collection-name validation (no path traversal), file ingestion confined to
`LEANN_INGEST_ROOT`, 64 MiB body cap, CORS allow-list, and every index/LLM
call wrapped in `spawn_blocking` so slow embedders can never stall the
runtime. See `examples/rest-demo.sh` for a full walkthrough.

## Storage format

One file, atomic writes (tmp + rename), mmap-loaded with zero-copy texts:

```
LEANNIDX | version | metric | dim | pq_m | ksub | meta-json
metadatas-json | text offsets + blob
pq codebook (f32) | pq codes (m bytes/chunk)
graph: levels + upper lists + CSR adjacency | tombstones bitmap
```

**Encrypted at rest** (optional, memvid-parity): `save_encrypted(path, passphrase)`
seals the entire payload with ChaCha20-Poly1305 under an Argon2id-derived key
(`LEANNENC` magic). Texts, codes, graph and metadata are all inside the seal —
a stolen file leaks nothing, and tampering fails authentication:

```rust
idx.save_encrypted("bank.leann", "correct horse battery staple")?;
let idx = LeannIndex::open_with_passphrase(
    "bank.leann", "correct horse battery staple", provider)?;
```

## Configuration

Embedder / LLM / transcriber / guardian all resolve from env when not given
explicitly: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LEANN_EMBED_MODEL`,
`LEANN_LLM_BASE_URL`, `LEANN_LLM_API_KEY`, `LEANN_LLM_MODEL`,
`ANTHROPIC_API_KEY`, `LEANN_EMBEDDER=local` (offline mode).

## Performance

Measured with the criterion suite (`cargo bench -p leann-core`), 2-core
x86_64 (AVX-512), quick mode:

| Kernel | Result | Note |
|---|---|---|
| `dot` dim=768 (AVX-512) | **40 ns** (0.05 ns/dim) | 2.5× faster than portable |
| `dot` dim=1536 | 87 ns | 2.4× faster than portable |
| `l2_sq` dim=768 | 48 ns | 2.1× faster than portable |
| PQ ADC distance (m=32) | **26 ns** per vector | LUT: `m` adds + lookups |
| PQ ADC table build | 68 µs | once per query |
| Two-level search, 10k chunks, warm cache | **91 µs** | reused scratch: −10 µs alloc overhead |
| Hybrid search (vector + BM25 + RRF), 5k chunks | 274 µs | |
| SimHash (near-dup check) | 2.8 µs per chunk | |
| BM25 search, 5k chunks | 362 µs | full corpus scan, top-10 |
| RRF fusion (2 × 10 ranked lists) | **192 ns** | |
| Real-repo self-ingest corpus, vector search | 21 µs | 90 chunks of this repo's own source |
| Real-repo self-ingest corpus, hybrid search | 28 µs | incl. BM25 + fusion |
| Argon2id key derivation | 17 ms | once per open, by design |
| ChaCha20-Poly1305 seal/unseal, 64 KiB | 49 µs | |

Steady-state search is **allocation-free**: the hot loop runs on
generation-stamped flat arrays (no HashMap/HashSet traffic), the embed cache
is sharded across threads, and batch queries fan out across rayon workers
with per-thread scratch. `searchBatch` embeds all queries in one provider
call, then searches them in parallel — on this 2-core box a 64-query batch
runs in 10.4 ms (~160 µs/query amortized); scaling follows cores.

```sh
cargo bench -p leann-core   # SIMD kernels, PQ, graph, two-level search,
                            # hot-path (fresh vs reused scratch), batch
                            # parallelism, dedup + crypto, BM25 + fusion,
                            # real_repo (this repository's own source)
```

## Testing

131 unit + integration tests across the workspace, including:
- SIMD kernels agree with portable loops across dims (incl. odd/tail lengths)
- scratch-reuse search ≡ fresh-scratch search (generation-stamp correctness)
- compaction: tombstone GC + id remap + save/load roundtrip (regression)
- encryption: roundtrip, wrong-passphrase, bit-flip, truncation rejection
- dedup: exact + near (calibrated thresholds), short-text fallback
- URL validation (SSRF), filter depth/size limits, path traversal
- MCP server schema + live JSON-RPC session
- recall@k of two-level search vs brute force (≥0.8 asserted)
- hub-prune recall preservation, sharded ≈ single-build parity
- save/load byte-identical results, add/delete dynamics
- parsers for every format (real DOCX/EPUB zips built in-test)
- BM25 ranking, RRF fusion, filters, context expansion
- guardian block structure, score parsing, sentence attribution,
  grounded rewriting, all four guard policies, repair loops
- export/import: JSON round-trip, RFC 4180 CSV quoting, Markdown dump
- QR-video memory: MP4 (ffmpeg) + GIF (pure-Rust) frame roundtrips,
  multi-frame chunk stitching (CJK/emoji), UTF-8-safe framing regression,
  deleted-chunk skipping, capacity guards, search-from-video
- REST API: full lifecycle (create → add → search/hybrid → filter delete →
  stats → export/import → guard → visual build/search), auth (401 paths,
  constant-time compare), rate limiting (429), collection-name validation,
  path-scoped file ingestion (403 outside LEANN_INGEST_ROOT)
- real-data E2E (`scripts/e2e_self.sh`): the repo ingests its own source
  (356 chunks), searches it, builds a 356-frame MP4 memory, retrieves text
  from the video, and drives the REST server over HTTP

## License

Apache-2.0. Based on the public research of the LEANN authors (UC Berkeley et
al.) and the IBM Granite Guardian team; all code here is an independent
implementation.
