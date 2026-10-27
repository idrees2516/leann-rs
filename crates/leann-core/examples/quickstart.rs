//! End-to-end LEANN example: build a hybrid index over mixed-format
//! documents and run guarded RAG.
//!
//! ```sh
//! cargo run --release --example quickstart
//! ```

use serde_json::json;
use std::sync::Arc;

use leann_chat::rag::GuardSettings;
use leann_core::index::{HybridOptions, LeannIndex, SearchOptions};
use leann_core::metric::Metric;
use leann_embedder::LocalHashEmbedder;
use leann_guard::{GuardPolicy, Guardian};
use leann_ingest::{ingest_text, IngestOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Prepare mixed-format content (normally: leann_ingest::ingest_files
    //    for pdf/docx/epub/audio/video/code).
    let mut texts: Vec<String> = Vec::new();
    let mut metas: Vec<serde_json::Value> = Vec::new();
    let opts = IngestOptions::default();

    for (i, (doc, fmt)) in [
        ("# Storage\n\nLEANN prunes the graph: hubs keep up to M edges, others m=M/5. Storage drops by 50x.", "markdown"),
        ("Plain text: two-level search interleaves PQ approximate distances with exact recomputation, alpha=0.08.", "txt"),
        ("fn main() { println!(\"code chunks keep line ranges\"); }", "code"),
    ]
    .iter()
    .enumerate()
    {
        for c in ingest_text(doc, &format!("example-{i}.{fmt}"), fmt, &opts) {
            texts.push(c.text);
            let mut m = c.metadata;
            m["seq"] = json!(i);
            metas.push(m);
        }
    }

    // 2. Build the index (local zero-dependency embedder for the example).
    let embedder = Arc::new(LocalHashEmbedder::new(384));
    let cfg = leann_core::index::IndexConfig {
        metric: Metric::Cosine,
        ..Default::default()
    };
    let idx = LeannIndex::build(&texts, Some(metas), embedder, cfg)?;
    println!("built index: {} chunks", idx.len());

    // 3. Pure vector search.
    let res = idx.search("hub pruning storage", &SearchOptions { k: 3, ..Default::default() })?;
    println!("\nvector search: {} hits in {:.2}ms", res.hits.len(), res.took_ms);
    for h in &res.hits {
        let preview: String = h.text.as_deref().unwrap_or("").chars().take(60).collect();
        println!("  id={} sim={:.3} :: {}", h.id, h.similarity, preview);
    }

    // 4. Hybrid search (vector + BM25 + RRF).
    let hres = idx.hybrid_search("graph pruning M edges", &HybridOptions::default())?;
    println!("\nhybrid search: {} hits in {:.2}ms", hres.hits.len(), hres.took_ms);
    for h in &hres.hits {
        println!(
            "  id={} fused={:.4} (v{:?} l{:?})",
            h.id, h.fused_score, h.vector_rank, h.lexical_rank
        );
    }

    // 5. Save / reload as a single portable file.
    let dir = std::env::temp_dir().join("leann-example");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("demo.leann");
    idx.save(&path)?;
    let provider = Arc::new(LocalHashEmbedder::new(384));
    let reloaded = LeannIndex::open(&path, provider)?;
    println!("\nreloaded {} chunks from {:?}", reloaded.len(), path);

    // 6. Guard an answer (IBM Granite Guardian elimination, offline mode).
    let guardian = Guardian::new(); // deterministic local checks in this example
    let verdict = guardian.check_groundedness(
        "LEANN prunes hubs to m = M/5 edges and achieves 50x storage reduction in 2031.",
        &texts,
    )?;
    println!(
        "\nguard groundedness: triggered={} confidence={:.2}",
        verdict.triggered, verdict.confidence
    );
    let rewritten = guardian.grounded_rewrite(
        "LEANN prunes hubs to m = M/5 edges. It was invented in 2031 in Paris.",
        &texts,
    )?;
    println!("grounded rewrite: {rewritten}");
    let _ = GuardPolicy::Abstain;
    let _ = GuardSettings::default();
    Ok(())
}
