//! # leann CLI
//!
//! ```text
//! leann init --embedder local-hash            # create an index
//! leann add docs/*.pdf src/**/*.rs            # ingest (any format)
//! leann search "graph pruning"                # vector search
//! leann search "graph pruning" --hybrid       # + BM25 fusion
//! leann chat "How does LEANN reduce storage?" # RAG (needs --llm)
//! leann guard "answer.txt" --context docs/    # IBM Granite Guardian audit
//! leann stats
//! ```

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

use leann_chat::{create_llm, LlmConfig, LlmKind};
use leann_core::filter::parse_filter_checked;
use leann_core::index::{HybridOptions, LeannIndex, SearchOptions};
use leann_core::metric::Metric;
use leann_embedder::{create_embedder, EmbedderConfig, EmbedderKind};
use leann_guard::{GuardPolicy, Guardian, OpenAiJudge, RagGuard};
use leann_ingest::{ingest_files, IngestOptions};
mod mcp;

#[derive(Parser)]
#[command(name = "leann", version, about = "LEANN: storage-efficient vector search + agentic RAG (Rust)")]
struct Cli {
    /// Index file (default: ./index.leann)
    #[arg(long, global = true, default_value = "index.leann")]
    index: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build/add documents to the index
    Add {
        files: Vec<PathBuf>,
        /// Storage-efficient sharded build (for very large corpora)
        #[arg(long)]
        sharded: bool,
        /// Shard size for --sharded
        #[arg(long, default_value = "50000")]
        shard_size: usize,
        #[arg(long, default_value = "512")]
        chunk_tokens: usize,
        #[arg(long, default_value = "64")]
        overlap_tokens: usize,
    },
    /// Vector search
    Search {
        query: String,
        #[arg(long, default_value = "10")]
        k: usize,
        /// Hybrid search (vector + BM25 + RRF fusion)
        #[arg(long)]
        hybrid: bool,
        /// JSON metadata filter
        #[arg(long)]
        filter: Option<String>,
        /// Show neighbor context
        #[arg(long)]
        expand: bool,
    },
    /// RAG chat over the index (requires an LLM endpoint)
    Chat {
        query: String,
        #[arg(long, default_value = "gpt-4o-mini")]
        model: String,
        #[arg(long, default_value = "https://api.openai.com/v1")]
        llm_base_url: String,
        /// Verify answers with the IBM Granite Guardian (see leann-guard)
        #[arg(long)]
        guard: bool,
        #[arg(long, default_value = "http://localhost:11434/v1")]
        guard_base_url: String,
        #[arg(long, default_value = "ibm-granite/granite-guardian-3.3-8b")]
        guard_model: String,
    },
    /// Audit a text file against context files with the guardian
    Guard {
        /// Answer text file
        answer: PathBuf,
        /// Context files (any format; ingested as chunks)
        context: Vec<PathBuf>,
        #[arg(long, default_value = "http://localhost:11434/v1")]
        base_url: String,
        #[arg(long, default_value = "ibm-granite/granite-guardian-3.3-8b")]
        model: String,
        #[arg(long, default_value = "repair")]
        policy: String,
        /// Offline mode: deterministic local checks (no judge server needed)
        #[arg(long)]
        local: bool,
    },
    /// Print index statistics
    Stats,
    /// Run as an MCP (Model Context Protocol) stdio tool server
    Mcp {
        /// Passphrase for encrypted indexes (also LEANN_PASSPHRASE env)
        #[arg(long)]
        passphrase: Option<String>,
    },
    /// Delete chunks matching a JSON filter
    Delete { filter: String },
    /// Export the live corpus as JSON / CSV / Markdown
    Export {
        /// json | csv | md
        #[arg(default_value = "json")]
        format: String,
        /// Output file (default: stdout)
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// QR-code video memory (memvid-style): corpus in, portable video out
    Visual {
        #[command(subcommand)]
        cmd: VisualCmd,
    },
    /// Serve the REST API + web playground for real-world integrations
    Serve {
        #[arg(long, default_value = "8080")]
        port: u16,
        #[arg(long, default_value = "./leann-data")]
        data_dir: PathBuf,
        /// Require this API key (also LEANN_API_KEY env; unset = open server)
        #[arg(long)]
        api_key: Option<String>,
        /// Requests per client per minute (0 = unlimited)
        #[arg(long, default_value = "240")]
        rate_limit: u32,
        /// Allowed CORS origin (repeatable)
        #[arg(long = "allow-origin")]
        allow_origins: Vec<String>,
    },
}

#[derive(Subcommand)]
enum VisualCmd {
    /// Build a QR-code video from the index's live chunks
    Build {
        /// Output video path (.mp4 or .gif)
        #[arg(long, default_value = "memory.mp4")]
        output: PathBuf,
        /// auto | mp4 | gif (auto = mp4 when ffmpeg is installed)
        #[arg(long, default_value = "auto")]
        format: String,
        /// Pixels per QR module
        #[arg(long, default_value = "8")]
        scale: u32,
    },
    /// Semantic search; hit text is decoded straight out of the video
    Query {
        query: String,
        /// Video memory file
        #[arg(long, default_value = "memory.mp4")]
        video: PathBuf,
        #[arg(long, default_value = "5")]
        k: usize,
    },
}

fn embedder_from_env() -> EmbedderConfig {
    let local = std::env::var("LEANN_EMBEDDER").map(|v| v == "local" || v == "local-hash").unwrap_or(false);
    if local {
        EmbedderConfig {
            kind: EmbedderKind::LocalHash,
            dim: 384,
            ..Default::default()
        }
    } else {
        EmbedderConfig::default()
    }
}

fn main() {
    let cli = Cli::parse();
    // embedder is only needed for index operations (not for `guard`)
    let needs_embedder = !matches!(cli.cmd, Cmd::Guard { .. });
    let provider = if needs_embedder {
        let cfg = embedder_from_env();
        match create_embedder(&cfg) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("embedder error: {e}\nhint: set OPENAI_API_KEY, or use LEANN_EMBEDDER=local");
                std::process::exit(2);
            }
        }
    } else {
        // placeholder provider, never used by `guard`
        create_embedder(&EmbedderConfig {
            kind: EmbedderKind::LocalHash,
            dim: 8,
            ..Default::default()
        })
        .expect("local embedder cannot fail")
    };

    let r = match cli.cmd {
        Cmd::Add { files, sharded, shard_size, chunk_tokens, overlap_tokens } => {
            cmd_add(&cli.index, provider, files, sharded, shard_size, chunk_tokens, overlap_tokens)
        }
        Cmd::Search { query, k, hybrid, filter, expand } => {
            cmd_search(&cli.index, provider, query, k, hybrid, filter, expand)
        }
        Cmd::Chat { query, model, llm_base_url, guard, guard_base_url, guard_model } => {
            cmd_chat(&cli.index, provider, query, model, llm_base_url, guard, guard_base_url, guard_model)
        }
        Cmd::Guard { answer, context, base_url, model, policy, local } => {
            cmd_guard(answer, context, base_url, model, policy, local)
        }
        Cmd::Stats => cmd_stats(&cli.index, provider),
        Cmd::Delete { filter } => cmd_delete(&cli.index, provider, filter),
        Cmd::Mcp { passphrase } => {
            let _ = passphrase; // encrypted MCP indexes: LEANN_PASSPHRASE env
            std::process::exit(mcp::run(cli.index));
        }
        Cmd::Export { format, out } => cmd_export(&cli.index, provider, &format, out),
        Cmd::Visual { cmd } => cmd_visual(&cli.index, provider, cmd),
        Cmd::Serve { port, data_dir, api_key, rate_limit, allow_origins } => {
            let mut cfg = leann_server::ServerConfig {
                data_dir,
                ..Default::default()
            };
            cfg.api_key = api_key.or_else(|| std::env::var("LEANN_API_KEY").ok());
            cfg.rate_limit_per_min = rate_limit;
            if !allow_origins.is_empty() {
                cfg.allow_origins = allow_origins;
            }
            let addr = format!("0.0.0.0:{port}");
            leann_server::serve(&addr, cfg);
            Ok(())
        }
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn open_or_hint(path: &PathBuf, provider: Arc<dyn leann_core::cache::EmbeddingProvider>) -> Result<LeannIndex, String> {
    if path.exists() {
        LeannIndex::open(path, provider).map_err(|e| e.to_string())
    } else {
        Err(format!(
            "index {} not found — run `leann add <files>` first",
            path.display()
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_add(
    index_path: &PathBuf,
    provider: Arc<dyn leann_core::cache::EmbeddingProvider>,
    files: Vec<PathBuf>,
    sharded: bool,
    shard_size: usize,
    chunk_tokens: usize,
    overlap_tokens: usize,
) -> Result<(), String> {
    if files.is_empty() {
        return Err("no files given".into());
    }
    // expand directories
    let mut expanded: Vec<PathBuf> = Vec::new();
    for f in files {
        if f.is_dir() {
            collect_files(&f, &mut expanded);
        } else {
            expanded.push(f);
        }
    }
    eprintln!("ingesting {} files...", expanded.len());
    let opts = IngestOptions {
        chunker: leann_ingest::chunker::ChunkerConfig {
            chunk_tokens,
            overlap_tokens,
            ..Default::default()
        },
        ..Default::default()
    };
    let chunks = ingest_files(&expanded, &opts, None).map_err(|e| e.to_string())?;
    eprintln!("{} chunks produced", chunks.len());
    if chunks.is_empty() {
        return Err("no text extracted from the given files".into());
    }
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let metas: Vec<serde_json::Value> = chunks.iter().map(|c| c.metadata.clone()).collect();

    let existing = index_path.exists();
    let idx = if existing {
        let mut i = open_or_hint(index_path, provider.clone())?;
        i.add(&texts, Some(metas)).map_err(|e| e.to_string())?;
        i
    } else {
        let cfg = leann_core::index::IndexConfig::default();
        if sharded {
            LeannIndex::build_sharded(&texts, Some(metas), provider.clone(), cfg, shard_size)
                .map_err(|e| e.to_string())?
        } else {
            LeannIndex::build(&texts, Some(metas), provider.clone(), cfg).map_err(|e| e.to_string())?
        }
    };
    idx.save(index_path).map_err(|e| e.to_string())?;
    eprintln!("index saved: {} ({} chunks)", index_path.display(), idx.len());
    Ok(())
}

fn collect_files(dir: &PathBuf, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_files(&p, out);
            } else {
                out.push(p);
            }
        }
    }
}

fn cmd_search(
    index_path: &PathBuf,
    provider: Arc<dyn leann_core::cache::EmbeddingProvider>,
    query: String,
    k: usize,
    hybrid: bool,
    filter: Option<String>,
    expand: bool,
) -> Result<(), String> {
    let idx = open_or_hint(index_path, provider)?;
    let f = filter
        .as_deref()
        .map(parse_filter_checked)
        .transpose()
        .map_err(|e| format!("bad filter: {e}"))?;

    if hybrid || expand {
        let mut opts = HybridOptions {
            base: SearchOptions {
                k,
                filter: f,
                ..Default::default()
            },
            ..Default::default()
        };
        if expand {
            use leann_core::context::ContextExpansionOptions;
            opts.expand = Some(ContextExpansionOptions { before: 1, after: 1 });
        }
        let res = idx.hybrid_search(&query, &opts).map_err(|e| e.to_string())?;
        println!("took {:.2} ms", res.took_ms);
        for (i, h) in res.hits.iter().enumerate() {
            println!(
                "#{:<3} id={:<6} score={:.4} vrank={:?} lrank={:?}",
                i + 1,
                h.id,
                h.fused_score,
                h.vector_rank,
                h.lexical_rank
            );
            if let Some(t) = &h.text {
                println!("     {}", preview(t, 160));
            }
        }
        if !res.expanded.is_empty() {
            println!("--- context ---");
            for e in &res.expanded {
                let neigh: Vec<String> = e
                    .neighbors
                    .iter()
                    .map(|(id, off)| format!("{id}({off:+})"))
                    .collect();
                println!("around {}: {}", e.primary, neigh.join(" "));
            }
        }
    } else {
        let res = idx
            .search(
                &query,
                &SearchOptions {
                    k,
                    filter: f,
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
        println!(
            "took {:.2} ms ({} vectors re-embedded of {})",
            res.took_ms, res.stats.vectors_embedded, idx.len()
        );
        for (i, h) in res.hits.iter().enumerate() {
            println!("#{:<3} id={:<6} sim={:.4}", i + 1, h.id, h.similarity);
            if let Some(t) = &h.text {
                println!("     {}", preview(t, 160));
            }
        }
    }
    Ok(())
}

fn cmd_chat(
    index_path: &PathBuf,
    provider: Arc<dyn leann_core::cache::EmbeddingProvider>,
    query: String,
    model: String,
    llm_base_url: String,
    guard: bool,
    guard_base_url: String,
    guard_model: String,
) -> Result<(), String> {
    let idx = open_or_hint(index_path, provider)?;
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let llm = create_llm(&LlmConfig {
        kind: LlmKind::OpenAi,
        base_url: Some(llm_base_url),
        api_key: Some(api_key),
        model: Some(model),
        ..Default::default()
    })
    .map_err(|e| e.to_string())?;

    // retrieve context
    let res = idx
        .search(&query, &SearchOptions { k: 6, ..Default::default() })
        .map_err(|e| e.to_string())?;
    let docs: Vec<String> = res.hits.iter().map(|h| h.text.clone().unwrap_or_default()).collect();

    let mut cfg = leann_chat::rag::RagConfig::default();
    let mut guardian = None;
    if guard {
        let judge = OpenAiJudge::new(&guard_base_url, "", &guard_model);
        guardian = Some(Guardian::with_judge(judge));
        cfg.guard = Some(leann_chat::rag::GuardSettings::default());
    }
    let ans = leann_chat::answer_with_context_guarded(
        llm.as_ref(),
        &query,
        &res.hits
            .iter()
            .map(|h| {
                (
                    h.id,
                    h.text.clone().unwrap_or_default(),
                    h.metadata.clone().unwrap_or(serde_json::Value::Null),
                )
            })
            .collect::<Vec<_>>(),
        &[],
        &cfg,
        guardian.as_ref(),
    )
    .map_err(|e| e.to_string())?;

    println!("{}", ans.answer);
    if let Some(g) = &ans.guard {
        println!(
            "\n[guard] triggered={} action={:?} ungrounded={}/{}",
            g.risk_triggered,
            g.action,
            g.sentences.iter().filter(|s| !s.keep).count(),
            g.sentences.len()
        );
    }
    let _ = docs;
    Ok(())
}

fn cmd_guard(
    answer: PathBuf,
    context: Vec<PathBuf>,
    base_url: String,
    model: String,
    policy: String,
    local: bool,
) -> Result<(), String> {
    let answer_text = std::fs::read_to_string(&answer).map_err(|e| e.to_string())?;
    let opts = IngestOptions::default();
    let chunks = ingest_files(&context, &opts, None).map_err(|e| e.to_string())?;
    let docs: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    eprintln!("auditing {} chars against {} context chunks", answer_text.len(), docs.len());

    let guardian = if local {
        // offline: deterministic lexical/numeric grounding heuristic
        Guardian::new()
    } else {
        let judge = OpenAiJudge::new(&base_url, "", &model);
        Guardian::with_judge(judge)
    };
    let p = match policy.as_str() {
        "abstain" => GuardPolicy::Abstain,
        "annotate" => GuardPolicy::Annotate,
        "log" => GuardPolicy::LogOnly,
        _ => GuardPolicy::Repair,
    };
    let guard = RagGuard::new(guardian).with_policy(p);
    let decision = guard
        .check_rag_answer("(audit)", &answer_text, &docs)
        .map_err(|e| e.to_string())?;
    println!("{}", decision.answer);
    println!(
        "--- report: triggered={} action={:?} risks={:?} repaired={}",
        decision.report.risk_triggered,
        decision.report.action,
        decision.report.risks,
        decision.report.repaired
    );
    for s in &decision.report.sentences {
        if !s.keep {
            println!("DROPPED: {}", preview(&s.text, 100));
        }
    }
    Ok(())
}

fn cmd_stats(index_path: &PathBuf, provider: Arc<dyn leann_core::cache::EmbeddingProvider>) -> Result<(), String> {
    let idx = open_or_hint(index_path, provider)?;
    let s = idx.stats();
    println!("chunks:            {}", s.chunks);
    println!("alive / deleted:   {} / {}", s.alive, s.deleted);
    println!("dim / metric:      {} / {:?}", s.dim, s.metric);
    println!("pq m / compress:   {} / {:.0}x", s.pq_m, s.pq_compression);
    println!("graph edges:       {} (avg degree {:.1})", s.graph_edges, s.avg_degree);
    println!("bm25 vocab:        {}", s.bm25_vocab);
    println!("text bytes:        {}", s.text_bytes);
    println!("codes bytes:       {}", s.codes_bytes);
    println!(
        "vs conventional:   {} bytes of raw embeddings avoided ({:.1}% saved)",
        s.conventional_vector_bytes,
        100.0 * (1.0 - (s.codes_bytes + s.graph_bytes + s.codebook_bytes) as f64
            / s.conventional_vector_bytes as f64)
    );
    if let Some(f) = s.file_bytes {
        println!("file size:         {} bytes", f);
    }
    let _ = Metric::Cosine;
    Ok(())
}

fn cmd_delete(
    index_path: &PathBuf,
    provider: Arc<dyn leann_core::cache::EmbeddingProvider>,
    filter: String,
) -> Result<(), String> {
    let mut idx = open_or_hint(index_path, provider)?;
    let f = parse_filter_checked(&filter).map_err(|e| format!("bad filter: {e}"))?;
    let n = idx.delete_where(&f);
    idx.save(index_path).map_err(|e| e.to_string())?;
    println!("deleted {} chunks", n);
    Ok(())
}

fn cmd_export(
    index_path: &PathBuf,
    provider: Arc<dyn leann_core::cache::EmbeddingProvider>,
    format: &str,
    out: Option<PathBuf>,
) -> Result<(), String> {
    use leann_core::export::ExportFormat;
    let fmt = ExportFormat::parse(format).ok_or_else(|| format!("unknown format '{format}' (json|csv|md)"))?;
    let idx = open_or_hint(index_path, provider)?;
    let body = leann_core::export::export_string(&idx, fmt);
    match out {
        Some(path) => {
            std::fs::write(&path, body).map_err(|e| e.to_string())?;
            eprintln!("exported {} live chunks to {}", idx.alive_count(), path.display());
        }
        None => print!("{body}"),
    }
    Ok(())
}

fn cmd_visual(
    index_path: &PathBuf,
    provider: Arc<dyn leann_core::cache::EmbeddingProvider>,
    cmd: VisualCmd,
) -> Result<(), String> {
    use leann_visual::{build_visual_from_index, VideoFormat, VisualOptions, VisualRetriever};
    match cmd {
        VisualCmd::Build { output, format, scale } => {
            let idx = open_or_hint(index_path, provider)?;
            let fmt = VideoFormat::parse(&format)
                .ok_or_else(|| format!("unknown format '{format}' (auto|mp4|gif)"))?;
            let opts = VisualOptions { scale, format: fmt, ..Default::default() };
            let stats = build_visual_from_index(&idx, &output, opts).map_err(|e| e.to_string())?;
            eprintln!(
                "video memory: {} chunks → {} ({} bytes, {:.2} bytes per text byte)",
                stats.chunks,
                stats.video_path.display(),
                stats.video_bytes,
                stats.bytes_per_text_byte
            );
            eprintln!("sidecar: {}", stats.sidecar_path.display());
            Ok(())
        }
        VisualCmd::Query { query, video, k } => {
            let idx = open_or_hint(index_path, provider)?;
            let r = VisualRetriever::open(&video).map_err(|e| e.to_string())?;
            let hits = r.search(&idx, &query, k).map_err(|e| e.to_string())?;
            println!("container: {} ({} chunks)", r.container(), r.chunk_count());
            for (i, h) in hits.iter().enumerate() {
                println!("#{:<3} id={:<6} sim={:.4} frame={:<5}", i + 1, h.id, h.similarity, h.frame);
                println!("     {}", preview(&h.text, 160));
            }
            Ok(())
        }
    }
}

fn preview(s: &str, n: usize) -> String {
    let t: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.chars().count() <= n {
        t
    } else {
        let cut: String = t.chars().take(n).collect();
        format!("{cut}...")
    }
}
