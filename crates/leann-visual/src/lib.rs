//! # leann-visual — QR-code video memory (memvid, in Rust)
//!
//! Packs a RAG corpus into a **portable video file**: every text chunk is
//! rendered as a QR code on one lossless video frame, and a small JSON sidecar
//! maps chunk ids → frame numbers. The chunk text lives *only* in the video —
//! retrieval seeks to a frame and decodes the QR, giving you a knowledge base
//! you can email, drop on a USB stick, play in a browser, or ship to an
//! air-gapped machine.
//!
//! ```no_run
//! use leann_visual::{build_visual, VisualOptions};
//! use leann_core::index::{LeannIndex, SearchOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use std::path::Path;
//! let chunks = vec![
//!     ("The LEANN paper is arXiv 2506.08276.".to_string(), serde_json::json!({"src": "readme"})),
//!     ("HNSW graphs enable sub-linear ANN search.".to_string(), serde_json::json!({"src": "core"})),
//! ];
//! // 1) encode the corpus into memory.mp4 (+ memory.leannvid.json sidecar)
//! let stats = build_visual(&chunks, Path::new("memory.mp4"), VisualOptions::default())?;
//!
//! // 2) pair it with any LEANN index over the same corpus for semantic search
//! let idx: LeannIndex = todo!("open your index (LeannIndex::open)");
//! let retriever = leann_visual::VisualRetriever::open(Path::new("memory.mp4"))?;
//! for hit in retriever.search(&idx, "what paper?", 3)? {
//!     println!("{} (frame {}): {}", hit.id, hit.frame, hit.text);
//! }
//! # Ok(()) }
//! ```
//!
//! ## Containers
//!
//! * **MP4** (default when `ffmpeg` is installed): lossless H.264, all-intra
//!   frames → instant random access, excellent compression on QR patterns.
//! * **GIF** (pure-Rust fallback): zero system dependencies, always works.
//!
//! ## Safety properties
//!
//! * Chunk payloads are capped at [`qr::MAX_CHUNK_BYTES`] (QR v40/L capacity);
//!   larger chunks must be split before encoding.
//! * The sidecar's video reference is validated to live next to the sidecar
//!   itself (no path traversal from a crafted sidecar).
//! * ffmpeg is invoked with an argument vector — never a shell string.

pub mod codec;
pub mod qr;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use thiserror::Error;

use codec::{FrameReader, FrameWriter, GifReader, GifWriter, Mp4Reader, Mp4Writer};
pub use codec::VideoFormat;
use leann_core::index::{LeannIndex, SearchOptions};

/// Errors produced by the visual memory layer.
#[derive(Error, Debug)]
pub enum VisualError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("chunk is empty")]
    EmptyChunk,
    #[error("chunk of {bytes} bytes exceeds QR capacity ({max}); split it before encoding")]
    ChunkTooLarge { bytes: usize, max: usize },
    #[error("QR encode error: {0}")]
    Qr(String),
    #[error("QR decode error: {0}")]
    Decode(String),
    #[error("no QR code found in frame")]
    NoQr,
    #[error("frame error: {0}")]
    Frame(String),
    #[error("ffmpeg is required for MP4 memory (install it or use --format gif)")]
    NoFfmpeg,
    #[error("ffmpeg error: {0}")]
    Ffmpeg(String),
    #[error("gif error: {0}")]
    Gif(String),
    #[error("sidecar error: {0}")]
    Sidecar(String),
    #[error("chunk id {0} not present in this memory")]
    UnknownChunk(u32),
    #[error("invalid scale {0} (must be 1..=64)")]
    BadScale(u32),
}

pub type Result<T> = std::result::Result<T, VisualError>;

/// Tuning knobs for building a visual memory.
#[derive(Debug, Clone)]
pub struct VisualOptions {
    /// Pixels per QR module (higher = more robust scanning, bigger frames).
    pub scale: u32,
    /// Quiet-zone border in modules (QR spec minimum: 4).
    pub quiet_zone: u32,
    /// Container format (auto = MP4 when ffmpeg exists, else GIF).
    pub format: VideoFormat,
    /// Logical frame rate; only affects seek math and GIF timing.
    pub fps: f64,
}

impl Default for VisualOptions {
    fn default() -> Self {
        Self { scale: 8, quiet_zone: 4, format: VideoFormat::Auto, fps: 2.0 }
    }
}

impl VisualOptions {
    pub fn with_format(mut self, f: VideoFormat) -> Self {
        self.format = f;
        self
    }
}

/// Sidecar document: `<video>.leannvid.json`.
///
/// Maps chunk ids to frames and stores per-chunk metadata. Chunk **text is
/// intentionally absent** — the video is the database.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Sidecar {
    format: String,
    video: String,
    container: String,
    width: u32,
    height: u32,
    fps: f64,
    chunks: Vec<SidecarChunk>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SidecarChunk {
    id: u32,
    /// First video frame of this chunk.
    frame: usize,
    /// How many consecutive frames the chunk spans (1 for small chunks).
    #[serde(default = "one")]
    frames: usize,
    #[serde(default)]
    metadata: Value,
}

fn one() -> usize {
    1
}

/// Where the sidecar for `video` lives.
pub fn sidecar_path(video: &Path) -> PathBuf {
    let mut name = video.file_name().unwrap_or_default().to_os_string();
    name.push(".leannvid.json");
    video.with_file_name(name)
}

/// Statistics from a successful build.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VisualBuildStats {
    pub chunks: usize,
    pub frames: usize,
    pub container: &'static str,
    pub video_bytes: u64,
    pub text_bytes: u64,
    /// video bytes per text byte (lower is better; < 1.0 means the codec
    /// compresses the QR frames below the size of the raw text).
    pub bytes_per_text_byte: f64,
    pub video_path: PathBuf,
    pub sidecar_path: PathBuf,
}

/// Incremental builder (memvid `MemvidEncoder` equivalent).
pub struct VisualBuilder {
    opts: VisualOptions,
    entries: Vec<(u32, String, Value)>,
}

impl VisualBuilder {
    pub fn new(opts: VisualOptions) -> Self {
        Self { opts, entries: Vec::new() }
    }

    /// Queue one chunk; ids are assigned in insertion order (0, 1, 2, ...).
    pub fn add_text(&mut self, text: &str, metadata: Value) -> Result<()> {
        if text.as_bytes().is_empty() {
            return Err(VisualError::EmptyChunk);
        }
        let id = self.entries.len() as u32;
        self.entries.push((id, text.to_string(), metadata));
        Ok(())
    }

    pub fn add_texts(&mut self, texts: &[String], metadatas: Option<&[Value]>) -> Result<()> {
        for (i, t) in texts.iter().enumerate() {
            let meta = metadatas.map(|m| m.get(i).cloned().unwrap_or(Value::Null)).unwrap_or(Value::Null);
            self.add_text(t, meta)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Encode everything into `video_path` (+ sidecar). Consumes the builder.
    pub fn build(self, video_path: &Path) -> Result<VisualBuildStats> {
        build_frames(self.entries, video_path, self.opts)
    }
}

fn build_frames(
    entries: Vec<(u32, String, Value)>,
    video_path: &Path,
    opts: VisualOptions,
) -> Result<VisualBuildStats> {
    if entries.is_empty() {
        return Err(VisualError::Sidecar("no chunks to encode".into()));
    }
    let container = opts.format.resolve();
    if container == VideoFormat::Mp4 && codec::ffmpeg_path().is_none() {
        return Err(VisualError::NoFfmpeg);
    }

    // Pass 1: frame every chunk (split oversized chunks across consecutive
    // QR frames) and compute the worst-case canvas size.
    let mut framed: Vec<(u32, usize /*frames*/, Value, Vec<Vec<u8>>)> = Vec::with_capacity(entries.len());
    let mut side_modules = 0u32;
    let mut total_frames = 0usize;
    for (id, text, meta) in &entries {
        if text.as_bytes().is_empty() {
            return Err(VisualError::EmptyChunk);
        }
        let frame_bytes = qr::frame_chunk(text)?;
        for bytes in &frame_bytes {
            let w = qr::qr_module_width(bytes)?;
            side_modules = side_modules.max(w + 2 * opts.quiet_zone);
        }
        total_frames += frame_bytes.len();
        framed.push((*id, frame_bytes.len(), meta.clone(), frame_bytes));
    }
    let side = side_modules * opts.scale;

    // Pass 2: render + stream frames, reusing one canvas buffer.
    std::fs::create_dir_all(
        video_path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new(".")),
    )
    .map_err(VisualError::Io)?;
    let mut writer: Box<dyn FrameWriter> = match container {
        VideoFormat::Mp4 => Box::new(Mp4Writer::create(video_path, side, opts.fps, total_frames)?),
        _ => Box::new(GifWriter::create(video_path, side, opts.fps)?),
    };
    let mut canvas = vec![255u8; (side * side) as usize];
    let mut text_bytes = 0u64;
    let mut next_frame = 0usize;
    let chunks = framed
        .iter()
        .map(|(id, frames, meta, frame_bytes)| {
            for bytes in frame_bytes {
                let frame = qr::render_qr(bytes, opts.scale, opts.quiet_zone)?;
                qr::blit_to_canvas(&frame, &mut canvas, side);
                writer.write_frame(&canvas).map_err(VisualError::from)?;
            }
            let entry = SidecarChunk {
                id: *id,
                frame: next_frame,
                frames: *frames,
                metadata: meta.clone(),
            };
            text_bytes += frame_bytes.iter().map(|b| b.len() - qr::FRAME_HEADER).sum::<usize>() as u64;
            next_frame += frame_bytes.len();
            Ok(entry)
        })
        .collect::<Result<Vec<_>>>()?;
    writer.finish()?;

    let video_bytes = std::fs::metadata(video_path).map(|m| m.len()).unwrap_or(0);
    let sc = Sidecar {
        format: "leann-visual/v1".into(),
        video: video_path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
        container: match container {
            VideoFormat::Mp4 => "mp4",
            _ => "gif",
        }
        .into(),
        width: side,
        height: side,
        fps: opts.fps,
        chunks,
    };
    let sc_path = sidecar_path(video_path);
    std::fs::write(&sc_path, serde_json::to_vec_pretty(&sc)?).map_err(VisualError::Io)?;

    Ok(VisualBuildStats {
        chunks: entries.len(),
        frames: total_frames,
        container: match container {
            VideoFormat::Mp4 => "mp4",
            _ => "gif",
        },
        video_bytes,
        text_bytes,
        bytes_per_text_byte: if text_bytes > 0 { video_bytes as f64 / text_bytes as f64 } else { 0.0 },
        video_path: video_path.to_path_buf(),
        sidecar_path: sc_path,
    })
}

/// One-shot build from `(text, metadata)` pairs.
pub fn build_visual(
    chunks: &[(String, Value)],
    video_path: &Path,
    opts: VisualOptions,
) -> Result<VisualBuildStats> {
    let mut b = VisualBuilder::new(opts);
    for (t, m) in chunks {
        b.add_text(t, m.clone())?;
    }
    b.build(video_path)
}

/// Build from the live (non-deleted) chunks of an existing LEANN index.
pub fn build_visual_from_index(
    idx: &LeannIndex,
    video_path: &Path,
    opts: VisualOptions,
) -> Result<VisualBuildStats> {
    let chunks = leann_core::export::alive_chunks(idx)
        .into_iter()
        .map(|(_, t, m)| (t, m))
        .collect::<Vec<_>>();
    build_visual(&chunks, video_path, opts)
}

/// One chunk fetched back from the video.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VisualChunk {
    pub id: u32,
    pub frame: usize,
    pub text: String,
    pub metadata: Value,
}

/// One search hit whose text was decoded out of the video.
#[derive(Debug, Clone, serde::Serialize)]
pub struct VisualHit {
    pub id: u32,
    pub frame: usize,
    pub similarity: f32,
    pub text: String,
    pub metadata: Value,
}

/// Random-access reader over a visual memory (memvid `MemvidRetriever`
/// equivalent). Cheap to clone-by-Arc; safe to share across threads.
pub struct VisualRetriever {
    sidecar: Sidecar,
    by_id: HashMap<u32, usize>,
    reader: Box<dyn FrameReader>,
}

impl VisualRetriever {
    /// Open `video` and its sidecar. The sidecar must reference a video file
    /// in the same directory (anti-path-traversal rule).
    pub fn open(video_path: &Path) -> Result<Self> {
        let sc_path = sidecar_path(video_path);
        let raw = std::fs::read_to_string(&sc_path)
            .map_err(|e| VisualError::Sidecar(format!("cannot read {}: {e}", sc_path.display())))?;
        let sidecar: Sidecar = serde_json::from_str(&raw)
            .map_err(|e| VisualError::Sidecar(format!("bad sidecar JSON: {e}")))?;
        if sidecar.format != "leann-visual/v1" {
            return Err(VisualError::Sidecar(format!("unsupported sidecar {}", sidecar.format)));
        }
        // The referenced video must resolve inside the sidecar's own directory.
        let dir = match sc_path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let video = dir.join(&sidecar.video);
        let canon = video
            .canonicalize()
            .map_err(|e| VisualError::Sidecar(format!("video {} missing: {e}", video.display())))?;
        let dir_canon = dir
            .canonicalize()
            .map_err(|e| VisualError::Sidecar(format!("bad dir: {e}")))?;
        if !canon.starts_with(&dir_canon) {
            return Err(VisualError::Sidecar(
                "sidecar video reference escapes its own directory".into(),
            ));
        }
        let reader: Box<dyn FrameReader> = if sidecar.container == "mp4" {
            Box::new(Mp4Reader { path: canon, side: sidecar.width, fps: sidecar.fps })
        } else {
            Box::new(GifReader::new(&canon, sidecar.width))
        };
        let by_id = sidecar
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id, i))
            .collect();
        Ok(Self { sidecar, by_id, reader })
    }

    pub fn chunk_count(&self) -> usize {
        self.sidecar.chunks.len()
    }

    pub fn container(&self) -> &str {
        &self.sidecar.container
    }

    /// Fetch chunk `i` (position order) — decodes and stitches its QR frames.
    pub fn get_chunk(&self, i: usize) -> Result<VisualChunk> {
        let entry = self
            .sidecar
            .chunks
            .get(i)
            .ok_or_else(|| VisualError::Sidecar(format!("chunk index {i} out of range")))?;
        let mut stitched = Vec::new();
        for k in 0..entry.frames {
            let frame_no = entry.frame + k;
            let pixels = self.reader.read_frame(frame_no)?;
            let bytes = qr::decode_qr(&pixels, self.sidecar.width, self.sidecar.height)?;
            let (seq, last, payload) = qr::parse_frame_bytes(&bytes)?;
            if seq != k {
                return Err(VisualError::Frame(format!(
                    "chunk {i}: frame sequence gap (expected {k}, got {seq})"
                )));
            }
            stitched.extend_from_slice(payload);
            if last && k + 1 != entry.frames {
                return Err(VisualError::Frame(format!(
                    "chunk {i}: premature last-frame flag at {k}"
                )));
            }
        }
        let text = String::from_utf8(stitched)
            .map_err(|_| VisualError::Frame("frame decoded to invalid UTF-8".into()))?;
        Ok(VisualChunk { id: entry.id, frame: entry.frame, text, metadata: entry.metadata.clone() })
    }

    /// Fetch by original chunk id (as used by the paired LEANN index).
    pub fn get_by_id(&self, id: u32) -> Result<VisualChunk> {
        let i = *self.by_id.get(&id).ok_or(VisualError::UnknownChunk(id))?;
        self.get_chunk(i)
    }

    /// Semantic search over the paired index, fetching hit text from the video.
    pub fn search(&self, idx: &LeannIndex, query: &str, k: usize) -> Result<Vec<VisualHit>> {
        self.search_with(idx, query, &SearchOptions { k, ..Default::default() })
    }

    /// [`Self::search`] with full [`SearchOptions`].
    pub fn search_with(
        &self,
        idx: &LeannIndex,
        query: &str,
        opts: &SearchOptions,
    ) -> Result<Vec<VisualHit>> {
        let res = idx
            .search(query, opts)
            .map_err(|e| VisualError::Sidecar(format!("index search failed: {e}")))?;
        let mut hits = Vec::with_capacity(res.hits.len());
        for h in res.hits {
            if let Some(&i) = self.by_id.get(&h.id) {
                let chunk = self.get_chunk(i)?;
                hits.push(VisualHit {
                    id: h.id,
                    frame: chunk.frame,
                    similarity: h.similarity,
                    text: chunk.text,
                    metadata: chunk.metadata,
                });
            }
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leann_core::cache::EmbeddingProvider;
    use leann_core::index::IndexConfig;
    use std::sync::Arc;

    /// Bag-of-words hashing embedder (test only): every word hashes to a
    /// dimension, so shared vocabulary produces high cosine similarity.
    struct BagProvider;
    impl EmbeddingProvider for BagProvider {
        fn name(&self) -> &str {
            "bag"
        }
        fn dim(&self) -> usize {
            64
        }
        fn embed_batch(&self, texts: &[&str]) -> leann_core::Result<Vec<Vec<f32>>> {
            fn word_dim(w: &str) -> usize {
                let mut h: u64 = 0xcbf29ce484222325;
                for b in w.bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                (h % 64) as usize
            }
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0f32; 64];
                    for w in t.split(|c: char| !c.is_alphanumeric()) {
                        if !w.is_empty() {
                            v[word_dim(w)] += 1.0;
                        }
                    }
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    v.iter().map(|x| x / n).collect()
                })
                .collect())
        }
    }

    fn corpus() -> Vec<(String, Value)> {
        [
            "Product quantization compresses vectors by roughly one hundred x.",
            "The HNSW graph is pruned: hubs keep M edges, others keep M over five.",
            "IBM Granite Guardian checks answers for groundedness before release.",
            "视频内存 — memory in video frames, QR encoded.",
            "The two level search balances an approximate queue and an exact queue.",
        ]
        .iter()
        .enumerate()
        .map(|(i, t)| (t.to_string(), serde_json::json!({"i": i})))
        .collect()
    }

    fn build_index(corpus: &[(String, Value)]) -> LeannIndex {
        let texts: Vec<String> = corpus.iter().map(|(t, _)| t.clone()).collect();
        let metas: Vec<Value> = corpus.iter().map(|(_, m)| m.clone()).collect();
        LeannIndex::build(&texts, Some(metas), Arc::new(BagProvider), IndexConfig::default())
            .unwrap()
    }

    #[test]
    fn gif_memory_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = corpus();
        let video = dir.path().join("memory.gif");
        let stats = build_visual(&corpus, &video, VisualOptions::default().with_format(VideoFormat::Gif))
            .unwrap();
        assert_eq!(stats.chunks, 5);
        assert_eq!(stats.container, "gif");
        assert!(stats.video_bytes > 0);
        assert!(sidecar_path(&video).exists());

        let r = VisualRetriever::open(&video).unwrap();
        assert_eq!(r.chunk_count(), 5);
        for (i, (text, meta)) in corpus.iter().enumerate() {
            let c = r.get_chunk(i).unwrap();
            assert_eq!(c.text, *text, "chunk {i} text mismatch");
            assert_eq!(c.metadata, *meta);
            assert_eq!(c.frame, i);
        }
        let by_id = r.get_by_id(3).unwrap();
        assert!(by_id.text.contains("视频内存"));
    }

    #[test]
    fn mp4_memory_roundtrip() {
        if codec::ffmpeg_path().is_none() {
            eprintln!("skipping: ffmpeg not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let corpus = corpus();
        let video = dir.path().join("memory.mp4");
        let stats = build_visual(&corpus, &video, VisualOptions::default().with_format(VideoFormat::Mp4))
            .unwrap();
        assert_eq!(stats.container, "mp4");

        let r = VisualRetriever::open(&video).unwrap();
        for (i, (text, _)) in corpus.iter().enumerate() {
            let c = r.get_chunk(i).unwrap();
            assert_eq!(c.text, *text, "mp4 chunk {i} text mismatch");
        }
    }

    #[test]
    fn search_fetches_text_from_video() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = corpus();
        let idx = build_index(&corpus);
        let video = dir.path().join("m.gif");
        build_visual(&corpus, &video, VisualOptions::default().with_format(VideoFormat::Gif)).unwrap();

        let r = VisualRetriever::open(&video).unwrap();
        let hits = r.search(&idx, "graph hubs edges", 3).unwrap();
        assert!(!hits.is_empty());
        // all decoded texts must be corpus members, and the graph chunk must
        // appear somewhere in the top hits
        for h in &hits {
            assert!(corpus.iter().any(|(t, _)| t == &h.text), "decoded text must be a corpus chunk");
        }
        assert!(
            hits.iter().any(|h| h.text.contains("pruned") || h.text.contains("graph")),
            "graph chunk missing from top hits: {:?}",
            hits.iter().map(|h| h.text.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn build_from_index_skips_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = corpus();
        let mut idx = build_index(&corpus);
        idx.delete(&[0, 1]);
        let video = dir.path().join("m.gif");
        let stats = build_visual_from_index(&idx, &video, VisualOptions::default().with_format(VideoFormat::Gif))
            .unwrap();
        assert_eq!(stats.chunks, 3);
        let r = VisualRetriever::open(&video).unwrap();
        assert_eq!(r.chunk_count(), 3);
    }

    #[test]
    fn oversized_chunk_splits_across_frames() {
        let dir = tempfile::tempdir().unwrap();
        // ~3x the single-frame QR capacity, with multi-byte characters that
        // straddle piece boundaries
        let big: String = "颗粒视频内存 - mémoire 🎬 ".repeat(300);
        assert!(big.len() > qr::MAX_CHUNK_BYTES * 2);
        let chunks = vec![(big.clone(), Value::Null)];
        let stats =
            build_visual(&chunks, &dir.path().join("m.gif"), VisualOptions::default().with_format(VideoFormat::Gif))
                .unwrap();
        assert!(stats.frames > stats.chunks, "oversized chunk must span multiple frames");
        let r = VisualRetriever::open(&dir.path().join("m.gif")).unwrap();
        let back = r.get_chunk(0).unwrap();
        assert_eq!(back.text, big, "stitched text must roundtrip byte-for-byte");
    }

    #[test]
    fn empty_build_rejected() {
        let err = build_visual(&[], Path::new("/tmp/never.gif"), VisualOptions::default());
        assert!(matches!(err, Err(VisualError::Sidecar(_))));
    }

    #[test]
    fn unknown_id_errors() {
        let dir = tempfile::tempdir().unwrap();
        let corpus = corpus();
        let video = dir.path().join("m.gif");
        build_visual(&corpus, &video, VisualOptions::default().with_format(VideoFormat::Gif)).unwrap();
        let r = VisualRetriever::open(&video).unwrap();
        assert!(matches!(r.get_by_id(99), Err(VisualError::UnknownChunk(99))));
    }
}
