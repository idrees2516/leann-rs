//! # leann-ingest
//!
//! Multi-format document ingestion for agentic RAG:
//!
//! | Format | Extraction | Chunking |
//! |--------|-----------|----------|
//! | PDF | `pdf-extract` (pure Rust) | paragraph + page metadata |
//! | TXT / LOG | direct | paragraph |
//! | Markdown | direct | heading-aware, fenced blocks atomic |
//! | HTML | built-in extractor (script/style stripped, entities decoded) | paragraph + title |
//! | DOCX | zip + OOXML | paragraph + page metadata |
//! | EPUB | zip + spine order | chapter sections |
//! | JSON | value / JSONL pretty print | paragraph |
//! | Code (30+ langs) | direct | definition-boundary aware + line ranges |
//! | Audio (mp3/wav/m4a/...) | ffmpeg -> 16k wav -> Whisper-compatible API | timestamped segments |
//! | Video (mp4/mkv/...) | audio track transcription + optional VLM keyframes | interleaved timeline |
//!
//! Every chunk carries uniform metadata (`doc_id`, `source`, `format`,
//! `chunk_index`, `total_chunks`, + format-specific fields) so LEANN's
//! context expansion and metadata filtering work across all formats.

pub mod chunker;
pub mod detect;
pub mod parsers;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};

use chunker::{estimate_tokens, Chunk, ChunkerConfig};
use detect::Format;
pub use parsers::media::{transcribe_media, FrameDescriber, Transcriber};
use parsers::ParsedDoc;
use rayon::prelude::*;

pub use parsers::media::{OpenAiTranscriber, Segment, Transcript};

#[derive(Debug, thiserror::Error)]
pub enum LeannIngestError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("media error: {0}")]
    Media(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, LeannIngestError>;

/// Pipeline options.
#[derive(Debug, Clone)]
pub struct IngestOptions {
    pub chunker: ChunkerConfig,
    /// Extra metadata merged into every chunk (e.g. {"user": "ada"}).
    pub extra_metadata: Value,
    /// Keyframe cadence for video VLM passes; None = frames skipped.
    pub video_frame_interval: Option<f64>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            chunker: ChunkerConfig::default(),
            extra_metadata: Value::Null,
            video_frame_interval: None,
        }
    }
}

/// One ingestable chunk with metadata.
#[derive(Debug, Clone)]
pub struct IngestedChunk {
    pub text: String,
    pub metadata: Value,
}

/// Ingest a single document file (any supported format).
pub fn ingest_file(
    path: &Path,
    opts: &IngestOptions,
    transcriber: Option<&dyn Transcriber>,
) -> Result<Vec<IngestedChunk>> {
    let format = detect::detect(path);
    let doc_id = doc_id_for(path);
    match format {
        Format::Audio => {
            let t = transcriber.ok_or_else(|| {
                LeannIngestError::Unsupported(
                    "audio ingestion requires a Transcriber (e.g. OpenAiTranscriber)".into(),
                )
            })?;
            let transcript = transcribe_media(path, t)?;
            let chunks = chunks_from_segments(
                &transcript.segments,
                path,
                "audio",
                &doc_id,
                opts,
                &[
                    ("duration_sec", json!(path_duration(path))),
                    ("language", transcript.language.clone().map(Value::String).unwrap_or(Value::Null)),
                ],
            );
            Ok(chunks)
        }
        Format::Video => {
            let t = transcriber.ok_or_else(|| {
                LeannIngestError::Unsupported(
                    "video ingestion requires a Transcriber (e.g. OpenAiTranscriber)".into(),
                )
            })?;
            let transcript = transcribe_media(path, t)?;
            let chunks = chunks_from_segments(
                &transcript.segments,
                path,
                "video",
                &doc_id,
                opts,
                &[("duration_sec", json!(path_duration(path)))],
            );
            // optional frame descriptions are appended by callers with a
            // FrameDescriber via `describe_video_frames`
            let _ = &opts.video_frame_interval;
            Ok(chunks)
        }
        f => {
            let parsed = parsers::parse_file(path, f)?;
            Ok(chunks_from_doc(&parsed, path, &doc_id, opts))
        }
    }
}

/// Ingest many files in parallel (rayon).
pub fn ingest_files(
    paths: &[PathBuf],
    opts: &IngestOptions,
    transcriber: Option<Arc<dyn Transcriber>>,
) -> Result<Vec<IngestedChunk>> {
    let results: Vec<Result<Vec<IngestedChunk>>> = paths
        .par_iter()
        .map(|p| {
            let t: Option<&dyn Transcriber> = transcriber.as_deref();
            ingest_file(p, opts, t)
        })
        .collect();
    let mut out = Vec::new();
    for r in results {
        out.extend(r?);
    }
    Ok(out)
}

/// Ingest raw bytes with an explicit name (extension used for detection).
pub fn ingest_bytes(
    name: &str,
    bytes: &[u8],
    opts: &IngestOptions,
) -> Result<Vec<IngestedChunk>> {
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let format = if Format::from_ext(ext) != Format::Unknown {
        Format::from_ext(ext)
    } else {
        Format::sniff(bytes)
    };
    if format == Format::Unknown {
        // default to text
        return Ok(chunks_from_doc(
            &parsers::parse_txt(&String::from_utf8_lossy(bytes)),
            Path::new(name),
            &doc_id_for(Path::new(name)),
            opts,
        ));
    }
    let parsed = parsers::parse_bytes(name, bytes, format)?;
    Ok(chunks_from_doc(&parsed, Path::new(name), &doc_id_for(Path::new(name)), opts))
}

/// Ingest an inline text blob (chat memory, notes, pasted content).
pub fn ingest_text(text: &str, source: &str, format: &str, opts: &IngestOptions) -> Vec<IngestedChunk> {
    let doc = match format {
        "markdown" | "md" => parsers::parse_markdown(text),
        "html" => parsers::parse_html(text),
        _ => parsers::parse_txt(text),
    };
    let doc_id = short_hash(source.as_bytes());
    chunks_from_doc(&doc, Path::new(source), &doc_id, opts)
}

/// Optional second pass: describe video keyframes with a VLM and append
/// description chunks on the timeline.
pub fn describe_video_frames(
    video: &Path,
    describer: &dyn FrameDescriber,
    interval_secs: f64,
    opts: &IngestOptions,
) -> Result<Vec<IngestedChunk>> {
    let frames = parsers::media::extract_keyframes(video, interval_secs)?;
    let doc_id = doc_id_for(video);
    let mut out = Vec::new();
    for (frame, ts) in frames {
        if let Ok(desc) = describer.describe(&frame, ts) {
            if desc.trim().is_empty() {
                continue;
            }
            let mut meta = base_meta(video, "video-frame", &doc_id, opts);
            meta["timestamp_sec"] = json!(ts);
            meta["text_kind"] = json!("frame-description");
            out.push(IngestedChunk {
                text: format!("[{ts:.1}s] {desc}"),
                metadata: meta,
            });
        }
    }
    // index + total
    let total = out.len();
    for (i, c) in out.iter_mut().enumerate() {
        c.metadata["chunk_index"] = json!(i);
        c.metadata["total_chunks"] = json!(total);
    }
    Ok(out)
}

// ---------------- internals ----------------

fn doc_id_for(path: &Path) -> String {
    short_hash(path.to_string_lossy().as_bytes())
}

fn short_hash(bytes: &[u8]) -> String {
    // FNV-1a 64 -> 16 hex chars; stable across runs/platforms
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    format!("{h:016x}")
}

fn path_duration(path: &Path) -> Option<f64> {
    parsers::media::media_duration(path)
}

fn base_meta(path: &Path, format: &str, doc_id: &str, opts: &IngestOptions) -> Value {
    let mut m = json!({
        "doc_id": doc_id,
        "source": path.to_string_lossy(),
        "format": format,
    });
    if let Value::Object(extra) = &opts.extra_metadata {
        if let Value::Object(dst) = &mut m {
            for (k, v) in extra {
                dst.insert(k.clone(), v.clone());
            }
        }
    }
    m
}

fn chunks_from_doc(doc: &ParsedDoc, path: &Path, doc_id: &str, opts: &IngestOptions) -> Vec<IngestedChunk> {
    let lang = doc
        .extra
        .iter()
        .find(|(k, _)| k == "language")
        .and_then(|(_, v)| v.as_str())
        .unwrap_or("")
        .to_string();

    let chunks: Vec<Chunk> = if doc.format == "markdown" {
        chunker::chunk_markdown(&doc.text, &opts.chunker)
    } else if doc.format == "code" && !lang.is_empty() {
        chunker::chunk_code(&doc.text, &lang, &opts.chunker)
    } else {
        chunker::chunk_text(&doc.text, &opts.chunker)
            .into_iter()
            .map(|text| Chunk {
                text,
                metadata: Value::Null,
            })
            .collect()
    };

    let total = chunks.len();
    let has_pages = !doc.pages.is_empty();
    let page_bounds = page_char_bounds(&doc.pages, &doc.text);

    chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let mut meta = base_meta(path, &doc.format, doc_id, opts);
            meta["chunk_index"] = json!(i);
            meta["total_chunks"] = json!(total);
            meta["tokens"] = json!(estimate_tokens(&c.text));
            if !c.metadata.is_null() {
                if let (Value::Object(dst), Value::Object(src)) = (&mut meta, &c.metadata) {
                    for (k, v) in src {
                        dst.insert(k.clone(), v.clone());
                    }
                }
            }
            for (k, v) in &doc.extra {
                meta[k.as_str()] = v.clone();
            }
            // page attribution via char position
            if has_pages {
                let pos = find_char_pos(&doc.text, &c.text);
                if let Some(page) = page_for(&page_bounds, pos) {
                    meta["page"] = json!(page);
                }
            }
            IngestedChunk {
                text: c.text,
                metadata: meta,
            }
        })
        .collect()
}

/// Map chunk text back to a char position in the doc (first occurrence of
/// its first 40 chars).
fn find_char_pos(doc_text: &str, chunk_text: &str) -> usize {
    let probe: String = chunk_text.chars().take(40).collect();
    if probe.is_empty() {
        return 0;
    }
    doc_text.find(&probe).unwrap_or(0)
}

/// Cumulative char offsets where each page starts within the full text.
fn page_char_bounds(pages: &[(usize, String)], full: &str) -> Vec<(usize, usize)> {
    // pages' texts appear in order in `full`; find each page's start offset
    let mut bounds = Vec::with_capacity(pages.len());
    let mut search_from = 0usize;
    for (_, ptext) in pages {
        let probe: String = ptext.chars().take(40).collect();
        if probe.is_empty() {
            continue;
        }
        if let Some(pos) = full[search_from..].find(&probe) {
            bounds.push((search_from + pos, search_from + pos + ptext.len()));
            search_from += pos + 1;
        }
    }
    bounds
}

fn page_for(bounds: &[(usize, usize)], pos: usize) -> Option<usize> {
    // number of page starts at or before pos
    let mut page = 0;
    for (i, (start, _)) in bounds.iter().enumerate() {
        if *start <= pos {
            page = i + 1;
        } else {
            break;
        }
    }
    if bounds.is_empty() {
        None
    } else {
        Some(page)
    }
}

/// Build timestamped chunks from transcript segments.
fn chunks_from_segments(
    segments: &[Segment],
    path: &Path,
    format: &str,
    doc_id: &str,
    opts: &IngestOptions,
    extra: &[(&str, Value)],
) -> Vec<IngestedChunk> {
    if segments.is_empty() {
        // single chunk from full text if we have one
        return Vec::new();
    }
    // group segments until token target
    let mut groups: Vec<(f64, f64, Vec<&Segment>)> = Vec::new();
    let mut cur_start = segments[0].start;
    let mut cur_end = segments[0].end;
    let mut cur: Vec<&Segment> = Vec::new();
    let mut tokens = 0;
    for s in segments {
        let t = estimate_tokens(&s.text);
        if !cur.is_empty() && tokens + t > opts.chunker.chunk_tokens {
            groups.push((cur_start, cur_end, std::mem::take(&mut cur)));
            tokens = 0;
        }
        if cur.is_empty() {
            cur_start = s.start;
        }
        cur.push(s);
        cur_end = s.end;
        tokens += t;
    }
    if !cur.is_empty() {
        groups.push((cur_start, cur_end, cur));
    }

    let total = groups.len();
    groups
        .into_iter()
        .enumerate()
        .map(|(i, (start, end, segs))| {
            let mut meta = base_meta(path, format, doc_id, opts);
            meta["chunk_index"] = json!(i);
            meta["total_chunks"] = json!(total);
            meta["start_sec"] = json!(start);
            meta["end_sec"] = json!(end);
            meta["text_kind"] = json!("transcript");
            for (k, v) in extra {
                if !v.is_null() {
                    meta[*k] = v.clone();
                }
            }
            let text = segs
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            IngestedChunk {
                text,
                metadata: meta,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpfile(name: &str, content: &[u8]) -> PathBuf {
        let p = std::env::temp_dir().join(format!("leann-ingest-{name}"));
        std::fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn txt_pipeline() {
        let p = tmpfile("a.txt", b"First paragraph here.\n\nSecond paragraph with more text in it.\n\nThird one follows.");
        let chunks = ingest_file(&p, &IngestOptions::default(), None).unwrap();
        assert!(!chunks.is_empty());
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.metadata["chunk_index"].as_u64().unwrap() as usize, i);
            assert_eq!(c.metadata["format"], "txt");
            assert_eq!(c.metadata["doc_id"].as_str().unwrap().len(), 16);
            assert!(c.metadata["total_chunks"].as_u64().unwrap() >= 1);
        }
    }

    #[test]
    fn markdown_pipeline() {
        let md = "# Head\n\npara one.\n\n## Sub\n\npara two here.\n";
        let p = tmpfile("doc.md", md.as_bytes());
        let chunks = ingest_file(&p, &IngestOptions::default(), None).unwrap();
        assert!(chunks
            .iter()
            .any(|c| c.metadata.get("heading").and_then(|h| h.as_str()) == Some("Head")));
    }

    #[test]
    fn html_pipeline() {
        let html = r#"<html><head><title>My Page</title><style>body{}</style></head>
        <body><h1>Title</h1><p>First <b>bold</b> paragraph.</p><script>evil()</script>
        <p>Second paragraph &amp; entities.</p></body></html>"#;
        let p = tmpfile("page.html", html.as_bytes());
        let chunks = ingest_file(&p, &IngestOptions::default(), None).unwrap();
        let all = chunks.iter().map(|c| c.text.as_str()).collect::<String>();
        assert!(all.contains("First bold paragraph"), "got: {all}");
        assert!(all.contains("entities"));
        assert!(!all.contains("evil"), "script must be stripped");
        assert!(!all.contains("body{}"), "style must be stripped");
    }

    #[test]
    fn code_pipeline() {
        let code = "fn one() {}\n\nfn two() {}\n\nclass Fake:\n    pass\n";
        let p = tmpfile("lib.rs", code.as_bytes());
        let chunks = ingest_file(&p, &IngestOptions::default(), None).unwrap();
        assert!(chunks.iter().all(|c| c.metadata["format"] == "code"));
        assert!(chunks.iter().any(|c| c.metadata["language"] == "rust"));
    }

    #[test]
    fn json_pipeline() {
        let data = r#"[{"name":"ada","role":"admin"},{"name":"bob","role":"user"}]"#;
        let p = tmpfile("data.json", data.as_bytes());
        let chunks = ingest_file(&p, &IngestOptions::default(), None).unwrap();
        let all = chunks.iter().map(|c| c.text.as_str()).collect::<String>();
        assert!(all.contains("ada") && all.contains("bob"));
    }

    #[test]
    fn docx_pipeline() {
        // minimal docx: zip with word/document.xml
        let p = std::env::temp_dir().join("leann-ingest-t.docx");
        {
            let f = std::fs::File::create(&p).unwrap();
            let mut w = zip::ZipWriter::new(f);
            w.start_file("word/document.xml", zip::write::SimpleFileOptions::default())
                .unwrap();
            let xml = r#"<?xml version="1.0"?><w:document><w:body><w:p><w:r><w:t>Hello docx world.</w:t></w:r></w:p><w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p></w:body></w:document>"#;
            std::io::Write::write_all(&mut w, xml.as_bytes()).unwrap();
            w.finish().unwrap();
        }
        let chunks = ingest_file(&p, &IngestOptions::default(), None).unwrap();
        let all = chunks.iter().map(|c| c.text.as_str()).collect::<String>();
        assert!(all.contains("Hello docx world"), "got: {all}");
        assert_eq!(chunks[0].metadata["format"], "docx");
    }

    #[test]
    fn epub_pipeline() {
        let p = std::env::temp_dir().join("leann-ingest-t.epub");
        {
            let f = std::fs::File::create(&p).unwrap();
            let mut w = zip::ZipWriter::new(f);
            // mimetype must be first, stored
            w.start_file(
                "mimetype",
                zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
            )
            .unwrap();
            std::io::Write::write_all(&mut w, b"application/epub+zip").unwrap();
            w.start_file("META-INF/container.xml", zip::write::SimpleFileOptions::default())
                .unwrap();
            let container = r#"<?xml version="1.0"?><container><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#;
            std::io::Write::write_all(&mut w, container.as_bytes()).unwrap();
            w.start_file("OEBPS/content.opf", zip::write::SimpleFileOptions::default())
                .unwrap();
            let opf = r#"<?xml version="1.0"?><package xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>Test Book</dc:title></metadata><manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/><item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>"#;
            std::io::Write::write_all(&mut w, opf.as_bytes()).unwrap();
            w.start_file("OEBPS/c1.xhtml", zip::write::SimpleFileOptions::default())
                .unwrap();
            std::io::Write::write_all(&mut w, b"<html><body><p>Chapter one text.</p></body></html>").unwrap();
            w.start_file("OEBPS/c2.xhtml", zip::write::SimpleFileOptions::default())
                .unwrap();
            std::io::Write::write_all(&mut w, b"<html><body><p>Chapter two text.</p></body></html>").unwrap();
            w.finish().unwrap();
        }
        let chunks = ingest_file(&p, &IngestOptions::default(), None).unwrap();
        let all = chunks.iter().map(|c| c.text.as_str()).collect::<String>();
        assert!(all.contains("Chapter one text"), "got: {all}");
        assert!(all.contains("Chapter two text"));
        assert!(chunks.iter().any(|c| c.metadata.get("title") == Some(&json!("Test Book"))));
    }

    #[test]
    fn bytes_and_text_paths() {
        let chunks = ingest_bytes(
            "note.txt",
            b"some inline note text",
            &IngestOptions::default(),
        )
        .unwrap();
        assert!(!chunks.is_empty());
        let chunks2 = ingest_text("hello memory", "chat://session-1", "txt", &IngestOptions::default());
        assert!(!chunks2.is_empty());
        assert_eq!(chunks2[0].metadata["source"], "chat://session-1");
    }

    #[test]
    fn extra_metadata_propagates() {
        let opts = IngestOptions {
            extra_metadata: json!({"user": "ada", "project": "demo"}),
            ..Default::default()
        };
        let chunks = ingest_text("body text", "mem://1", "txt", &opts);
        assert_eq!(chunks[0].metadata["user"], "ada");
    }
}
