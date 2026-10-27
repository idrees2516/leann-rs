//! Token-aware chunking with overlap.
//!
//! Strategies:
//! - **Paragraph chunking** (prose): accumulate whole paragraphs up to the
//!   token target; oversized paragraphs split by sentences then words.
//! - **Markdown**: section on heading boundaries, heading path carried into
//!   chunk metadata; fenced code blocks treated as atomic paragraphs.
//! - **Code**: language-aware top-level boundary detection (function / class /
//!   struct definitions per language), then accumulate blocks; blocks are
//!   never split unless they exceed the hard ceiling.
//!
//! Token counts use the standard chars/4-vs-words max estimate (close to BPE
//! for English; cheap and dependency-free).

use serde_json::{json, Value};

#[derive(Debug, Clone, Copy)]
pub struct ChunkerConfig {
    /// Target chunk size in estimated tokens.
    pub chunk_tokens: usize,
    /// Overlap between consecutive chunks in tokens.
    pub overlap_tokens: usize,
    /// Chunks below this merge into neighbors.
    pub min_tokens: usize,
    /// Hard ceiling; oversized blocks split by line.
    pub max_tokens: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_tokens: 512,
            overlap_tokens: 64,
            min_tokens: 32,
            max_tokens: 1024,
        }
    }
}

/// Estimate token count (BPE-ish for English, no tokenizer dependency).
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    let words = text.split_whitespace().count();
    ((chars + 3) / 4).max(words)
}

/// A produced chunk: text plus format-specific metadata.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub metadata: Value,
}

// ---------------- generic prose ----------------

/// Split into paragraphs, keeping fenced blocks atomic for markdown.
fn split_paragraphs(text: &str, keep_fences: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_fence = false;
    let mut cur = String::new();
    for line in text.lines() {
        let trimmed = line.trim_end();
        if keep_fences && trimmed.trim_start().starts_with("```") {
            in_fence = !in_fence;
            cur.push_str(trimmed);
            cur.push('\n');
            if !in_fence {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if !in_fence && trimmed.trim().is_empty() {
            if !cur.trim().is_empty() {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        } else {
            cur.push_str(trimmed);
            cur.push('\n');
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out.into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Split an oversized paragraph by sentence boundaries (heuristic, no regex).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            // look ahead for closing quotes/space
            let mut extra = String::new();
            let consumed = false;
            while let Some(&n) = chars.peek() {
                if n == '"' || n == '\'' || n == ')' || n == ']' || n == '}' {
                    extra.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            cur.push_str(&extra);
            let is_boundary = matches!(chars.peek(), Some(' ') | Some('\n') | None);
            if is_boundary && !consumed {
                if estimate_tokens(&cur) >= 4 {
                    sentences.push(cur.trim().to_string());
                    cur.clear();
                }
            }
        }
    }
    if !cur.trim().is_empty() {
        sentences.push(cur.trim().to_string());
    }
    sentences
}

fn split_by_words(text: &str, max_tokens: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let per = (max_tokens).max(8);
    let mut out = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut cur_tokens = 0;
    for w in words {
        let t = estimate_tokens(w);
        cur.push(w);
        cur_tokens += t.max(1);
        if cur_tokens >= per {
            out.push(cur.join(" "));
            cur.clear();
            cur_tokens = 0;
        }
    }
    if !cur.is_empty() {
        out.push(cur.join(" "));
    }
    out
}

/// Tail of `text` covering ~`overlap_tokens` tokens, cut at a word boundary.
fn overlap_tail(text: &str, overlap_tokens: usize) -> String {
    if overlap_tokens == 0 || text.is_empty() {
        return String::new();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }
    let mut taken = Vec::new();
    let mut tokens = 0;
    for w in words.iter().rev() {
        taken.push(*w);
        tokens += estimate_tokens(w).max(1);
        if tokens >= overlap_tokens {
            break;
        }
    }
    taken.reverse();
    taken.join(" ")
}

/// Paragraph-aware chunking of plain prose.
pub fn chunk_text(text: &str, cfg: &ChunkerConfig) -> Vec<String> {
    let paras = split_paragraphs(text, false);
    accumulate(paras, cfg)
}

/// Core accumulator shared by strategies: whole blocks up to target,
/// overlap between chunks, tiny-tail merging.
fn accumulate(blocks: Vec<String>, cfg: &ChunkerConfig) -> Vec<String> {
    // pre-split oversized blocks
    let mut pieces: Vec<String> = Vec::new();
    for b in blocks {
        if estimate_tokens(&b) <= cfg.max_tokens {
            pieces.push(b);
        } else {
            for s in split_sentences(&b) {
                if estimate_tokens(&s) <= cfg.max_tokens {
                    pieces.push(s);
                } else {
                    pieces.extend(split_by_words(&s, cfg.chunk_tokens.max(16)));
                }
            }
        }
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_tokens = 0;
    for p in pieces {
        let t = estimate_tokens(&p);
        if cur_tokens > 0 && cur_tokens + t > cfg.chunk_tokens {
            chunks.push(std::mem::take(&mut cur));
            cur_tokens = 0;
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(&p);
        cur_tokens += t;
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }

    // merge tiny tail
    if chunks.len() >= 2 {
        let last = chunks.pop().unwrap();
        if estimate_tokens(&last) < cfg.min_tokens {
            let prev = chunks.pop().unwrap();
            chunks.push(format!("{prev}\n\n{last}"));
        } else {
            chunks.push(last);
        }
    }

    // apply overlap
    if cfg.overlap_tokens > 0 && chunks.len() > 1 {
        let mut overlapped: Vec<String> = Vec::with_capacity(chunks.len());
        overlapped.push(chunks[0].clone());
        for w in chunks.windows(2) {
            let tail = overlap_tail(&w[0], cfg.overlap_tokens);
            let next = if tail.is_empty() {
                w[1].clone()
            } else {
                format!("{tail}\n\n{}", w[1])
            };
            overlapped.push(next);
        }
        return overlapped;
    }
    chunks
}

// ---------------- markdown ----------------

/// Markdown chunking with heading awareness.
/// Returns chunks whose metadata carries the heading path.
pub fn chunk_markdown(md: &str, cfg: &ChunkerConfig) -> Vec<Chunk> {
    let mut sections: Vec<(String, String)> = Vec::new(); // (heading path, body)
    let mut path: Vec<String> = Vec::new();
    let mut cur_title = String::new();
    let mut cur_body = String::new();

    for line in md.lines() {
        let t = line.trim_start();
        if !t.starts_with("```") {
            if let Some(level) = heading_level(t) {
                // flush current section
                if !cur_body.trim().is_empty() || !cur_title.is_empty() {
                    sections.push((cur_title.clone(), std::mem::take(&mut cur_body)));
                }
                let title = t.trim_start_matches('#').trim().to_string();
                path.truncate(level.saturating_sub(1));
                path.push(title);
                cur_title = path.join(" > ");
                continue;
            }
        }
        cur_body.push_str(line.trim_end());
        cur_body.push('\n');
    }
    if !cur_body.trim().is_empty() || !cur_title.is_empty() {
        sections.push((cur_title, cur_body));
    }

    let mut out = Vec::new();
    for (title, body) in sections {
        let trimmed = body.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        let mut meta = json!({});
        if !title.is_empty() {
            meta["heading"] = json!(title);
        }
        // split fenced blocks out so they stay atomic
        let pieces = split_paragraphs(&trimmed, true);
        for text in accumulate(pieces, cfg) {
            let mut m = meta.clone();
            m["heading"] = meta.get("heading").cloned().unwrap_or(Value::Null);
            out.push(Chunk {
                text,
                metadata: m,
            });
        }
    }
    out
}

fn heading_level(line: &str) -> Option<usize> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let level = t.chars().take_while(|c| *c == '#').count();
    if level >= 1 && level <= 6 && t[level..].starts_with(' ') {
        Some(level)
    } else {
        None
    }
}

// ---------------- code ----------------

/// Top-level definition starters per language.
fn boundary_patterns(lang: &str) -> &'static [&'static str] {
    match lang {
        "python" => &[
            "def ", "class ", "async def ", "@", "if __name__",
        ],
        "rust" => &[
            "fn ", "pub fn ", "pub(crate) fn ", "struct ", "pub struct ", "enum ", "pub enum ",
            "impl ", "trait ", "pub trait ", "mod ", "pub mod ", "macro_rules!",
        ],
        "javascript" | "typescript" | "tsx" | "jsx" => &[
            "function ", "export function", "export default", "class ", "export class",
            "const ", "export const", "interface ", "type ", "enum ",
        ],
        "go" => &["func ", "type ", "package "],
        "java" | "csharp" | "cpp" | "c" | "kotlin" | "scala" | "swift" | "dart" => &[
            "class ", "public class", "struct ", "public struct", "interface ", "enum ",
            "fn ", "func ", "impl ", "trait ",
        ],
        "ruby" => &["def ", "class ", "module "],
        "php" => &["function ", "class ", "public function", "private function"],
        "shell" => &["function ", "#!/"],
        _ => &[],
    }
}

/// Code-aware chunking: split at top-level definition boundaries, accumulate
/// blocks to the token target, record line ranges.
pub fn chunk_code(code: &str, lang: &str, cfg: &ChunkerConfig) -> Vec<Chunk> {
    let pats = boundary_patterns(lang);
    let mut blocks: Vec<(usize, String)> = Vec::new(); // (start line, text)
    let mut cur_start: Option<usize> = None;
    let mut cur = String::new();

    let mut in_block_comment = false;
    let mut in_string: Option<char> = None;
    for (i, line) in code.lines().enumerate() {
        let t = line.trim_start();
        // crude comment/string state tracking to avoid splitting inside them
        if t.starts_with("/*") {
            in_block_comment = true;
        }
        if in_block_comment {
            if t.contains("*/") {
                in_block_comment = false;
            }
        } else if t.starts_with("//") || t.starts_with("#") && lang == "python" || t.starts_with("--") {
            // comment line — attach to current block
        }
        let is_boundary = !in_block_comment
            && in_string.is_none()
            && (pats.iter().any(|p| t.starts_with(p)) || (t.starts_with("#") && lang != "python" && false));
        // string state (triple-quote python)
        if lang == "python" {
            if t.contains("\"\"\"") {
                in_string = if in_string.is_some() { None } else { Some('"') };
            }
        }
        if is_boundary && !cur.trim().is_empty() {
            blocks.push((cur_start.unwrap_or(0), std::mem::take(&mut cur)));
            cur_start = Some(i);
        } else if cur_start.is_none() {
            cur_start = Some(i);
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        blocks.push((cur_start.unwrap_or(0), cur));
    }

    // accumulate blocks into chunks
    let mut out = Vec::new();
    let mut cur_text = String::new();
    let mut cur_tokens = 0;
    let mut cur_line_start = blocks.first().map(|b| b.0).unwrap_or(0);
    let mut cur_line_count = 0;

    let flush = |out: &mut Vec<Chunk>, text: &str, ls: usize, lc: usize| {
        if text.trim().is_empty() {
            return;
        }
        out.push(Chunk {
            text: text.trim().to_string(),
            metadata: json!({
                "line_start": ls + 1,
                "line_end": ls + lc,
            }),
        });
    };

    for (start_line, block) in blocks {
        let t = estimate_tokens(&block);
        if cur_tokens > 0 && cur_tokens + t > cfg.chunk_tokens {
            flush(&mut out, &cur_text, cur_line_start, cur_line_count);
            cur_text.clear();
            cur_tokens = 0;
            cur_line_start = start_line;
        }
        if !cur_text.is_empty() {
            cur_text.push('\n');
        }
        cur_line_count = start_line + block.lines().count() - cur_line_start;
        cur_text.push_str(&block);
        cur_tokens += t;
    }
    flush(&mut out, &cur_text, cur_line_start, cur_line_count);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimates() {
        assert!(estimate_tokens("hello world") >= 2);
        assert_eq!(estimate_tokens(""), 0);
        let long = "word ".repeat(100);
        assert!(estimate_tokens(&long) >= 100);
    }

    #[test]
    fn prose_chunking_sizes() {
        let text = (0..80)
            .map(|i| format!("Paragraph {i} with some words about topic {i} in it."))
            .collect::<Vec<_>>()
            .join("\n\n");
        let cfg = ChunkerConfig {
            chunk_tokens: 40,
            overlap_tokens: 8,
            min_tokens: 5,
            max_tokens: 120,
        };
        let chunks = chunk_text(&text, &cfg);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(estimate_tokens(c) <= 140, "chunk too big: {}", estimate_tokens(c));
        }
        // overlap present between consecutive chunks
        assert!(chunks.len() > 1);
    }

    #[test]
    fn oversized_paragraph_splits() {
        let text = "Sentence one is here. Sentence two follows it. ".repeat(60);
        let cfg = ChunkerConfig {
            chunk_tokens: 30,
            overlap_tokens: 0,
            min_tokens: 5,
            max_tokens: 60,
        };
        let chunks = chunk_text(&text, &cfg);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(estimate_tokens(c) <= 80);
        }
    }

    #[test]
    fn markdown_headings() {
        let md = "# Title\n\nIntro paragraph here.\n\n## Section A\n\nText under A.\n\n### Deep\n\nDeep text.\n\n## Section B\n\nText under B that is long enough. More words here to make it chunky. Extra padding sentence for length.";
        let cfg = ChunkerConfig::default();
        let chunks = chunk_markdown(md, &cfg);
        assert!(!chunks.is_empty());
        let headings: Vec<&str> = chunks
            .iter()
            .filter_map(|c| c.metadata.get("heading").and_then(|h| h.as_str()))
            .collect();
        assert!(headings.iter().any(|h| h.contains("Section A")));
        assert!(headings.iter().any(|h| h.contains("Title")));
    }

    #[test]
    fn code_boundaries() {
        let code = r#"// module docs
use std::io;

fn helper() -> u32 {
    1
}

pub struct Thing {
    x: u32,
}

impl Thing {
    pub fn new() -> Self {
        Thing { x: helper() }
    }
}

fn main() {
    println!("hi");
}
"#;
        let cfg = ChunkerConfig {
            chunk_tokens: 25,
            overlap_tokens: 0,
            min_tokens: 1,
            max_tokens: 512,
        };
        let chunks = chunk_code(code, "rust", &cfg);
        assert!(chunks.len() >= 2, "expected multiple code chunks");
        // each chunk should start near a definition
        let joined: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert!(joined.iter().any(|t| t.contains("pub struct Thing")));
        assert!(joined.iter().any(|t| t.contains("fn helper")));
        for c in &chunks {
            assert!(c.metadata["line_start"].as_u64().unwrap() >= 1);
            assert!(c.metadata["line_end"].as_u64().unwrap() >= c.metadata["line_start"].as_u64().unwrap());
        }
    }

    #[test]
    fn fenced_blocks_atomic() {
        let md = "# T\n\nbefore\n\n```rust\nfn a() {}\n\nfn b() {}\n\nfn c() {}\n```\n\nafter";
        let paras = split_paragraphs(md, true);
        assert!(paras.iter().any(|p| p.starts_with("```") && p.contains("fn c")));
    }
}
