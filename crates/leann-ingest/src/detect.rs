//! Format detection by file extension + content sniffing.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Pdf,
    Txt,
    Markdown,
    Html,
    Docx,
    Epub,
    Code,
    Json,
    Audio,
    Video,
    Unknown,
}

impl Format {
    pub fn as_str(&self) -> &'static str {
        match self {
            Format::Pdf => "pdf",
            Format::Txt => "txt",
            Format::Markdown => "markdown",
            Format::Html => "html",
            Format::Docx => "docx",
            Format::Epub => "epub",
            Format::Code => "code",
            Format::Json => "json",
            Format::Audio => "audio",
            Format::Video => "video",
            Format::Unknown => "unknown",
        }
    }

    pub fn from_ext(ext: &str) -> Format {
        match ext.to_ascii_lowercase().as_str() {
            "pdf" => Format::Pdf,
            "txt" | "text" | "log" => Format::Txt,
            "md" | "markdown" | "mdx" => Format::Markdown,
            "html" | "htm" | "xhtml" => Format::Html,
            "docx" => Format::Docx,
            "epub" => Format::Epub,
            "rs" | "py" | "js" | "mjs" | "cjs" | "ts" | "tsx" | "jsx" | "go" | "java" | "c"
            | "h" | "cpp" | "cc" | "hpp" | "cs" | "rb" | "php" | "swift" | "kt" | "scala"
            | "sh" | "bash" | "zsh" | "sql" | "lua" | "r" | "m" | "pl" | "ex" | "exs"
            | "hs" | "clj" | "dart" | "zig" | "nim" | "v" | "vue" | "svelte" => Format::Code,
            "json" => Format::Json,
            "mp3" | "wav" | "m4a" | "aac" | "flac" | "ogg" | "opus" | "wma" | "aiff" => {
                Format::Audio
            }
            "mp4" | "mkv" | "mov" | "avi" | "webm" | "flv" | "wmv" | "m4v" => Format::Video,
            _ => Format::Unknown,
        }
    }

    /// Programming language for code files (metadata only).
    pub fn code_language(ext: &str) -> Option<&'static str> {
        Some(match ext.to_ascii_lowercase().as_str() {
            "rs" => "rust",
            "py" => "python",
            "js" | "mjs" | "cjs" => "javascript",
            "ts" => "typescript",
            "tsx" => "tsx",
            "jsx" => "jsx",
            "go" => "go",
            "java" => "java",
            "c" | "h" => "c",
            "cpp" | "cc" | "hpp" => "cpp",
            "cs" => "csharp",
            "rb" => "ruby",
            "php" => "php",
            "swift" => "swift",
            "kt" => "kotlin",
            "scala" => "scala",
            "sh" | "bash" | "zsh" => "shell",
            "sql" => "sql",
            "lua" => "lua",
            "r" => "r",
            "pl" => "perl",
            "hs" => "haskell",
            "clj" => "clojure",
            "dart" => "dart",
            "zig" => "zig",
            "vue" => "vue",
            "svelte" => "svelte",
            _ => return None,
        })
    }

    /// Sniff format from leading bytes.
    pub fn sniff(bytes: &[u8]) -> Format {
        if bytes.starts_with(b"%PDF") {
            return Format::Pdf;
        }
        if bytes.starts_with(b"PK\x03\x04") {
            // zip: docx vs epub vs other
            if let Some(f) = sniff_zip(bytes) {
                return f;
            }
            return Format::Unknown;
        }
        let head = String::from_utf8_lossy(&bytes[..bytes.len().min(512)]);
        let t = head.trim_start();
        if t.starts_with("<!DOCTYPE html")
            || t.starts_with("<html")
            || t.starts_with("<HTML")
            || t.starts_with("<?xml")
        {
            return Format::Html;
        }
        Format::Unknown
    }
}

fn sniff_zip(bytes: &[u8]) -> Option<Format> {
    // Look for epub mimetype or docx [Content_Types].xml in the first 2KB
    let window = &bytes[..bytes.len().min(4096)];
    let s = String::from_utf8_lossy(window);
    if s.contains("mimetypeapplication/epub+zip") || s.contains("application/epub") {
        return Some(Format::Epub);
    }
    if s.contains("word/") || s.contains("[Content_Types].xml") {
        return Some(Format::Docx);
    }
    None
}

/// Detect from a file path (extension first, sniff as fallback).
pub fn detect(path: &Path) -> Format {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let f = Format::from_ext(ext);
    if f != Format::Unknown {
        return f;
    }
    if let Ok(mut file) = std::fs::File::open(path) {
        use std::io::Read;
        let mut buf = [0u8; 4096];
        if let Ok(n) = file.read(&mut buf) {
            return Format::sniff(&buf[..n]);
        }
    }
    Format::Unknown
}
