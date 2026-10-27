//! Format parsers: bytes/paths -> extracted plain text (+ page boundaries
//! where available). Each returns `ParsedDoc` with format name and optional
//! per-page texts so chunks can carry `page` metadata.

use std::path::Path;

use serde_json::Value;

use crate::detect::Format;
use crate::{LeannIngestError, Result};

pub mod media;

/// A parsed document.
pub struct ParsedDoc {
    pub format: String,
    /// Full extracted text.
    pub text: String,
    /// Optional page/section boundaries: (label, text) — pages for pdf/docx.
    pub pages: Vec<(usize, String)>,
    /// Extra metadata discovered during parsing (title, language, duration...).
    pub extra: Vec<(String, Value)>,
}

// ---------------- txt / markdown / json ----------------

pub fn parse_txt(text: &str) -> ParsedDoc {
    ParsedDoc {
        format: "txt".into(),
        text: text.to_string(),
        pages: Vec::new(),
        extra: Vec::new(),
    }
}

pub fn parse_markdown(text: &str) -> ParsedDoc {
    ParsedDoc {
        format: "markdown".into(),
        text: text.to_string(),
        pages: Vec::new(),
        extra: Vec::new(),
    }
}

pub fn parse_json(bytes: &[u8]) -> Result<ParsedDoc> {
    let v: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| LeannIngestError::Parse(format!("invalid JSON: {e}")))?;
    // pretty-print into retrievable text; arrays of objects become JSONL
    let text = match &v {
        Value::Array(arr) => arr
            .iter()
            .map(|o| serde_json::to_string(o).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => serde_json::to_string_pretty(&v).unwrap_or_default(),
    };
    Ok(ParsedDoc {
        format: "json".into(),
        text,
        pages: Vec::new(),
        extra: Vec::new(),
    })
}

// ---------------- pdf ----------------

pub fn parse_pdf_bytes(bytes: &[u8]) -> Result<ParsedDoc> {
    let text = pdf_extract::extract_text_from_mem(bytes)
        .map_err(|e| LeannIngestError::Parse(format!("pdf extraction failed: {e}")))?;
    let pages = split_pdf_pages(&text);
    Ok(ParsedDoc {
        format: "pdf".into(),
        text: normalize_ws(&text),
        pages,
        extra: Vec::new(),
    })
}

pub fn parse_pdf_file(path: &Path) -> Result<ParsedDoc> {
    let text = pdf_extract::extract_text(path)
        .map_err(|e| LeannIngestError::Parse(format!("pdf extraction failed: {e}")))?;
    let pages = split_pdf_pages(&text);
    Ok(ParsedDoc {
        format: "pdf".into(),
        text: normalize_ws(&text),
        pages,
        extra: Vec::new(),
    })
}

/// pdf-extract emits form feeds between pages.
fn split_pdf_pages(text: &str) -> Vec<(usize, String)> {
    let parts: Vec<&str> = text.split('\u{0c}').collect();
    if parts.len() <= 1 {
        return Vec::new();
    }
    parts
        .iter()
        .enumerate()
        .map(|(i, p)| (i + 1, normalize_ws(p)))
        .filter(|(_, t)| !t.is_empty())
        .collect()
}

fn normalize_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_nl = false;
    let mut last_sp = false;
    for ch in s.chars() {
        match ch {
            '\n' => {
                if !last_nl {
                    out.push('\n');
                }
                last_nl = true;
                last_sp = false;
            }
            ' ' | '\t' | '\r' => {
                if !last_sp && !last_nl {
                    out.push(' ');
                }
                last_sp = true;
            }
            c => {
                out.push(c);
                last_nl = false;
                last_sp = false;
            }
        }
    }
    out.trim().to_string()
}

// ---------------- html ----------------

/// Lightweight HTML text extractor (no DOM dependency): strips script/style,
/// converts block tags to newlines, decodes entities.
pub fn parse_html(html: &str) -> ParsedDoc {
    let mut title = String::new();
    let mut text = String::new();

    // title
    if let Some(start) = find_ci(html, "<title") {
        if let Some(gt) = html[start..].find('>') {
            let rest = &html[start + gt + 1..];
            if let Some(end) = find_ci(rest, "</title") {
                title = decode_entities(rest[..end].trim());
            }
        }
    }

    // Remove script/style/noscript/template/svg blocks and comments.
    let mut cleaned = html.to_string();
    loop {
        let mut changed = false;
        for tag in ["script", "style", "noscript", "template", "svg"] {
            let end_tag = format!("</{tag}>");
            if let Some(s) = find_ci(&cleaned, &format!("<{tag}")) {
                // find closing tag (case-insensitive) after the opener
                if let Some(e) = find_ci(&cleaned[s..], &end_tag) {
                    let close_end = s + e + end_tag.len();
                    cleaned.replace_range(s..close_end.min(cleaned.len()), "");
                    changed = true;
                    break;
                } else {
                    // unterminated block: drop to end
                    cleaned.truncate(s);
                }
            }
        }
        // comments
        while let Some(s) = cleaned.find("<!--") {
            if let Some(e) = cleaned[s..].find("-->") {
                let end = s + e + 3;
                cleaned.replace_range(s..end, "");
                changed = true;
            } else {
                cleaned.truncate(s);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // tag -> text conversion
    let mut chars = cleaned.char_indices().peekable();
    let mut in_tag = false;
    let mut tag_name = String::new();
    while let Some((_, ch)) = chars.next() {
        if ch == '<' {
            in_tag = true;
            tag_name.clear();
            continue;
        }
        if in_tag {
            if ch == '>' {
                in_tag = false;
                let t = tag_name.to_ascii_lowercase();
                let t = t.split(|c: char| c.is_whitespace()).next().unwrap_or("");
                if matches!(
                    t,
                    "p" | "div" | "br" | "li" | "tr" | "table" | "section" | "article"
                        | "header" | "footer" | "nav" | "main" | "h1" | "h2" | "h3" | "h4"
                        | "h5" | "h6" | "ul" | "ol" | "blockquote" | "pre" | "hr"
                ) {
                    text.push('\n');
                }
            } else {
                tag_name.push(ch);
            }
        } else {
            text.push(ch);
        }
    }

    let decoded = decode_entities(&text);
    let mut extra = Vec::new();
    if !title.is_empty() {
        extra.push(("title".to_string(), Value::String(title)));
    }
    ParsedDoc {
        format: "html".into(),
        text: normalize_ws(&decoded),
        pages: Vec::new(),
        extra,
    }
}

fn find_ci(hay: &str, needle: &str) -> Option<usize> {
    let h = hay.to_ascii_lowercase();
    let n = needle.to_ascii_lowercase();
    h.find(&n)
}

pub fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let after = &rest[i..];
        if let Some(semi) = after.find(';').filter(|&x| x <= 12) {
            let ent = &after[1..semi];
            let rep = match ent {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "nbsp" => Some(' '),
                "mdash" => Some('—'),
                "ndash" => Some('–'),
                "hellip" => Some('…'),
                "laquo" => Some('«'),
                "raquo" => Some('»'),
                _ => {
                    if let Some(hex) = ent.strip_prefix("#x").or_else(|| ent.strip_prefix("#X")) {
                        u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
                    } else if let Some(dec) = ent.strip_prefix('#') {
                        dec.parse::<u32>().ok().and_then(char::from_u32)
                    } else {
                        None
                    }
                }
            };
            match rep {
                Some(c) => {
                    out.push(c);
                    rest = &after[semi + 1..];
                }
                None => {
                    out.push('&');
                    rest = &after[1..];
                }
            }
        } else {
            out.push('&');
            rest = &after[1..];
        }
    }
    out.push_str(rest);
    out
}

// ---------------- docx ----------------

pub fn parse_docx(bytes: &[u8]) -> Result<ParsedDoc> {
    let cursor = std::io::Cursor::new(bytes);
    let mut zip = zip::ZipArchive::new(cursor)
        .map_err(|e| LeannIngestError::Parse(format!("docx (zip) open failed: {e}")))?;
    let mut doc = zip
        .by_name("word/document.xml")
        .map_err(|_| LeannIngestError::Parse("docx: missing word/document.xml".into()))?;
    let mut xml = String::new();
    std::io::Read::read_to_string(&mut doc, &mut xml)
        .map_err(|e| LeannIngestError::Parse(format!("docx read failed: {e}")))?;

    // minimal OOXML text extraction: <w:t> runs, </w:p> paragraphs,
    // page breaks become page boundaries
    let mut text = String::new();
    let mut pages: Vec<String> = vec![String::new()];
    let mut i = 0;
    let bytes = xml.as_bytes();
    while i < xml.len() {
        if bytes[i] == b'<' {
            let end = xml[i..].find('>').map(|x| i + x + 1).unwrap_or(xml.len());
            let tag = &xml[i..end];
            if tag.starts_with("<w:t") {
                // text run
                if let Some(close) = xml[end..].find("</w:t>") {
                    let content = &xml[end..end + close];
                    text.push_str(&decode_xml(content));
                    pages.last_mut().unwrap().push_str(&decode_xml(content));
                }
            } else if tag.starts_with("</w:p>") || tag.starts_with("<w:br") || tag.starts_with("</w:tc>") {
                let page_break = tag.contains("w:type=\"page\"");
                text.push('\n');
                let last = pages.last_mut().unwrap();
                if page_break {
                    pages.push(String::new());
                } else if !last.is_empty() {
                    last.push('\n');
                }
            }
            i = end;
        } else {
            let next = xml[i..].find('<').map(|x| i + x).unwrap_or(xml.len());
            i = next;
        }
    }
    let page_list: Vec<(usize, String)> = pages
        .iter()
        .enumerate()
        .map(|(i, p)| (i + 1, normalize_ws(p)))
        .filter(|(_, t)| !t.is_empty())
        .collect();
    Ok(ParsedDoc {
        format: "docx".into(),
        text: normalize_ws(&text),
        pages: page_list,
        extra: Vec::new(),
    })
}

fn decode_xml(s: &str) -> String {
    decode_entities(s)
}

// ---------------- epub ----------------

pub fn parse_epub(bytes: &[u8]) -> Result<ParsedDoc> {
    let cursor = std::io::Cursor::new(bytes);
    let mut zip = zip::ZipArchive::new(cursor)
        .map_err(|e| LeannIngestError::Parse(format!("epub (zip) open failed: {e}")))?;

    // 1) container.xml -> rootfile path
    let mut container = zip
        .by_name("META-INF/container.xml")
        .map_err(|_| LeannIngestError::Parse("epub: missing container.xml".into()))?;
    let mut cxml = String::new();
    std::io::Read::read_to_string(&mut container, &mut cxml)?;
    drop(container);
    let opf_path = extract_attr(&cxml, "rootfile", "full-path")
        .ok_or_else(|| LeannIngestError::Parse("epub: no rootfile".into()))?;

    // 2) OPF: metadata + manifest + spine
    let mut opf = zip
        .by_name(&opf_path)
        .map_err(|_| LeannIngestError::Parse("epub: missing OPF".into()))?;
    let mut opf_xml = String::new();
    std::io::Read::read_to_string(&mut opf, &mut opf_xml)?;
    drop(opf);
    let base_dir = match opf_path.rfind('/') {
        Some(i) => format!("{}/", &opf_path[..i]),
        None => String::new(),
    };

    let title = extract_tag_text(&opf_xml, "dc:title");
    let mut extra = Vec::new();
    if let Some(t) = &title {
        extra.push(("title".to_string(), Value::String(t.clone())));
    }

    // manifest: id -> href
    let mut manifest: Vec<(String, String)> = Vec::new();
    for item_xml in find_elements(&opf_xml, "item") {
        let id = extract_attr(&item_xml, "item", "id").unwrap_or_default();
        let href = extract_attr(&item_xml, "item", "href").unwrap_or_default();
        let media = extract_attr(&item_xml, "item", "media-type").unwrap_or_default();
        if media.contains("html") || media.contains("xml") {
            manifest.push((id, href));
        }
    }
    // spine order
    let spine: Vec<String> = find_elements(&opf_xml, "itemref")
        .iter()
        .filter_map(|r| extract_attr(r, "itemref", "idref"))
        .collect();

    // 3) read spine documents in order
    let mut text = String::new();
    let mut sections: Vec<(usize, String)> = Vec::new();
    for (idx, idref) in spine.iter().enumerate() {
        let href = match manifest.iter().find(|(id, _)| id == idref) {
            Some((_, h)) => h,
            None => continue,
        };
        let path = format!("{base_dir}{href}");
        if let Ok(mut f) = zip.by_name(&path) {
            let mut html = String::new();
            if std::io::Read::read_to_string(&mut f, &mut html).is_ok() {
                let parsed = parse_html(&html);
                if !parsed.text.is_empty() {
                    sections.push((idx + 1, parsed.text.clone()));
                    text.push_str(&parsed.text);
                    text.push_str("\n\n");
                }
            }
        }
    }
    Ok(ParsedDoc {
        format: "epub".into(),
        text: normalize_ws(&text),
        pages: sections, // chapters treated as "pages"
        extra,
    })
}

fn extract_attr(xml: &str, _elem: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=");
    let i = xml.find(&needle)?;
    let rest = &xml[i + needle.len()..];
    let q = rest.chars().next()?;
    if q == '"' || q == '\'' {
        let end = rest[1..].find(q)? + 1;
        Some(decode_xml(&rest[1..end]))
    } else {
        None
    }
}

fn find_elements<'a>(xml: &'a str, elem: &str) -> Vec<String> {
    let mut out = Vec::new();
    let open = format!("<{elem}");
    let close = format!("</{elem}>");
    let mut rest = xml;
    while let Some(s) = rest.find(&open) {
        let after = &rest[s..];
        // self-closing?
        if let Some(gt) = after.find('>') {
            if after[..gt].ends_with('/') {
                out.push(after[..=gt].to_string());
                rest = &after[gt + 1..];
                continue;
            }
        }
        if let Some(e) = after.find(&close) {
            out.push(after[..e + close.len()].to_string());
            rest = &after[e + close.len()..];
        } else {
            break;
        }
    }
    out
}

fn extract_tag_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let s = xml.find(&open)?;
    let gt = xml[s..].find('>')? + s + 1;
    let e = xml[gt..].find(&close)? + gt;
    Some(decode_xml(xml[gt..e].trim()))
}

// ---------------- code ----------------

pub fn parse_code(code: &str, language: &str) -> ParsedDoc {
    let mut extra = Vec::new();
    if !language.is_empty() {
        extra.push(("language".to_string(), Value::String(language.to_string())));
    }
    extra.push((
        "lines".to_string(),
        Value::from(code.lines().count() as u64),
    ));
    ParsedDoc {
        format: "code".into(),
        text: code.to_string(),
        pages: Vec::new(),
        extra,
    }
}

// ---------------- entry point ----------------

pub fn parse_bytes(name: &str, bytes: &[u8], format: Format) -> Result<ParsedDoc> {
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match format {
        Format::Pdf => parse_pdf_bytes(bytes),
        Format::Txt => Ok(parse_txt(&String::from_utf8_lossy(bytes))),
        Format::Markdown => Ok(parse_markdown(&String::from_utf8_lossy(bytes))),
        Format::Html => Ok(parse_html(&String::from_utf8_lossy(bytes))),
        Format::Docx => parse_docx(bytes),
        Format::Epub => parse_epub(bytes),
        Format::Json => parse_json(bytes),
        Format::Code => Ok(parse_code(
            &String::from_utf8_lossy(bytes),
            Format::code_language(ext).unwrap_or(""),
        )),
        Format::Audio | Format::Video | Format::Unknown => Err(LeannIngestError::Unsupported(format!(
            "format {format:?} needs a file path (media) or is unknown: {name}"
        ))),
    }
}

pub fn parse_file(path: &Path, format: Format) -> Result<ParsedDoc> {
    let _ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match format {
        Format::Pdf => parse_pdf_file(path),
        _ => {
            let bytes = std::fs::read(path)?;
            parse_bytes(&path.to_string_lossy(), &bytes, format)
        }
    }
}
