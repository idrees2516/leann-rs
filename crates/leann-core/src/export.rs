//! Export / import helpers for LEANN indexes.
//!
//! Real-world RAG systems rarely live in isolation: corpora move between
//! environments, get audited in spreadsheets, feed other pipelines, or are
//! archived in human-readable form. This module converts the alive chunks of a
//! [`LeannIndex`](crate::index::LeannIndex) into three portable formats:
//!
//! * **JSON** — lossless, machine-readable, round-trips through
//!   [`import_json`] into any other index.
//! * **CSV** — spreadsheet-friendly (RFC 4180 quoting, one row per chunk).
//! * **Markdown** — human-readable documentation dumps.
//!
//! Deleted (tombstoned) chunks are always skipped; exports reflect the live
//! corpus only.

use serde_json::{json, Map, Value};

use crate::index::LeannIndex;

/// Export format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    Csv,
    Markdown,
}

impl ExportFormat {
    /// Parse a format name (case-insensitive); accepts `md` as an alias.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Some(Self::Json),
            "csv" => Some(Self::Csv),
            "markdown" | "md" => Some(Self::Markdown),
            _ => None,
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Markdown => "md",
        }
    }
}

/// Collect the alive chunks of an index as `(id, text, metadata)` triples.
pub fn alive_chunks(idx: &LeannIndex) -> Vec<(u32, String, Value)> {
    let mut out = Vec::with_capacity(idx.alive_count());
    for id in 0..idx.len() as u32 {
        if idx.is_alive(id) {
            out.push((id, idx.text(id).to_string(), idx.metadata(id).clone()));
        }
    }
    out
}

/// Export the live corpus as a self-describing JSON document.
///
/// ```json
/// { "format": "leann-export/v1", "count": 2, "chunks": [ ... ] }
/// ```
pub fn export_json(idx: &LeannIndex) -> Value {
    let chunks = alive_chunks(idx);
    json!({
        "format": "leann-export/v1",
        "count": chunks.len(),
        "chunks": chunks
            .iter()
            .map(|(id, text, meta)| json!({
                "id": id,
                "text": text,
                "metadata": meta,
            }))
            .collect::<Vec<_>>(),
    })
}

/// Export the live corpus as CSV (header row + one row per chunk).
///
/// Columns: `id,text,metadata`. Fields are RFC 4180 quoted on demand so
/// embedded newlines/commas/quotes never break the layout.
pub fn export_csv(idx: &LeannIndex) -> String {
    fn esc(field: &str) -> String {
        if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r')
        {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    }
    let mut out = String::from("id,text,metadata\n");
    for (id, text, meta) in alive_chunks(idx) {
        let meta_str = serde_json::to_string(&meta).unwrap_or_else(|_| "{}".into());
        out.push_str(&format!("{},{},{}\n", id, esc(&text), esc(&meta_str)));
    }
    out
}

/// Export the live corpus as a Markdown document (one section per chunk).
pub fn export_markdown(idx: &LeannIndex) -> String {
    let mut out = String::from("# LEANN corpus export\n\n");
    let chunks = alive_chunks(idx);
    out.push_str(&format!("_{} live chunks_\n\n", chunks.len()));
    for (id, text, meta) in chunks {
        out.push_str(&format!("## chunk {id}\n\n"));
        let meta_str = serde_json::to_string_pretty(&meta).unwrap_or_else(|_| "{}".into());
        out.push_str(&format!("```json\n{meta_str}\n```\n\n{text}\n\n"));
    }
    out
}

/// Export into an in-memory string for the requested format.
pub fn export_string(idx: &LeannIndex, format: ExportFormat) -> String {
    match format {
        ExportFormat::Json => serde_json::to_string_pretty(&export_json(idx)).unwrap_or_default(),
        ExportFormat::Csv => export_csv(idx),
        ExportFormat::Markdown => export_markdown(idx),
    }
}

/// Parsed chunk from a JSON export.
#[derive(Debug, Clone)]
pub struct ImportedChunk {
    pub text: String,
    pub metadata: Value,
}

/// Parse a `leann-export/v1` JSON document (or a bare array of chunks) back
/// into ingestable `(text, metadata)` pairs. `id` fields are ignored: a fresh
/// index assigns its own ids.
pub fn import_json(json_str: &str) -> Result<Vec<ImportedChunk>, String> {
    let v: Value = serde_json::from_str(json_str).map_err(|e| format!("invalid JSON: {e}"))?;
    let empty = Map::new();
    let items: Vec<Value> = match &v {
        Value::Array(items) => items.clone(),
        Value::Object(map) => match map.get("chunks") {
            Some(Value::Array(items)) => items.clone(),
            _ => {
                return Err(
                    "expected a top-level array or an object with a \"chunks\" array".into(),
                )
            }
        },
        _ => return Err("expected a JSON array or export object".into()),
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let obj = item.as_object().unwrap_or(&empty);
        let text = obj
            .get("text")
            .and_then(Value::as_str)
            .ok_or("chunk missing string field \"text\"")?;
        if text.trim().is_empty() {
            return Err("chunk has empty text".into());
        }
        let metadata = obj.get("metadata").cloned().unwrap_or(Value::Null);
        out.push(ImportedChunk {
            text: text.to_string(),
            metadata,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::EmbeddingProvider;
    use crate::index::{IndexConfig, LeannIndex};
    use std::sync::Arc;

    /// Deterministic 8-dim character-bag embedder (test only).
    struct BagProvider;
    impl EmbeddingProvider for BagProvider {
        fn name(&self) -> &str {
            "bag"
        }
        fn dim(&self) -> usize {
            8
        }
        fn embed_batch(&self, texts: &[&str]) -> crate::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0f32; 8];
                    for (i, b) in t.bytes().enumerate() {
                        v[i % 8] += b as f32;
                    }
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                    v.iter().map(|x| x / n).collect()
                })
                .collect())
        }
    }

    fn tiny_index() -> LeannIndex {
        LeannIndex::build(
            &["hello, \"world\"\nsecond line".to_string(), "plain chunk two".to_string()],
            None,
            Arc::new(BagProvider),
            IndexConfig::default(),
        )
        .expect("build")
    }

    #[test]
    fn export_json_roundtrip() {
        let idx = tiny_index();
        let json = export_json(&idx);
        assert_eq!(json["count"], 2);
        let s = serde_json::to_string(&json).unwrap();
        let imported = import_json(&s).expect("import");
        assert_eq!(imported.len(), 2);
        assert_eq!(imported[0].text, "hello, \"world\"\nsecond line");
    }

    #[test]
    fn export_csv_quotes_correctly() {
        let idx = tiny_index();
        let csv = export_csv(&idx);
        assert!(csv.starts_with("id,text,metadata\n"));
        // embedded quotes are doubled, embedded newline is kept inside quotes
        assert!(csv.contains("\"hello, \"\"world\"\"\nsecond line\""));
    }

    #[test]
    fn export_markdown_renders_sections() {
        let idx = tiny_index();
        let md = export_markdown(&idx);
        assert!(md.contains("# LEANN corpus export"));
        assert!(md.contains("## chunk 0"));
        assert!(md.contains("plain chunk two"));
    }

    #[test]
    fn export_string_dispatch() {
        let idx = tiny_index();
        assert!(export_string(&idx, ExportFormat::Json).starts_with('{'));
        assert!(export_string(&idx, ExportFormat::Csv).contains("id,text"));
        assert!(export_string(&idx, ExportFormat::Markdown).contains("# LEANN"));
        assert_eq!(ExportFormat::parse("MD"), Some(ExportFormat::Markdown));
        assert_eq!(ExportFormat::parse("nope"), None);
    }

    #[test]
    fn import_rejects_bad_documents() {
        assert!(import_json("not json").is_err());
        assert!(import_json("[{\"metadata\":{}}]").is_err()); // missing text
        assert!(import_json("[{\"text\":\"ok\"}]").is_ok());
    }
}
