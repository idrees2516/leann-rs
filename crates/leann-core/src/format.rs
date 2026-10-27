//! Single-file on-disk index format with mmap support.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! magic "LEANNIDX" | u32 version | u32 metric | u32 dim | u32 pq_m | u32 ksub
//! u64 meta_len | meta JSON (config, counts, embedder info)
//! --- sections ---
//! metadatas: u64 len | JSON array of per-chunk metadata
//! texts:      u64 n | u64 offsets[n+1] | utf8 blob
//! codebook:   f32[m * ksub * subdim]
//! codes:      u8[n * m]
//! graph:      u32 n | u8 max_level | u32 entry | u8 levels[n]
//!             u64 upper_lists | { u32 node u8 layer u32 len u32 ids[len] }*
//!             u32 indptr[n+1] | u32 indices[E]
//! tombstones: u64 words | u64 words[n]
//! ```
//!
//! The file is intentionally a **single portable artifact** — texts, PQ
//! codebook + codes, graph and deletion bitset in one blob, no side files.
//! (This is our answer to memvid's "one MP4 file" story, at ~5% of the size
//! of a conventional vector index.)
//!
//! Texts are read zero-copy from the mmap; typed arrays are copied once at
//! load (they are tiny by design — PQ codes are `m` bytes per chunk).

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use serde_json::Value;

use crate::graph::HnswGraph;
use crate::index::IndexConfig;
use crate::metric::Metric;
use crate::pq::PqCodec;
use crate::{LeannError, Result};

pub const MAGIC: &[u8; 8] = b"LEANNIDX";
pub const FORMAT_VERSION: u32 = 1;

/// Chunk text storage: owned strings or zero-copy mmap view.
pub enum TextStore {
    Owned(Vec<String>),
    Mapped {
        mmap: Arc<Mmap>,
        base: usize,
        offsets: Vec<u64>,
    },
}

impl TextStore {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TextStore::Owned(v) => v.len(),
            TextStore::Mapped { offsets, .. } => offsets.len().saturating_sub(1),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn text(&self, id: u32) -> &str {
        match self {
            TextStore::Owned(v) => &v[id as usize],
            TextStore::Mapped { mmap, base, offsets } => {
                let s = offsets[id as usize] as usize;
                let e = offsets[id as usize + 1] as usize;
                let bytes = &mmap[base + s..base + e];
                std::str::from_utf8(bytes).expect("stored texts are valid utf8")
            }
        }
    }

    /// Convert to owned storage (needed before appending new chunks).
    pub fn to_owned(&mut self) {
        if let TextStore::Mapped { .. } = self {
            let owned: Vec<String> = (0..self.len() as u32).map(|i| self.text(i).to_string()).collect();
            *self = TextStore::Owned(owned);
        }
    }

    pub fn push(&mut self, s: String) {
        self.to_owned();
        if let TextStore::Owned(v) = self {
            v.push(s);
        }
    }

    pub fn total_bytes(&self) -> usize {
        match self {
            TextStore::Owned(v) => v.iter().map(|s| s.len()).sum(),
            TextStore::Mapped { offsets, .. } => {
                offsets.last().copied().unwrap_or(0) as usize
            }
        }
    }
}

/// Borrowed view of everything needed to persist an index (no clones).
pub struct IndexFile<'a> {
    pub config: &'a IndexConfig,
    pub texts: &'a TextStore,
    pub metadatas: &'a [Value],
    pub codec: &'a PqCodec,
    pub codes: &'a [u8],
    pub graph: &'a HnswGraph,
    pub tombstones: &'a [u64],
    pub embedder_name: &'a str,
}

// ---------- writer ----------

struct W<B: Write> {
    out: B,
}

impl<B: Write> W<B> {
    fn u32(&mut self, v: u32) -> std::io::Result<()> {
        self.out.write_all(&v.to_le_bytes())
    }
    fn u64(&mut self, v: u64) -> std::io::Result<()> {
        self.out.write_all(&v.to_le_bytes())
    }
    fn u8(&mut self, v: u8) -> std::io::Result<()> {
        self.out.write_all(&[v])
    }
    fn bytes(&mut self, v: &[u8]) -> std::io::Result<()> {
        self.out.write_all(v)
    }
    fn f32s(&mut self, v: &[f32]) -> std::io::Result<()> {
        for x in v {
            self.out.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
    fn u32s(&mut self, v: &[u32]) -> std::io::Result<()> {
        for x in v {
            self.out.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
    fn u64s(&mut self, v: &[u64]) -> std::io::Result<()> {
        for x in v {
            self.out.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
}

/// Write an index atomically (tmp file + rename).
pub fn write_index(path: &Path, data: IndexFile) -> Result<()> {
    let n = data.texts.len();
    if n != data.metadatas.len() || n != data.graph.n {
        return Err(LeannError::Invalid(format!(
            "inconsistent index parts: texts={} metas={} graph={}",
            n,
            data.metadatas.len(),
            data.graph.n
        )));
    }
    let tmp = path.with_extension("tmp");
    {
        let file = File::create(&tmp)?;
        let mut w = W {
            out: BufWriter::with_capacity(1 << 20, file),
        };

        // header
        w.bytes(MAGIC)?;
        w.u32(FORMAT_VERSION)?;
        w.u32(data.config.metric as u32)?;
        w.u32(data.codec.dim as u32)?;
        w.u32(data.codec.m as u32)?;
        w.u32(data.codec.ksub as u32)?;

        // meta json
        let meta = serde_json::json!({
            "config": data.config,
            "n": n,
            "deleted": count_bits(data.tombstones),
            "embedder": data.embedder_name,
            "format_version": FORMAT_VERSION,
        });
        let meta_bytes = serde_json::to_vec(&meta)?;
        w.u64(meta_bytes.len() as u64)?;
        w.bytes(&meta_bytes)?;

        // metadatas (JSON array)
        let metas = serde_json::to_vec(&data.metadatas)?;
        w.u64(metas.len() as u64)?;
        w.bytes(&metas)?;

        // texts
        w.u64(n as u64)?;
        let mut offsets: Vec<u64> = Vec::with_capacity(n + 1);
        let mut blob: Vec<u8> = Vec::new();
        let mut off = 0u64;
        offsets.push(0);
        for i in 0..n as u32 {
            let t = data.texts.text(i);
            blob.extend_from_slice(t.as_bytes());
            off += t.len() as u64;
            offsets.push(off);
        }
        w.u64s(&offsets)?;
        w.bytes(&blob)?;

        // codebook (flattened)
        let flat_cb: Vec<f32> = data.codec.codebook.iter().flatten().copied().collect();
        w.f32s(&flat_cb)?;

        // codes
        debug_assert_eq!(data.codes.len(), n * data.codec.m);
        w.bytes(&data.codes)?;

        // graph
        let g = &data.graph;
        w.u32(g.n as u32)?;
        w.u8(g.max_level)?;
        w.u32(g.entry)?;
        for &lv in &g.levels {
            w.u8(lv)?;
        }
        // upper-layer lists (only nodes with level >= 1)
        let mut upper_lists = 0u64;
        for (node, lists) in g.upper.iter().enumerate() {
            for (li, lst) in lists.iter().enumerate() {
                if !lst.is_empty() {
                    upper_lists += 1;
                }
                let _ = (node, li);
            }
        }
        w.u64(upper_lists)?;
        for (node, lists) in g.upper.iter().enumerate() {
            for (li, lst) in lists.iter().enumerate() {
                if !lst.is_empty() {
                    w.u32(node as u32)?;
                    w.u8((li + 1) as u8)?;
                    w.u32(lst.len() as u32)?;
                    w.u32s(lst)?;
                }
            }
        }
        // layer-0 CSR
        let mut indptr: Vec<u32> = Vec::with_capacity(g.n + 1);
        indptr.push(0);
        let mut indices: Vec<u32> = Vec::new();
        for a in &g.adj0 {
            indices.extend_from_slice(a);
            indptr.push(indices.len() as u32);
        }
        w.u32s(&indptr)?;
        w.u32s(&indices)?;

        // tombstones
        w.u64(data.tombstones.len() as u64)?;
        w.u64s(&data.tombstones)?;

        w.out.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn count_bits(words: &[u64]) -> u64 {
    words.iter().map(|w| w.count_ones() as u64).sum()
}

// ---------- reader ----------

struct R<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(LeannError::Corrupt(format!(
                "unexpected EOF at {} (+{n}) of {}",
                self.pos,
                self.buf.len()
            )));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        let b = self.take(n * 4)?;
        Ok(b
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    fn u64s(&mut self, n: usize) -> Result<Vec<u64>> {
        let b = self.take(n * 8)?;
        Ok(b
            .chunks_exact(8)
            .map(|c| {
                u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            })
            .collect())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let b = self.take(n * 4)?;
        Ok(b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// Loaded index parts plus the retained mmap (for zero-copy texts).
pub struct LoadedIndex {
    pub config: IndexConfig,
    pub texts: TextStore,
    pub metadatas: Vec<Value>,
    pub codec: PqCodec,
    pub codes: Vec<u8>,
    pub graph: HnswGraph,
    pub tombstones: Vec<u64>,
    pub embedder_name: String,
    pub deleted_count: u64,
    pub file_size: u64,
}

/// Memory-map and parse an index file.
pub fn read_index(path: &Path) -> Result<LoadedIndex> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mmap = Arc::new(unsafe { Mmap::map(&file)? });
    let buf: &[u8] = &mmap;
    let mut r = R { buf, pos: 0 };

    // header
    let magic = r.take(8)?;
    if magic != MAGIC {
        return Err(LeannError::Corrupt("bad magic".into()));
    }
    let version = r.u32()?;
    if version != FORMAT_VERSION {
        return Err(LeannError::Version {
            found: version,
            expected: FORMAT_VERSION,
        });
    }
    let metric_id = r.u32()?;
    let metric = Metric::try_from(metric_id).map_err(|_| LeannError::Corrupt("bad metric id".into()))?;
    let dim = r.u32()? as usize;
    let pq_m = r.u32()? as usize;
    let ksub = r.u32()? as usize;

    // meta json
    let meta_len = r.u64()? as usize;
    let meta: Value = serde_json::from_slice(r.take(meta_len)?)?;
    let n = meta["n"].as_u64().ok_or_else(|| LeannError::Corrupt("missing n".into()))? as usize;
    let deleted_count = meta["deleted"].as_u64().unwrap_or(0);
    let embedder_name = meta["embedder"].as_str().unwrap_or("unknown").to_string();
    let config: IndexConfig =
        serde_json::from_value(meta["config"].clone()).map_err(|e| LeannError::Corrupt(format!("config: {e}")))?;

    // metadatas
    let metas_len = r.u64()? as usize;
    let metadatas: Vec<Value> = serde_json::from_slice(r.take(metas_len)?)
        .map_err(|e| LeannError::Corrupt(format!("metadatas: {e}")))?;

    // texts: record the section base for zero-copy reads
    let n_stored = r.u64()? as usize;
    if n_stored != n {
        return Err(LeannError::Corrupt(format!("text count mismatch {n_stored} != {n}")));
    }
    let offsets = r.u64s(n + 1)?;
    let blob_len = *offsets.last().ok_or_else(|| LeannError::Corrupt("empty offsets".into()))? as usize;
    let blob = r.take(blob_len)?;
    let texts_base = blob.as_ptr() as usize - buf.as_ptr() as usize;

    // codebook
    if dim == 0 || pq_m == 0 || dim % pq_m != 0 {
        return Err(LeannError::Corrupt("bad pq geometry".into()));
    }
    let subdim = dim / pq_m;
    let cb_len = pq_m * ksub * subdim;
    let flat_cb = r.f32s(cb_len)?;
    let mut codebook = Vec::with_capacity(pq_m);
    for j in 0..pq_m {
        codebook.push(flat_cb[j * ksub * subdim..(j + 1) * ksub * subdim].to_vec());
    }
    let codec = PqCodec {
        m: pq_m,
        ksub,
        dim,
        subdim,
        codebook,
        metric,
    };

    // codes
    let codes = r.take(n * pq_m)?.to_vec();

    // graph
    let gn = r.u32()? as usize;
    if gn != n {
        return Err(LeannError::Corrupt(format!("graph n mismatch {gn} != {n}")));
    }
    let max_level = r.u8()?;
    let entry = r.u32()?;
    let levels = r.take(n)?.to_vec();
    let mut graph = HnswGraph::with_capacity(n);
    graph.max_level = max_level;
    graph.entry = entry;
    graph.levels = levels;
    graph.mark_complete(n);

    let upper_lists = r.u64()? as usize;
    for _ in 0..upper_lists {
        let node = r.u32()? as usize;
        let layer = r.u8()? as usize;
        let len = r.u32()? as usize;
        let ids = r.u32s(len)?;
        if node >= n || layer == 0 || layer > 24 {
            return Err(LeannError::Corrupt("bad upper list".into()));
        }
        while graph.upper[node].len() < layer {
            graph.upper[node].push(Vec::new());
        }
        graph.upper[node][layer - 1] = ids;
    }
    let indptr = r.u32s(n + 1)?;
    if *indptr.last().unwrap_or(&0) as usize > buf.len() {
        return Err(LeannError::Corrupt("bad indptr".into()));
    }
    let e_total = *indptr.last().ok_or_else(|| LeannError::Corrupt("empty indptr".into()))? as usize;
    let indices = r.u32s(e_total)?;
    for i in 0..n {
        let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
        graph.adj0[i] = indices[s..e].to_vec();
    }

    // tombstones
    let tw = r.u64()? as usize;
    let tombstones = r.u64s(tw)?;

    if metadatas.len() != n {
        return Err(LeannError::Corrupt(format!("metadata count mismatch {}", metadatas.len())));
    }

    Ok(LoadedIndex {
        config,
        texts: TextStore::Mapped {
            mmap,
            base: texts_base,
            offsets,
        },
        metadatas,
        codec,
        codes,
        graph,
        tombstones,
        embedder_name,
        deleted_count,
        file_size,
    })
}

/// Open with a buffered reader wrapper (unused mmap path kept for API parity).
#[allow(dead_code)]
pub fn read_index_buffered(path: &Path) -> Result<LoadedIndex> {
    // mmap is strictly better here; kept for completeness
    read_index(path)
}

/// Quick inspection without full parse (magic + meta JSON).
pub fn peek_meta(path: &Path) -> Result<Value> {
    let file = File::open(path)?;
    let mut br = BufReader::new(file);
    let mut header = [0u8; 8 + 4 * 5 + 8];
    std::io::Read::read_exact(&mut br, &mut header)?;
    if &header[0..8] != MAGIC {
        return Err(LeannError::Corrupt("bad magic".into()));
    }
    let meta_len = u64::from_le_bytes(
        header[28..36]
            .try_into()
            .map_err(|_| LeannError::Corrupt("short header".into()))?,
    ) as usize;
    let mut meta_buf = vec![0u8; meta_len];
    std::io::Read::read_exact(&mut br, &mut meta_buf)?;
    Ok(serde_json::from_slice(&meta_buf)?)
}

impl Metric {
    fn try_from(id: u32) -> std::result::Result<Metric, ()> {
        match id {
            0 => Ok(Metric::L2),
            1 => Ok(Metric::Cosine),
            2 => Ok(Metric::InnerProduct),
            _ => Err(()),
        }
    }
}
