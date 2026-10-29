//! QR encoding / rendering / decoding.
//!
//! Text chunks become QR codes rendered as 8-bit grayscale frames (one QR per
//! video frame). Frames are decoded back with the pure-Rust `rqrr` decoder, so
//! retrieval needs no network and no external services — exactly the property
//! that makes video-as-database viable for offline and air-gapped deployments.

use qrcode::{EcLevel, QrCode};

use crate::VisualError;

/// Hard cap that fits a QR version 40 / EC-L codeword in byte mode.
pub const MAX_CHUNK_BYTES: usize = 2900;

/// Per-frame framing header: `[flags:1][seq_hi:1][seq_lo:1][len_hi:1][len_lo:1]`.
/// `flags & 1` marks the final frame of a chunk. The 16-bit `seq` lets the
/// reader detect stitching errors, and `len` guards truncated payloads.
pub const FRAME_HEADER: usize = 5;

/// Maximum payload bytes carried by one QR frame.
pub const MAX_PAYLOAD: usize = MAX_CHUNK_BYTES - FRAME_HEADER;

/// Error-correction level used for every code. Videos are written losslessly,
/// so level L maximizes payload while still absorbing minor pixel damage.
pub const EC_LEVEL: EcLevel = EcLevel::L;

/// Frame a whole chunk into QR-encodable byte vectors.
///
/// The qrcode crate's automatic mode segmentation (numeric / alphanumeric /
/// byte runs) adds per-run overhead, so a nominally in-capacity payload of
/// mixed content can still overflow a QR version. This function tries a
/// ladder of shrinking piece sizes until every frame encodes cleanly —
/// deterministic, always terminating, and exact (stitched payloads reproduce
/// the input byte-for-byte).
pub fn frame_chunk(text: &str) -> Result<Vec<Vec<u8>>, VisualError> {
    let data = text.as_bytes();
    if data.is_empty() {
        return Err(VisualError::EmptyChunk);
    }
    for &cap in &[MAX_PAYLOAD, MAX_PAYLOAD / 2, MAX_PAYLOAD / 4, 512usize, 128, 24] {
        if let Some(framed) = try_frame_with_cap(data, cap) {
            return Ok(framed);
        }
    }
    Err(VisualError::Qr("chunk cannot be framed even at minimum frame size".into()))
}

fn try_frame_with_cap(data: &[u8], cap: usize) -> Option<Vec<Vec<u8>>> {
    let mut pieces: Vec<&[u8]> = Vec::new();
    let mut start = 0usize;
    while start < data.len() {
        let mut end = (start + cap).min(data.len());
        while end > start && end < data.len() && data[end] & 0xC0 == 0x80 {
            end -= 1; // keep UTF-8 characters whole
        }
        pieces.push(&data[start..end]);
        start = end;
    }
    let last = pieces.len() - 1;
    let mut framed = Vec::with_capacity(pieces.len());
    for (seq, piece) in pieces.iter().enumerate() {
        let bytes = build_frame_bytes(piece, seq, seq == last);
        // Two hard requirements, enforced for every frame:
        //  1. QR-encodable (capacity incl. mode-segmentation overhead)
        //  2. valid UTF-8 as a whole (string-returning QR decoders)
        if QrCode::with_error_correction_level(&bytes, EC_LEVEL).is_err()
            || std::str::from_utf8(&bytes).is_err()
        {
            return None; // retry with a smaller cap
        }
        framed.push(bytes);
    }
    Some(framed)
}

/// Build the framed bytes for frame `seq` of a chunk (`last` = final frame).
///
/// Every header byte is ≤ 0x7F (ASCII-safe): `seq` and `len` are stored as
/// 7-bit big-endian pairs, so a framed buffer is valid UTF-8 whenever its
/// payload is — a hard requirement for QR decoders (rqrr, phone cameras)
/// that return strings, and one that raw 16-bit little fields would violate
/// for ~27% of all lengths.
pub fn build_frame_bytes(piece: &[u8], seq: usize, last: bool) -> Vec<u8> {
    debug_assert!(seq < 1 << 14, "chunk spans at most 16383 frames");
    debug_assert!(piece.len() < 1 << 14, "piece fits 14 bits");
    let mut out = Vec::with_capacity(FRAME_HEADER + piece.len());
    out.push(if last { b'1' } else { b'0' });
    out.push((seq >> 7) as u8);
    out.push((seq & 0x7f) as u8);
    out.push((piece.len() >> 7) as u8);
    out.push((piece.len() & 0x7f) as u8);
    out.extend_from_slice(piece);
    out
}

/// Parse a decoded frame back into `(seq, last, payload)`.
pub fn parse_frame_bytes(bytes: &[u8]) -> Result<(usize, bool, &[u8]), VisualError> {
    if bytes.len() < FRAME_HEADER {
        return Err(VisualError::Frame("frame shorter than framing header".into()));
    }
    let last = bytes[0] == b'1';
    let seq = ((bytes[1] as usize) << 7) | bytes[2] as usize;
    let len = ((bytes[3] as usize) << 7) | bytes[4] as usize;
    let payload = &bytes[FRAME_HEADER..];
    if payload.len() != len {
        return Err(VisualError::Frame(format!(
            "frame payload length mismatch: header says {len}, got {}",
            payload.len()
        )));
    }
    Ok((seq, last, payload))
}

/// A rendered QR frame: grayscale pixels, `0 = black`, `255 = white`.
pub struct QrFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Build a QR code for `data` and report its module width (incl. nothing else).
pub fn qr_module_width(data: &[u8]) -> Result<u32, VisualError> {
    if data.is_empty() {
        return Err(VisualError::EmptyChunk);
    }
    if data.len() > MAX_CHUNK_BYTES {
        return Err(VisualError::ChunkTooLarge {
            bytes: data.len(),
            max: MAX_CHUNK_BYTES,
        });
    }
    let code = QrCode::with_error_correction_level(data, EC_LEVEL)
        .map_err(|e| VisualError::Qr(e.to_string()))?;
    Ok(code.width() as u32)
}

/// Render `data` into a grayscale QR frame with the given pixel scale and
/// quiet-zone border (in modules). The frame is square.
pub fn render_qr(data: &[u8], scale: u32, quiet_zone: u32) -> Result<QrFrame, VisualError> {
    if !(1..=64).contains(&scale) {
        return Err(VisualError::BadScale(scale));
    }
    let code = QrCode::with_error_correction_level(data, EC_LEVEL)
        .map_err(|e| VisualError::Qr(e.to_string()))?;
    let modules = code.width() as u32;
    let side = (modules + 2 * quiet_zone) * scale;
    let mut pixels = vec![255u8; (side * side) as usize];
    for my in 0..modules {
        for mx in 0..modules {
            if code[(mx as usize, my as usize)] == qrcode::Color::Dark {
                let py = (my + quiet_zone) * scale;
                let px = (mx + quiet_zone) * scale;
                for dy in 0..scale {
                    let row = ((py + dy) * side + px) as usize;
                    pixels[row..row + scale as usize].fill(0);
                }
            }
        }
    }
    Ok(QrFrame { width: side, height: side, pixels })
}

/// Blit `frame` into a larger square canvas (top-left anchored, white padding).
/// Reuses the canvas buffer across frames to avoid per-frame allocation.
pub fn blit_to_canvas(frame: &QrFrame, canvas: &mut [u8], canvas_side: u32) {
    debug_assert_eq!(canvas.len(), (canvas_side * canvas_side) as usize);
    canvas.fill(255);
    let copy_w = frame.width.min(canvas_side);
    for y in 0..copy_w {
        let src = (y * frame.width) as usize;
        let dst = (y * canvas_side) as usize;
        canvas[dst..dst + copy_w as usize]
            .copy_from_slice(&frame.pixels[src..src + copy_w as usize]);
    }
}

/// Decode the first QR code found in a grayscale frame.
pub fn decode_qr(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, VisualError> {
    let img = image::GrayImage::from_raw(width, height, pixels.to_vec())
        .ok_or_else(|| VisualError::Frame("frame buffer does not match dimensions".into()))?;
    let mut prepared = rqrr::PreparedImage::prepare(img);
    let grids = prepared.detect_grids();
    let mut last_err = VisualError::NoQr;
    for grid in grids {
        match grid.decode() {
            Ok((_, text)) => return Ok(text.into_bytes()),
            Err(e) => last_err = VisualError::Decode(e.to_string()),
        }
    }
    Err(last_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qr_roundtrip_ascii() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let frame = render_qr(data, 8, 4).unwrap();
        assert!(frame.width >= (21 + 8) * 8);
        let back = decode_qr(&frame.pixels, frame.width, frame.height).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn qr_roundtrip_unicode() {
        let data = "颗粒嵌入视频帧 — memory in video 🎬 ελληνικά".as_bytes();
        let frame = render_qr(data, 8, 4).unwrap();
        let back = decode_qr(&frame.pixels, frame.width, frame.height).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn frame_framing_roundtrip() {
        let data = vec![b'z'; MAX_PAYLOAD * 2 + 123];
        let frames = frame_chunk(std::str::from_utf8(&data).unwrap()).unwrap();
        assert!(frames.len() >= 3);
        // every frame must be QR-encodable (this is the frame_chunk contract)
        for f in &frames {
            assert!(QrCode::with_error_correction_level(f, EC_LEVEL).is_ok());
        }
        let mut stitched = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            let (seq, last, payload) = parse_frame_bytes(f).unwrap();
            assert_eq!(seq, i);
            assert_eq!(last, i == frames.len() - 1);
            stitched.extend_from_slice(payload);
        }
        assert_eq!(stitched, data);
    }

    #[test]
    fn frame_chunk_handles_mixed_content() {
        // alternating alphanumeric / binary runs stress mode segmentation
        let text: String = "abc123 \u{4e2d}\u{6587} xyz\n".repeat(400);
        let frames = frame_chunk(&text).unwrap();
        let mut stitched = Vec::new();
        for (k, f) in frames.iter().enumerate() {
            let (seq, last, payload) = parse_frame_bytes(f).unwrap();
            assert_eq!(seq, k);
            assert_eq!(last, k == frames.len() - 1);
            stitched.extend_from_slice(payload);
        }
        assert_eq!(stitched, text.as_bytes());
    }

    /// Regression: 16-bit little-endian length fields used to emit 0x80..0xBF
    /// continuation bytes, silently breaking every string-based QR decoder.
    /// Every framed buffer — header included — must be valid UTF-8, and must
    /// survive a full QR render/decode roundtrip.
    #[test]
    fn framed_bytes_are_always_utf8_safe() {
        for len in [1921usize, 1922, 1923, 1924, 1925, 500, 89, 2895, 2896, 120] {
            let text: String = "a".repeat(len);
            let frames = frame_chunk(&text).unwrap();
            for f in &frames {
                assert!(std::str::from_utf8(f).is_ok(), "len {len}: framed bytes not UTF-8");
                let rendered = render_qr(f, 4, 4).unwrap();
                let back = decode_qr(&rendered.pixels, rendered.width, rendered.height)
                    .unwrap_or_else(|e| panic!("len {len}: QR decode failed: {e}"));
                assert_eq!(&back, f);
            }
        }
        // a real-world-shaped payload (source code, punctuation, CJK)
        let rustish = "fn main() { let x = 1921; println!(\"颗粒 {}\", x); }\n".repeat(60);
        for f in frame_chunk(&rustish).unwrap() {
            assert!(std::str::from_utf8(&f).is_ok());
        }
    }

    #[test]
    fn oversize_rejected() {
        let big = vec![b'x'; MAX_CHUNK_BYTES + 1];
        assert!(matches!(
            qr_module_width(&big),
            Err(VisualError::ChunkTooLarge { .. })
        ));
    }

    #[test]
    fn blit_pads_canvas_white() {
        let frame = render_qr(b"pad me", 4, 4).unwrap();
        let side = frame.width + 40;
        let mut canvas = vec![0u8; (side * side) as usize];
        blit_to_canvas(&frame, &mut canvas, side);
        // padding stays white
        assert!(canvas[((side - 1) * side) as usize] == 255);
        // original frame region still decodes
        let back = decode_qr(&canvas, side, side).unwrap();
        assert_eq!(back, b"pad me");
    }
}
