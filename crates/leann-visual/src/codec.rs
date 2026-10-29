//! Video container codecs.
//!
//! Two interchangeable backends implement [`FrameReader`]:
//!
//! * [`Mp4Codec`] — pipes raw grayscale frames into `ffmpeg` (H.264 **lossless**
//!   `-qp 0`, all-intra `-g 1` so `-ss` seeks are instant). Best compression,
//!   needs the `ffmpeg` binary on PATH (or `LEANN_FFMPEG`).
//! * [`GifCodec`] — pure-Rust `gif` crate, zero system dependencies. Always
//!   available; ideal for tests, edge devices, and constrained deployments.
//!
//! [`VideoFormat::Auto`] picks MP4 when ffmpeg is present and GIF otherwise,
//! so `leann-visual` works out of the box everywhere while still taking
//! advantage of stronger codecs when they exist.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::VisualError;

/// Container choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoFormat {
    /// MP4 when ffmpeg is available, GIF otherwise.
    Auto,
    Mp4,
    Gif,
}

impl VideoFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "mp4" | "h264" => Some(Self::Mp4),
            "gif" => Some(Self::Gif),
            _ => None,
        }
    }

    /// Resolve `Auto` against ffmpeg availability.
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto => {
                if ffmpeg_path().is_some() {
                    Self::Mp4
                } else {
                    Self::Gif
                }
            }
            other => other,
        }
    }
}

/// Locate the ffmpeg binary (`LEANN_FFMPEG` overrides PATH lookup).
pub fn ffmpeg_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LEANN_FFMPEG") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let out = Command::new("ffmpeg").arg("-version").stdout(Stdio::piped()).stderr(Stdio::null()).output().ok()?;
    out.status.success().then(|| PathBuf::from("ffmpeg"))
}

/// Reads decoded grayscale frames out of a video.
pub trait FrameReader: Send + Sync {
    /// Read frame `i` (0-based) as a `width * height` grayscale buffer.
    fn read_frame(&self, i: usize) -> Result<Vec<u8>, VisualError>;
    fn container(&self) -> &'static str;
}

/// Writes grayscale frames into a video container.
pub(crate) trait FrameWriter {
    fn write_frame(&mut self, pixels: &[u8]) -> Result<(), VisualError>;
    fn finish(self: Box<Self>) -> Result<(), VisualError>;
}

// ---------------------------------------------------------------- MP4 (ffmpeg)

/// Stream raw gray frames into a lossless H.264 MP4.
///
/// Every frame is a keyframe (`-g 1`), so random access via `-ss` costs one
/// container seek + one frame decode — the sub-100ms retrieval property that
/// makes video-as-database practical.
pub(crate) struct Mp4Writer {
    child: std::process::Child,
    side: u32,
}

impl Mp4Writer {
    pub fn create(path: &Path, side: u32, fps: f64, frames: usize) -> Result<Self, VisualError> {
        let ffmpeg = ffmpeg_path().ok_or(VisualError::NoFfmpeg)?;
        let size = format!("{side}x{side}");
        let framerate = format!("{fps:.3}");
        // -framerate on input is required for rawvideo; total frames inferred from stream.
        let mut cmd = Command::new(&ffmpeg);
        cmd.args([
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "gray8",
            "-video_size", &size,
            "-framerate", &framerate,
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-qp", "0",              // mathematically lossless
            "-pix_fmt", "yuv444p",   // keeps grayscale exact
            "-profile:v", "high444",
            "-g", "1",               // all-intra: instant -ss seeks
            "-movflags", "+faststart",
            "-frames:v", &frames.to_string(),
        ])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
        let child = cmd.spawn().map_err(|e| VisualError::Ffmpeg(e.to_string()))?;
        Ok(Self { child, side })
    }

}

impl FrameWriter for Mp4Writer {
    fn write_frame(&mut self, pixels: &[u8]) -> Result<(), VisualError> {
        let expect = (self.side * self.side) as usize;
        if pixels.len() != expect {
            return Err(VisualError::Frame("frame buffer size mismatch".into()));
        }
        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| VisualError::Ffmpeg("ffmpeg stdin closed".into()))?;
        stdin.write_all(pixels).map_err(|e| VisualError::Ffmpeg(e.to_string()))?;
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<(), VisualError> {
        drop(self.child.stdin.take());
        let out = self
            .child
            .wait_with_output()
            .map_err(|e| VisualError::Ffmpeg(e.to_string()))?;
        if !out.status.success() {
            let msg = String::from_utf8_lossy(&out.stderr);
            return Err(VisualError::Ffmpeg(format!("encoder failed: {msg}")));
        }
        Ok(())
    }
}

/// Random-access MP4 frame reader backed by `ffmpeg -ss`.
pub struct Mp4Reader {
    pub(crate) path: PathBuf,
    pub(crate) side: u32,
    pub(crate) fps: f64,
}

impl Mp4Reader {
    fn ffmpeg(&self) -> Result<PathBuf, VisualError> {
        ffmpeg_path().ok_or(VisualError::NoFfmpeg)
    }
}

impl FrameReader for Mp4Reader {
    fn read_frame(&self, i: usize) -> Result<Vec<u8>, VisualError> {
        let ffmpeg = self.ffmpeg()?;
        // ffmpeg input seeking keeps the first frame with pts >= target, so
        // aim half a slot BEFORE frame i: (i - 0.5) / fps is strictly between
        // frames i-1 and i for every i >= 1 (i == 0 needs no -ss at all).
        // All frames are keyframes (-g 1), so the demuxer lands instantly.
        let mut args: Vec<String> = vec!["-loglevel".into(), "error".into()];
        if i > 0 {
            let t = (i as f64 - 0.5) / self.fps.max(0.001);
            args.extend(["-ss".into(), format!("{t:.6}")]);
        }
        args.extend([
            "-i".into(),
            self.path.to_string_lossy().into_owned(),
            "-frames:v".into(),
            "1".into(),
            "-f".into(),
            "rawvideo".into(),
            "-pix_fmt".into(),
            "gray8".into(),
            "pipe:1".into(),
        ]);
        let out = Command::new(&ffmpeg)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| VisualError::Ffmpeg(e.to_string()))?;
        if !out.status.success() {
            let msg = String::from_utf8_lossy(&out.stderr);
            return Err(VisualError::Ffmpeg(format!("frame {i} decode failed: {msg}")));
        }
        let expect = (self.side * self.side) as usize;
        if out.stdout.len() != expect {
            return Err(VisualError::Frame(format!(
                "frame {i}: expected {expect} bytes, got {}",
                out.stdout.len()
            )));
        }
        Ok(out.stdout)
    }

    fn container(&self) -> &'static str {
        "mp4"
    }
}

// ---------------------------------------------------------------- GIF (pure Rust)

/// Pure-Rust GIF writer (2-color global palette: index 0 white, 1 black).
pub(crate) struct GifWriter {
    encoder: gif::Encoder<File>,
    delay_cs: u16,
    side: u32,
}

impl GifWriter {
    pub fn create(path: &Path, side: u32, fps: f64) -> Result<Self, VisualError> {
        if side > 65_535 {
            return Err(VisualError::Frame(format!(
                "gif canvas {side} exceeds 65535px; lower the scale"
            )));
        }
        let file = File::create(path).map_err(VisualError::Io)?;
        // 2-entry global palette: [white, black]
        let palette: &[u8] = &[0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00];
        let mut encoder = gif::Encoder::new(file, side as u16, side as u16, palette)
            .map_err(|e| VisualError::Gif(e.to_string()))?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .map_err(|e| VisualError::Gif(e.to_string()))?;
        let delay_cs = ((100.0 / fps.max(0.001)).round() as i64).clamp(1, 65_535) as u16;
        Ok(Self { encoder, delay_cs, side })
    }
}

impl FrameWriter for GifWriter {
    fn write_frame(&mut self, pixels: &[u8]) -> Result<(), VisualError> {
        // gray (0=black,255=white) → palette indices (0=white, 1=black)
        let indices: Vec<u8> = pixels.iter().map(|&p| if p == 0 { 1 } else { 0 }).collect();
        let frame = gif::Frame {
            delay: self.delay_cs,
            width: self.side as u16,
            height: self.side as u16,
            buffer: std::borrow::Cow::Owned(indices),
            ..Default::default()
        };
        self.encoder
            .write_frame(&frame)
            .map_err(|e| VisualError::Gif(e.to_string()))
    }

    fn finish(self: Box<Self>) -> Result<(), VisualError> {
        drop(self.encoder);
        Ok(())
    }
}

/// Sequential GIF frame reader (decodes up to frame `i`, then caches it).
pub struct GifReader {
    path: PathBuf,
    side: u32,
    /// Cache of the last decoded frame: (index, pixels).
    last: std::sync::Mutex<Option<(usize, Vec<u8>)>>,
}

impl GifReader {
    pub fn new(path: &Path, side: u32) -> Self {
        Self { path: path.to_path_buf(), side, last: std::sync::Mutex::new(None) }
    }
}

impl FrameReader for GifReader {
    fn read_frame(&self, i: usize) -> Result<Vec<u8>, VisualError> {
        {
            let cache = self.last.lock().unwrap();
            if let Some((idx, pixels)) = cache.as_ref() {
                if *idx == i {
                    return Ok(pixels.clone());
                }
            }
        }
        let file = File::open(&self.path).map_err(VisualError::Io)?;
        let mut opts = gif::DecodeOptions::new();
        opts.set_color_output(gif::ColorOutput::Indexed);
        let mut decoder = opts
            .read_info(file)
            .map_err(|e| VisualError::Gif(e.to_string()))?;
        // Map palette indices to luminance via the global palette.
        let palette: Vec<u8> = decoder
            .global_palette()
            .map(|p| p.to_vec())
            .unwrap_or_else(|| vec![0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00]);
        let lum = |idx: u8| -> u8 {
            let base = idx as usize * 3;
            if base + 2 < palette.len() {
                let r = palette[base] as u32;
                let g = palette[base + 1] as u32;
                let b = palette[base + 2] as u32;
                ((r * 299 + g * 587 + b * 114) / 1000).clamp(0, 255) as u8
            } else {
                255
            }
        };
        let expect = (self.side * self.side) as usize;
        let mut target: Option<Vec<u8>> = None;
        let mut n = 0usize;
        while let Some(frame) = decoder
            .read_next_frame()
            .map_err(|e| VisualError::Gif(e.to_string()))?
        {
            if n == i {
                let mut pixels = vec![255u8; expect];
                // frames are always full-canvas when written by us; be defensive
                let fw = frame.width as usize;
                let fh = frame.height as usize;
                let (left, top) = (frame.left as usize, frame.top as usize);
                for y in 0..fh.min(self.side as usize - top) {
                    for x in 0..fw.min(self.side as usize - left) {
                        let idx = frame.buffer[y * fw + x];
                        pixels[(y + top) * self.side as usize + (x + left)] = lum(idx);
                    }
                }
                target = Some(pixels);
                break;
            }
            n += 1;
        }
        let pixels = target.ok_or_else(|| VisualError::Frame(format!("frame {i} not in gif")))?;
        *self.last.lock().unwrap() = Some((i, pixels.clone()));
        Ok(pixels)
    }

    fn container(&self) -> &'static str {
        "gif"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gif_write_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.gif");
        let side = 64u32;
        let mut w = GifWriter::create(&path, side, 2.0).unwrap();
        let mut frames = Vec::new();
        for i in 0..5u8 {
            let mut px = vec![255u8; (side * side) as usize];
            // paint a unique corner stripe per frame
            for y in 0..(i as u32 + 1) * 4 {
                for x in 0..8 {
                    px[(y * side + x) as usize] = 0;
                }
            }
            frames.push(px.clone());
            w.write_frame(&px).unwrap();
        }
        Box::new(w).finish().unwrap();
        let r = GifReader::new(&path, side);
        for (i, px) in frames.iter().enumerate() {
            assert_eq!(&r.read_frame(i).unwrap(), px, "frame {i} mismatch");
        }
    }

    #[test]
    fn mp4_write_read_roundtrip() {
        if ffmpeg_path().is_none() {
            eprintln!("skipping: ffmpeg not available");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.mp4");
        let side = 96u32;
        let mut w = Mp4Writer::create(&path, side, 2.0, 5).unwrap();
        let mut frames = Vec::new();
        for i in 0..5u8 {
            let mut px = vec![255u8; (side * side) as usize];
            for y in 0..(i as u32 + 1) * 8 {
                for x in 0..16 {
                    px[(y * side + x) as usize] = 0;
                }
            }
            frames.push(px.clone());
            w.write_frame(&px).unwrap();
        }
        Box::new(w).finish().unwrap();
        let r = Mp4Reader { path: path.clone(), side, fps: 2.0 };
        for (i, px) in frames.iter().enumerate() {
            assert_eq!(&r.read_frame(i).unwrap(), px, "frame {i} mismatch");
        }
    }
}
