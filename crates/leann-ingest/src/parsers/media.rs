//! Audio and video ingestion.
//!
//! Audio: `ffmpeg` (if present) converts any format to 16 kHz mono WAV,
//! then a [`Transcriber`] turns it into timestamped segments. Chunks are
//! grouped from segments with time metadata.
//!
//! Video: the audio track is transcribed the same way; keyframes can
//! optionally be extracted for a [`FrameDescriber`] (VLM) pass, producing
//! descriptive chunks interleaved on the timeline.
//!
//! The bundled [`OpenAiTranscriber`] speaks the OpenAI
//! `/audio/transcriptions` multipart protocol (works with OpenAI,
//! Groq, local whisper servers).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{LeannIngestError, Result};

/// One transcribed segment with timestamps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub segments: Vec<Segment>,
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
}

/// Anything that can transcribe an audio file.
pub trait Transcriber: Send + Sync {
    fn transcribe(&self, audio_path: &Path) -> Result<Transcript>;
}

/// Optional VLM pass over extracted keyframes.
pub trait FrameDescriber: Send + Sync {
    /// Describe one frame image file.
    fn describe(&self, image_path: &Path, timestamp_sec: f64) -> Result<String>;
}

// ---------------- ffmpeg helpers ----------------

fn ffmpeg_bin() -> Option<PathBuf> {
    let candidates = ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() || which(c) {
            return Some(p);
        }
    }
    None
}

fn which(bin: &str) -> bool {
    std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {bin}"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Convert any media file to 16 kHz mono WAV. Returns the temp path.
pub fn to_wav16k(input: &Path) -> Result<PathBuf> {
    let ffmpeg = ffmpeg_bin().ok_or_else(|| {
        LeannIngestError::Media(
            "ffmpeg not found — install ffmpeg to ingest audio/video".into(),
        )
    })?;
    let out = std::env::temp_dir().join(format!(
        "leann-{:x}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let status = std::process::Command::new(&ffmpeg)
        .args(["-y", "-i"])
        .arg(input)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "wav"])
        .arg(&out)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| LeannIngestError::Media(format!("ffmpeg spawn failed: {e}")))?;
    if !status.success() || !out.exists() {
        return Err(LeannIngestError::Media(format!(
            "ffmpeg failed ({status}) for {}",
            input.display()
        )));
    }
    Ok(out)
}

/// Extract keyframes at `fps` (e.g. 1.0 = one per second) into a temp dir.
/// Returns (dir, sorted frame paths with timestamps).
pub fn extract_keyframes(video: &Path, one_every_secs: f64) -> Result<Vec<(PathBuf, f64)>> {
    let ffmpeg = ffmpeg_bin().ok_or_else(|| {
        LeannIngestError::Media("ffmpeg not found — required for frame extraction".into())
    })?;
    let dir = std::env::temp_dir().join(format!(
        "leann-frames-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir)?;
    let status = std::process::Command::new(&ffmpeg)
        .args(["-y", "-i"])
        .arg(video)
        .arg("-vf")
        .arg(format!("fps=1/{one_every_secs}"))
        .arg(dir.join("f%06d.jpg"))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| LeannIngestError::Media(format!("ffmpeg spawn failed: {e}")))?;
    if !status.success() {
        return Err(LeannIngestError::Media("frame extraction failed".into()));
    }
    let mut frames: Vec<(PathBuf, f64)> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "jpg").unwrap_or(false))
        .filter_map(|p| {
            let stem = p.file_stem()?.to_str()?;
            let idx: f64 = stem.trim_start_matches('f').parse().ok()?;
            Some((p, (idx - 1.0).max(0.0) * one_every_secs))
        })
        .collect();
    frames.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    Ok(frames)
}

/// Total duration of a media file in seconds (via ffprobe if available).
pub fn media_duration(path: &Path) -> Option<f64> {
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "ffprobe -v error -show_entries format=duration -of csv=p=0 {}",
            shell_escape(&path.to_string_lossy())
        ))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

// ---------------- OpenAI-compatible transcriber ----------------

#[derive(Debug, Clone)]
pub struct OpenAiTranscriber {
    agent: ureq::Agent,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub language: Option<String>,
}

#[derive(Deserialize)]
struct VerboseTranscript {
    #[serde(default)]
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    segments: Vec<VerboseSegment>,
}

#[derive(Deserialize)]
struct VerboseSegment {
    start: f64,
    end: f64,
    #[serde(default)]
    text: String,
}

impl OpenAiTranscriber {
    /// Default: OpenAI whisper-1.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base("https://api.openai.com/v1", api_key, "whisper-1")
    }

    pub fn with_base(base_url: &str, api_key: impl Into<String>, model: &str) -> Self {
        Self {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(600))
                .build(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.to_string(),
            language: None,
        }
    }

    pub fn with_language(mut self, lang: Option<String>) -> Self {
        self.language = lang;
        self
    }
}

impl Transcriber for OpenAiTranscriber {
    fn transcribe(&self, audio_path: &Path) -> Result<Transcript> {
        let bytes = std::fs::read(audio_path)
            .map_err(|e| LeannIngestError::Media(format!("read audio failed: {e}")))?;
        let mime = match audio_path.extension().and_then(|e| e.to_str()) {
            Some("wav") => "audio/wav",
            Some("mp3") => "audio/mpeg",
            Some("m4a") => "audio/mp4",
            Some("flac") => "audio/flac",
            Some("ogg") | Some("opus") => "audio/ogg",
            Some("webm") => "audio/webm",
            _ => "audio/wav",
        };
        let boundary = format!("----leann{:x}", rand_boundary());
        let mut body: Vec<u8> = Vec::with_capacity(bytes.len() + 2048);

        let field = |body: &mut Vec<u8>, name: &str, value: &str| {
            body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
        };
        field(&mut body, "model", &self.model);
        field(&mut body, "response_format", "verbose_json");
        if let Some(lang) = &self.language {
            field(&mut body, "language", lang);
        }
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio\"\r\nContent-Type: {mime}\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(&bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

        let url = format!("{}/audio/transcriptions", self.base_url);
        let resp = self
            .agent
            .post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
            .send_bytes(&body)
            .map_err(|e| LeannIngestError::Media(format!("transcription request failed: {e}")))?;

        let mut resp_body = String::new();
        resp.into_reader()
            .read_to_string(&mut resp_body)
            .map_err(|e| LeannIngestError::Media(format!("read transcript failed: {e}")))?;
        let parsed: VerboseTranscript = serde_json::from_str(&resp_body)
            .map_err(|e| LeannIngestError::Media(format!("bad transcript JSON: {e}: {}", &resp_body[..resp_body.len().min(300)])))?;

        let segments: Vec<Segment> = parsed
            .segments
            .into_iter()
            .map(|s| Segment {
                start: s.start,
                end: s.end,
                text: s.text.trim().to_string(),
            })
            .filter(|s| !s.text.is_empty())
            .collect();
        let text = if parsed.text.is_empty() {
            segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ")
        } else {
            parsed.text
        };
        Ok(Transcript {
            segments,
            text,
            language: parsed.language,
        })
    }
}

fn rand_boundary() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5eed)
}

/// Transcribe any audio/video file by path (converts to wav16k first).
pub fn transcribe_media(path: &Path, t: &dyn Transcriber) -> Result<Transcript> {
    let wav = to_wav16k(path)?;
    let result = t.transcribe(&wav);
    let _ = std::fs::remove_file(&wav);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_available_or_clean_error() {
        // On CI/containers without ffmpeg we just verify the error path.
        match to_wav16k(Path::new("/nonexistent.mp3")) {
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("ffmpeg") || msg.contains("failed"));
            }
        }
    }
}
