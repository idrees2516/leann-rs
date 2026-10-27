//! Input validation helpers shared across the crates.
//!
//! Threat model: LEANN embeds user-supplied text, calls user-configured HTTP
//! endpoints, parses user-supplied JSON filters, and writes index files to
//! user-supplied paths. These helpers close the classic holes:
//!
//! - **SSRF / scheme confusion**: base URLs must be `http://` or `https://`,
//!   with no userinfo (`http://attacker@host/`) and no embedded credentials.
//! - **Resource exhaustion**: JSON filters are depth- and node-limited, so a
//!   hostile filter can't blow the stack or build a billion-node tree.
//! - **Path traversal**: collection/user names can't escape their root dir.

use crate::{LeannError, Result};

/// Maximum JSON filter nesting depth.
pub const MAX_FILTER_DEPTH: usize = 16;
/// Maximum total nodes in one filter tree.
pub const MAX_FILTER_NODES: usize = 256;
/// Maximum accepted base-URL length.
pub const MAX_URL_LEN: usize = 2048;

/// Validate a user-supplied HTTP base URL (SSRF hardening: scheme allowlist,
/// no userinfo, no credentials embedded in the URL, sane length).
///
/// Returns the URL unchanged on success (no allocation) so hot paths can use
/// it as a cheap gate at construction time.
pub fn validate_base_url(url: &str) -> Result<&str> {
    if url.len() > MAX_URL_LEN {
        return Err(LeannError::Invalid(format!(
            "base URL too long ({} > {MAX_URL_LEN})",
            url.len()
        )));
    }
    let lower = url.to_ascii_lowercase();
    let has_scheme = lower.starts_with("http://") || lower.starts_with("https://");
    if !has_scheme {
        return Err(LeannError::Invalid(
            "base URL must start with http:// or https://".into(),
        ));
    }
    let rest = &url[lower.find("://").unwrap() + 3..];
    // reject userinfo (user:pass@host) — credentials belong in headers/env
    if let Some(at) = rest.find('@') {
        // an @ before the first / or :port means userinfo
        let host_part = rest.split(['/', '?', '#']).next().unwrap_or("");
        if at < host_part.len() {
            return Err(LeannError::Invalid(
                "base URL must not contain userinfo (user:pass@host)".into(),
            ));
        }
    }
    if rest.is_empty() || rest.starts_with('/') {
        return Err(LeannError::Invalid("base URL has no host".into()));
    }
    Ok(url)
}

/// Validate and normalize a collection / user name for filesystem use:
/// strips path separators, traversal sequences, and control characters.
/// Returns a safe replacement string.
pub fn safe_component(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        match ch {
            '/' | '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*' => out.push('_'),
            c if (c as u32) < 0x20 => out.push('_'),
            c => out.push(c),
        }
    }
    let trimmed = out.trim().trim_matches('.').to_string();
    if trimmed.is_empty() {
        "_".to_string()
    } else if trimmed.len() > 128 {
        trimmed[..128].to_string()
    } else {
        trimmed
    }
}

/// Check a JSON filter tree for depth/size limits before deserializing it
/// into a [`crate::filter::Filter`]. Cheap single pass, no allocation beyond
/// the recursion stack (bounded by `MAX_FILTER_DEPTH`).
pub fn check_filter_limits(v: &serde_json::Value) -> Result<()> {
    let mut nodes = 0usize;
    fn walk(v: &serde_json::Value, depth: usize, nodes: &mut usize) -> Result<()> {
        *nodes += 1;
        if *nodes > MAX_FILTER_NODES {
            return Err(LeannError::Invalid(format!(
                "filter too large (more than {MAX_FILTER_NODES} nodes)"
            )));
        }
        if depth > MAX_FILTER_DEPTH {
            return Err(LeannError::Invalid(format!(
                "filter nesting deeper than {MAX_FILTER_DEPTH}"
            )));
        }
        match v {
            serde_json::Value::Object(map) => {
                for (_, child) in map {
                    if child.is_object() || child.is_array() {
                        walk(child, depth + 1, nodes)?;
                    }
                }
            }
            serde_json::Value::Array(items) => {
                for child in items {
                    walk(child, depth + 1, nodes)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    walk(v, 0, &mut nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_scheme_allowlist() {
        assert!(validate_base_url("http://localhost:11434/v1").is_ok());
        assert!(validate_base_url("https://api.openai.com/v1").is_ok());
        assert!(validate_base_url("HTTPS://EXAMPLE.COM/v1").is_ok());
        assert!(validate_base_url("ftp://x/").is_err());
        assert!(validate_base_url("file:///etc/passwd").is_err());
        assert!(validate_base_url("javascript:alert(1)").is_err());
        assert!(validate_base_url("unix:///var/run/sock").is_err());
        assert!(validate_base_url("").is_err());
        assert!(validate_base_url("https://").is_err());
        let long = format!("https://{}.com", "a".repeat(3000));
        assert!(validate_base_url(&long).is_err());
    }

    #[test]
    fn url_userinfo_rejected() {
        assert!(validate_base_url("http://user:pass@evil.com/v1").is_err());
        assert!(validate_base_url("http://ak@host/v1").is_err());
        // @ in a path is fine
        assert!(validate_base_url("http://host/v1/@me").is_ok());
    }

    #[test]
    fn path_traversal_blocked() {
        // separators become '_' so ".." can never traverse; a component that
        // is *entirely* dots collapses to "_"
        assert_eq!(safe_component("../../etc/passwd"), "_.._etc_passwd");
        assert_eq!(safe_component("a/b\\c"), "a_b_c");
        assert_eq!(safe_component(".."), "_");
        assert_eq!(safe_component("."), "_");
        assert_eq!(safe_component("./"), "_");
        assert_eq!(safe_component(""), "_");
        assert_eq!(safe_component("   ..   "), "_");
        assert_eq!(safe_component("normal-name"), "normal-name");
        assert_eq!(safe_component("user@domain"), "user@domain");
        assert_eq!(safe_component(&"x".repeat(300)).len(), 128);
        // no separator survives sanitization
        assert!(!safe_component("a/b\\c:d").contains(['/', '\\', ':']));
    }

    #[test]
    fn filter_limits() {
        let ok: serde_json::Value =
            serde_json::from_str(r#"{"op":"and","filters":[{"op":"eq","field":"a","value":1}]}"#)
                .unwrap();
        assert!(check_filter_limits(&ok).is_ok());

        // deep nesting: 20 levels
        let mut v = serde_json::json!({"op":"eq","field":"x","value":1});
        for _ in 0..20 {
            v = serde_json::json!({"op":"not","filter":v});
        }
        assert!(check_filter_limits(&v).is_err());

        // huge flat list
        let items: Vec<serde_json::Value> = (0..300)
            .map(|i| serde_json::json!({"op":"eq","field":"f","value":i}))
            .collect();
        let big = serde_json::json!({"op":"and","filters":items});
        assert!(check_filter_limits(&big).is_err());
    }
}
