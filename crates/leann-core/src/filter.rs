//! JSON metadata filtering.
//!
//! Filters are plain JSON (so they can be authored from TypeScript/Python
//! without a DSL), deserialized into a small expression tree:
//!
//! ```json
//! { "op": "and", "filters": [
//!     { "op": "eq",  "field": "format",     "value": "pdf" },
//!     { "op": "gte", "field": "page",       "value": 10 },
//!     { "op": "in",  "field": "doc.author", "value": ["ada", "grace"] }
//! ]}
//! ```
//!
//! `field` supports dot-paths into nested objects. Numeric comparisons
//! coerce strings to f64 when possible.

use serde_json::Value;

/// One filter operation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Filter {
    And {
        filters: Vec<Filter>,
    },
    Or {
        filters: Vec<Filter>,
    },
    Not {
        filter: Box<Filter>,
    },
    Eq {
        field: String,
        value: Value,
    },
    Ne {
        field: String,
        value: Value,
    },
    Gt {
        field: String,
        value: Value,
    },
    Gte {
        field: String,
        value: Value,
    },
    Lt {
        field: String,
        value: Value,
    },
    Lte {
        field: String,
        value: Value,
    },
    In {
        field: String,
        value: Vec<Value>,
    },
    /// String contains (substring) or array contains value.
    Contains {
        field: String,
        value: Value,
    },
    Exists {
        field: String,
    },
}

/// Supported operations for manual construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    And,
    Or,
    Not,
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Contains,
    Exists,
}

/// Parse a filter from JSON (single object or array treated as AND).
pub fn parse_filter(json: &str) -> Result<Filter, serde_json::Error> {
    let v: Value = serde_json::from_str(json)?;
    Filter::from_value(&v)
}

/// Parse with resource-exhaustion guards (depth/size limits).
pub fn parse_filter_checked(json: &str) -> Result<Filter, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    crate::validate::check_filter_limits(&v).map_err(|e| e.to_string())?;
    Filter::from_value(&v).map_err(|e| e.to_string())
}

impl Filter {
    pub fn from_value(v: &Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(v.clone())
    }

    /// Evaluate against a metadata object.
    pub fn eval(&self, meta: &Value) -> bool {
        match self {
            Filter::And { filters } => filters.iter().all(|f| f.eval(meta)),
            Filter::Or { filters } => filters.iter().any(|f| f.eval(meta)),
            Filter::Not { filter } => !filter.eval(meta),
            Filter::Eq { field, value } => {
                lookup(meta, field).map(|v| values_equal(&v, value)).unwrap_or(false)
            }
            Filter::Ne { field, value } => {
                lookup(meta, field).map(|v| !values_equal(&v, value)).unwrap_or(true)
            }
            Filter::Gt { field, value } => cmp(field, value, meta, |o| o == std::cmp::Ordering::Greater),
            Filter::Gte { field, value } => {
                cmp(field, value, meta, |o| {
                    o == std::cmp::Ordering::Greater || o == std::cmp::Ordering::Equal
                })
            }
            Filter::Lt { field, value } => cmp(field, value, meta, |o| o == std::cmp::Ordering::Less),
            Filter::Lte { field, value } => cmp(field, value, meta, |o| {
                o == std::cmp::Ordering::Less || o == std::cmp::Ordering::Equal
            }),
            Filter::In { field, value } => match lookup(meta, field) {
                // array field: membership test
                Some(Value::Array(arr)) => {
                    value.iter().any(|x| arr.iter().any(|a| values_equal(a, x)))
                }
                Some(v) => value.iter().any(|x| values_equal(v, x)),
                None => false,
            },
            Filter::Contains { field, value } => match lookup(meta, field) {
                Some(Value::String(s)) => match value {
                    Value::String(sub) => s.contains(sub.as_str()),
                    other => s.contains(&other.to_string()),
                },
                Some(Value::Array(a)) => a.iter().any(|x| values_equal(x, value)),
                _ => false,
            },
            Filter::Exists { field } => lookup(meta, field).is_some(),
        }
    }
}

fn lookup<'a>(meta: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = meta;
    for part in path.split('.') {
        cur = cur.get(part)?;
    }
    Some(cur)
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            x.as_f64().zip(y.as_f64()).map(|(x, y)| (x - y).abs() < 1e-9).unwrap_or(false)
        }
        _ => a == b,
    }
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        Value::Bool(b) => Some(*b as i64 as f64),
        _ => None,
    }
}

fn cmp(
    field: &str,
    value: &Value,
    meta: &Value,
    ok: impl Fn(std::cmp::Ordering) -> bool,
) -> bool {
    match (lookup(meta, field), as_f64(value)) {
        (Some(f), Some(v)) => match as_f64(f) {
            Some(fv) => fv.partial_cmp(&v).map(&ok).unwrap_or(false),
            None => false,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn filters_json() {
        let meta = json!({
            "format": "pdf",
            "page": 12,
            "doc": { "author": "ada", "tags": ["ml", "sys"] },
            "title": "Neural Approaches to Retrieval"
        });

        let f = parse_filter(r#"{"op":"eq","field":"format","value":"pdf"}"#).unwrap();
        assert!(f.eval(&meta));
        let f = parse_filter(r#"{"op":"gte","field":"page","value":10}"#).unwrap();
        assert!(f.eval(&meta));
        let f = parse_filter(r#"{"op":"in","field":"doc.tags","value":["sys","db"]}"#).unwrap();
        assert!(f.eval(&meta));
        let f = parse_filter(r#"{"op":"contains","field":"doc.author","value":"ad"}"#).unwrap();
        assert!(f.eval(&meta));
        let f = parse_filter(
            r#"{"op":"and","filters":[{"op":"eq","field":"format","value":"pdf"},{"op":"lt","field":"page","value":5}]}"#,
        )
        .unwrap();
        assert!(!f.eval(&meta));
        let f = parse_filter(r#"{"op":"not","filter":{"op":"exists","field":"nope"}}"#).unwrap();
        assert!(f.eval(&meta));
    }
}
