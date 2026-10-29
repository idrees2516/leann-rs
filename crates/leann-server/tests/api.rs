//! End-to-end API tests using `tower::ServiceExt::oneshot` (no sockets).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use leann_server::{build_router, ServerConfig, SharedState, AppState};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tower::util::ServiceExt;

fn state(dir: &std::path::Path, api_key: Option<&str>, rate: u32) -> SharedState {
    let mut cfg = ServerConfig {
        data_dir: dir.to_path_buf(),
        ..Default::default()
    };
    cfg.api_key = api_key.map(str::to_string);
    cfg.rate_limit_per_min = rate;
    cfg.llm_base_url = None; // no network in tests
    Arc::new(AppState::new(cfg).expect("state"))
}

async fn call(app: &mut axum::Router, method: &str, uri: &str, body: Option<Value>, key: Option<&str>) -> (StatusCode, Value, String) {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(k) = key {
        req = req.header("x-api-key", k);
    }
    let req = match body {
        Some(b) => req.header("content-type", "application/json").body(Body::from(b.to_string())).unwrap(),
        None => req.body(Body::empty()).unwrap(),
    };
    let res = app.oneshot(req).await.unwrap();
    let status = res.status();
    let ct = res
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, v, ct)
}

#[tokio::test]
async fn full_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let mut app = build_router(state(dir.path(), None, 0));

    // health
    let (st, v, _) = call(&mut app, "GET", "/health", None, None).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["status"], "ok");

    // create collection
    let (st, v, _) = call(&mut app, "POST", "/collections", Some(json!({"name": "docs"})), None).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    // duplicate → 409
    let (st, _, _) = call(&mut app, "POST", "/collections", Some(json!({"name": "docs"})), None).await;
    assert_eq!(st, StatusCode::CONFLICT);
    // invalid name → 400
    let (st, _, _) = call(&mut app, "POST", "/collections", Some(json!({"name": "../escape"})), None).await;
    assert_eq!(st, StatusCode::BAD_REQUEST);

    // add texts
    let texts: Vec<String> = (0..40)
        .map(|i| {
            format!(
                "chunk number {i} about topic {} — the leann index stores product quantization codes and hnsw graph edges",
                ["storage", "search", "graphs", "embeddings"][i % 4]
            )
        })
        .collect();
    let (st, v, _) = call(
        &mut app,
        "POST",
        "/collections/docs/texts",
        Some(json!({ "texts": texts, "metadatas": (0..40).map(|i| json!({"i": i})).collect::<Vec<_>>() })),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["added"], 40);

    // search
    let (st, v, _) = call(&mut app, "POST", "/collections/docs/search", Some(json!({"query": "hnsw graph edges", "k": 3})), None).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["hits"].as_array().unwrap().len(), 3);

    // hybrid search
    let (st, v, _) = call(&mut app, "POST", "/collections/docs/hybrid", Some(json!({"query": "quantization codes", "k": 5})), None).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert!(v["hits"].as_array().unwrap().len() >= 1);

    // filter + delete
    let (st, v, _) = call(
        &mut app,
        "POST",
        "/collections/docs/delete",
        Some(json!({"filter": {"op": "eq", "field": "i", "value": 3}})),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["deleted"], 1);

    // stats
    let (st, v, _) = call(&mut app, "GET", "/collections/docs/stats", None, None).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["alive"], 39);

    // export json → import into second collection
    let (st, body, _) = call(&mut app, "GET", "/collections/docs/export?format=json", None, None).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(body["count"], 39);
    let (st, _, _) = call(&mut app, "POST", "/collections", Some(json!({"name": "copy"})), None).await;
    assert_eq!(st, StatusCode::OK);
    let (st, v, _) = call(&mut app, "POST", "/collections/copy/import", Some(json!({"data": body})), None).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["imported"], 39);

    // csv export sanity
    let (st, _, ct) = call(&mut app, "GET", "/collections/docs/export?format=csv", None, None).await;
    assert_eq!(st, StatusCode::OK);
    assert!(ct.starts_with("text/csv"));

    // guard (offline deterministic)
    let (st, v, _) = call(
        &mut app,
        "POST",
        "/collections/docs/guard",
        Some(json!({"query": "quantization codes", "answer": "LEANN uses product quantization codes.", "k": 5})),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert!(v["groundedness"].is_object());

    // chat without LLM → 503
    let (st, v, _) = call(&mut app, "POST", "/collections/docs/chat", Some(json!({"query": "hello"})), None).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "{v}");

    // visual memory (GIF to stay test-env independent)
    let (st, v, _) = call(&mut app, "POST", "/collections/docs/visual", Some(json!({"format": "gif"})), None).await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert_eq!(v["container"], "gif");
    assert_eq!(v["chunks"], 39);

    // visual search
    let (st, v, _) = call(
        &mut app,
        "POST",
        "/collections/docs/visual-search",
        Some(json!({"query": "hnsw graph edges", "k": 3})),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    let hits = v["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0]["text"].as_str().unwrap().contains("leann index"));

    // collections list
    let (st, v, _) = call(&mut app, "GET", "/collections", None, None).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["collections"].as_array().unwrap().len(), 2);

    // playground served
    let (st, _, ct) = call(&mut app, "GET", "/", None, None).await;
    assert_eq!(st, StatusCode::OK);
    assert!(ct.starts_with("text/html"));

    // unknown collection → 404
    let (st, _, _) = call(&mut app, "GET", "/collections/nope/stats", None, None).await;
    assert_eq!(st, StatusCode::NOT_FOUND);

    // delete collection
    let (st, _, _) = call(&mut app, "DELETE", "/collections/copy", None, None).await;
    assert_eq!(st, StatusCode::OK);
    let (st, v, _) = call(&mut app, "GET", "/collections", None, None).await;
    assert_eq!(v["collections"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn auth_required_when_configured() {
    let dir = tempfile::tempdir().unwrap();
    let mut app = build_router(state(dir.path(), Some("sekrit"), 0));

    // no key → 401
    let (st, _, _) = call(&mut app, "GET", "/health", None, None).await;
    assert_eq!(st, StatusCode::UNAUTHORIZED);
    // wrong key → 401
    let (st, _, _) = call(&mut app, "GET", "/health", None, Some("wrong")).await;
    assert_eq!(st, StatusCode::UNAUTHORIZED);
    // right key → 200
    let (st, v, _) = call(&mut app, "GET", "/health", None, Some("sekrit")).await;
    assert_eq!(st, StatusCode::OK);
    assert_eq!(v["auth"], "bearer");
    // bearer form also works
    let mut app2 = build_router(state(dir.path(), Some("sekrit"), 0));
    let req = Request::builder()
        .uri("/health")
        .header("authorization", "Bearer sekrit")
        .body(Body::empty())
        .unwrap();
    let res = app2.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn rate_limit_kicks_in() {
    let dir = tempfile::tempdir().unwrap();
    // bucket of 3 per minute (no refill during the test window)
    let mut app = build_router(state(dir.path(), None, 3));
    let mut codes = Vec::new();
    for _ in 0..6 {
        let (st, _, _) = call(&mut app, "GET", "/health", None, None).await;
        codes.push(st);
    }
    assert_eq!(codes.iter().filter(|s| **s == StatusCode::OK).count(), 3);
    assert_eq!(codes.iter().filter(|s| **s == StatusCode::TOO_MANY_REQUESTS).count(), 3);
}

#[tokio::test]
async fn file_ingestion_path_rules() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("note.txt"), "leann is a storage efficient vector index with pq codes\n").unwrap();

    // without ingest root: absolute paths rejected
    let mut app = build_router(state(dir.path().join("data1").as_path(), None, 0));
    call(&mut app, "POST", "/collections", Some(json!({"name": "c"})), None).await;
    let (st, v, _) = call(
        &mut app,
        "POST",
        "/collections/c/files",
        Some(json!({"paths": [root.join("note.txt").to_string_lossy()]})),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::FORBIDDEN, "{v}");

    // with ingest root: allowed
    let mut cfg_data = dir.path().join("data2");
    let mut app2 = {
        let mut cfg = ServerConfig {
            data_dir: cfg_data.clone(),
            ..Default::default()
        };
        cfg.ingest_root = Some(root.clone());
        cfg.rate_limit_per_min = 0;
        cfg.llm_base_url = None;
        build_router(Arc::new(AppState::new(cfg).unwrap()))
    };
    let _ = &mut cfg_data;
    call(&mut app2, "POST", "/collections", Some(json!({"name": "c"})), None).await;
    let (st, v, _) = call(
        &mut app2,
        "POST",
        "/collections/c/files",
        Some(json!({"paths": [root.join("note.txt").to_string_lossy()]})),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{v}");
    assert!(v["added_chunks"].as_i64().unwrap() >= 1);

    // path outside root → 403
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();
    let (st, _, _) = call(
        &mut app2,
        "POST",
        "/collections/c/files",
        Some(json!({"paths": [outside.path().join("secret.txt").to_string_lossy()]})),
        None,
    )
    .await;
    assert_eq!(st, StatusCode::FORBIDDEN);
}
