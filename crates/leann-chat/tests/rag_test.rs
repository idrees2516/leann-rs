use serde_json::json;

use leann_chat::provider::{ChatMessage, ChatRequest, LlmProvider};
use leann_chat::{answer_with_context, RagConfig, Result};

/// Echo provider: returns a deterministic answer mentioning the prompt length.
struct MockLlm;
impl LlmProvider for MockLlm {
    fn name(&self) -> &str {
        "mock"
    }
    fn model(&self) -> &str {
        "mock-1"
    }
    fn chat(&self, req: &ChatRequest) -> Result<String> {
        let last_user = req
            .messages
            .last()
            .map(|m| m.content.clone())
            .unwrap_or_default();
        Ok(format!(
            "MOCK_ANSWER ({} chars of context received)",
            last_user.len()
        ))
    }
}

#[test]
fn rag_prompt_assembly() {
    let cfg = RagConfig::default();
    let ctx = vec![
        (
            3u32,
            "LEANN stores no embeddings and recomputes them at query time.".to_string(),
            json!({"source": "leann-paper.pdf", "page": 1}),
        ),
        (
            7u32,
            "BM25 gives exact keyword matching power to hybrid retrieval.".to_string(),
            json!({"source": "notes.md", "heading": "Hybrid"}),
        ),
    ];
    let ans = answer_with_context(&MockLlm, "What does LEANN store?", &ctx, &[], &cfg).unwrap();
    assert!(ans.answer.contains("MOCK_ANSWER"));
    assert_eq!(ans.sources.len(), 2);
    assert_eq!(ans.sources[0].chunk_id, 3);
    assert_eq!(ans.sources[0].source.as_deref(), Some("leann-paper.pdf"));
    assert_eq!(ans.query, "What does LEANN store?");
}

#[test]
fn history_and_system_prompt() {
    struct Inspector;
    impl LlmProvider for Inspector {
        fn name(&self) -> &str {
            "inspector"
        }
        fn model(&self) -> &str {
            "insp-1"
        }
        fn chat(&self, req: &ChatRequest) -> Result<String> {
            assert!(req.messages[0].content.contains("helpful assistant"));
            // history message present
            assert!(req.messages.iter().any(|m| m.content == "earlier question"));
            assert!(req.messages.last().unwrap().content.contains("Question: now what"));
            Ok("ok".into())
        }
    }
    let cfg = RagConfig::default();
    let ctx = vec![(1u32, "some context".to_string(), json!({}))];
    let history = vec![
        ChatMessage::user("earlier question"),
        ChatMessage::assistant("earlier answer"),
    ];
    let ans = answer_with_context(&Inspector, "now what", &ctx, &history, &cfg).unwrap();
    assert_eq!(ans.answer, "ok");
}

#[test]
fn context_budget_respected() {
    let cfg = RagConfig {
        max_context_chars: 500,
        ..Default::default()
    };
    let big: String = "x".repeat(10_000);
    let ctx = vec![
        (1u32, big.clone(), json!({})),
        (2u32, big, json!({})),
        (3u32, "small tail".to_string(), json!({})),
    ];
    struct Cap;
    impl LlmProvider for Cap {
        fn name(&self) -> &str {
            "cap"
        }
        fn model(&self) -> &str {
            "cap-1"
        }
        fn chat(&self, req: &ChatRequest) -> Result<String> {
            let user = req.messages.last().unwrap();
            assert!(
                user.content.len() < 2000,
                "context must be budgeted: {}",
                user.content.len()
            );
            Ok("ok".into())
        }
    }
    let ans = answer_with_context(&Cap, "q", &ctx, &[], &cfg).unwrap();
    assert_eq!(ans.answer, "ok");
}
