use super::*;

#[test]
fn rank_order_sorts_descending_and_keeps_input_order_on_ties() {
    assert_eq!(rank_order(&[0.1, 0.9, 0.5], None), vec![1, 2, 0]);
    // Stable sort: equal scores keep input order.
    assert_eq!(rank_order(&[0.5, 0.5, 0.9], None), vec![2, 0, 1]);
    assert_eq!(rank_order(&[], None), Vec::<usize>::new());
}

#[test]
fn rank_order_truncates_to_top_n() {
    assert_eq!(rank_order(&[0.1, 0.9, 0.5], Some(2)), vec![1, 2]);
    assert_eq!(rank_order(&[0.1, 0.9], Some(0)), Vec::<usize>::new());
    // top_n beyond the document count is a no-op, not an error.
    assert_eq!(rank_order(&[0.1, 0.9], Some(10)), vec![1, 0]);
}

#[test]
fn rerank_request_deserializes_with_optional_fields_defaulted() {
    let req: RerankRequest =
        serde_json::from_str(r#"{"query": "q", "documents": ["a", "b"]}"#).unwrap();
    assert_eq!(req.query, "q");
    assert_eq!(req.documents, vec!["a", "b"]);
    assert!(req.instruction.is_none());
    assert!(req.top_n.is_none());

    let req: RerankRequest = serde_json::from_str(
        r#"{"model": "x", "query": "q", "documents": ["a"], "instruction": "find code", "top_n": 3}"#,
    )
    .unwrap();
    assert_eq!(req.instruction.as_deref(), Some("find code"));
    assert_eq!(req.top_n, Some(3));
}

#[test]
fn rerank_request_rejects_missing_required_fields() {
    assert!(serde_json::from_str::<RerankRequest>(r#"{"documents": ["a"]}"#).is_err());
    assert!(serde_json::from_str::<RerankRequest>(r#"{"query": "q"}"#).is_err());
}
