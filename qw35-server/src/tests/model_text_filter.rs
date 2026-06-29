    #[test]
    fn generated_text_filter_emit_reasoning_passes_markers_through() {
        let mut filter = super::GeneratedTextFilter::new(true);
        assert_eq!(
            filter.visible_text_delta("\nthinking text"),
            "\nthinking text"
        );
        assert_eq!(
            filter.visible_text_delta("</think>\n\nanswer"),
            "</think>\n\nanswer"
        );
    }

    #[test]
    fn generated_text_filter_handles_leading_newlines_and_think_blocks() {
        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(filter.visible_text_delta("\n"), "");
        assert!(filter.trim_leading_newlines);
        assert_eq!(filter.visible_text_delta("\r\n\nHello"), "Hello");
        assert!(!filter.trim_leading_newlines);
        assert_eq!(filter.visible_text_delta("\nkept"), "\nkept");

        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(filter.visible_text_delta(" Hello"), " Hello");
        assert!(!filter.trim_leading_newlines);

        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(filter.visible_text_delta("\n<think>\nprivate"), "");
        assert!(filter.suppress_thinking);
        assert_eq!(filter.visible_text_delta(" reasoning"), "");
        assert_eq!(filter.visible_text_delta("</think>\nVisible"), "Visible");
        assert!(!filter.suppress_thinking);

        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(
            filter.visible_text_delta("<think>private</think>\nVisible"),
            "Visible"
        );

        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(filter.visible_text_delta("</think>\nVisible"), "Visible");
    }
