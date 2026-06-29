    #[test]
    fn stop_sequence_watcher_matches_across_delta_boundaries() {
        let sequences = vec!["STOP".to_string()];
        let mut watcher = super::StopSequenceWatcher::new(&sequences);
        assert_eq!(watcher.feed("hello S"), ("hello ".to_string(), false));
        assert_eq!(watcher.feed("TO"), (String::new(), false));
        assert_eq!(watcher.feed("P world"), (String::new(), true));

        // Partial prefix that never completes is flushed at finish.
        let mut watcher = super::StopSequenceWatcher::new(&sequences);
        assert_eq!(watcher.feed("a ST"), ("a ".to_string(), false));
        assert_eq!(watcher.feed("Ow"), ("STOw".to_string(), false));
        assert_eq!(watcher.feed("x S"), ("x ".to_string(), false));
        assert_eq!(watcher.finish(), "S".to_string());

        // Single-delta match drops the stop string and the tail after it.
        let mut watcher = super::StopSequenceWatcher::new(&sequences);
        assert_eq!(watcher.feed("one STOP two"), ("one ".to_string(), true));

        // Earliest match wins across multiple sequences.
        let sequences = vec!["xyz".to_string(), "lo w".to_string()];
        let mut watcher = super::StopSequenceWatcher::new(&sequences);
        assert_eq!(watcher.feed("hello world"), ("hel".to_string(), true));

        // No sequences: pure passthrough.
        let mut watcher = super::StopSequenceWatcher::new(&[]);
        assert_eq!(watcher.feed("anything"), ("anything".to_string(), false));
        assert_eq!(watcher.finish(), String::new());

        // Multi-byte text around the holdback split point stays intact.
        let sequences = vec!["##".to_string()];
        let mut watcher = super::StopSequenceWatcher::new(&sequences);
        assert_eq!(watcher.feed("héé#"), ("héé".to_string(), false));
        assert_eq!(watcher.feed("é"), ("#é".to_string(), false));
    }
