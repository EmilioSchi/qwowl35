    use super::{contains_bytes, REQUIRED_NATIVE_KERNELS, WARMUP_METALLIB};

    #[test]
    fn qw35_metallib_contains_native_kernel_names() {
        if !cfg!(target_os = "macos") {
            return;
        }

        for name in REQUIRED_NATIVE_KERNELS {
            assert!(
                contains_bytes(WARMUP_METALLIB, name.as_bytes()),
                "Qw35 Metal library is missing {name}"
            );
        }
    }
