use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

const METAL_SOURCES: &[&str] = &[
    "src/metal/qw35_common.metal",
    "src/metal/qw35_types.metal",
    "src/metal/qw35_core.metal",
    "src/metal/qw35_matvec.metal",
    "src/metal/qw35_gf4.metal",
    "src/metal/qw35_gf2.metal",
    "src/metal/qw35_ssm.metal",
    "src/metal/qw35_attention.metal",
    "src/metal/qw35_output.metal",
    "src/metal/qw35_tiled_mm.metal",
];

const BRIDGE_DIR: &str = "src/metal/bridge";

// Floor for the release: lowering the deployment target widens OS compatibility
// (down to macOS 14 / Apple-silicon M1+) without changing which kernels run or
// their numerics — the GPU driver still compiles the same AIR. Without these
// flags the toolchain stamps the metallib at the build machine's latest target.
const MACOS_DEPLOY_TARGET: &str = "14.0";
const METAL_STD: &str = "metal3.0";

fn main() {
    for source in METAL_SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-changed={BRIDGE_DIR}");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let metallib = out_dir.join("qw35.metallib");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        // Lower the linked binary's Mach-O minimum-OS so it loads on the floor.
        println!("cargo:rustc-link-arg=-mmacosx-version-min={MACOS_DEPLOY_TARGET}");

        let combined = out_dir.join("qw35_all.metal");
        let mut source_text = String::new();
        for source in METAL_SOURCES {
            let text = fs::read_to_string(source)
                .unwrap_or_else(|err| panic!("failed to read Metal source {source}: {err}"));
            source_text.push_str("\n// appended ");
            source_text.push_str(source);
            source_text.push('\n');
            source_text.push_str(&text);
            source_text.push('\n');
        }
        fs::write(&combined, source_text).expect("write combined Metal source");

        let module_cache = out_dir.join("clang-module-cache");
        fs::create_dir_all(&module_cache).expect("create clang module cache");
        let module_cache_flag = format!("-fmodules-cache-path={}", module_cache.display());
        let metal_tool = find_tool("metal");
        let clang_tool = find_tool("clang");
        let metallib_tool = metal_tool
            .parent()
            .map(|parent| parent.join("metallib"))
            .filter(|path| path.exists())
            .unwrap_or_else(|| find_tool("metallib"));

        compile_metal_to_metallib(
            &metal_tool,
            &metallib_tool,
            &module_cache,
            &module_cache_flag,
            &combined,
            &out_dir.join("qw35.air"),
            &metallib,
        );

        compile_objc_bridge(&clang_tool, &out_dir);
    } else {
        fs::write(&metallib, []).expect("write placeholder metallib");
    }
}

fn compile_objc_bridge(clang_tool: &Path, out_dir: &Path) {
    let sdk_path = xcrun_output(&["--show-sdk-path"]);
    let mut objects = Vec::new();

    let mut sources: Vec<PathBuf> = fs::read_dir(BRIDGE_DIR)
        .expect("read Metal bridge directory")
        .filter_map(|entry| {
            let path = entry.expect("read bridge dir entry").path();
            (path.extension().and_then(|ext| ext.to_str()) == Some("m")).then_some(path)
        })
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no .m sources found in {BRIDGE_DIR}");

    for source in &sources {
        let stem = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("bridge source has a UTF-8 file stem");
        let object = out_dir.join(format!("{stem}.o"));
        let status = Command::new(clang_tool)
            .args(["-fobjc-arc", "-O2"])
            .arg(format!("-mmacosx-version-min={MACOS_DEPLOY_TARGET}"))
            .arg("-isysroot")
            .arg(sdk_path.trim())
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(&object)
            .status()
            .unwrap_or_else(|err| panic!("failed to run clang for {}: {err}", source.display()));
        assert!(status.success(), "clang failed for {}", source.display());
        objects.push(object);
    }

    let archive = out_dir.join("libqw35bridge.a");
    let _ = fs::remove_file(&archive);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&archive)
        .args(&objects)
        .status()
        .expect("failed to run ar for Metal bridge archive");
    assert!(status.success(), "ar failed for Metal bridge archive");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=qw35bridge");
    // The runtime is split into Objective-C categories (Qw35MetalRuntime+*.m),
    // one per forward-pass stage. A category defines no symbol the linker
    // references directly, so without -ObjC ld would dead-strip those object
    // files out of the static archive and the category methods would be missing
    // at runtime (unrecognized selector). -ObjC force-loads archive members that
    // contain Objective-C classes or categories.
    println!("cargo:rustc-link-arg=-ObjC");
}

fn compile_metal_to_metallib(
    metal_tool: &Path,
    metallib_tool: &Path,
    module_cache: &Path,
    module_cache_flag: &str,
    source: &Path,
    air: &Path,
    metallib: &Path,
) {
    let metal_status = Command::new(metal_tool)
        .arg(module_cache_flag)
        .arg(format!("-std={METAL_STD}"))
        .arg(format!("-mmacosx-version-min={MACOS_DEPLOY_TARGET}"))
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(air)
        .env("CLANG_MODULE_CACHE_PATH", module_cache)
        .status()
        .expect("failed to run metal");
    assert!(
        metal_status.success(),
        "metal failed for {}",
        source.display()
    );

    let metallib_status = Command::new(metallib_tool)
        .arg(air)
        .arg("-o")
        .arg(metallib)
        .status()
        .expect("failed to run metallib");
    assert!(
        metallib_status.success(),
        "metallib failed for {}",
        source.display()
    );
}

fn find_tool(name: &str) -> PathBuf {
    let text = xcrun_output(&["--find", name]);
    PathBuf::from(text.trim())
}

fn xcrun_output(args: &[&str]) -> String {
    let output = Command::new("xcrun")
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("failed to run xcrun {}: {err}", args.join(" ")));
    assert!(output.status.success(), "xcrun {} failed", args.join(" "));
    String::from_utf8(output.stdout).expect("xcrun output is UTF-8")
}
