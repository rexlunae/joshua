//! Run capture checks in fresh processes: tracing configuration is cached once
//! per process and must not change the environment of concurrent model tests.
mod common;

use std::path::Path;

fn captures(root: &Path) -> Vec<serde_json::Value> {
    if !root.exists() {
        return Vec::new();
    }
    std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|s| s == "json"))
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
        .collect()
}

#[test]
fn qwen_positional_capture_includes_only_requested_forward() {
    for position in [0, 2] {
        let root = common::model_dir(&format!("qwen-position-trace-{position}"));
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "qwen_capture_child", "--ignored", "--nocapture"])
            .env("JOSHUA_TRACE_TEST_ROOT", &root)
            .env("JOSHUA_ACTIVATION_TRACE", root.join("trace"))
            .env("JOSHUA_ACTIVATION_TRACE_POSITION", position.to_string())
            .env("JOSHUA_ACTIVATION_TRACE_LIMIT", "512")
            .output()
            .unwrap();
        std::fs::remove_dir_all(&root).unwrap();
        assert!(
            output.status.success(),
            "position {position}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

#[test]
#[ignore = "subprocess helper for qwen_positional_capture_includes_only_requested_forward"]
fn qwen_capture_child() {
    let root = std::path::PathBuf::from(std::env::var_os("JOSHUA_TRACE_TEST_ROOT").unwrap());
    let trace = root.join("trace");
    let position: usize = std::env::var("JOSHUA_ACTIVATION_TRACE_POSITION")
        .unwrap()
        .parse()
        .unwrap();
    let path = root.join("model.gguf");
    common::write_tiny_qwen_gguf(&path, "qwen3");
    let mut model = common::load_model(&path, false);
    if position == 2 {
        common::logits(&mut model, &[1, 4], 0);
        assert!(captures(&trace).is_empty(), "prefill must be filtered");
    }
    common::logits(&mut model, &[2], position);
    let selected = captures(&trace);
    for label in ["qwen-input", "qwen-q-rope", "qwen-output"] {
        assert!(
            selected.iter().any(|capture| capture["label"] == label),
            "position {position}: missing {label} in {selected:?}",
        );
    }
    common::logits(&mut model, &[7], position + 1);
    assert_eq!(
        captures(&trace).len(),
        selected.len(),
        "continuation must be filtered"
    );
}
