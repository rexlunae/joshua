//! End-to-end phase-0 harness smoke test, without downloads or accelerator jobs.
mod common;

#[test]
fn tiny_qwen_locality_harness_checks_output_and_reuse() {
    let directory = common::model_dir("prefix-locality");
    let model = directory.join("locality.gguf");
    common::write_tiny_qwen3_layers(&model, 1);
    let output = directory.join("measurement");
    let status = std::process::Command::new("python3")
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/scripts/prefix_locality.py"
        ))
        .args(["--binary", env!("CARGO_BIN_EXE_joshua")])
        .arg("--model")
        .arg(&model)
        .arg("--output")
        .arg(&output)
        .args([
            "--prefix-counts",
            "2,6,12",
            "--concurrency",
            "1,2,4",
            "--rounds",
            "2",
            "--repeats",
            "2",
            "--prefix-words",
            "8",
            "--max-tokens",
            "4",
            "--n-ctx",
            "256",
            "--root-min",
            "16",
            "--threads",
            "1",
        ])
        .status()
        .expect("python3 runs the stdlib harness");
    assert!(status.success(), "harness output: {}", output.display());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["all_outputs_match"], true);
    assert_eq!(report["completed"], true);
    let calibrations = report["calibration"].as_array().unwrap();
    assert_eq!(calibrations.len(), 2);
    for calibration in calibrations {
        assert_eq!(calibration["outputs_match"], true);
        assert_eq!(
            calibration["engine_completion"]["cold"]["reused_tokens"],
            0.0
        );
        assert!(
            calibration["engine_completion"]["warm"]["reused_tokens"]
                .as_f64()
                .unwrap()
                > 0.0
        );
    }
    let cells = report["cells"].as_array().unwrap();
    assert_eq!(cells.len(), 18);
    let mut root_resumes = 0;
    for cell in cells {
        assert_eq!(cell["outputs_match"], true);
        let expected_order = if cell["repeat"] == 0 {
            serde_json::json!(["load", "oracle"])
        } else {
            serde_json::json!(["oracle", "load"])
        };
        assert_eq!(cell["policy_order"], expected_order);
        for policy in ["load", "oracle"] {
            let result = &cell["policies"][policy];
            root_resumes += result["reuse"]["root_resume_events"].as_u64().unwrap();
            assert_eq!(result["errors"], 0);
            assert_eq!(
                result["requests_per_worker"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|count| count.as_u64().unwrap())
                    .sum::<u64>(),
                result["successful_requests"].as_u64().unwrap()
            );
            assert!(result["reuse"]["reused_tokens"].as_f64().unwrap() > 0.0);
            assert!(result["ttft_ms"]["p50"].as_f64().unwrap() > 0.0);
            assert_eq!(
                result["reuse"]["completion_records"],
                result["successful_requests"]
            );
            if policy == "load" {
                assert_eq!(result["queue_ms"]["records"], result["successful_requests"]);
            } else {
                assert!(result["queue_ms"]["p50"].is_null());
            }
        }
    }
    assert!(
        root_resumes > 0,
        "exercise root restores, not only pool rewinds"
    );
    std::fs::remove_dir_all(directory).unwrap();
}
