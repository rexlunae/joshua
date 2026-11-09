//! Write the tiny Qwen3 model the test suite runs on (random weights and a
//! tokenizer made of JSON characters, so constrained JSON output can be
//! produced) into a directory, so a real server can be started on it: `cargo run --example tiny_model -- <dir>` writes
//! `<dir>/tiny.gguf` and `<dir>/tokenizer.json`.  The client suite under
//! `clients/` serves it to the OpenAI and Ollama client libraries.

#[allow(dead_code, unused_variables, unused_imports)]
#[path = "../tests/common/mod.rs"]
mod common;

fn main() {
    let dir = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .expect("usage: tiny_model <dir>");
    std::fs::create_dir_all(&dir).expect("creating the directory");
    std::fs::write(dir.join("tokenizer.json"), common::JSON_TOKENIZER_JSON).expect("writing tokenizer.json");
    common::write_tiny_qwen_gguf(&dir.join("tiny.gguf"), "qwen3");
    println!("{}", dir.join("tiny.gguf").display());
}
