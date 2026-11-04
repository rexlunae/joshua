//! Tool calls through the models' real chat templates.
//!
//! For each family: the template is recognised as its tool-call format, a
//! conversation whose assistant turn carries a tool call with OpenAI's
//! JSON-string `arguments` renders (the arguments reach the template as an
//! object), and the call the template writes into the prompt parses back to
//! the same name and arguments, so the parser reads exactly what the model
//! was trained to write.

mod common;

use joshua::template::ChatTemplate;
use joshua::tools::{parse_tool_calls, ToolCallFormat};
use joshua::types::{FunctionDef, GenerationOptions, Tool};
use joshua::{ChatMessage, Engine, EngineOptions};
use serde_json::{json, Value};

fn template(file: &str) -> ChatTemplate {
    let path = format!(
        "{}/tests/fixtures/chat_templates/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    let source = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    // Three template features the renderer does not have, none of which
    // matters to the tool calls under test: gpt-oss dates its system
    // message with transformers' `strftime_now` (pinned here); LFM2.5 marks
    // assistant turns with transformers' `{% generation %}` training-mask
    // block, which renders its body; and Kimi K3 leaves its tool-result
    // loop with `{% break %}`, which this conversation never reaches.
    let source = source
        .replace("strftime_now(\"%Y-%m-%d\")", "\"2026-01-01\"")
        .replace("{%- generation -%}", "{%- if true -%}")
        .replace("{%- endgeneration -%}", "{%- endif -%}")
        .replace("{%- break -%}", "");
    ChatTemplate::new(source, "<s>", "</s>")
}

fn weather_tool() -> Tool {
    Tool {
        tool_type: "function".into(),
        function: FunctionDef {
            name: "get_weather".into(),
            description: Some("Look up the weather".into()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                },
                "required": ["city"]
            })),
        },
    }
}

/// A user question, the assistant's call (arguments as the JSON string
/// OpenAI clients send back), and the tool's result.
fn conversation(call_id: &str) -> Vec<ChatMessage> {
    let mut call = ChatMessage::text("assistant", "");
    call.tool_calls = Some(json!([{
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 3}"}
    }]));
    let mut result = ChatMessage::text("tool", "{\"forecast\": \"sunny\"}");
    result.tool_call_id = Some(call_id.into());
    result.name = Some("get_weather".into());
    vec![
        ChatMessage::text("user", "What is the weather in Paris?"),
        call,
        result,
    ]
}

struct Case {
    file: &'static str,
    format: ToolCallFormat,
    /// The id the template expects back (Kimi K2 and Mistral check theirs).
    call_id: &'static str,
    /// Where the rendered call starts, after the user's question.
    start: &'static str,
    /// Where it ends: the text after `end` is not part of it.
    end: &'static str,
}

const CASES: &[Case] = &[
    Case {
        file: "Qwen-Qwen3-0.6B.jinja",
        format: ToolCallFormat::Generic,
        call_id: "call_1",
        start: "<tool_call>",
        end: "</tool_call>",
    },
    Case {
        file: "Qwen3-Coder.jinja",
        format: ToolCallFormat::Generic,
        call_id: "call_1",
        start: "<tool_call>",
        end: "</tool_call>",
    },
    Case {
        file: "mistralai-Mistral-Nemo-Instruct-2407.jinja",
        format: ToolCallFormat::Mistral,
        call_id: "a1b2c3d4e",
        start: "[TOOL_CALLS]",
        end: "</s>",
    },
    Case {
        file: "unsloth-mistral-Devstral-Small-2507.jinja",
        format: ToolCallFormat::Mistral,
        call_id: "a1b2c3d4e",
        start: "[TOOL_CALLS]",
        end: "</s>",
    },
    Case {
        file: "meta-llama-Llama-3.1-8B-Instruct.jinja",
        format: ToolCallFormat::LlamaJson,
        call_id: "call_1",
        start: "{\"name\"",
        end: "<|eot_id|>",
    },
    Case {
        file: "NVIDIA-Nemotron-Nano-v2.jinja",
        format: ToolCallFormat::NemotronV2,
        call_id: "call_1",
        start: "<TOOLCALL>",
        end: "</TOOLCALL>",
    },
    Case {
        file: "LFM2.5-8B-A1B.jinja",
        format: ToolCallFormat::Lfm2,
        call_id: "call_1",
        start: "<|tool_call_start|>",
        end: "<|tool_call_end|>",
    },
    Case {
        file: "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja",
        format: ToolCallFormat::DeepSeekV3,
        call_id: "call_1",
        start: "<｜tool▁calls▁begin｜>",
        end: "<｜tool▁calls▁end｜>",
    },
    Case {
        file: "deepseek-ai-DeepSeek-V3.1.jinja",
        format: ToolCallFormat::DeepSeekV3,
        call_id: "call_1",
        start: "<｜tool▁calls▁begin｜>",
        end: "<｜tool▁calls▁end｜>",
    },
    Case {
        file: "deepseek-ai-DeepSeek-V3.2.jinja",
        format: ToolCallFormat::DeepSeekDsml,
        call_id: "call_1",
        start: "<｜DSML｜function_calls>",
        end: "</｜DSML｜function_calls>",
    },
    Case {
        file: "deepseek-ai-DeepSeek-V4.jinja",
        format: ToolCallFormat::DeepSeekDsml,
        call_id: "call_1",
        start: "<｜DSML｜tool_calls>",
        end: "</｜DSML｜tool_calls>",
    },
    Case {
        file: "moonshotai-Kimi-K2.jinja",
        format: ToolCallFormat::KimiK2,
        call_id: "functions.get_weather:0",
        start: "<|tool_calls_section_begin|>",
        end: "<|tool_calls_section_end|>",
    },
    Case {
        file: "Kimi-K3.jinja",
        format: ToolCallFormat::KimiK3,
        call_id: "call_1",
        start: "<|open|>tools<|sep|>",
        end: "<|close|>tools<|sep|>",
    },
    Case {
        file: "GLM-4.6.jinja",
        format: ToolCallFormat::Glm45,
        call_id: "call_1",
        start: "<tool_call>",
        end: "</tool_call>",
    },
    Case {
        file: "GLM-4.7-Flash.jinja",
        format: ToolCallFormat::Glm45,
        call_id: "call_1",
        start: "<tool_call>",
        end: "</tool_call>",
    },
    Case {
        file: "openai-gpt-oss-120b.jinja",
        format: ToolCallFormat::Harmony,
        call_id: "call_1",
        start: "<|start|>assistant to=",
        end: "<|call|>",
    },
    Case {
        file: "vllm-functiongemma.jinja",
        format: ToolCallFormat::FunctionGemma,
        call_id: "call_1",
        start: "<start_function_call>",
        end: "<end_function_call>",
    },
];

#[test]
fn templates_are_recognised_as_their_tool_call_format() {
    for case in CASES {
        assert_eq!(
            template(case.file).tool_call_format(),
            Some(case.format),
            "{}",
            case.file
        );
    }
}

#[test]
fn tool_call_history_renders_and_parses_back() {
    let tools = [weather_tool()];
    for case in CASES {
        let rendered = template(case.file)
            .render(&conversation(case.call_id), Some(&tools))
            .unwrap_or_else(|e| panic!("{}: {e}", case.file));
        let question = rendered
            .find("What is the weather in Paris?")
            .unwrap_or_else(|| panic!("{}: question missing:\n{rendered}", case.file));
        let start = question
            + rendered[question..]
                .find(case.start)
                .unwrap_or_else(|| panic!("{}: no call in:\n{rendered}", case.file));
        let len = rendered[start..]
            .find(case.end)
            .unwrap_or_else(|| panic!("{}: unterminated call in:\n{rendered}", case.file));
        let end = start
            + len
            + if case.end == "</s>" || case.end == "<|eot_id|>" {
                0
            } else {
                case.end.len()
            };
        let written = &rendered[start..end];
        let (prose, calls) = parse_tool_calls(case.format, &tools, written);
        assert_eq!(prose, "", "{}: {written}", case.file);
        assert_eq!(calls.len(), 1, "{}: {written}", case.file);
        assert_eq!(calls[0].name, "get_weather", "{}", case.file);
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(
            args,
            json!({"city": "Paris", "days": 3}),
            "{}: {written}",
            case.file
        );
    }
}

/// Mistral's templates raise on a call id that is not 9 alphanumerics, so
/// the ids minted for a Mistral model's calls are such ids, and a history
/// holding one renders (a `call_…` id would fall back to the ChatML prompt).
#[test]
fn mistral_templates_take_back_the_ids_minted_for_them() {
    let tools = [weather_tool()];
    let nemo = template("mistralai-Mistral-Nemo-Instruct-2407.jinja");
    let format = nemo.tool_call_format().unwrap();
    let id = format.call_id("call_");
    assert!(id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric()));
    let rendered = nemo
        .render(&conversation(&id), Some(&tools))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        rendered.contains(&format!("\"id\": \"{id}\"")),
        "{rendered}"
    );
    let other = ToolCallFormat::Generic.call_id("call_");
    assert!(nemo.render(&conversation(&other), Some(&tools)).is_err());
    // Devstral's template does not check them, and renders either.
    let devstral = template("unsloth-mistral-Devstral-Small-2507.jinja");
    assert_eq!(devstral.tool_call_format(), Some(ToolCallFormat::Mistral));
    devstral.render(&conversation(&id), Some(&tools)).unwrap();
}

/// JSON-string arguments reach the template as an object, not a string it
/// would double-encode (`tojson`) or fail to iterate (`items`).
#[test]
fn string_arguments_reach_templates_as_objects() {
    let tools = [weather_tool()];
    let llama = template("meta-llama-Llama-3.1-8B-Instruct.jinja")
        .render(&conversation("call_1"), Some(&tools))
        .unwrap();
    assert!(llama.contains(r#""parameters": {"#), "{llama}");
    assert!(!llama.contains(r#"\"city\""#), "double-encoded: {llama}");
    let qwen3_coder = template("Qwen3-Coder.jinja")
        .render(&conversation("call_1"), Some(&tools))
        .unwrap();
    assert!(
        qwen3_coder.contains("<parameter=city>\nParis\n</parameter>"),
        "{qwen3_coder}"
    );
    let glm = template("GLM-4.7-Flash.jinja")
        .render(&conversation("call_1"), Some(&tools))
        .unwrap();
    assert!(
        glm.contains("<arg_key>city</arg_key><arg_value>Paris</arg_value>"),
        "{glm}"
    );
}

/// Functionary v3.2 joins the arguments to text with `+`, which an object
/// cannot take: its history gets the client's string, as llama.cpp's
/// capability probe leaves it for this template, instead of the request
/// falling back to ChatML.
#[test]
fn templates_that_concatenate_arguments_take_the_string() {
    let rendered = template("meetkai-functionary-medium-v3.2.jinja")
        .render(&conversation("call_1"), Some(&[weather_tool()]))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        rendered.contains(">>>get_weather\n{\"city\": \"Paris\", \"days\": 3}<|eot_id|>"),
        "{rendered}"
    );
}

/// Qwen 2.5 / 3 write tools and call arguments with `tojson`, which writes
/// what transformers' does (as vLLM and llama.cpp render them): the
/// client's own `json.dumps` text comes back as it was sent, keys in their
/// order and nothing HTML-escaped.
#[test]
fn qwen3_renders_tools_and_arguments_as_python_json() {
    let tools = [weather_tool()];
    let sent = r#"{"path": "a.rs", "code": "if a < b && c > 0 { println!(\"don't\") }"}"#;
    let mut messages = conversation("call_1");
    messages[1].tool_calls = Some(json!([{
        "id": "call_1",
        "type": "function",
        "function": {"name": "edit", "arguments": sent}
    }]));
    let qwen3 = template("Qwen-Qwen3-0.6B.jinja")
        .render(&messages, Some(&tools))
        .unwrap();
    assert!(
        qwen3.contains(&format!(
            "<tool_call>\n{{\"name\": \"edit\", \"arguments\": {sent}}}\n</tool_call>"
        )),
        "{qwen3}"
    );
    assert!(
        qwen3.contains(
            r#"{"type": "function", "function": {"name": "get_weather", "description": "Look up the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}"#
        ),
        "{qwen3}"
    );
}

/// A tool-call history and its tools render to exactly the prompt
/// transformers' `apply_chat_template` gives (what vLLM sends the model),
/// for templates that write tools and calls with `tojson`.  The expected
/// prompts were rendered with transformers 5.22.1 from the same templates,
/// with the arguments as a dict (vLLM's `_postprocess_messages`); joshua
/// gets them as the JSON string an OpenAI client sends.  (Qwen3-Coder's and
/// GLM's also match, but only once transformers' `trim_blocks` /
/// `lstrip_blocks` whitespace control, which this renderer does not apply,
/// is set aside.)
#[test]
fn tool_histories_render_as_transformers_renders_them() {
    let path = format!(
        "{}/tests/fixtures/chat_templates/transformers-tool-renders.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let expected: serde_json::Map<String, Value> =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let tools: Vec<Tool> = serde_json::from_value(json!([{
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "code": {"type": "string"},
                    "n": {"type": "integer"},
                    "ok": {"type": "boolean"}
                },
                "required": ["path", "code"]
            }
        }
    }]))
    .unwrap();
    let mut differ = Vec::new();
    for (file, prompt) in &expected {
        let call_id = if file.contains("istral") {
            "a1b2c3d4e"
        } else {
            "call_1"
        };
        let mut call = ChatMessage::text("assistant", "");
        call.tool_calls = Some(json!([{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "edit",
                "arguments": r#"{"path": "a.rs", "code": "if a < b && c > 0 { println!(\"don't\") }", "n": 3, "ok": true}"#
            }
        }]));
        let mut result = ChatMessage::text("tool", "done");
        result.tool_call_id = Some(call_id.into());
        result.name = Some("edit".into());
        let messages = [ChatMessage::text("user", "Fix a.rs"), call, result];
        let rendered = template(file)
            .render(&messages, Some(&tools))
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        if rendered != prompt.as_str().unwrap() {
            eprintln!("{file}:\n{rendered:?}\n{prompt}");
            differ.push(file.as_str());
        }
    }
    assert!(differ.is_empty(), "{differ:?}");
}

/// The engine parses in the format its GGUF's template teaches, the
/// architecture's when the template tells nothing (the tiny llama: the
/// generic reader), and the operator's `--tool-call-parser` over both.
#[test]
fn engine_tool_call_format_follows_template_and_override() {
    let dir = common::model_dir("tool-format");
    let model = dir.join("model.gguf");
    common::write_tiny_llama_gguf(&model);
    let engine = Engine::with_n_ctx(&dir, 64).unwrap();
    assert_eq!(engine.tool_call_format(), ToolCallFormat::Generic);
    drop(engine);

    common::replace_chat_template(
        &model,
        "{% for m in messages %}hello {{ m.content }} {% endfor %}\
         {# <tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call> #}world",
    );
    let engine = Engine::with_n_ctx(&dir, 64).unwrap();
    assert_eq!(engine.tool_call_format(), ToolCallFormat::Glm45);
    drop(engine);

    let options = EngineOptions::with_n_ctx(64).tool_call_parser(Some(ToolCallFormat::KimiK2));
    let engine = Engine::with_options(&dir, options).unwrap();
    assert_eq!(engine.tool_call_format(), ToolCallFormat::KimiK2);
    drop(engine);

    // A template that cannot render a request with tools leaves it to the
    // ChatML prompt, which teaches Hermes calls.
    common::replace_chat_template(
        &model,
        "{{ undefined_function() }}{# <arg_key>key</arg_key> #}",
    );
    let engine = Engine::with_n_ctx(&dir, 64).unwrap();
    assert_eq!(engine.tool_call_format(), ToolCallFormat::Glm45);
    assert_eq!(
        engine
            .tool_call_format_for_request(
                &[ChatMessage::text("user", "weather?")],
                Some(&[weather_tool()]),
                &GenerationOptions::default(),
            )
            .unwrap(),
        ToolCallFormat::Generic
    );
    drop(engine);
    std::fs::remove_dir_all(&dir).ok();
}

/// Arguments that are not a JSON object stay as the client sent them, and
/// an empty `tool_calls` list leaves the assistant turn a plain one.
#[test]
fn unusual_tool_call_histories_still_render() {
    let source = "{% for m in messages %}{{ m.role }}:{% if m.tool_calls %}\
                  {% for c in m.tool_calls %}{{ c.function.name }}({% if c.function.arguments is string %}str:{{ c.function.arguments }}{% else %}{{ c.function.arguments | tojson }}{% endif %}){% endfor %}\
                  {% elif m.tool_calls is defined %}EMPTY{% else %}{{ m.content }}{% endif %};{% endfor %}";
    let t = ChatTemplate::new(source, "", "");
    let mut call = ChatMessage::text("assistant", "");
    call.tool_calls = Some(json!([
        {"type": "function", "function": {"name": "a", "arguments": "not json"}},
        {"type": "function", "function": {"name": "b", "arguments": ""}},
        {"type": "function", "function": {"name": "c", "arguments": "[1, 2]"}},
        {"type": "function", "function": {"name": "d", "arguments": {"x": 1}}},
        {"type": "function", "function": {"name": "e"}}
    ]));
    let mut empty = ChatMessage::text("assistant", "plain");
    empty.tool_calls = Some(json!([]));
    let out = t.render(&[call, empty], None).unwrap();
    assert_eq!(
        out,
        r#"assistant:a(str:not json)b({})c(str:[1, 2])d({"x": 1})e({});assistant:plain;"#
    );
}
