//! Chat-template rendering for GGUF models.
//!
//! GGUF files converted by llama.cpp embed the model's chat template as
//! Jinja2 source under the `tokenizer.chat_template` metadata key — the same
//! template `transformers` ships in `tokenizer_config.json`.  Rendering it
//! produces exactly the prompt format the model was trained on (Llama 3
//! headers, Gemma turns, ChatML, …) instead of assuming one fixed format.
//!
//! Rendering is done with [`minijinja`], a pure-Rust Jinja2 engine, extended
//! with Python method emulation (`.strip()`, `.title()`, …) that HuggingFace
//! templates routinely use, and the `raise_exception` helper they call for
//! unsupported message sequences.

use minijinja::{context, Environment, Error, ErrorKind, Value};
use serde::Serialize;

use crate::types::{ChatMessage, Tool};

/// A chat template extracted from GGUF metadata, plus the special-token
/// strings templates interpolate.
pub struct ChatTemplate {
    source: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Wrap raw Jinja source with the model's BOS/EOS token strings.
    ///
    /// Pass empty strings for tokens the model does not define; templates
    /// that never reference them are unaffected.
    pub fn new(
        source: impl Into<String>,
        bos_token: impl Into<String>,
        eos_token: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            bos_token: bos_token.into(),
            eos_token: eos_token.into(),
        }
    }

    /// The raw Jinja source of the template.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render the conversation to a prompt string, appending the generation
    /// prompt for the assistant turn (`add_generation_prompt = true`).
    ///
    /// When `tools` is provided the definitions are exposed to the template
    /// as the standard `tools` variable (OpenAI wire format), which tool-
    /// aware templates fold into their system prompt.
    ///
    /// The rendered prompt already contains every special token the model
    /// expects (including BOS where the template emits one), so it should be
    /// tokenised *without* adding special tokens again.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<String, String> {
        #[derive(Serialize)]
        struct Msg<'a> {
            role: &'a str,
            content: &'a str,
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tool_calls: Option<&'a serde_json::Value>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tool_call_id: Option<&'a str>,
        }

        let msgs: Vec<Msg> = messages
            .iter()
            .map(|m| Msg {
                role: &m.role,
                content: &m.content,
                name: m.name.as_deref(),
                tool_calls: m.tool_calls.as_ref(),
                tool_call_id: m.tool_call_id.as_deref(),
            })
            .collect();

        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<(), Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template("chat", &self.source)
            .map_err(|e| format!("chat template failed to parse: {e}"))?;

        // Only define `tools` when the request supplied some: templates
        // distinguish "no tools" via both `is defined` and truthiness checks.
        let ctx: Value = match tools {
            Some(tools) => context! {
                messages => msgs,
                add_generation_prompt => true,
                bos_token => self.bos_token,
                eos_token => self.eos_token,
                tools => tools,
            },
            None => context! {
                messages => msgs,
                add_generation_prompt => true,
                bos_token => self.bos_token,
                eos_token => self.eos_token,
            },
        };

        env.get_template("chat")
            .expect("template was just added")
            .render(ctx)
            .map_err(|e| format!("chat template failed to render: {e}"))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage::text(role, content)
    }

    /// The ChatML template as shipped in Qwen2.5 GGUF files (simplified: no
    /// tool branch taken when `tools` is undefined).
    const QWEN_CHATML: &str = "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";

    /// The Llama 3 instruct template (uses `.strip()` → needs pycompat).
    const LLAMA3: &str = "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";

    /// The Gemma template: rejects system messages via raise_exception and
    /// maps the assistant role to "model".
    const GEMMA: &str = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}{{ '<start_of_turn>' + role + '\\n' + message['content'].strip() + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}";

    #[test]
    fn chatml_style_template_renders() {
        let t = ChatTemplate::new(QWEN_CHATML, "", "<|im_end|>");
        let out = t
            .render(&[msg("system", "Be brief."), msg("user", "Hi!")], None)
            .unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nBe brief.<|im_end|>\n\
             <|im_start|>user\nHi!<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn llama3_style_template_renders_with_bos() {
        let t = ChatTemplate::new(LLAMA3, "<|begin_of_text|>", "<|eot_id|>");
        let out = t.render(&[msg("user", "  Hi!  ")], None).unwrap();
        assert_eq!(
            out,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn gemma_template_uses_pycompat_strip_and_role_mapping() {
        let t = ChatTemplate::new(GEMMA, "<bos>", "<eos>");
        let out = t
            .render(&[msg("user", " Hi! "), msg("assistant", "Hello."), msg("user", "Bye")], None)
            .unwrap();
        assert_eq!(
            out,
            "<bos><start_of_turn>user\nHi!<end_of_turn>\n\
             <start_of_turn>model\nHello.<end_of_turn>\n\
             <start_of_turn>user\nBye<end_of_turn>\n\
             <start_of_turn>model\n"
        );
    }

    #[test]
    fn raise_exception_surfaces_as_render_error() {
        let t = ChatTemplate::new(GEMMA, "<bos>", "<eos>");
        let err = t.render(&[msg("system", "nope")], None).unwrap_err();
        assert!(err.contains("System role not supported"), "got: {err}");
    }

    #[test]
    fn invalid_template_syntax_is_a_parse_error() {
        let t = ChatTemplate::new("{% for m in messages %}unclosed", "", "");
        let err = t.render(&[msg("user", "hi")], None).unwrap_err();
        assert!(err.contains("parse"), "got: {err}");
    }

    #[test]
    fn tools_are_exposed_to_the_template() {
        use crate::types::{FunctionDef, Tool};

        // Qwen-style: fold tool JSON into the system prompt when tools exist.
        let source = "{% if tools %}<tools>{% for tool in tools %}{{ tool.function.name }};{% endfor %}</tools>{% endif %}{% for message in messages %}[{{ message.role }}]{{ message.content }}{% endfor %}";
        let t = ChatTemplate::new(source, "", "");
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: Some("Look up weather".to_string()),
                parameters: None,
            },
        }];

        let with = t.render(&[msg("user", "hi")], Some(&tools)).unwrap();
        assert_eq!(with, "<tools>get_weather;</tools>[user]hi");

        // Without tools the `tools` variable is undefined → branch skipped.
        let without = t.render(&[msg("user", "hi")], None).unwrap();
        assert_eq!(without, "[user]hi");
    }

    #[test]
    fn tool_role_and_tool_calls_reach_the_template() {
        let source = "{% for m in messages %}{% if m.tool_calls %}calls:{{ m.tool_calls | length }}{% elif m.role == 'tool' %}result[{{ m.tool_call_id }}]:{{ m.content }}{% else %}{{ m.content }}{% endif %};{% endfor %}";
        let t = ChatTemplate::new(source, "", "");

        let mut assistant = msg("assistant", "");
        assistant.tool_calls = Some(serde_json::json!([
            {"id": "call_1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}
        ]));
        let mut tool_result = msg("tool", "42");
        tool_result.tool_call_id = Some("call_1".to_string());

        let out = t
            .render(&[msg("user", "go"), assistant, tool_result], None)
            .unwrap();
        assert_eq!(out, "go;calls:1;result[call_1]:42;");
    }
}
