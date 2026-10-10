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

use std::collections::BTreeMap;

use minijinja::{Environment, Error, ErrorKind, Value};
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
        self.render_with_vars(messages, tools, &serde_json::Map::new())
    }

    /// [`ChatTemplate::render`] with extra template variables — the
    /// `chat_template_kwargs` that carry per-model switches such as Qwen3's
    /// `enable_thinking` or gpt-oss's `reasoning_effort`.
    ///
    /// The engine-supplied variables (`messages`, `tools`,
    /// `add_generation_prompt`, `bos_token`, `eos_token`) always win over a
    /// same-named entry in `vars`.
    pub fn render_with_vars(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        vars: &serde_json::Map<String, serde_json::Value>,
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

        let mut ctx: BTreeMap<String, Value> = vars
            .iter()
            .map(|(k, v)| (k.clone(), Value::from_serialize(v)))
            .collect();
        ctx.insert("messages".into(), Value::from_serialize(&msgs));
        ctx.insert("add_generation_prompt".into(), Value::from(true));
        ctx.insert("bos_token".into(), Value::from(self.bos_token.as_str()));
        ctx.insert("eos_token".into(), Value::from(self.eos_token.as_str()));
        // Only define `tools` when the request supplied some: templates
        // distinguish "no tools" via both `is defined` and truthiness checks.
        match tools {
            Some(tools) => {
                ctx.insert("tools".into(), Value::from_serialize(tools));
            }
            None => {
                ctx.remove("tools");
            }
        }

        env.get_template("chat")
            .expect("template was just added")
            .render(Value::from(ctx))
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

    // ── Reasoning switches (#144) ──────────────────────────────────────────

    use crate::types::{reasoning_template_kwargs, ReasoningEffort};

    /// The generation-prompt tail of Qwen3's hybrid-thinking template.
    const QWEN3_TAIL: &str = "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}";

    /// DeepSeek-V3.1's switch: thinking defaults off unless `thinking` is set.
    const DEEPSEEK_V31_TAIL: &str = "{% if not thinking is defined %}{% set thinking = false %}{% endif %}{% for message in messages %}<｜User｜>{{ message.content }}{% endfor %}{% if add_generation_prompt %}<｜Assistant｜>{% if thinking %}<think>{% else %}</think>{% endif %}{% endif %}";

    /// gpt-oss's system header: `Reasoning: {{ reasoning_effort }}`, medium
    /// by default.
    const GPT_OSS_HEAD: &str = "<|start|>system<|message|>Reasoning: {% if reasoning_effort is defined %}{{ reasoning_effort }}{% else %}medium{% endif %}<|end|>{% for message in messages %}<|start|>{{ message.role }}<|message|>{{ message.content }}<|end|>{% endfor %}<|start|>assistant";

    /// Seed-OSS reads an integer `thinking_budget`.
    const SEED_OSS_BUDGET: &str = "{% if thinking_budget is defined %}budget={{ thinking_budget }}{% else %}unlimited{% endif %}";

    fn kwargs(
        effort: Option<ReasoningEffort>,
        enabled: Option<bool>,
    ) -> serde_json::Map<String, serde_json::Value> {
        reasoning_template_kwargs(effort, enabled, None)
    }

    #[test]
    fn qwen3_enable_thinking_switch() {
        let t = ChatTemplate::new(QWEN3_TAIL, "", "<|im_end|>");
        let m = [msg("user", "hi")];
        let default = t.render(&m, None).unwrap();
        let on = t
            .render_with_vars(&m, None, &kwargs(None, Some(true)))
            .unwrap();
        let off = t
            .render_with_vars(&m, None, &kwargs(None, Some(false)))
            .unwrap();
        let effort_none = t
            .render_with_vars(&m, None, &kwargs(Some(ReasoningEffort::None), None))
            .unwrap();
        assert_eq!(
            default,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(on, default, "thinking on is the Qwen3 default");
        assert_eq!(off, format!("{default}<think>\n\n</think>\n\n"));
        assert_eq!(effort_none, off, "reasoning_effort=none turns thinking off");
    }

    #[test]
    fn deepseek_v31_thinking_switch() {
        let t = ChatTemplate::new(DEEPSEEK_V31_TAIL, "", "");
        let m = [msg("user", "hi")];
        assert!(t
            .render(&m, None)
            .unwrap()
            .ends_with("<｜Assistant｜></think>"));
        let high = t
            .render_with_vars(&m, None, &kwargs(Some(ReasoningEffort::High), None))
            .unwrap();
        assert!(high.ends_with("<｜Assistant｜><think>"), "got: {high}");
        let off = t
            .render_with_vars(&m, None, &kwargs(None, Some(false)))
            .unwrap();
        assert!(off.ends_with("<｜Assistant｜></think>"), "got: {off}");
    }

    #[test]
    fn gpt_oss_reasoning_effort_levels() {
        let t = ChatTemplate::new(GPT_OSS_HEAD, "", "");
        let m = [msg("user", "hi")];
        let level = |vars| {
            let out = t.render_with_vars(&m, None, &vars).unwrap();
            out["<|start|>system<|message|>Reasoning: ".len()..]
                .split('<')
                .next()
                .unwrap()
                .to_string()
        };
        assert_eq!(level(serde_json::Map::new()), "medium");
        for (effort, want) in [
            (ReasoningEffort::None, "low"),
            (ReasoningEffort::Minimal, "low"),
            (ReasoningEffort::Low, "low"),
            (ReasoningEffort::Medium, "medium"),
            (ReasoningEffort::High, "high"),
            (ReasoningEffort::Xhigh, "high"),
        ] {
            assert_eq!(level(kwargs(Some(effort), None)), want, "{effort:?}");
        }
        // Thinking switched off without an effort: as low as gpt-oss goes.
        assert_eq!(level(kwargs(None, Some(false))), "low");
        // Thinking switched on without an effort keeps the template default.
        assert_eq!(level(kwargs(None, Some(true))), "medium");
    }

    #[test]
    fn templates_without_a_switch_ignore_reasoning_vars() {
        let t = ChatTemplate::new(LLAMA3, "<|begin_of_text|>", "<|eot_id|>");
        let m = [msg("user", "Hi!")];
        let plain = t.render(&m, None).unwrap();
        for vars in [
            kwargs(Some(ReasoningEffort::High), None),
            kwargs(None, Some(false)),
            reasoning_template_kwargs(None, None, Some(512)),
        ] {
            assert_eq!(t.render_with_vars(&m, None, &vars).unwrap(), plain);
        }
    }

    #[test]
    fn thinking_budget_reaches_seed_oss_style_templates() {
        let t = ChatTemplate::new(SEED_OSS_BUDGET, "", "");
        let m = [msg("user", "hi")];
        assert_eq!(t.render(&m, None).unwrap(), "unlimited");
        let vars = reasoning_template_kwargs(None, None, Some(512));
        assert_eq!(t.render_with_vars(&m, None, &vars).unwrap(), "budget=512");
    }

    #[test]
    fn engine_variables_win_over_template_kwargs() {
        let t = ChatTemplate::new(
            "{{ bos_token }}|{{ add_generation_prompt }}|{{ messages | length }}|{{ tools is defined }}|{{ custom }}",
            "<s>",
            "",
        );
        let mut vars = serde_json::Map::new();
        vars.insert("bos_token".into(), "X".into());
        vars.insert("add_generation_prompt".into(), false.into());
        vars.insert("messages".into(), serde_json::json!([]));
        vars.insert("tools".into(), serde_json::json!([1]));
        vars.insert("custom".into(), "ok".into());
        let out = t
            .render_with_vars(&[msg("user", "hi")], None, &vars)
            .unwrap();
        assert_eq!(out, "<s>|True|1|False|ok");
    }
}
