//! Broken markup in every format streams as it parses.
//!
//! A model's markup can break anywhere: a stray quote or brace, a line
//! break in a header, a closing tag too early.  Each sample output below is
//! cut off or has a piece of junk spliced in at every position, and every
//! result must stream, in pieces of any size, to the calls and content the
//! one-shot parse returns.

use super::tests::assert_streams_like_parse;
use super::ToolCallFormat as F;
use crate::types::{FunctionDef, Tool};

/// One sample output per shape of each format's markup.
const SAMPLES: &[(F, &str)] = &[
    (
        F::Generic,
        "Both.\n<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\n\
         <tool_call>{\"name\": \"b\", \"arguments\": {\"x\": 1}}</tool_call>",
    ),
    (
        F::Generic,
        "[TOOL_CALLS][{\"name\": \"get_time\", \"arguments\": {\"tz\": \"UTC\"}}, \
         {\"name\": \"b\", \"arguments\": {}}]",
    ),
    (
        F::Generic,
        "Hi[TOOL_CALLS]special_function[CALL_ID]000000001[ARGS]{\"arg1\": 1}\
         [TOOL_CALLS]g[ARGS]{\"a\": 2}",
    ),
    (
        F::Generic,
        "Calling now.\n<function=search>\n<parameter=query>\nrust async\n</parameter>\n\
         <parameter=limit>5</parameter>\n</function>\n<function=b>\n<parameter=x>\n1\n\
         </parameter>\n</function>",
    ),
    (
        F::Generic,
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n\
         </function>\n</tool_call> Done.",
    ),
    (
        F::Generic,
        "<function=get_weather>{\"location\": \"Paris\"}</function> Done.",
    ),
    (
        F::Generic,
        "{\"name\": \"lookup\", \"parameters\": {\"q\": \"rust\"}}",
    ),
    (
        F::LlamaJson,
        "{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Tokyo\"}}; \
         {\"name\": \"add\", \"parameters\": {\"x\": 1, \"y\": 2}}",
    ),
    (
        F::LlamaJson,
        "Let me look.<|python_tag|>{\"name\": \"get_weather\", \"parameters\": \
         {\"city\": \"Paris\"}}; {\"name\": \"b\", \"parameters\": {}}<|eom_id|>",
    ),
    (
        F::Pythonic,
        "[get_weather(city='San Francisco', metric='celsius'), register_user(name='John', \
         age=37, address={'city': 'SF'}, aliases=['J', 'Jo'], scores=(1.5, -2), ok=True)]",
    ),
    (
        F::Pythonic,
        "<|python_start|>[get_weather(city=\"LA\", metric=\"F\")]<|python_end|> Done.",
    ),
    (
        F::Lfm2,
        "Let me check.<|tool_call_start|>[get_time(city=\"Paris\"), \
         get_weather(city=\"Paris\")]<|tool_call_end|> Done.",
    ),
    (
        F::NemotronV2,
        "Sure.<TOOLCALL>[{\"name\": \"a\", \"arguments\": {}}, \
         {\"name\": \"b\", \"arguments\": {\"x\": 1}}]</TOOLCALL> Done.",
    ),
    (
        F::Functools,
        "functools[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, \
         {\"name\": \"get_time\", \"arguments\": {}}]",
    ),
    (
        F::DeepSeekV3,
        "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>\
         get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜>\n\
         <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{\"tz\": \"CET\"}\n```\
         <｜tool▁call▁end｜><｜tool▁calls▁end｜> Done.",
    ),
    (
        F::DeepSeekV3,
        "CONTENT<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>\
         {\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_weather\
         <｜tool▁sep｜>{\"city\": \"Paris\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ),
    (
        F::DeepSeekDsml,
        "Hi<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n\
         <｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n\
         <｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n\
         </｜DSML｜invoke>\n<｜DSML｜invoke name=\"b\">\n</｜DSML｜invoke>\n\
         </｜DSML｜tool_calls> Done.",
    ),
    (
        F::KimiK2,
        "Hi<|tool_calls_section_begin|><|tool_call_begin|>functions.special_function:0\
         <|tool_call_argument_begin|>{\"arg1\": 1}<|tool_call_end|><|tool_call_begin|>\
         functions.g:1<|tool_call_argument_begin|>{\"arg1\": 1, \"arg2\": 2}\
         <|tool_call_end|><|tool_calls_section_end|> Done.",
    ),
    (
        F::KimiK3,
        "<|open|>response<|sep|>On it.<|close|>response<|sep|><|open|>tools<|sep|>\
         <|open|>call tool=\"special_function\" index=\"1\"<|sep|>\
         <|open|>argument key=\"arg1\" type=\"number\"<|sep|>1<|close|>argument<|sep|>\
         <|close|>call<|sep|><|open|>call tool=\"g\" index=\"2\"<|sep|>\
         <|open|>argument key=\"arg1\" type=\"number\"<|sep|>1<|close|>argument<|sep|>\
         <|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>",
    ),
    (
        F::Glm45,
        "Preparing.<tool_call>magic_int<arg_key>ref</arg_key><arg_value>42</arg_value>\
         <arg_key>name</arg_key><arg_value>上海</arg_value></tool_call><tool_call>toggle\
         <arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>\
         <tool_call>empty_args</tool_call> Done.",
    ),
    (
        F::Harmony,
        "<|channel|>commentary<|message|>Checking.<|end|><|start|>assistant \
         to=functions.get_weather<|channel|>commentary json<|message|>{\"city\": \"Paris\"}\
         <|call|><|start|>assistant<|channel|>commentary to=functions.b <|constrain|>json\
         <|message|>{}<|call|>",
    ),
    (
        F::FunctionGemma,
        "Let me check. <start_function_call>call:set_alarm{days:[<escape>mon<escape>,\
         <escape>tue<escape>],enabled:true,hour:7,label:<escape>Wake up<escape>,\
         snooze:{minutes:5}}<end_function_call><start_function_call>call:get_time\
         {timezone:<escape>UTC<escape>}<end_function_call> Done.",
    ),
];

/// What a model may write where its markup breaks (nothing: it is cut off).
const JUNK: &[&str] = &[
    "", "x", "'", "\"", "\\", "{", "}", ")", ";", "<", "\n", " Done.", "True",
];

/// The functions the samples call, and their parameters.
const FUNCTIONS: &[&str] = &[
    "a",
    "b",
    "g",
    "get_time",
    "get_weather",
    "search",
    "special_function",
    "set_alarm",
    "lookup",
    "add",
    "register_user",
    "magic_int",
    "toggle",
    "empty_args",
];
const PARAMETERS: &[&str] = &[
    "city", "query", "limit", "x", "location", "q", "tz", "arg1", "arg2", "name", "ref", "enabled",
    "label", "timezone", "metric", "a", "days", "hour",
];

/// Tools whose every parameter in the samples is a string, so the formats
/// that type values by schema stream string values as they are written.
fn string_tools() -> Vec<Tool> {
    let properties: serde_json::Map<String, serde_json::Value> = PARAMETERS
        .iter()
        .map(|key| (key.to_string(), serde_json::json!({"type": "string"})))
        .collect();
    FUNCTIONS
        .iter()
        .map(|name| Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: name.to_string(),
                description: None,
                parameters: Some(serde_json::json!({"type": "object", "properties": properties})),
            },
        })
        .collect()
}

/// Each position gets every `SPREAD`th kind of junk, in rotation, to keep
/// the test to a few seconds.
const SPREAD: usize = 3;

#[test]
fn broken_markup_streams_as_it_parses() {
    let string_tools = string_tools();
    std::thread::scope(|scope| {
        for &(format, sample) in SAMPLES {
            let string_tools = &string_tools;
            scope.spawn(move || {
                let cuts = (0..=sample.len()).filter(|&at| sample.is_char_boundary(at));
                for (n, at) in cuts.enumerate() {
                    let (head, tail) = sample.split_at(at);
                    let junk = JUNK.iter().skip(n % SPREAD).step_by(SPREAD);
                    for text in
                        junk.flat_map(|j| [format!("{head}{j}"), format!("{head}{j}{tail}")])
                    {
                        for tools in [&[][..], &string_tools[..]] {
                            let checked = std::panic::catch_unwind(|| {
                                assert_streams_like_parse(format, tools, &text)
                            });
                            if let Err(panic) = checked {
                                eprintln!("{format} with {} tools: {text:?}", tools.len());
                                std::panic::resume_unwind(panic);
                            }
                        }
                    }
                }
            });
        }
    });
}
