Model chat templates for `tests/tool_template_tests.rs`, copied unchanged
from llama.cpp's `models/templates` (commit
4e910a86e8e8818e3619738ae372563a4e0cb42b), which takes them from each
model's Hugging Face repository.

`vllm-functiongemma.jinja` is vLLM's
`examples/tool_chat_template_functiongemma.jinja` (commit
5689c2790c911a55c75825aac5166305df0b591d), unchanged: llama.cpp has no
FunctionGemma template.

`transformers-tool-renders.json` holds the prompts transformers 5.22.1's
`apply_chat_template` renders from some of these templates for the tool-call
conversation in `tool_histories_render_as_transformers_renders_them`.
