//! Keeping a client's prompt head byte-identical from request to request.
//!
//! Reuse of a cached prompt state — a pooled session or a prefix-cache root
//! — needs the new prompt to start with exactly the cached tokens.  Two
//! things clients do break that for no semantic reason, and both cost a
//! re-read of the whole system prompt and tool list (often 10K+ tokens):
//!
//! * **Tool order.**  Some clients send the same set of tools in a different
//!   order from one request to the next.  [`ToolOrders`] renders a set it has
//!   seen before in the order it was first seen (Strata #1624).
//! * **A volatile block in the system prompt.**  opencode writes an `<env>`
//!   block (working directory, today's date) near the start of its system
//!   message, so a new folder or a new day changes the prompt from there on.
//!   [`env_to_user`] moves the block to the front of the first user message
//!   (opt-in: it changes the prompt the model sees).

use std::collections::{BTreeSet, VecDeque};
use std::sync::Mutex;

use crate::types::{ChatMessage, Tool};

/// Distinct tool sets remembered.
const MAX_TOOL_SETS: usize = 64;

/// The order each recently seen set of tools first arrived in.
#[derive(Default)]
pub struct ToolOrders {
    seen: Mutex<VecDeque<(BTreeSet<String>, Vec<String>)>>,
}

impl ToolOrders {
    /// `tools` in the order this set of names was first seen, or `None` when
    /// they already are in that order (or the set is new, or names repeat).
    pub fn stable(&self, tools: &[Tool]) -> Option<Vec<Tool>> {
        let names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
        let set: BTreeSet<String> = names.iter().cloned().collect();
        if set.len() != names.len() || names.len() < 2 {
            return None;
        }
        let mut seen = self.seen.lock().unwrap_or_else(|p| p.into_inner());
        let Some(i) = seen.iter().position(|(s, _)| *s == set) else {
            seen.push_back((set, names));
            if seen.len() > MAX_TOOL_SETS {
                seen.pop_front();
            }
            return None;
        };
        // Most recently used last, so the oldest set is the one forgotten.
        let entry = seen.remove(i).expect("position is in range");
        let reordered = (entry.1 != names).then(|| {
            entry
                .1
                .iter()
                .map(|n| {
                    tools
                        .iter()
                        .find(|t| &t.function.name == n)
                        .expect("same set of names")
                        .clone()
                })
                .collect()
        });
        seen.push_back(entry);
        reordered
    }
}

/// The `<env>` block in `text`: its byte range, including an introducing
/// line ending in "environment you are running in:" and the block's trailing
/// newline.
fn env_block(text: &str) -> Option<std::ops::Range<usize>> {
    let open = text.find("<env>\n")?;
    let close = open + text[open..].find("\n</env>")?;
    let mut end = close + "\n</env>".len();
    if text[end..].starts_with('\n') {
        end += 1;
    }
    let mut start = open;
    let line_start = text[..open.saturating_sub(1)]
        .rfind('\n')
        .map_or(0, |i| i + 1);
    if open > 0
        && text[..open].ends_with('\n')
        && text[line_start..open - 1].ends_with("environment you are running in:")
    {
        start = line_start;
    }
    Some(start..end)
}

/// `messages` with the system message's `<env>` block moved to the front of
/// the first user message (separated by a blank line), or `None` when the
/// conversation has no such block or no user message to take it.
pub fn env_to_user(messages: &[ChatMessage]) -> Option<Vec<ChatMessage>> {
    let first = messages.first().filter(|m| m.role == "system")?;
    let range = env_block(&first.content)?;
    let user = messages.iter().position(|m| m.role == "user")?;
    let block = first.content[range.clone()]
        .trim_end_matches('\n')
        .to_string();
    let mut out = messages.to_vec();
    out[0].content = format!(
        "{}{}",
        &first.content[..range.start],
        &first.content[range.end..]
    );
    out[user].content = format!("{block}\n\n{}", messages[user].content);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionDef;

    fn tool(name: &str) -> Tool {
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: name.into(),
                description: None,
                parameters: None,
            },
        }
    }

    fn names(tools: &[Tool]) -> Vec<&str> {
        tools.iter().map(|t| t.function.name.as_str()).collect()
    }

    #[test]
    fn a_seen_set_keeps_its_first_order() {
        let orders = ToolOrders::default();
        assert!(orders.stable(&[tool("a"), tool("b"), tool("c")]).is_none());
        let again = orders.stable(&[tool("c"), tool("a"), tool("b")]).unwrap();
        assert_eq!(names(&again), ["a", "b", "c"]);
        // Already in order: nothing to change.
        assert!(orders.stable(&[tool("a"), tool("b"), tool("c")]).is_none());
        // Another set is its own entry.
        assert!(orders.stable(&[tool("b"), tool("a")]).is_none());
        assert_eq!(
            names(&orders.stable(&[tool("a"), tool("b")]).unwrap()),
            ["b", "a"]
        );
        // Repeated names are left alone.
        assert!(orders.stable(&[tool("x"), tool("x")]).is_none());
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        serde_json::from_value(serde_json::json!({ "role": role, "content": content })).unwrap()
    }

    #[test]
    fn env_block_moves_to_the_first_user_message() {
        let sys = "You are opencode.\nHere is some useful information about the environment you are running in:\n<env>\n  Working directory: /tmp/x\n  Today's date: Thu Oct 15 2026\n</env>\nRules follow.";
        let out = env_to_user(&[msg("system", sys), msg("user", "hi")]).unwrap();
        assert_eq!(out[0].content, "You are opencode.\nRules follow.");
        assert_eq!(
            out[1].content,
            "Here is some useful information about the environment you are running in:\n<env>\n  Working directory: /tmp/x\n  Today's date: Thu Oct 15 2026\n</env>\n\nhi"
        );
        // Without the introducing line only the block moves.
        let out =
            env_to_user(&[msg("system", "A\n<env>\nx\n</env>\nB"), msg("user", "q")]).unwrap();
        assert_eq!(
            (out[0].content.as_str(), out[1].content.as_str()),
            ("A\nB", "<env>\nx\n</env>\n\nq")
        );
        // Nothing to move, or nowhere to move it.
        assert!(env_to_user(&[msg("system", "plain"), msg("user", "q")]).is_none());
        assert!(env_to_user(&[msg("system", "<env>\nx\n</env>")]).is_none());
        assert!(env_to_user(&[msg("user", "<env>\nx\n</env>")]).is_none());
    }
}
