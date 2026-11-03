//! Fill-in-the-middle (FIM) prompts for code completion, built the way
//! llama-server's `/infill` builds them (`format_prompt_infill` in
//! `tools/server/server-common.cpp`).
//!
//! A FIM-trained model (Qwen2.5-Coder, DeepSeek-Coder, CodeLlama, Granite,
//! …) completes the gap between a prefix and a suffix when they are framed
//! by its FIM tokens.  With the repo-level pattern of
//! <https://arxiv.org/pdf/2409.12186>, extra files go first:
//!
//! ```text
//! [FIM_REP]myproject
//! [FIM_SEP]filename0
//! extra chunk 0
//! [FIM_SEP]filename1
//! extra chunk 1
//! [FIM_SEP]filename
//! [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
//! ```
//!
//! [`FimTokens`] reads the token ids from GGUF metadata, falling back to
//! the well-known token strings llama.cpp's vocabulary loader searches for.

use std::collections::HashMap;

use candle_core::quantized::gguf_file;
use serde::{Deserialize, Serialize};

/// A model's fill-in-the-middle token ids.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct FimTokens {
    /// Starts the prefix (`tokenizer.ggml.fim_pre_token_id`).
    pub pre: Option<u32>,
    /// Starts the suffix (`fim_suf`).
    pub suf: Option<u32>,
    /// Starts the middle the model writes (`fim_mid`).
    pub mid: Option<u32>,
    /// Padding (`fim_pad`).
    pub pad: Option<u32>,
    /// Repository name marker (`fim_rep`).
    pub rep: Option<u32>,
    /// File separator (`fim_sep`).
    pub sep: Option<u32>,
}

/// Token strings llama.cpp's `llama_vocab::impl::load` accepts for each FIM
/// role when the GGUF does not name the id.
const PRE_NAMES: &[&str] = &[
    "<|fim_prefix|>",
    "<fim-prefix>",
    "<fim_prefix>",
    "<｜fim▁begin｜>",
    "<PRE>",
    "▁<PRE>",
    "<|code_prefix|>",
    "<|prefix|>",
    "<|plamo:fim_prefix|>",
];
const SUF_NAMES: &[&str] = &[
    "<|fim_suffix|>",
    "<fim-suffix>",
    "<fim_suffix>",
    "<｜fim▁hole｜>",
    "<SUF>",
    "▁<SUF>",
    "<|code_suffix|>",
    "<|suffix|>",
    "<|plamo:fim_suffix|>",
];
const MID_NAMES: &[&str] = &[
    "<|fim_middle|>",
    "<fim-middle>",
    "<fim_middle>",
    "<｜fim▁end｜>",
    "<MID>",
    "▁<MID>",
    "<|code_middle|>",
    "<|middle|>",
    "<|plamo:fim_middle|>",
];
const PAD_NAMES: &[&str] = &["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"];
const REP_NAMES: &[&str] = &[
    "<|fim_repo|>",
    "<|repo_name|>",
    "<fim-repo>",
    "<REPO>",
    "<reponame>",
];
const SEP_NAMES: &[&str] = &["<|file_sep|>", "<|plamo:file_separator|>"];

/// The chunk separator llama-server writes between extra files when the
/// model has no file-separator token.
const CHUNK_SEPARATOR: &str = "\n\n--- snippet ---\n\n";

fn metadata_id(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<u32> {
    match metadata.get(key)? {
        gguf_file::Value::U32(id) => Some(*id),
        gguf_file::Value::I32(id) => u32::try_from(*id).ok(),
        gguf_file::Value::U64(id) => u32::try_from(*id).ok(),
        gguf_file::Value::I64(id) => u32::try_from(*id).ok(),
        _ => None,
    }
}

impl FimTokens {
    /// Read the ids from GGUF metadata (`tokenizer.ggml.fim_*_token_id`,
    /// or the older `prefix/suffix/middle_token_id` keys), then look up the
    /// known token strings for any still missing.  `token_id` maps a token
    /// string to its id.
    pub fn from_gguf(
        metadata: &HashMap<String, gguf_file::Value>,
        token_id: &dyn Fn(&str) -> Option<u32>,
    ) -> Self {
        let pick = |keys: &[&str], names: &[&str]| {
            keys.iter()
                .find_map(|key| metadata_id(metadata, key))
                .or_else(|| names.iter().find_map(|name| token_id(name)))
        };
        Self {
            pre: pick(
                &[
                    "tokenizer.ggml.fim_pre_token_id",
                    "tokenizer.ggml.prefix_token_id",
                ],
                PRE_NAMES,
            ),
            suf: pick(
                &[
                    "tokenizer.ggml.fim_suf_token_id",
                    "tokenizer.ggml.suffix_token_id",
                ],
                SUF_NAMES,
            ),
            mid: pick(
                &[
                    "tokenizer.ggml.fim_mid_token_id",
                    "tokenizer.ggml.middle_token_id",
                ],
                MID_NAMES,
            ),
            pad: pick(&["tokenizer.ggml.fim_pad_token_id"], PAD_NAMES),
            rep: pick(&["tokenizer.ggml.fim_rep_token_id"], REP_NAMES),
            sep: pick(&["tokenizer.ggml.fim_sep_token_id"], SEP_NAMES),
        }
    }

    /// Why the model cannot do infill, if it cannot: llama-server's
    /// "prefix/suffix/middle token is missing" check.
    pub fn missing(&self) -> Option<String> {
        let mut err = String::new();
        if self.pre.is_none() {
            err.push_str("prefix token is missing. ");
        }
        if self.suf.is_none() {
            err.push_str("suffix token is missing. ");
        }
        if self.mid.is_none() {
            err.push_str("middle token is missing. ");
        }
        (!err.is_empty()).then(|| err.trim_end().to_string())
    }
}

/// An extra file for repo-level FIM (`input_extra` entries).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InfillChunk {
    /// File name; llama-server defaults it to `tmp`.
    #[serde(default = "InfillChunk::default_filename")]
    pub filename: String,
    /// File contents.
    pub text: String,
}

impl InfillChunk {
    fn default_filename() -> String {
        "tmp".to_string()
    }
}

/// Size limits for [`format_infill`] (llama-server's `n_batch`, `n_predict`
/// and the slot's `n_ctx`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InfillLimits {
    /// Logical batch size: the prefix gets three quarters of it, the suffix
    /// one quarter (less the prompt).
    pub n_batch: usize,
    /// Requested output length, `-1` for none (it reserves room for the
    /// answer when extra context is cut).
    pub n_predict: i64,
    /// Context window.
    pub n_ctx: usize,
}

/// llama-server's default `--batch-size`.
pub const DEFAULT_INFILL_BATCH: usize = 2048;

/// Build a FIM prompt (llama-server's `format_prompt_infill`).
///
/// `prefix`, `suffix` and `prompt` are already tokenized (without special
/// tokens); `tokenize` encodes the file names and extra chunks the same
/// way.  `bos` is prepended to the FIM part when the tokenizer adds a BOS.
/// `spm` selects the suffix-prefix-middle order.  Errors when the model
/// lacks a prefix, suffix or middle token.
#[allow(clippy::too_many_arguments)]
pub fn format_infill(
    fim: &FimTokens,
    tokenize: &dyn Fn(&str) -> Vec<u32>,
    mut prefix: Vec<u32>,
    mut suffix: Vec<u32>,
    extra: &[InfillChunk],
    prompt: &[u32],
    limits: InfillLimits,
    spm: bool,
    bos: Option<u32>,
) -> Result<Vec<u32>, String> {
    if let Some(missing) = fim.missing() {
        return Err(format!("Infill is not supported by this model: {missing}"));
    }
    let (pre, suf, mid) = (fim.pre.unwrap(), fim.suf.unwrap(), fim.mid.unwrap());

    let mut extra_tokens: Vec<u32> = Vec::new();
    if let Some(rep) = fim.rep {
        extra_tokens.push(rep);
        extra_tokens.extend(tokenize("myproject\n"));
    }
    for chunk in extra {
        match fim.sep {
            Some(sep) => {
                extra_tokens.push(sep);
                extra_tokens.extend(tokenize(&format!("{}\n", chunk.filename)));
            }
            None => extra_tokens.extend(tokenize(CHUNK_SEPARATOR)),
        }
        extra_tokens.extend(tokenize(&chunk.text));
    }
    if let Some(sep) = fim.sep {
        extra_tokens.push(sep);
        extra_tokens.extend(tokenize("filename\n"));
    }

    // The FIM context fits one batch, prefix:suffix = 3:1.
    let n_batch = limits.n_batch as i64;
    let n_prefix_take = (prefix.len() as i64).min(3 * (n_batch / 4)).max(0) as usize;
    let n_suffix_take = (suffix.len() as i64)
        .min((n_batch / 4 - (2 + prompt.len() as i64)).max(0))
        .max(0) as usize;
    // The rest of the context goes to extra chunks, newest last.
    let n_extra_take = (limits.n_ctx as i64 - n_batch - 2 * limits.n_predict)
        .max(0)
        .min(extra_tokens.len() as i64) as usize;

    prefix.drain(..prefix.len() - n_prefix_take);
    suffix.truncate(n_suffix_take);
    prefix.insert(0, pre);
    prefix.extend_from_slice(prompt);
    suffix.insert(0, suf);

    let (mut input, end) = if spm {
        (suffix, prefix)
    } else {
        (prefix, suffix)
    };
    if let Some(bos) = bos {
        input.insert(0, bos);
    }
    let mut out = Vec::with_capacity(n_extra_take + input.len() + end.len() + 1);
    out.extend_from_slice(&extra_tokens[extra_tokens.len() - n_extra_take..]);
    out.extend(input);
    out.extend(end);
    out.push(mid);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRE: u32 = 1000;
    const SUF: u32 = 1001;
    const MID: u32 = 1002;
    const REP: u32 = 1003;
    const SEP: u32 = 1004;
    const BOS: u32 = 1;

    /// One token per byte.
    fn bytes(s: &str) -> Vec<u32> {
        s.bytes().map(u32::from).collect()
    }

    fn qwen() -> FimTokens {
        FimTokens {
            pre: Some(PRE),
            suf: Some(SUF),
            mid: Some(MID),
            pad: None,
            rep: Some(REP),
            sep: Some(SEP),
        }
    }

    fn limits() -> InfillLimits {
        InfillLimits {
            n_batch: 2048,
            n_predict: -1,
            n_ctx: 8192,
        }
    }

    #[test]
    fn repo_level_prompt_layout() {
        let extra = [InfillChunk {
            filename: "a.py".into(),
            text: "x=1".into(),
        }];
        let got = format_infill(
            &qwen(),
            &bytes,
            bytes("def f("),
            bytes(")\n"),
            &extra,
            &bytes("a"),
            limits(),
            false,
            None,
        )
        .unwrap();
        let mut want = vec![REP];
        want.extend(bytes("myproject\n"));
        want.push(SEP);
        want.extend(bytes("a.py\n"));
        want.extend(bytes("x=1"));
        want.push(SEP);
        want.extend(bytes("filename\n"));
        want.push(PRE);
        want.extend(bytes("def f("));
        want.extend(bytes("a"));
        want.push(SUF);
        want.extend(bytes(")\n"));
        want.push(MID);
        assert_eq!(got, want);
    }

    #[test]
    fn spm_order_bos_and_no_separator_token() {
        let fim = FimTokens {
            rep: None,
            sep: None,
            ..qwen()
        };
        let extra = [InfillChunk {
            filename: "ignored".into(),
            text: "y".into(),
        }];
        let got = format_infill(
            &fim,
            &bytes,
            bytes("p"),
            bytes("s"),
            &extra,
            &[],
            limits(),
            true,
            Some(BOS),
        )
        .unwrap();
        let mut want = bytes(CHUNK_SEPARATOR);
        want.extend(bytes("y"));
        want.extend([BOS, SUF]);
        want.extend(bytes("s"));
        want.push(PRE);
        want.extend(bytes("p"));
        want.push(MID);
        assert_eq!(got, want);
    }

    #[test]
    fn long_inputs_are_cut_to_the_batch_and_context() {
        let prefix = vec![7u32; 100];
        let suffix = vec![8u32; 100];
        let tight = InfillLimits {
            n_batch: 16,
            n_predict: 4,
            n_ctx: 30,
        };
        let extra = [InfillChunk {
            filename: "f".into(),
            text: "0123456789".into(),
        }];
        let got = format_infill(
            &qwen(),
            &bytes,
            prefix,
            suffix,
            &extra,
            &[9],
            tight,
            false,
            None,
        )
        .unwrap();
        // Prefix: last 12 (3/4 of 16); suffix: 16/4 - (2 + 1) = 1 token;
        // extra: 30 - 16 - 8 = 6 of the newest extra tokens.
        let fim_start = got.iter().position(|&t| t == PRE).unwrap();
        assert_eq!(fim_start, 6, "{got:?}");
        assert_eq!(&got[fim_start + 1..fim_start + 13], &[7u32; 12]);
        assert_eq!(got[fim_start + 13], 9);
        assert_eq!(&got[fim_start + 14..], &[SUF, 8, MID]);
        // The newest extra tokens are kept: the `filename\n` trailer.
        assert_eq!(&got[..6], &bytes("ename\n")[..]);
    }

    #[test]
    fn missing_tokens_are_reported_like_llama_server() {
        let err = format_infill(
            &FimTokens::default(),
            &bytes,
            vec![],
            vec![],
            &[],
            &[],
            limits(),
            false,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            "Infill is not supported by this model: prefix token is missing. \
             suffix token is missing. middle token is missing."
        );
    }

    #[test]
    fn ids_come_from_metadata_then_known_strings() {
        let mut md = HashMap::new();
        md.insert(
            "tokenizer.ggml.fim_pre_token_id".to_string(),
            gguf_file::Value::U32(5),
        );
        md.insert(
            "tokenizer.ggml.suffix_token_id".to_string(),
            gguf_file::Value::I32(6),
        );
        let vocab = |s: &str| match s {
            "<|fim_middle|>" => Some(7),
            "<|file_sep|>" => Some(8),
            "<|fim_prefix|>" => Some(99),
            _ => None,
        };
        let fim = FimTokens::from_gguf(&md, &vocab);
        assert_eq!(fim.pre, Some(5), "metadata wins over the token string");
        assert_eq!(fim.suf, Some(6), "the legacy key is read");
        assert_eq!(fim.mid, Some(7));
        assert_eq!(fim.sep, Some(8));
        assert_eq!(fim.rep, None);
        assert!(fim.missing().is_none());
        assert!(FimTokens::from_gguf(&HashMap::new(), &|_| None)
            .missing()
            .is_some());
    }
}
