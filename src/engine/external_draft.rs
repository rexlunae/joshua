//! Local-only, Engine-bound deterministic proposal verification.
//!
//! This is not a wire protocol or a deployable remote worker. In particular,
//! generation IDs are local handles, not model/tokenizer artifact identities.
//! Learned stochastic proposals require a different acceptance algorithm.

use std::time::{Duration, Instant};

use uuid::Uuid;

use super::{DecodeText, Engine, GenSession, InFlightGuard, SamplingWorkspace};
use crate::{
    error::{JoshuaError, Result},
    speculative::Verdict,
    GenerationOptions,
};

/// Hard bound on verification rows and the single retained request/response.
pub const MAX_EXTERNAL_DRAFT: usize = 32;

/// A deterministic candidate prefix of the next output, not proposal probabilities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyRequest {
    pub generation: Uuid,
    pub sequence: u64,
    /// Number of tokens committed to KV, including the prompt.
    pub offset: usize,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    Stop,
    MaxTokens,
    Context,
}

/// Contains no logits, probability vectors or mutable sampling state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyResponse {
    pub generation: Uuid,
    pub sequence: u64,
    /// Matching draft tokens, including a matching EOS that is not emitted.
    pub accepted: usize,
    /// Accepted tokens followed by a correction/bonus, excluding EOS.
    pub tokens: Vec<u32>,
    /// Tokens fed to KV, including the prompt. An accepted stop token was
    /// already fed; a correction/bonus stop token and EOS are not fed.
    pub offset: usize,
    pub completion_tokens: u32,
    pub finish: Option<FinishReason>,
    pub stop_sequence: Option<String>,
}

/// One admitted local generation. Dropping, cancelling, ending or explicitly
/// expiring it destroys its KV; it is never returned to the warm pool.
///
/// The owner must call `expire` to reclaim an idle handle at its deadline (or
/// drop it). There is deliberately no background reaper or network registry.
/// Methods serialize mutations through `&mut self`. Exactly the latest successful
/// request is replayable, including its final response, until closure/expiry.
pub struct DraftSession<'a> {
    engine: &'a Engine,
    model: Option<GenSession>,
    permit: Option<InFlightGuard<'a>>,
    generation: Uuid,
    sequence: u64,
    offset: usize,
    context: usize,
    max_draft: usize,
    deadline: Instant,
    options: GenerationOptions,
    seed: u64,
    logits: Vec<f32>,
    sampler: SamplingWorkspace,
    text: DecodeText<'a>,
    replay: Option<(VerifyRequest, VerifyResponse)>,
}

impl Engine {
    /// Begin a bounded, local deterministic verification session using token IDs
    /// from this Engine's tokenizer. No remote backend or constrained decoding
    /// is supported. A capability failure is explicit; callers may use ordinary
    /// completion instead. Options and the position-keyed seed are fixed here.
    pub fn begin_draft_session(
        &self,
        prompt: &[u32],
        options: &GenerationOptions,
        max_draft: usize,
        lifetime: Duration,
    ) -> Result<DraftSession<'_>> {
        if !(1..=MAX_EXTERNAL_DRAFT).contains(&max_draft)
            || lifetime.is_zero()
            || lifetime > Duration::from_secs(300)
            || options.constraint.is_some()
            || self.npu.is_some()
        {
            return Err(invalid(
                "invalid draft limits, constraint, or non-local backend",
            ));
        }
        let context = self.request_n_ctx(options)?;
        if prompt.is_empty() || prompt.len() >= context {
            return Err(invalid("prompt must be nonempty and shorter than context"));
        }
        if prompt
            .iter()
            .any(|&token| token as usize >= self.vocab_size())
        {
            return Err(invalid("prompt token outside tokenizer vocabulary"));
        }
        let permit = InFlightGuard::acquire_with(
            &self.in_flight,
            self.max_concurrency,
            0,
            Duration::ZERO,
            None,
        )?;
        let (mut model, reused) = self.acquire_session(prompt, false, options.cache_namespace())?;
        if !model.supports_speculative() {
            return Err(invalid("model lacks all-position logits and safe rollback"));
        }
        let logits = self.prefill(&mut model, &prompt[reused..], reused)?;
        let options = self.clamp_options(options, Some(prompt.len()));
        let seed = options.seed.unwrap_or_else(rand::random);
        Ok(DraftSession {
            engine: self,
            model: Some(model),
            permit: Some(permit),
            generation: Uuid::new_v4(),
            sequence: 0,
            offset: prompt.len(),
            context,
            max_draft,
            deadline: Instant::now() + lifetime,
            options,
            seed,
            logits,
            sampler: SamplingWorkspace::default(),
            text: DecodeText::new(prompt, &self.tokenizer),
            replay: None,
        })
    }
}

fn invalid(message: &str) -> JoshuaError {
    JoshuaError::InvalidRequest(message.into())
}

impl DraftSession<'_> {
    pub fn generation(&self) -> Uuid {
        self.generation
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn text(&self) -> &str {
        &self.text.response
    }

    /// Reclaim an idle session using a monotonic clock. Invalid requests do not
    /// extend the fixed lifetime. Returns whether the deadline has elapsed.
    pub fn expire(&mut self) -> bool {
        if Instant::now() >= self.deadline {
            self.cancel();
            true
        } else {
            false
        }
    }

    pub fn cancel(&mut self) {
        self.release();
        self.replay = None;
    }

    pub fn end(&mut self) {
        self.cancel();
    }

    fn release(&mut self) {
        self.model = None;
        self.permit = None;
        self.logits = Vec::new();
        self.sampler = SamplingWorkspace::default();
    }

    /// Validate before touching KV. An exact retry returns the stored response
    /// without forwarding or sampling again. Any execution error fails closed.
    pub fn verify(&mut self, request: &VerifyRequest) -> Result<VerifyResponse> {
        if self.expire() {
            return Err(invalid("draft session expired"));
        }
        if request.generation != self.generation || request.tokens.len() > self.max_draft {
            return Err(invalid("generation mismatch or oversized draft"));
        }
        if let Some((previous, response)) = &self.replay {
            if request == previous {
                return Ok(response.clone());
            }
        }
        if self.model.is_none()
            || request.sequence != self.sequence
            || request.offset != self.offset
        {
            return Err(invalid(
                "closed session, stale sequence or committed offset",
            ));
        }
        if request
            .tokens
            .iter()
            .any(|&token| token as usize >= self.logits.len())
        {
            return Err(invalid("draft token outside model vocabulary"));
        }
        let next_sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid("sequence exhausted"))?;
        match self.execute(request) {
            Ok(response) => {
                self.sequence = next_sequence;
                if response.finish.is_some() {
                    self.release();
                }
                self.replay = Some((request.clone(), response.clone()));
                Ok(response)
            }
            Err(error) => {
                self.cancel();
                Err(error)
            }
        }
    }

    fn limit(&self) -> Option<FinishReason> {
        if self.offset >= self.context {
            Some(FinishReason::Context)
        } else if self.text.n_decoded >= self.options.max_tokens {
            Some(FinishReason::MaxTokens)
        } else {
            None
        }
    }

    fn emit(&mut self, token: u32, response: &mut VerifyResponse) -> Result<bool> {
        if self.engine.eos_token_ids.contains(&token) {
            response.finish = Some(FinishReason::Eos);
            return Ok(false);
        }
        response.tokens.push(token);
        if let Some(stop) = self
            .engine
            .emit_token(&mut self.text, token, &self.options)?
        {
            response.stop_sequence = Some(stop);
            response.finish = Some(FinishReason::Stop);
            return Ok(false);
        }
        Ok(true)
    }

    fn execute(&mut self, request: &VerifyRequest) -> Result<VerifyResponse> {
        let mut response = VerifyResponse {
            generation: self.generation,
            sequence: request.sequence,
            accepted: 0,
            tokens: Vec::with_capacity(request.tokens.len() + 1),
            offset: self.offset,
            completion_tokens: self.text.n_decoded,
            finish: self.limit(),
            stop_sequence: None,
        };
        if response.finish.is_some() {
            return Ok(response);
        }
        let count = request
            .tokens
            .len()
            .min((self.options.max_tokens - self.text.n_decoded) as usize)
            .min(self.context - self.offset);
        let rows = if count == 0 {
            Vec::new()
        } else {
            self.model.as_mut().unwrap().forward_tokens_all(
                &request.tokens[..count],
                self.offset,
                &self.engine.dense_device,
            )?
        };
        let mut kept = 0;
        let mut correction = None;
        for (index, &draft) in request.tokens[..count].iter().enumerate() {
            let logits = if index == 0 {
                &self.logits
            } else {
                &rows[index - 1]
            };
            let verdict = self
                .sampler
                .distribution(
                    logits,
                    &self.options,
                    &self.text.recent_tokens,
                    None,
                    self.offset as u64,
                )?
                .verify_at(draft, self.seed, self.offset as u64)?;
            match verdict {
                Verdict::Accept => {
                    response.accepted += 1;
                    if !self.engine.eos_token_ids.contains(&draft) {
                        kept += 1;
                        self.offset += 1;
                    }
                    if !self.emit(draft, &mut response)? {
                        break;
                    }
                    response.finish = self.limit();
                    if response.finish.is_some() {
                        break;
                    }
                }
                Verdict::Reject(token) => {
                    correction = Some(token);
                    break;
                }
            }
        }
        if kept < count && !self.model.as_mut().unwrap().truncate_to(self.offset) {
            return Err(JoshuaError::Inference(
                "external draft rollback failed".into(),
            ));
        }
        if response.finish.is_none() {
            let next = match correction {
                Some(token) => token,
                None => self.sampler.sample_at(
                    if count == 0 {
                        &self.logits
                    } else {
                        &rows[count - 1]
                    },
                    &self.options,
                    &self.text.recent_tokens,
                    None,
                    self.seed,
                    self.offset as u64,
                )?,
            };
            if self.emit(next, &mut response)? {
                self.logits = self.model.as_mut().unwrap().forward_tokens(
                    &[next],
                    self.offset,
                    &self.engine.dense_device,
                )?;
                self.offset += 1;
                response.finish = self.limit();
            }
        }
        self.model.as_mut().unwrap().discard_verify();
        response.offset = self.offset;
        response.completion_tokens = self.text.n_decoded;
        Ok(response)
    }
}
