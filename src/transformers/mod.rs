//! # Transformers
//!
//! The transformers module contains some basic implementation
//! of transformers based models used to process the user prompt
//! and generate the related embeddings. It also includes some
//! simple tokenization.

pub mod clip;

#[cfg(feature = "std")]
pub mod clip_tokenizer;

#[cfg(feature = "std")]
pub use clip_tokenizer::{SimpleTokenizer, SimpleTokenizerConfig, TokenizerError};
