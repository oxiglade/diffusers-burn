//! CLIP Tokenizer
//!
//! BPE tokenizer for CLIP text encoding. This module is only available
//! with the `std` feature enabled.
//!
//! The tokenizer uses byte-pair encoding (BPE) to convert text into tokens
//! that can be processed by the CLIP text encoder.

use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::Path;

/// Errors that can occur during tokenization.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),
    #[error("Invalid BPE file format: {0}")]
    InvalidFormat(String),
    #[error("Unknown padding character: {0}")]
    UnknownPadChar(String),
}

/// Mapping from bytes to unicode characters used by CLIP's BPE.
/// This allows representing any byte sequence as valid unicode.
const BYTES_TO_UNICODE: [(u8, char); 256] = [
    (33, '!'),
    (34, '"'),
    (35, '#'),
    (36, '$'),
    (37, '%'),
    (38, '&'),
    (39, '\''),
    (40, '('),
    (41, ')'),
    (42, '*'),
    (43, '+'),
    (44, ','),
    (45, '-'),
    (46, '.'),
    (47, '/'),
    (48, '0'),
    (49, '1'),
    (50, '2'),
    (51, '3'),
    (52, '4'),
    (53, '5'),
    (54, '6'),
    (55, '7'),
    (56, '8'),
    (57, '9'),
    (58, ':'),
    (59, ';'),
    (60, '<'),
    (61, '='),
    (62, '>'),
    (63, '?'),
    (64, '@'),
    (65, 'A'),
    (66, 'B'),
    (67, 'C'),
    (68, 'D'),
    (69, 'E'),
    (70, 'F'),
    (71, 'G'),
    (72, 'H'),
    (73, 'I'),
    (74, 'J'),
    (75, 'K'),
    (76, 'L'),
    (77, 'M'),
    (78, 'N'),
    (79, 'O'),
    (80, 'P'),
    (81, 'Q'),
    (82, 'R'),
    (83, 'S'),
    (84, 'T'),
    (85, 'U'),
    (86, 'V'),
    (87, 'W'),
    (88, 'X'),
    (89, 'Y'),
    (90, 'Z'),
    (91, '['),
    (92, '\\'),
    (93, ']'),
    (94, '^'),
    (95, '_'),
    (96, '`'),
    (97, 'a'),
    (98, 'b'),
    (99, 'c'),
    (100, 'd'),
    (101, 'e'),
    (102, 'f'),
    (103, 'g'),
    (104, 'h'),
    (105, 'i'),
    (106, 'j'),
    (107, 'k'),
    (108, 'l'),
    (109, 'm'),
    (110, 'n'),
    (111, 'o'),
    (112, 'p'),
    (113, 'q'),
    (114, 'r'),
    (115, 's'),
    (116, 't'),
    (117, 'u'),
    (118, 'v'),
    (119, 'w'),
    (120, 'x'),
    (121, 'y'),
    (122, 'z'),
    (123, '{'),
    (124, '|'),
    (125, '}'),
    (126, '~'),
    (161, '\u{00A1}'), // Inverted exclamation mark
    (162, '\u{00A2}'), // Cent sign
    (163, '\u{00A3}'), // Pound sign
    (164, '\u{00A4}'), // Currency sign
    (165, '\u{00A5}'), // Yen sign
    (166, '\u{00A6}'), // Broken bar
    (167, '\u{00A7}'), // Section sign
    (168, '\u{00A8}'), // Diaeresis
    (169, '\u{00A9}'), // Copyright sign
    (170, '\u{00AA}'), // Feminine ordinal indicator
    (171, '\u{00AB}'), // Left-pointing double angle quotation mark
    (172, '\u{00AC}'), // Not sign
    (174, '\u{00AE}'), // Registered sign
    (175, '\u{00AF}'), // Macron
    (176, '\u{00B0}'), // Degree sign
    (177, '\u{00B1}'), // Plus-minus sign
    (178, '\u{00B2}'), // Superscript two
    (179, '\u{00B3}'), // Superscript three
    (180, '\u{00B4}'), // Acute accent
    (181, '\u{00B5}'), // Micro sign
    (182, '\u{00B6}'), // Pilcrow sign
    (183, '\u{00B7}'), // Middle dot
    (184, '\u{00B8}'), // Cedilla
    (185, '\u{00B9}'), // Superscript one
    (186, '\u{00BA}'), // Masculine ordinal indicator
    (187, '\u{00BB}'), // Right-pointing double angle quotation mark
    (188, '\u{00BC}'), // Vulgar fraction one quarter
    (189, '\u{00BD}'), // Vulgar fraction one half
    (190, '\u{00BE}'), // Vulgar fraction three quarters
    (191, '\u{00BF}'), // Inverted question mark
    (192, '\u{00C0}'), // Latin capital letter A with grave
    (193, '\u{00C1}'), // Latin capital letter A with acute
    (194, '\u{00C2}'), // Latin capital letter A with circumflex
    (195, '\u{00C3}'), // Latin capital letter A with tilde
    (196, '\u{00C4}'), // Latin capital letter A with diaeresis
    (197, '\u{00C5}'), // Latin capital letter A with ring above
    (198, '\u{00C6}'), // Latin capital letter AE
    (199, '\u{00C7}'), // Latin capital letter C with cedilla
    (200, '\u{00C8}'), // Latin capital letter E with grave
    (201, '\u{00C9}'), // Latin capital letter E with acute
    (202, '\u{00CA}'), // Latin capital letter E with circumflex
    (203, '\u{00CB}'), // Latin capital letter E with diaeresis
    (204, '\u{00CC}'), // Latin capital letter I with grave
    (205, '\u{00CD}'), // Latin capital letter I with acute
    (206, '\u{00CE}'), // Latin capital letter I with circumflex
    (207, '\u{00CF}'), // Latin capital letter I with diaeresis
    (208, '\u{00D0}'), // Latin capital letter Eth
    (209, '\u{00D1}'), // Latin capital letter N with tilde
    (210, '\u{00D2}'), // Latin capital letter O with grave
    (211, '\u{00D3}'), // Latin capital letter O with acute
    (212, '\u{00D4}'), // Latin capital letter O with circumflex
    (213, '\u{00D5}'), // Latin capital letter O with tilde
    (214, '\u{00D6}'), // Latin capital letter O with diaeresis
    (215, '\u{00D7}'), // Multiplication sign
    (216, '\u{00D8}'), // Latin capital letter O with stroke
    (217, '\u{00D9}'), // Latin capital letter U with grave
    (218, '\u{00DA}'), // Latin capital letter U with acute
    (219, '\u{00DB}'), // Latin capital letter U with circumflex
    (220, '\u{00DC}'), // Latin capital letter U with diaeresis
    (221, '\u{00DD}'), // Latin capital letter Y with acute
    (222, '\u{00DE}'), // Latin capital letter Thorn
    (223, '\u{00DF}'), // Latin small letter sharp s
    (224, '\u{00E0}'), // Latin small letter a with grave
    (225, '\u{00E1}'), // Latin small letter a with acute
    (226, '\u{00E2}'), // Latin small letter a with circumflex
    (227, '\u{00E3}'), // Latin small letter a with tilde
    (228, '\u{00E4}'), // Latin small letter a with diaeresis
    (229, '\u{00E5}'), // Latin small letter a with ring above
    (230, '\u{00E6}'), // Latin small letter ae
    (231, '\u{00E7}'), // Latin small letter c with cedilla
    (232, '\u{00E8}'), // Latin small letter e with grave
    (233, '\u{00E9}'), // Latin small letter e with acute
    (234, '\u{00EA}'), // Latin small letter e with circumflex
    (235, '\u{00EB}'), // Latin small letter e with diaeresis
    (236, '\u{00EC}'), // Latin small letter i with grave
    (237, '\u{00ED}'), // Latin small letter i with acute
    (238, '\u{00EE}'), // Latin small letter i with circumflex
    (239, '\u{00EF}'), // Latin small letter i with diaeresis
    (240, '\u{00F0}'), // Latin small letter eth
    (241, '\u{00F1}'), // Latin small letter n with tilde
    (242, '\u{00F2}'), // Latin small letter o with grave
    (243, '\u{00F3}'), // Latin small letter o with acute
    (244, '\u{00F4}'), // Latin small letter o with circumflex
    (245, '\u{00F5}'), // Latin small letter o with tilde
    (246, '\u{00F6}'), // Latin small letter o with diaeresis
    (247, '\u{00F7}'), // Division sign
    (248, '\u{00F8}'), // Latin small letter o with stroke
    (249, '\u{00F9}'), // Latin small letter u with grave
    (250, '\u{00FA}'), // Latin small letter u with acute
    (251, '\u{00FB}'), // Latin small letter u with circumflex
    (252, '\u{00FC}'), // Latin small letter u with diaeresis
    (253, '\u{00FD}'), // Latin small letter y with acute
    (254, '\u{00FE}'), // Latin small letter thorn
    (255, '\u{00FF}'), // Latin small letter y with diaeresis
    // Extended characters for bytes 0-32 and 127-160
    (0, '\u{0100}'),   // Latin capital letter A with macron
    (1, '\u{0101}'),   // Latin small letter a with macron
    (2, '\u{0102}'),   // Latin capital letter A with breve
    (3, '\u{0103}'),   // Latin small letter a with breve
    (4, '\u{0104}'),   // Latin capital letter A with ogonek
    (5, '\u{0105}'),   // Latin small letter a with ogonek
    (6, '\u{0106}'),   // Latin capital letter C with acute
    (7, '\u{0107}'),   // Latin small letter c with acute
    (8, '\u{0108}'),   // Latin capital letter C with circumflex
    (9, '\u{0109}'),   // Latin small letter c with circumflex
    (10, '\u{010A}'),  // Latin capital letter C with dot above
    (11, '\u{010B}'),  // Latin small letter c with dot above
    (12, '\u{010C}'),  // Latin capital letter C with caron
    (13, '\u{010D}'),  // Latin small letter c with caron
    (14, '\u{010E}'),  // Latin capital letter D with caron
    (15, '\u{010F}'),  // Latin small letter d with caron
    (16, '\u{0110}'),  // Latin capital letter D with stroke
    (17, '\u{0111}'),  // Latin small letter d with stroke
    (18, '\u{0112}'),  // Latin capital letter E with macron
    (19, '\u{0113}'),  // Latin small letter e with macron
    (20, '\u{0114}'),  // Latin capital letter E with breve
    (21, '\u{0115}'),  // Latin small letter e with breve
    (22, '\u{0116}'),  // Latin capital letter E with dot above
    (23, '\u{0117}'),  // Latin small letter e with dot above
    (24, '\u{0118}'),  // Latin capital letter E with ogonek
    (25, '\u{0119}'),  // Latin small letter e with ogonek
    (26, '\u{011A}'),  // Latin capital letter E with caron
    (27, '\u{011B}'),  // Latin small letter e with caron
    (28, '\u{011C}'),  // Latin capital letter G with circumflex
    (29, '\u{011D}'),  // Latin small letter g with circumflex
    (30, '\u{011E}'),  // Latin capital letter G with breve
    (31, '\u{011F}'),  // Latin small letter g with breve
    (32, '\u{0120}'),  // Latin capital letter G with dot above
    (127, '\u{0121}'), // Latin small letter g with dot above
    (128, '\u{0122}'), // Latin capital letter G with cedilla
    (129, '\u{0123}'), // Latin small letter g with cedilla
    (130, '\u{0124}'), // Latin capital letter H with circumflex
    (131, '\u{0125}'), // Latin small letter h with circumflex
    (132, '\u{0126}'), // Latin capital letter H with stroke
    (133, '\u{0127}'), // Latin small letter h with stroke
    (134, '\u{0128}'), // Latin capital letter I with tilde
    (135, '\u{0129}'), // Latin small letter i with tilde
    (136, '\u{012A}'), // Latin capital letter I with macron
    (137, '\u{012B}'), // Latin small letter i with macron
    (138, '\u{012C}'), // Latin capital letter I with breve
    (139, '\u{012D}'), // Latin small letter i with breve
    (140, '\u{012E}'), // Latin capital letter I with ogonek
    (141, '\u{012F}'), // Latin small letter i with ogonek
    (142, '\u{0130}'), // Latin capital letter I with dot above
    (143, '\u{0131}'), // Latin small letter dotless i
    (144, '\u{0132}'), // Latin capital ligature IJ
    (145, '\u{0133}'), // Latin small ligature ij
    (146, '\u{0134}'), // Latin capital letter J with circumflex
    (147, '\u{0135}'), // Latin small letter j with circumflex
    (148, '\u{0136}'), // Latin capital letter K with cedilla
    (149, '\u{0137}'), // Latin small letter k with cedilla
    (150, '\u{0138}'), // Latin small letter kra
    (151, '\u{0139}'), // Latin capital letter L with acute
    (152, '\u{013A}'), // Latin small letter l with acute
    (153, '\u{013B}'), // Latin capital letter L with cedilla
    (154, '\u{013C}'), // Latin small letter l with cedilla
    (155, '\u{013D}'), // Latin capital letter L with caron
    (156, '\u{013E}'), // Latin small letter l with caron
    (157, '\u{013F}'), // Latin capital letter L with middle dot
    (158, '\u{0140}'), // Latin small letter l with middle dot
    (159, '\u{0141}'), // Latin capital letter L with stroke
    (160, '\u{0142}'), // Latin small letter l with stroke
    (173, '\u{0143}'), // Latin capital letter N with acute
];

/// Regex pattern for tokenizing text.
/// Matches special tokens, contractions, letters, numbers, and other characters.
const TOKENIZER_PATTERN: &str =
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+";

/// Configuration for the CLIP tokenizer.
#[derive(Debug, Clone)]
pub struct SimpleTokenizerConfig {
    /// Maximum sequence length (default: 77 for CLIP).
    pub max_position_embeddings: usize,
    /// Character to use for padding. If None, uses end-of-text token.
    pub pad_with: Option<String>,
}

impl Default for SimpleTokenizerConfig {
    fn default() -> Self {
        Self {
            max_position_embeddings: 77,
            pad_with: None,
        }
    }
}

impl SimpleTokenizerConfig {
    /// Create config for Stable Diffusion v1.5.
    pub fn v1_5() -> Self {
        Self {
            max_position_embeddings: 77,
            pad_with: None,
        }
    }

    /// Create config for Stable Diffusion v2.1.
    pub fn v2_1() -> Self {
        Self {
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
        }
    }
}

/// A BPE tokenizer for CLIP text encoding.
///
/// This tokenizer converts text into token IDs that can be processed
/// by the CLIP text encoder. It uses byte-pair encoding (BPE) with
/// a vocabulary derived from the original CLIP model.
pub struct SimpleTokenizer {
    regex: regex::Regex,
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    start_of_text_token: usize,
    end_of_text_token: usize,
    config: SimpleTokenizerConfig,
}

impl SimpleTokenizer {
    /// Create a new tokenizer from a BPE vocabulary file.
    ///
    /// # Arguments
    /// * `bpe_path` - Path to the BPE vocabulary file (e.g., bpe_simple_vocab_16e6.txt)
    /// * `config` - Tokenizer configuration
    ///
    /// # Returns
    /// A new tokenizer instance or an error if the file cannot be read.
    pub fn new<P: AsRef<Path>>(
        bpe_path: P,
        config: SimpleTokenizerConfig,
    ) -> Result<Self, TokenizerError> {
        let file = std::fs::File::open(bpe_path)?;
        let reader = std::io::BufReader::new(file);

        let bpe_lines: Result<Vec<String>, _> = reader.lines().collect();
        let bpe_lines = bpe_lines?;

        // Parse BPE merges (skip header, take 49152 - 256 - 2 merges)
        let merge_count = 49152 - 256 - 2;
        let bpe_merges: Result<Vec<(String, String)>, TokenizerError> = bpe_lines[1..=merge_count]
            .iter()
            .map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() != 2 {
                    return Err(TokenizerError::InvalidFormat(format!(
                        "Expected 2 tokens, got {}: '{}'",
                        parts.len(),
                        line
                    )));
                }
                Ok((parts[0].to_string(), parts[1].to_string()))
            })
            .collect();
        let bpe_merges = bpe_merges?;

        // Build vocabulary
        let mut vocab: Vec<String> = Vec::new();

        // Add base characters
        for (_, c) in BYTES_TO_UNICODE.iter() {
            vocab.push(c.to_string());
        }

        // Add base characters with end-of-word marker
        for (_, c) in BYTES_TO_UNICODE.iter() {
            vocab.push(format!("{}</w>", c));
        }

        // Add BPE merges
        for (first, second) in bpe_merges.iter() {
            vocab.push(format!("{}{}", first, second));
        }

        // Add special tokens
        let start_of_text_token = vocab.len();
        vocab.push("<|startoftext|>".to_string());
        let end_of_text_token = vocab.len();
        vocab.push("<|endoftext|>".to_string());

        // Build encoder/decoder mappings
        let encoder: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        let decoder: HashMap<usize, String> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        // Build BPE ranks
        let bpe_ranks: HashMap<(String, String), usize> = bpe_merges
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();

        let regex = regex::Regex::new(TOKENIZER_PATTERN)?;

        Ok(Self {
            regex,
            encoder,
            decoder,
            bpe_ranks,
            start_of_text_token,
            end_of_text_token,
            config,
        })
    }

    /// Get pairs of adjacent tokens in a word.
    fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        for i in 1..word.len() {
            pairs.insert((word[i - 1].clone(), word[i].clone()));
        }
        pairs
    }

    /// Apply BPE encoding to a single token.
    fn bpe(&self, token: &str) -> Vec<usize> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        if word.is_empty() {
            return Vec::new();
        }

        // Add end-of-word marker to last character
        let last_idx = word.len() - 1;
        word[last_idx] = format!("{}</w>", word[last_idx]);

        // Iteratively merge pairs with lowest BPE rank
        while word.len() > 1 {
            let pairs = Self::get_pairs(&word);

            // Find pair with lowest rank
            let best_pair = pairs
                .iter()
                .filter_map(|p| self.bpe_ranks.get(p).map(|rank| (rank, p)))
                .min_by_key(|(rank, _)| *rank)
                .map(|(_, p)| p.clone());

            let (first, second) = match best_pair {
                Some(p) => p,
                None => break,
            };

            // Merge the pair
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    new_word.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;
        }

        // Convert to token IDs
        word.iter()
            .filter_map(|w| self.encoder.get(w).copied())
            .collect()
    }

    /// Encode text to token IDs with optional padding.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `pad_to` - If Some, pad the result to this length
    ///
    /// # Returns
    /// A vector of token IDs.
    pub fn encode_with_padding(
        &self,
        text: &str,
        pad_to: Option<usize>,
    ) -> Result<Vec<usize>, TokenizerError> {
        let text = text.to_lowercase();
        let mut tokens = vec![self.start_of_text_token];

        // Tokenize each match
        for cap in self.regex.captures_iter(&text) {
            if let Some(m) = cap.get(0) {
                tokens.extend(self.bpe(m.as_str()));
            }
        }

        // Add end token
        tokens.push(self.end_of_text_token);

        // Apply padding if requested
        if let Some(target_len) = pad_to {
            // Truncate if necessary (keep room for end token)
            if tokens.len() > target_len {
                tokens.truncate(target_len - 1);
                tokens.push(self.end_of_text_token);
            }

            // Pad to target length
            let pad_token = match &self.config.pad_with {
                None => self.end_of_text_token,
                Some(pad_char) => self
                    .encoder
                    .get(pad_char)
                    .copied()
                    .ok_or_else(|| TokenizerError::UnknownPadChar(pad_char.clone()))?,
            };

            while tokens.len() < target_len {
                tokens.push(pad_token);
            }
        }

        Ok(tokens)
    }

    /// Encode text to token IDs, padding to max_position_embeddings.
    ///
    /// This is the main entry point for tokenization.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    ///
    /// # Returns
    /// A vector of token IDs padded to the configured max length.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, TokenizerError> {
        self.encode_with_padding(text, Some(self.config.max_position_embeddings))
    }

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    /// * `tokens` - The token IDs to decode
    ///
    /// # Returns
    /// The decoded text string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let text: String = tokens
            .iter()
            .filter_map(|t| self.decoder.get(t))
            .cloned()
            .collect();
        text.replace("</w>", " ")
    }

    /// Get the start-of-text token ID.
    pub fn start_of_text_token(&self) -> usize {
        self.start_of_text_token
    }

    /// Get the end-of-text token ID.
    pub fn end_of_text_token(&self) -> usize {
        self.end_of_text_token
    }

    /// Get the maximum sequence length.
    pub fn max_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require the BPE vocabulary file to be present.
    // The file can be downloaded from:
    // https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz

    #[test]
    fn test_tokenizer_config() {
        let config = SimpleTokenizerConfig::v1_5();
        assert_eq!(config.max_position_embeddings, 77);
        assert!(config.pad_with.is_none());

        let config = SimpleTokenizerConfig::v2_1();
        assert_eq!(config.max_position_embeddings, 77);
        assert_eq!(config.pad_with, Some("!".to_string()));
    }
}
