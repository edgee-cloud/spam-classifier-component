pub const SPAM_TRESHOLD: f32 = 0.80;
pub const DEFAULT_ALPHA: f32 = 1.0;

static MODEL: &[u8] = include_bytes!("../model.fst");

use fst::Streamer;

#[derive(Default, Debug, Clone, Copy)]
pub struct Counter {
    pub spam: u32,
    pub ham: u32,
}

impl Counter {
    pub fn from_u64(value: u64) -> Self {
        let spam = (value >> 32) as u32;
        let ham = (value & u32::MAX as u64) as u32;

        Self { spam, ham }
    }

    #[allow(dead_code)]
    pub fn to_u64(self) -> u64 {
        let spam = self.spam as u64;
        let ham = self.ham as u64;

        (spam << 32) | ham
    }
}

/// Naive Bayes classifier statistics
#[derive(Debug, Clone)]
pub struct ClassifierStats {
    pub total_spam: u32,
    pub total_ham: u32,
    pub total_tokens: u32,
    pub unique_tokens: u32,
}

impl ClassifierStats {
    pub fn new() -> Self {
        Self {
            total_spam: 0,
            total_ham: 0,
            total_tokens: 0,
            unique_tokens: 0,
        }
    }

    pub fn from_model<D: AsRef<[u8]>>(model: &fst::Map<D>) -> Self {
        let mut stats = Self::new();
        let mut stream = model.stream();

        while let Some((_, value)) = stream.next() {
            let counter = Counter::from_u64(value);
            stats.total_spam += counter.spam;
            stats.total_ham += counter.ham;
            stats.unique_tokens += 1;
        }

        stats.total_tokens = stats.total_spam + stats.total_ham;
        stats
    }

    /// Calculate prior probability P(spam)
    pub fn prior_spam(&self) -> f32 {
        if self.total_tokens == 0 {
            return 0.5; // Default to 50% if no data
        }
        self.total_spam as f32 / self.total_tokens as f32
    }

    /// Calculate prior probability P(ham)
    pub fn prior_ham(&self) -> f32 {
        1.0 - self.prior_spam()
    }
}

/// Optimized Naive Bayes classifier for spam detection
pub struct NaiveBayesClassifier<D> {
    model: fst::Map<D>,
    stats: ClassifierStats,
    alpha: f32,          // Laplace smoothing parameter
    spam_threshold: f32, // Spam classification threshold
}

impl NaiveBayesClassifier<&'static [u8]> {
    pub fn new() -> Self {
        let model = fst::Map::new(MODEL).unwrap();
        Self::from_model(model)
    }
}

impl<D: AsRef<[u8]>> NaiveBayesClassifier<D> {
    pub fn from_model(model: fst::Map<D>) -> Self {
        let stats = ClassifierStats::from_model(&model);

        Self {
            model,
            stats,
            alpha: DEFAULT_ALPHA,
            spam_threshold: SPAM_TRESHOLD,
        }
    }

    /// Calculate both spam and ham likelihoods from a single counter
    fn calculate_likelihoods(&self, counter: &Counter) -> (f32, f32) {
        let spam_numerator = counter.spam as f32 + self.alpha;
        let spam_denominator =
            self.stats.total_spam as f32 + (self.alpha * self.stats.unique_tokens as f32);
        let spam_likelihood = spam_numerator / spam_denominator;

        let ham_numerator = counter.ham as f32 + self.alpha;
        let ham_denominator =
            self.stats.total_ham as f32 + (self.alpha * self.stats.unique_tokens as f32);
        let ham_likelihood = ham_numerator / ham_denominator;

        (spam_likelihood, ham_likelihood)
    }

    /// Get token counter from the FST model
    fn get_token_counter(&self, word: &str) -> Counter {
        self.model
            .get(word)
            .map(Counter::from_u64)
            .unwrap_or_default()
    }

    /// Classify text and return spam probability
    pub fn classify(&self, text: &str) -> f32 {
        let tokens = tokenize(text);

        if tokens.is_empty() {
            return self.stats.prior_spam(); // Return prior if no tokens
        }

        // Calculate log probabilities to avoid numerical underflow
        let mut log_prob_spam = self.stats.prior_spam().ln();
        let mut log_prob_ham = self.stats.prior_ham().ln();

        for token in tokens {
            let counter = self.get_token_counter(&token);
            let (p_word_spam, p_word_ham) = self.calculate_likelihoods(&counter);

            // Add log probabilities instead of multiplying
            log_prob_spam += p_word_spam.ln();
            log_prob_ham += p_word_ham.ln();
        }

        // Convert back to probability using Bayes' theorem
        let log_denominator = (log_prob_spam.exp() + log_prob_ham.exp()).ln();
        let log_prob_spam_given_text = log_prob_spam - log_denominator;

        let result = log_prob_spam_given_text.exp();

        // Handle NaN and infinite values
        if result.is_nan() || result.is_infinite() {
            self.stats.prior_spam()
        } else {
            result
        }
    }

    /// Set the alpha value for Laplace smoothing
    #[allow(dead_code)]
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
    }

    /// Get the current alpha value
    #[allow(dead_code)]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Set the spam threshold value
    #[allow(dead_code)]
    pub fn set_spam_threshold(&mut self, threshold: f32) {
        self.spam_threshold = threshold;
    }

    /// Get the current spam threshold value
    #[allow(dead_code)]
    pub fn spam_threshold(&self) -> f32 {
        self.spam_threshold
    }

    /// Get detailed classification results
    pub fn classify_detailed(&self, text: &str) -> ClassificationResult {
        let spam_probability = self.classify(text);
        let is_spam = spam_probability >= self.spam_threshold;

        ClassificationResult {
            spam_probability,
            ham_probability: 1.0 - spam_probability,
            is_spam,
            confidence: if is_spam {
                spam_probability
            } else {
                1.0 - spam_probability
            },
        }
    }
}

/// Detailed classification result
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub spam_probability: f32,
    pub ham_probability: f32,
    pub is_spam: bool,
    pub confidence: f32,
}

pub fn tokenize(input: &str) -> Vec<String> {
    use unobtanium_segmenter::augmentation::{AugmentationClassify, AugmentationDetectLanguage};
    use unobtanium_segmenter::chain::{ChainAugmenter, ChainSegmenter, StartSegmentationChain};
    use unobtanium_segmenter::normalization::{NormalizationLowercase, NormalizationRustStemmers};
    use unobtanium_segmenter::segmentation::{UnicodeSentenceSplitter, UnicodeWordSplitter};
    use unobtanium_segmenter::SegmentedTokenKind;

    input
        .start_segmentation_chain()
        .chain_owned_segmenter(UnicodeSentenceSplitter::new())
        .chain_owned_augmenter(AugmentationDetectLanguage::new())
        .chain_owned_segmenter(UnicodeWordSplitter::new())
        .chain_owned_augmenter(AugmentationClassify::new())
        .chain_owned_augmenter(NormalizationLowercase::new())
        .chain_owned_augmenter(NormalizationRustStemmers::new())
        .filter(|token| token.kind == Some(SegmentedTokenKind::AlphaNumeric))
        .map(|token| token.get_text_prefer_normalized_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(input: &str) -> f32 {
        let classifier = NaiveBayesClassifier::new();
        classifier.classify(input)
    }

    #[test]
    fn test_tokenization() {
        let text = "Hello world! This is a test message.";
        let tokens = tokenize(text);

        println!("Tokens: {:?}", tokens);

        assert!(!tokens.is_empty());
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Check for either "message" or "messag" (stemmed version)
        assert!(
            tokens.contains(&"message".to_string()) || tokens.contains(&"messag".to_string()),
            "Expected 'message' or 'messag' in tokens: {:?}",
            tokens
        );
    }

    #[test]
    fn test_empty_input() {
        let score = classify("");
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_spam_indicators() {
        let spam_text = "FREE MONEY! Click here to win $1000000! Limited time offer!";
        let score = classify(spam_text);

        // Should have higher spam probability
        assert!(score > 0.5);
    }

    #[test]
    fn test_ham_indicators() {
        let ham_text = "Hello, how are you doing today? I hope you have a great day.";
        let score = classify(ham_text);

        // Should have lower spam probability
        assert!(score < 0.8);
    }

    #[test]
    fn test_classifier_stats() {
        let stats = ClassifierStats::new();
        assert_eq!(stats.total_spam, 0);
        assert_eq!(stats.total_ham, 0);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.unique_tokens, 0);
        assert_eq!(stats.prior_spam(), 0.5);
        assert_eq!(stats.prior_ham(), 0.5);
    }

    #[test]
    fn test_counter_serialization() {
        let counter = Counter { spam: 10, ham: 5 };
        let serialized = counter.to_u64();
        let deserialized = Counter::from_u64(serialized);

        assert_eq!(counter.spam, deserialized.spam);
        assert_eq!(counter.ham, deserialized.ham);
    }

    #[test]
    fn test_classification_result() {
        let result = ClassificationResult {
            spam_probability: 0.8,
            ham_probability: 0.2,
            is_spam: true,
            confidence: 0.8,
        };

        assert_eq!(result.spam_probability, 0.8);
        assert_eq!(result.ham_probability, 0.2);
        assert!(result.is_spam);
        assert_eq!(result.confidence, 0.8);
    }

    #[test]
    fn test_edge_cases() {
        // Test with very long text
        let long_text = "This is a very long text ".repeat(100);
        let score = classify(&long_text);
        println!("Long text score: {}", score);
        assert!(score >= 0.0 && score <= 1.0, "Long text score: {}", score);

        // Test with special characters
        let special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let score = classify(special_text);
        println!("Special chars score: {}", score);
        assert!(
            score >= 0.0 && score <= 1.0,
            "Special chars score: {}",
            score
        );

        // Test with numbers only
        let numbers = "1234567890";
        let score = classify(numbers);
        println!("Numbers score: {}", score);
        assert!(score >= 0.0 && score <= 1.0, "Numbers score: {}", score);
    }

    #[test]
    fn test_probability_bounds() {
        let texts = vec![
            "Normal email content",
            "FREE MONEY NOW!!!",
            "Meeting tomorrow at 3pm",
            "BUY VIAGRA CHEAP!!!",
            "Hello, how are you?",
        ];

        for text in texts {
            let score = classify(text);
            assert!(
                score >= 0.0 && score <= 1.0,
                "Score {} for text '{}' is out of bounds",
                score,
                text
            );
        }
    }

    #[test]
    fn test_classifier_instance() {
        let classifier = NaiveBayesClassifier::new();
        let result = classifier.classify_detailed("Test message");

        assert!(result.spam_probability >= 0.0 && result.spam_probability <= 1.0);
        assert!(result.ham_probability >= 0.0 && result.ham_probability <= 1.0);
        assert!((result.spam_probability + result.ham_probability - 1.0).abs() < 0.001);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_alpha_methods() {
        let mut classifier = NaiveBayesClassifier::new();

        // Test default alpha value
        assert_eq!(classifier.alpha(), DEFAULT_ALPHA);

        // Test setting alpha value
        classifier.set_alpha(2.0);
        assert_eq!(classifier.alpha(), 2.0);

        // Test classification still works with different alpha
        let result = classifier.classify("Test message");
        assert!(result >= 0.0 && result <= 1.0);
    }
}
