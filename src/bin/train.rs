use std::collections::HashMap;
use std::fs::File;
use std::io;

use fst::Streamer;

#[allow(dead_code)]
#[path = "../classifier.rs"]
mod classifier;

const TEXT_INDEX: usize = 0;
const LABEL_INDEX: usize = 1;

const LABEL_HAM: &str = "ham";
const LABEL_SPAM: &str = "spam";

#[derive(Debug)]
struct TrainingStats {
    total_samples: u32,
    spam_samples: u32,
    ham_samples: u32,
    total_tokens: u32,
    unique_tokens: u32,
    avg_tokens_per_sample: f32,
}

impl TrainingStats {
    fn new() -> Self {
        Self {
            total_samples: 0,
            spam_samples: 0,
            ham_samples: 0,
            total_tokens: 0,
            unique_tokens: 0,
            avg_tokens_per_sample: 0.0,
        }
    }

    fn print(&self) {
        println!("=== Training Statistics ===");
        println!("Total samples: {}", self.total_samples);
        println!(
            "Spam samples: {} ({:.1}%)",
            self.spam_samples,
            (self.spam_samples as f32 / self.total_samples as f32) * 100.0
        );
        println!(
            "Ham samples: {} ({:.1}%)",
            self.ham_samples,
            (self.ham_samples as f32 / self.total_samples as f32) * 100.0
        );
        println!("Total tokens: {}", self.total_tokens);
        println!("Unique tokens: {}", self.unique_tokens);
        println!(
            "Average tokens per sample: {:.1}",
            self.avg_tokens_per_sample
        );
        println!("==========================");
    }
}

fn main() {
    let input_path = std::env::args()
        .nth(1)
        .expect("Should have training dataset as first argument");
    let output_path = std::env::args()
        .nth(2)
        .expect("Should have output as second argument");

    // Build counters
    println!("Building token counters...");

    let mut counters: HashMap<String, classifier::Counter> = HashMap::with_capacity(256);
    let mut stats = TrainingStats::new();

    // Extend with model if exists
    if std::fs::exists(&output_path).unwrap() {
        println!("Loading existing model...");

        let data = std::fs::read(&output_path).unwrap();
        let map = fst::Map::new(data).unwrap();

        let mut stream = map.stream();
        while let Some((key, value)) = stream.next() {
            let key = String::from_utf8(key.to_vec()).unwrap();
            let counter = classifier::Counter::from_u64(value);

            counters.insert(key, counter);
            stats.total_tokens += counter.spam + counter.ham;
            stats.unique_tokens += 1;
        }
    }

    // Read dataset
    println!("Reading training dataset...");

    let file = File::open(input_path).expect("Could not open file");
    let reader = io::BufReader::new(file);

    let reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(reader);

    let records = reader.into_records().filter_map(|record| record.ok());
    for record in records {
        let text = record.get(TEXT_INDEX).unwrap();
        let label = record.get(LABEL_INDEX).unwrap();

        let (is_spam, is_ham) = (label == LABEL_SPAM, label == LABEL_HAM);

        if is_spam {
            stats.spam_samples += 1;
        } else if is_ham {
            stats.ham_samples += 1;
        }
        stats.total_samples += 1;

        let tokens = classifier::tokenize(text);
        stats.total_tokens += tokens.len() as u32;

        for token in tokens {
            let counter = counters.entry(token).or_default();

            if is_spam {
                counter.spam += 1;
            } else if is_ham {
                counter.ham += 1;
            }
        }
    }

    stats.unique_tokens = counters.len() as u32;
    stats.avg_tokens_per_sample = stats.total_tokens as f32 / stats.total_samples as f32;
    stats.print();

    let mut counters: Vec<_> = counters.into_iter().collect();
    counters.sort_by(|(left, _), (right, _)| left.cmp(right));

    // Build FST model
    println!("Building FST model...");

    let writer = io::BufWriter::new(File::create(&output_path).unwrap());
    let mut builder = fst::MapBuilder::new(writer).unwrap();

    for (word, counter) in counters {
        builder.insert(word, counter.to_u64()).unwrap();
    }

    builder.finish().unwrap();
    println!("Model saved to: {}", output_path);

    // Validate model
    println!("Validating model...");
    validate_model(&output_path, &stats);
}

fn validate_model(model_path: &str, _stats: &TrainingStats) {
    let data = std::fs::read(model_path).unwrap();
    let map = fst::Map::new(&data).unwrap();

    let mut total_spam = 0u32;
    let mut total_ham = 0u32;
    let mut unique_tokens = 0u32;

    let mut stream = map.stream();
    while let Some((_, value)) = stream.next() {
        let counter = classifier::Counter::from_u64(value);
        total_spam += counter.spam;
        total_ham += counter.ham;
        unique_tokens += 1;
    }

    println!("=== Model Validation ===");
    println!("Total spam tokens in model: {}", total_spam);
    println!("Total ham tokens in model: {}", total_ham);
    println!("Unique tokens in model: {}", unique_tokens);
    println!("Model size: {:.2} MB", data.len() as f32 / 1024.0 / 1024.0);

    // Calculate some basic statistics
    let prior_spam = total_spam as f32 / (total_spam + total_ham) as f32;
    println!("Prior P(spam): {:.3}", prior_spam);
    println!("Prior P(ham): {:.3}", 1.0 - prior_spam);
    println!("=======================");
}
