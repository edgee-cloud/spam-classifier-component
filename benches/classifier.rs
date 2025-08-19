use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

use classifier::{tokenize, NaiveBayesClassifier};

#[allow(dead_code)]
#[path = "../src/classifier.rs"]
mod classifier;

fn bench_classify_short_text(c: &mut Criterion) {
    let classifier = NaiveBayesClassifier::new();
    let text = "Hello world";
    let token_count = tokenize(text).len() as u64;

    let mut group = c.benchmark_group("classify_short_text");
    group.throughput(Throughput::Elements(token_count));
    group.bench_function("classify", |b| {
        b.iter(|| {
            black_box(classifier.classify(black_box(text)));
        })
    });
    group.finish();
}

fn bench_classify_medium_text(c: &mut Criterion) {
    let classifier = NaiveBayesClassifier::new();
    let text = "Hello world! This is a medium length message with several words to test classification performance.";
    let token_count = tokenize(text).len() as u64;

    let mut group = c.benchmark_group("classify_medium_text");
    group.throughput(Throughput::Elements(token_count));
    group.bench_function("classify", |b| {
        b.iter(|| {
            black_box(classifier.classify(black_box(text)));
        })
    });
    group.finish();
}

fn bench_classify_long_text(c: &mut Criterion) {
    let classifier = NaiveBayesClassifier::new();
    let text = "This is a very long text message that contains many words and tokens. \
                It is designed to test the performance of the classifier with longer inputs \
                that might be more representative of real-world email content. The text contains \
                various words including some that might be associated with spam like FREE, MONEY, \
                OFFER, and CLICK HERE, while also containing normal conversational text.";
    let token_count = tokenize(text).len() as u64;

    let mut group = c.benchmark_group("classify_long_text");
    group.throughput(Throughput::Elements(token_count));
    group.bench_function("classify", |b| {
        b.iter(|| {
            black_box(classifier.classify(black_box(text)));
        })
    });
    group.finish();
}

fn bench_classify_spam_text(c: &mut Criterion) {
    let classifier = NaiveBayesClassifier::new();
    let text = "FREE MONEY! Click here to win $1000000! Limited time offer! \
                Buy now! Guaranteed results! No payment required! Act fast!";
    let token_count = tokenize(text).len() as u64;

    let mut group = c.benchmark_group("classify_spam_text");
    group.throughput(Throughput::Elements(token_count));
    group.bench_function("classify", |b| {
        b.iter(|| {
            black_box(classifier.classify(black_box(text)));
        })
    });
    group.finish();
}

fn bench_tokenization(c: &mut Criterion) {
    let text = "Hello world! This is a test message with various punctuation marks, numbers 123, and symbols @#$%.";

    c.bench_function("tokenization", |b| {
        b.iter(|| {
            black_box(tokenize(black_box(text)));
        })
    });
}

criterion_group!(
    benches,
    bench_classify_short_text,
    bench_classify_medium_text,
    bench_classify_long_text,
    bench_classify_spam_text,
    bench_tokenization,
);
criterion_main!(benches);
