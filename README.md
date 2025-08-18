<div align="center">
<p align="center">
  <a href="https://www.edgee.cloud">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://cdn.edgee.cloud/img/component-dark.svg">
      <img src="https://cdn.edgee.cloud/img/component.svg" height="100" alt="Edgee">
    </picture>
  </a>
</p>
</div>

<h1 align="center">Spam Classifier component for Edgee</h1>

[![Coverage Status](https://coveralls.io/repos/github/edgee-cloud/spam-classifier-component/badge.svg)](https://coveralls.io/github/edgee-cloud/spam-classifier-component)
[![GitHub issues](https://img.shields.io/github/issues/edgee-cloud/spam-classifier-component.svg)](https://github.com/edgee-cloud/spam-classifier-component/issues)
[![Edgee Component Registry](https://img.shields.io/badge/Edgee_Component_Registry-Public-green.svg)](https://www.edgee.cloud/edgee/spam-classifier)

A high-performance spam classification component for Edgee, implementing optimized Naive Bayes algorithms with Finite State Transducers (FST) for efficient token storage and lookup, served directly at the edge.

This component allows you to map spam classification to a specific endpoint like `/classify` and can be invoked from your frontend code to classify text content in real-time.

## Quick Start

1. Download the latest component version
2. Place `component.wasm` on your server
3. Configure your `edgee.toml` file

## Configuration

Add the following configuration to your `edgee.toml`:

```toml
[[components.edge_functions]]
id = "spam-classifier"
file = "/var/edgee/components/component.wasm"
settings.edgee_path = "/classify"
settings.spam_threshold = "0.80"
settings.laplace_smoothing_factor = "1.0"
```

### Settings

- **spam_threshold** (optional): Threshold for spam classification (default: 0.80)
  - Values between 0.0 and 1.0
  - Higher values = more strict spam detection
  - Lower values = more sensitive spam detection

- **laplace_smoothing_factor** (optional): Laplace smoothing parameter for Naive Bayes (default: 1.0)
  - Values 0.0 and above
  - Higher values = more smoothing for unseen tokens
  - Lower values = less smoothing, may improve accuracy but reduce robustness

## Usage

### HTTP API

Send a POST request to classify text:

```bash
curl -X POST https://your-edge-function-url/classify \
  -H "Content-Type: application/json" \
  -d '{"input": "FREE MONEY! Click here to win $1000000!"}'
```

Response:
```json
{
  "text": "FREE MONEY! Click here to win $1000000!",
  "spam_probability": 0.8542,
  "ham_probability": 0.1458,
  "is_spam": true,
  "confidence": 0.8542
}
```

### JavaScript Example

```javascript
const response = await fetch('/classify', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    input: "Hello, how are you today?"
  })
});

const result = await response.json();
console.log(`Spam probability: ${result.spam_probability}`);
console.log(`Ham probability: ${result.ham_probability}`);
console.log(`Is spam: ${result.is_spam}`);
console.log(`Confidence: ${result.confidence}`);
```

## Features

### 🧠 **Machine Learning Core**
- **Naive Bayes classifier** with optimized likelihood calculations
- **Configurable Laplace smoothing** (default α=1.0) to handle unseen tokens
- **Prior probability calculation** from training data statistics
- **Detailed classification results** with spam/ham probabilities and confidence scores

### 📝 **Text Processing Pipeline**
- **Advanced tokenization** with Unicode sentence and word splitting via unobtanium-segmenter
- **Multi-language support** with automatic language detection
- **Text normalization** (lowercase conversion and stemming via rust-stemmers)
- **AlphaNumeric token filtering** to focus on meaningful content
- **Edge case handling** for empty input and special characters

### ⚡ **Performance Optimizations**
- **Finite State Transducer (FST)** for O(log n) token lookup with embedded model
- **64-bit packed counters** for space-efficient token storage
- **Log-space calculations** to prevent numerical underflow
- **Static model embedding** for fast initialization in Wasm environment

### 🛡️ **Reliability & Safety**
- **Probability bounds enforcement** (0.0-1.0) with NaN/infinite value protection
- **Fallback to prior probabilities** for edge cases
- **Extensive test coverage** with edge case validation
- **Benchmarking suite** for performance regression testing

## Development

### Prerequisites

- Rust (latest stable version)
- `wasm32-wasip2` target: `rustup target add wasm32-wasip2`

### Build Commands

Build the component:
```bash
edgee component build
```

Or build manually:
```bash
cargo build --target wasm32-wasip2 --release
cp target/wasm32-wasip2/release/spam_classifier_component.wasm component.wasm
```

### Testing

Run the test suite:
```bash
cargo test
```

Run performance benchmarks:
```bash
cargo bench
```

## Performance

Performance benchmarks on x86 (native, not WASM):

- **Short text** (2 tokens): ~28 µs → **72K tokens/sec**
- **Medium text** (15 tokens): ~66 µs → **227K tokens/sec** 
- **Long text** (62 tokens): ~128 µs → **484K tokens/sec**
- **Spam text** (19 tokens): ~204 µs → **93K tokens/sec**

*Note: WASM performance will be lower than native x86 benchmarks shown above*

Key optimizations:
- **Single model lookup per token** (reduced from 2 to 1 lookup)
- **FST-based token storage** for fast O(log n) lookups
- **Optimized tokenization pipeline** with stemming and normalization

## Model Training

The spam classifier includes a powerful training binary that can build and update models from CSV datasets.

### Training a New Model

```bash
cargo run --bin train --features training -- input.csv model.fst
```

Note: The training feature must be enabled to build the training binary.

### Dataset Format

Training data should be in CSV format with headers:
- **Column 1**: Text content to classify
- **Column 2**: Label (`spam` or `ham`)

Example CSV:
```csv
text,label
"FREE MONEY! Click here now!",spam
"Hello, how are you today?",ham
"URGENT: Your account will be closed!",spam
"Meeting scheduled for tomorrow at 3pm",ham
```

### Incremental Training

The trainer can extend existing models by loading and updating them:

```bash
# Train initial model
cargo run --bin train --features training -- dataset1.csv model.fst

# Update existing model with new data
cargo run --bin train --features training -- dataset2.csv model.fst
```

### Training Output

The trainer provides detailed statistics during training:

```
=== Training Statistics ===
Total samples: 10000
Spam samples: 4832 (48.3%)
Ham samples: 5168 (51.7%)
Total tokens: 125420
Unique tokens: 8934
Average tokens per sample: 12.5
==========================

=== Model Validation ===
Total spam tokens in model: 62108
Total ham tokens in model: 63312
Unique tokens in model: 8934
Model size: 2.31 MB
Prior P(spam): 0.495
Prior P(ham): 0.505
=======================
```

### Model Storage

- Models are stored as **Finite State Transducers (FST)** for optimal performance
- Binary format provides fast loading and efficient memory usage
- Token counters are packed into 64-bit values for space efficiency
- Alphabetically sorted keys enable O(log n) lookup times

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`cargo test`)
5. Submit a pull request

## Security

If you discover a security vulnerability, please email security@edgee.cloud. All security vulnerabilities will be promptly addressed.