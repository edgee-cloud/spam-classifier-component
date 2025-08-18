use std::collections::HashMap;

use anyhow::Result;

use bindings::wasi::http::types::{IncomingRequest, ResponseOutparam};
use helpers::body::Json;

mod bindings {
    wit_bindgen::generate!({
        path: ".edgee/wit",
        world: "edge-function",
        generate_all,
        pub_export_macro: true,
        default_bindings_module: "$crate::bindings",
    });
}
mod classifier;
mod helpers;

struct Component;
bindings::export!(Component);

impl bindings::exports::wasi::http::incoming_handler::Guest for Component {
    fn handle(req: IncomingRequest, response_out: ResponseOutparam) {
        helpers::run(req, response_out, handle);
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Input {
    input: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Output {
    text: String,
    spam_probability: f64,
    ham_probability: f64,
    is_spam: bool,
    confidence: f64,
}

fn handle(req: http::Request<Json<Input>>) -> Result<http::Response<Json<Output>>> {
    let Json(Input { ref input }) = req.body();

    let settings = Settings::from_req(&req)?;
    let mut classifier = classifier::NaiveBayesClassifier::new();
    classifier.set_spam_threshold(settings.spam_threshold);
    classifier.set_alpha(settings.laplace_smoothing_factor);
    let result = classifier.classify_detailed(input);

    http::Response::builder()
        .status(200)
        .body(Json(Output {
            text: input.clone(),
            spam_probability: result.spam_probability,
            ham_probability: result.ham_probability,
            is_spam: result.is_spam,
            confidence: result.confidence,
        }))
        .map_err(Into::into)
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct Settings {
    pub spam_threshold: f64,
    pub laplace_smoothing_factor: f64,
}

impl Settings {
    pub fn new(headers: &http::header::HeaderMap) -> Result<Self> {
        let value = headers
            .get("x-edgee-component-settings")
            .ok_or_else(|| anyhow::anyhow!("Missing 'x-edgee-component-settings' header"))
            .and_then(|value| value.to_str().map_err(Into::into))?;
        let data: HashMap<String, String> = serde_json::from_str(value)?;

        let spam_threshold = data
            .get("spam_threshold")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(classifier::SPAM_TRESHOLD);

        let laplace_smoothing_factor = data
            .get("laplace_smoothing_factor")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(classifier::DEFAULT_ALPHA);

        Ok(Self { spam_threshold, laplace_smoothing_factor })
    }

    pub fn from_req<B>(req: &http::Request<B>) -> Result<Self> {
        Self::new(req.headers())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_function() {
        // Create test input
        let input = Input {
            input: "Hello, this is a test message".to_string(),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("/")
            .header("x-edgee-component-settings", "{}")
            .body(Json(input))
            .unwrap();

        // Call handle function
        let response = handle(req).unwrap();

        // Verify response
        assert_eq!(response.status(), 200);

        let Json(output) = response.body();
        assert_eq!(output.text, "Hello, this is a test message");
        assert!(output.spam_probability >= 0.0 && output.spam_probability <= 1.0);
        assert!(output.ham_probability >= 0.0 && output.ham_probability <= 1.0);
        assert!((output.spam_probability + output.ham_probability - 1.0).abs() < 0.001);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
        assert_eq!(
            output.is_spam,
            output.spam_probability >= classifier::SPAM_TRESHOLD
        );
    }

    #[test]
    fn test_handle_spam_input() {
        let input = Input {
            input: "FREE MONEY! Click here to win $1000000!".to_string(),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("/")
            .header("x-edgee-component-settings", "{}")
            .body(Json(input))
            .unwrap();

        let response = handle(req).unwrap();

        assert_eq!(response.status(), 200);

        let Json(output) = response.body();
        assert_eq!(output.text, "FREE MONEY! Click here to win $1000000!");
        assert!(output.spam_probability > 0.5); // Should be high spam probability
        assert!(output.spam_probability >= 0.0 && output.spam_probability <= 1.0);
        assert!(output.ham_probability >= 0.0 && output.ham_probability <= 1.0);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_handle_ham_input() {
        let input = Input {
            input: "Good morning! How are you today?".to_string(),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("/")
            .header("x-edgee-component-settings", "{}")
            .body(Json(input))
            .unwrap();

        let response = handle(req).unwrap();

        assert_eq!(response.status(), 200);

        let Json(output) = response.body();
        assert_eq!(output.text, "Good morning! How are you today?");
        assert!(output.spam_probability >= 0.0 && output.spam_probability <= 1.0);
        assert!(output.ham_probability >= 0.0 && output.ham_probability <= 1.0);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_handle_empty_input() {
        let input = Input {
            input: "".to_string(),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("/")
            .header("x-edgee-component-settings", "{}")
            .body(Json(input))
            .unwrap();

        let response = handle(req).unwrap();

        assert_eq!(response.status(), 200);

        let Json(output) = response.body();
        assert_eq!(output.text, "");
        assert!(output.spam_probability >= 0.0 && output.spam_probability <= 1.0);
        assert!(output.ham_probability >= 0.0 && output.ham_probability <= 1.0);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_output_structure() {
        let input = Input {
            input: "Test message for structure validation".to_string(),
        };

        let req = http::Request::builder()
            .method("POST")
            .uri("/")
            .header("x-edgee-component-settings", "{}")
            .body(Json(input))
            .unwrap();

        let response = handle(req).unwrap();
        let Json(output) = response.body();

        // Test that all required fields are present and valid
        assert!(!output.text.is_empty());
        assert!(output.spam_probability.is_finite());
        assert!(output.ham_probability.is_finite());
        assert!(output.confidence.is_finite());

        // Test probability bounds
        assert!(output.spam_probability >= 0.0);
        assert!(output.spam_probability <= 1.0);
        assert!(output.ham_probability >= 0.0);
        assert!(output.ham_probability <= 1.0);
        assert!(output.confidence >= 0.0);
        assert!(output.confidence <= 1.0);

        // Test probability sum
        let prob_sum = output.spam_probability + output.ham_probability;
        assert!(
            (prob_sum - 1.0).abs() < 0.001,
            "Probabilities should sum to 1.0, got {}",
            prob_sum
        );

        // Test is_spam consistency
        assert_eq!(
            output.is_spam,
            output.spam_probability >= classifier::SPAM_TRESHOLD
        );

        // Test confidence calculation
        let expected_confidence = if output.is_spam {
            output.spam_probability
        } else {
            output.ham_probability
        };
        assert!((output.confidence - expected_confidence).abs() < 0.001);
    }
}
