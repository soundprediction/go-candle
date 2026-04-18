use std::ffi::CStr;
use std::os::raw::c_char;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{
    Config as DebertaV2Config, DebertaV2SeqClassificationModel,
};

use crate::{create_hf_repo, json_str, load_weight_files, parse_config_json, set_last_error};

/// Opaque wrapper for a sequence classification pipeline.
pub struct SeqClassificationPipelineWrapper {
    model: DebertaV2SeqClassificationModel,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    num_labels: usize,
}

/// Result of sequence classification — raw logits for each class.
#[repr(C)]
pub struct SeqClassificationResult {
    pub logits: *mut f32,
    pub num_classes: usize,
}

/// Batch result of sequence classification.
#[repr(C)]
pub struct BatchSeqClassificationResult {
    pub logits: *mut f32,
    pub num_classes: usize,
    pub count: usize,
}

fn softmax(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

fn load_seq_classification_model(
    config: &serde_json::Value,
    device: &Device,
) -> anyhow::Result<(
    DebertaV2SeqClassificationModel,
    tokenizers::Tokenizer,
    usize,
)> {
    let model_id = json_str(config, "model_id", "");
    if model_id.is_empty() {
        anyhow::bail!("model_id is required");
    }

    let cache_dir = config
        .get("cache_dir")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let repo = create_hf_repo(model_id, cache_dir.as_deref())?;

    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;

    let deberta_config: DebertaV2Config =
        serde_json::from_reader(std::fs::File::open(&config_path)?)?;

    let num_labels = deberta_config
        .id2label
        .as_ref()
        .map(|m| m.len())
        .unwrap_or(3); // Default to 3 for NLI (entailment/neutral/contradiction)

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;

    let weight_files = load_weight_files(&repo)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device)? };
    let model = DebertaV2SeqClassificationModel::load(vb, &deberta_config, None)?;

    Ok((model, tokenizer, num_labels))
}

fn classify_text(
    wrapper: &SeqClassificationPipelineWrapper,
    text: &str,
    apply_softmax: bool,
) -> anyhow::Result<Vec<f32>> {
    let encoding = wrapper
        .tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenization error: {e}"))?;

    let token_ids = encoding.get_ids();
    let input_ids = Tensor::new(token_ids, &wrapper.device)?.unsqueeze(0)?;
    let token_type_ids = Tensor::zeros_like(&input_ids)?;

    // Build attention mask (1 for real tokens)
    let attention_mask = Tensor::ones(input_ids.shape(), DType::I64, &wrapper.device)?;

    let logits = wrapper.model.forward(
        &input_ids,
        Some(token_type_ids),
        Some(attention_mask),
    )?;

    let mut result = logits.squeeze(0)?.to_vec1::<f32>()?;

    if apply_softmax {
        softmax(&mut result);
    }

    Ok(result)
}

fn classify_text_batch(
    wrapper: &SeqClassificationPipelineWrapper,
    texts: &[&str],
    apply_softmax: bool,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut all_results = Vec::with_capacity(texts.len());
    for text in texts {
        all_results.push(classify_text(wrapper, text, apply_softmax)?);
    }
    Ok(all_results)
}

/// Create a new sequence classification pipeline from JSON config.
///
/// Config fields:
/// - `model_id` (required): HF Hub model identifier (e.g. "cross-encoder/nli-deberta-v3-xsmall")
/// - `cache_dir` (optional): custom HF cache directory
/// - `apply_softmax` (optional): apply softmax to logits, default true
#[no_mangle]
pub extern "C" fn new_seq_classification_pipeline(
    config_json: *const c_char,
) -> *mut SeqClassificationPipelineWrapper {
    let config = match parse_config_json(config_json) {
        Ok(c) => c,
        Err(e) => {
            set_last_error(e);
            return std::ptr::null_mut();
        }
    };

    let device = Device::Cpu;

    match load_seq_classification_model(&config, &device) {
        Ok((model, tokenizer, num_labels)) => {
            let wrapper = SeqClassificationPipelineWrapper {
                model,
                tokenizer,
                device,
                num_labels,
            };
            Box::into_raw(Box::new(wrapper))
        }
        Err(e) => {
            set_last_error(format!("failed to load seq classification model: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Run sequence classification on a single text.
/// For NLI cross-encoders, text should be formatted as "premise [SEP] hypothesis"
/// or use the model's expected separator token.
///
/// Returns raw logits (or softmax probabilities if apply_softmax was set).
#[no_mangle]
pub extern "C" fn run_seq_classification(
    wrapper: *mut SeqClassificationPipelineWrapper,
    text: *const c_char,
    apply_softmax: bool,
) -> *mut SeqClassificationResult {
    if wrapper.is_null() || text.is_null() {
        set_last_error("null pointer argument".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let text_str = unsafe { CStr::from_ptr(text) }
        .to_str()
        .unwrap_or_default();

    match classify_text(wrapper, text_str, apply_softmax) {
        Ok(logits) => {
            let num_classes = logits.len();
            let mut boxed = logits.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);

            let result = SeqClassificationResult {
                logits: ptr,
                num_classes,
            };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("classification failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Run sequence classification on a batch of texts.
#[no_mangle]
pub extern "C" fn run_seq_classification_batch(
    wrapper: *mut SeqClassificationPipelineWrapper,
    texts: *const *const c_char,
    count: usize,
    apply_softmax: bool,
) -> *mut BatchSeqClassificationResult {
    if wrapper.is_null() || texts.is_null() || count == 0 {
        set_last_error("null pointer or zero count".to_string());
        return std::ptr::null_mut();
    }

    let wrapper = unsafe { &*wrapper };
    let text_ptrs = unsafe { std::slice::from_raw_parts(texts, count) };
    let text_strs: Vec<&str> = text_ptrs
        .iter()
        .map(|p| unsafe { CStr::from_ptr(*p) }.to_str().unwrap_or_default())
        .collect();

    match classify_text_batch(wrapper, &text_strs, apply_softmax) {
        Ok(results) => {
            if results.is_empty() {
                set_last_error("no classifications produced".to_string());
                return std::ptr::null_mut();
            }
            let num_classes = results[0].len();
            let count = results.len();

            let mut flat: Vec<f32> = Vec::with_capacity(num_classes * count);
            for r in &results {
                flat.extend_from_slice(r);
            }
            let mut boxed = flat.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);

            let result = BatchSeqClassificationResult {
                logits: ptr,
                num_classes,
                count,
            };
            Box::into_raw(Box::new(result))
        }
        Err(e) => {
            set_last_error(format!("batch classification failed: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free a sequence classification pipeline.
#[no_mangle]
pub extern "C" fn free_seq_classification_pipeline(
    wrapper: *mut SeqClassificationPipelineWrapper,
) {
    if !wrapper.is_null() {
        unsafe {
            drop(Box::from_raw(wrapper));
        }
    }
}

/// Free a sequence classification result.
#[no_mangle]
pub extern "C" fn free_seq_classification_result(result: *mut SeqClassificationResult) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.logits.is_null() {
                drop(Vec::from_raw_parts(r.logits, r.num_classes, r.num_classes));
            }
        }
    }
}

/// Free a batch sequence classification result.
#[no_mangle]
pub extern "C" fn free_batch_seq_classification_result(
    result: *mut BatchSeqClassificationResult,
) {
    if !result.is_null() {
        unsafe {
            let r = Box::from_raw(result);
            if !r.logits.is_null() {
                let total = r.num_classes * r.count;
                drop(Vec::from_raw_parts(r.logits, total, total));
            }
        }
    }
}
