package candle

/*
#include "candle.h"
*/
import "C"
import (
	"encoding/json"
	"errors"
	"runtime"
	"unsafe"
)

// SeqClassificationConfig configures a sequence classification pipeline.
type SeqClassificationConfig struct {
	ModelID  string `json:"model_id"`
	CacheDir string `json:"cache_dir,omitempty"`
}

// SeqClassificationPipeline wraps a Rust sequence classification pipeline.
// Supports NLI cross-encoder models that output entailment/neutral/contradiction logits.
type SeqClassificationPipeline struct {
	ptr *C.SeqClassificationPipelineWrapper
}

// NewSeqClassificationPipeline creates a new sequence classification pipeline.
// The model is automatically downloaded from HF Hub if not cached.
func NewSeqClassificationPipeline(cfg SeqClassificationConfig) (*SeqClassificationPipeline, error) {
	if !initialized {
		return nil, errors.New("candle library not initialized")
	}

	configJSON, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))

	ptr := C.call_new_seq_classification_pipeline(fnNewSeqClassificationPipeline, cConfig)
	if ptr == nil {
		return nil, errors.New(lastError())
	}

	p := &SeqClassificationPipeline{ptr: ptr}
	runtime.SetFinalizer(p, func(obj *SeqClassificationPipeline) {
		obj.Close()
	})
	return p, nil
}

// Classify runs sequence classification on a single text and returns softmax probabilities.
// For NLI, format the input as the model expects (typically premise + [SEP] + hypothesis,
// handled by the tokenizer when passed as a single string with separator).
func (p *SeqClassificationPipeline) Classify(text string, applySoftmax bool) ([]float32, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	res := C.call_run_seq_classification(fnRunSeqClassification, p.ptr, cText, C._Bool(applySoftmax))
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_seq_classification_result(fnFreeSeqClassificationResult, res)

	numClasses := int(res.num_classes)
	data := unsafe.Slice((*float32)(unsafe.Pointer(res.logits)), numClasses)
	result := make([]float32, numClasses)
	copy(result, data)

	return result, nil
}

// ClassifyBatch runs sequence classification on multiple texts.
// Returns a slice of logit/probability vectors, one per input text.
func (p *SeqClassificationPipeline) ClassifyBatch(texts []string, applySoftmax bool) ([][]float32, error) {
	if p.ptr == nil {
		return nil, errors.New("pipeline is closed")
	}
	if len(texts) == 0 {
		return nil, errors.New("texts cannot be empty")
	}

	cTexts := make([]*C.char, len(texts))
	for i, t := range texts {
		cTexts[i] = C.CString(t)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}

	res := C.call_run_seq_classification_batch(
		fnRunSeqClassificationBatch,
		p.ptr,
		&cTexts[0],
		C.size_t(len(texts)),
		C._Bool(applySoftmax),
	)
	if res == nil {
		return nil, errors.New(lastError())
	}
	defer C.call_free_batch_seq_classification_result(fnFreeBatchSeqClassificationResult, res)

	numClasses := int(res.num_classes)
	count := int(res.count)
	total := numClasses * count

	flatData := unsafe.Slice((*float32)(unsafe.Pointer(res.logits)), total)
	results := make([][]float32, count)
	for i := 0; i < count; i++ {
		results[i] = make([]float32, numClasses)
		copy(results[i], flatData[i*numClasses:(i+1)*numClasses])
	}

	return results, nil
}

// Close frees the underlying Rust resources. Safe to call multiple times.
func (p *SeqClassificationPipeline) Close() {
	if p.ptr != nil {
		C.call_free_seq_classification_pipeline(fnFreeSeqClassificationPipeline, p.ptr)
		p.ptr = nil
	}
}
