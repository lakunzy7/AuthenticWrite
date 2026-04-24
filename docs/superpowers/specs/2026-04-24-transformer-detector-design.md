# Design: Replace Statistical Model with Transformer Detector

**Date:** 2026-04-24
**Status:** Approved, ready for implementation plan

## Goal

Replace the scikit-learn statistical ensemble in `backend/app.py` with the Hugging Face `Hello-SimpleAI/chatgpt-detector-roberta` model to substantially improve AI-detection accuracy and close the gap with commercial detectors (Grammarly, GPTZero).

## Motivation

Current model uses ~25 hand-crafted statistical features on ~7.5k training samples — accuracy plateaus well below state-of-the-art. Transformers trained on hundreds of thousands of labeled AI/human pairs outperform statistical features by a wide margin on real-world text. Using a pre-trained open-source detector gets this uplift with no training cost and no ongoing fees.

## Scope

**In scope:**
- Swap the model in `backend/app.py`.
- Update API response shape (remove fields that no longer make sense).
- Update `requirements.txt`.
- Delete now-unused training code and pickle files.
- Verify frontend and extension still work against the new API.

**Out of scope:**
- UI redesign.
- Fine-tuning the model.
- Adding perplexity/burstiness features.
- Ensemble with the old model.

## Architecture

### Dependencies

**Add:** `transformers`, `torch` (CPU wheel), `safetensors`
**Remove:** `scikit-learn`, `scipy`, `joblib` — only if no non-ML code imports them. `features.py` will be deleted alongside the model, so these become unused.

### Model loading

At Flask import time, load tokenizer and model once into module-level globals:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
```

- First run: downloads ~500 MB to `~/.cache/huggingface/`.
- Subsequent runs: loaded from cache, ~5 s startup.
- Resident memory: ~1.5 GB.

If model load fails (no internet on first run, corrupt cache, etc.), Flask logs the error and exits with non-zero status. No degraded mode.

### Inference

On `POST /analyze`:
1. Validate `text` is non-empty (existing check).
2. Tokenize with `truncation=True, max_length=512, return_tensors="pt"`.
3. `torch.no_grad()` forward pass.
4. Softmax over logits. Map the model's `LABEL_1` (AI) probability to `ai_probability`.
5. Classify by threshold (mirrors current bands):
   - ≥ 80: "Likely AI-Generated"
   - 60–80: "Possibly AI-Generated"
   - 40–60: "Uncertain"
   - 20–40: "Possibly Human-Written"
   - < 20: "Likely Human-Written"
6. Confidence: "High" if probability is ≥ 80 or ≤ 20, "Medium" if 60–80 or 20–40, "Low" otherwise.

### Long-text handling

Model limit is 512 tokens (~400 English words). Strategy for longer inputs:

1. Tokenize the full text.
2. Chunk token IDs into 512-token windows with 64-token overlap.
3. Score each chunk.
4. Return the **mean** AI probability across chunks.

Single-chunk inputs skip the loop.

### API response

**Before:**
```json
{
  "ai_probability": 91.62,
  "classification": "Likely AI-Generated",
  "confidence": "High",
  "features": { ...25 fields... },
  "ml_model_used": true,
  "patterns": [...],
  "sentence_count": 1,
  "text_length": 116,
  "word_count": 21
}
```

**After:**
```json
{
  "ai_probability": 91.62,
  "classification": "Likely AI-Generated",
  "confidence": "High",
  "text_length": 116,
  "word_count": 21,
  "model": "chatgpt-detector-roberta"
}
```

Dropped: `features`, `patterns`, `ml_model_used`, `sentence_count`.
Kept: `text_length`, `word_count` (cheap, shown in UI).
Added: `model` (makes it explicit which detector produced the result).

### Error handling

- Model load failure at startup → log + exit.
- Empty/too-short text → 400 with existing error message.
- Inference exception → 500 with `{"error": "<message>"}`. No fallback.

## Files Affected

**Modified:**
- `backend/app.py` — swap detector implementation, update response shape.
- `backend/requirements.txt` — dependency swap.

**Deleted:**
- `backend/model.pkl`
- `backend/vectorizer.pkl`
- `backend/scaler.pkl`
- `backend/train_model.py`
- `backend/training_data.py`
- `backend/training_dataset.csv`
- `backend/features.py` (if no other importers remain; verify before delete)

**Verified (no changes expected):**
- `frontend/authenticwrite/src/App.tsx` — must still render with the new response shape.
- `browser-extension/popup.js` — same.
- If either reads `features`, `patterns`, `ml_model_used`, or `sentence_count`, we patch it as part of this change.

## Testing

Manual only (no test framework exists today):

1. **Smoke test — known AI text:** Paste a GPT-4-generated paragraph → expect probability > 70.
2. **Smoke test — known human text:** Paste a handwritten personal note → expect probability < 40.
3. **Edge case — short text (< 50 words):** Existing validation still rejects.
4. **Edge case — long text (> 1000 words):** Chunking path runs, returns a reasonable mean.
5. **Frontend check:** Submit via http://localhost:7000 — UI renders without errors.
6. **Extension check:** Select text on an external page, analyze — popup renders.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| First-run model download requires internet | Document in README; download completes in ~2 min on normal connections |
| CPU inference latency (0.5–2 s) | Acceptable for this app; if it becomes a problem later, switch to GPU or quantized model |
| Frontend unexpectedly reads a dropped field | Grep App.tsx and popup.js during implementation; patch if needed |
| Transformers/torch install size (~2 GB) | One-time cost; document in README |

## Success Criteria

- `POST /analyze` returns the new response shape for both short and long texts.
- Frontend and extension work without modification (or with the minimal patches needed).
- Manual smoke tests give intuitively correct results (AI samples score high, human samples score low).
- Old pickle files and training code removed from repo.
