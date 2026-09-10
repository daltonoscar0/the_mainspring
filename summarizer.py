"""
summarizer.py — Local BART model loading and text summarization.

This module is the core of the app's privacy guarantee: all AI inference
runs on your own hardware using a locally-cached model. No text is ever
sent to an external API. The model is downloaded once from HuggingFace
(~1.6 GB) and cached in ~/.cache/huggingface/; subsequent runs load it
from disk instantly.

Model: sshleifer/distilbart-cnn-12-6
  A distilled version of Facebook's BART-large fine-tuned on CNN/DailyMail
  news articles. It produces fluent, factual summaries well-suited to news.
  Distillation makes it ~2x faster than BART-large with minimal quality loss.

Dev mode:
  Set DEV_MODE=1 to skip the model entirely. summarize() returns a fake
  "[DEV] ..." placeholder drawn from the first 30 words of the input. This
  lets you develop and test the UI without waiting for model downloads or
  inference.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Check the environment variable once at import time. Avoids re-checking on
# every call to summarize().
DEV_MODE = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")

# Module-level model and tokenizer singletons. None until load_model() is called.
# Using module-level globals means the model is loaded once per process and
# shared across all worker threads without reloading.
_model = None
_tokenizer = None


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

def load_model():
    """Download (if needed) and load the DistilBART model into memory.

    Called once during application startup from main.py's lifespan handler.
    Subsequent calls are no-ops (the global check prevents double-loading).

    In DEV_MODE the model is never loaded, keeping startup instant and
    avoiding the ~1.6 GB download.

    Raises an exception if the model fails to load (e.g., disk full, network
    error during first download) so startup fails loudly rather than silently.
    """
    global _model, _tokenizer

    if DEV_MODE:
        logger.info("DEV_MODE enabled — skipping BART model load.")
        return

    # Guard against being called more than once (e.g., via /api/refresh).
    if _model is not None:
        return

    try:
        from transformers import BartForConditionalGeneration, BartTokenizer

        logger.info("Loading sshleifer/distilbart-cnn-12-6 (this may take a moment)...")

        # from_pretrained() checks the local HuggingFace cache first and only
        # downloads if the model is not already present.
        _tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        _model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")

        # Switch to eval mode: disables dropout and other training-only behaviors,
        # making inference faster and deterministic.
        _model.eval()

        logger.info("BART model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load BART model: {e}")
        raise


# ---------------------------------------------------------------------------
# Dev mode shim
# ---------------------------------------------------------------------------

def _fake_summary(text: str) -> str:
    """Return a placeholder summary for dev mode.

    Takes the first 30 words of the input so the card is visually meaningful
    during UI development. Prefixed with [DEV] to make it obvious in the UI
    that real summarization is not running.
    """
    words = text.split()
    snippet = " ".join(words[:30]) if len(words) >= 30 else text
    return f"[DEV] {snippet}..."


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    """Summarize text using DistilBART and return the summary string.

    Args:
        text:        The input text to summarize. Should be at least 100
                     characters; shorter inputs are returned as-is.
        max_length:  Maximum number of output tokens. Default 150 (~2–3 sentences).
                     The daily digest uses 250 to allow a broader overview.
        min_length:  Minimum number of output tokens. Prevents the model from
                     producing a one-word answer for short inputs.

    Returns:
        A string containing the summary, or an empty string on failure.

    Generation parameters explained:
        num_beams=4          — Beam search with 4 candidates; trades speed for
                               summary quality vs. greedy decoding.
        no_repeat_ngram_size=3 — Prevents the model from repeating any 3-gram,
                               which reduces redundant phrases in the output.
        length_penalty=1.0   — Neutral; doesn't push toward shorter or longer
                               outputs beyond min/max_length.
        early_stopping=True  — Stop beam search once all beams produce an EOS
                               token (avoids unnecessary extra compute).
    """
    # Skip very short inputs, not enough context for BART to summarize.
    if not text or len(text.strip()) < 100:
        return text.strip() if text else ""

    # In dev mode, return a fake summary immediately without touching the model.
    if DEV_MODE:
        return _fake_summary(text)

    # Sanity check: model must be loaded before calling summarize().
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    try:
        import torch

        # Tokenize the input. BART's maximum input sequence length is 1024
        # tokens; truncation=True silently drops anything beyond that.
        # return_tensors="pt" returns PyTorch tensors ready for the model.
        inputs = _tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        # torch.no_grad() disables gradient computation, which is unnecessary
        # for inference and saves significant memory and compute.
        with torch.no_grad():
            summary_ids = _model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                max_length=max_length,
                min_length=min_length,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        # Decode the output token IDs back to a string.
        # skip_special_tokens=True removes <s>, </s>, <pad>, etc.
        return _tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return ""
