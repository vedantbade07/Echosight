"""
Real-time captioning, built directly on Qualcomm's Whisper export
(qualcomm/ai-hub-models, whisper_base_en) run on the Hexagon NPU via
onnxruntime-qnn.

Tensor I/O names below are taken from Qualcomm AI Hub's own published
profiling jobs for whisper_base_en (aihub.qualcomm.com/jobs/j5qryreop,
.../jpewz3y0p) — re-verify against your own exported graph with
scripts/introspect_model.py, since export options can change shapes.

Encoder (HfWhisperEncoder):
  input:  input_features            float32[1, 80, 3000]   (30s log-mel audio)
  output: k_cache_cross_{0..5}      float16[8, 1, 64, 1500] (one per layer)
          v_cache_cross_{0..5}      float16[8, 1, 1500, 64]

Decoder (HfWhisperDecoder), run autoregressively, one token per call:
  input:  input_ids                 int32[1, 1]
          position_ids              int32[1]
          attention_mask            float16[1, 1, 1, 200]
          k_cache_self_{0..5}_in    float16[8, 1, 64, 199]  (rolling cache)
          v_cache_self_{0..5}_in    float16[8, 1, 199, 64]
          k_cache_cross_{0..5}      (from encoder output, fixed per utterance)
          v_cache_cross_{0..5}      (from encoder output, fixed per utterance)
  output: logits over vocab for the next token, plus updated self-cache

Reference integration pattern: qualcomm/ai-hub-apps "Whisper Speech-to-Text"
sample app. Token <-> text decoding uses the standard Whisper tokenizer
(e.g. `transformers.WhisperTokenizer.from_pretrained("openai/whisper-base.en")`).
"""

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

NUM_LAYERS = 6  # whisper-base.en has 6 decoder layers
MAX_DECODE_TOKENS = 200  # matches attention_mask's 200-length dimension above


@dataclass
class CaptionChunk:
    text: str
    is_final: bool


class WhisperCaptioner:
    """
    Thin wrapper around the exported Whisper encoder/decoder ONNX graphs.
    Swap `model_dir` for wherever `scripts/fetch_models.py` cached the
    compiled Whisper assets.
    """

    def __init__(self, model_dir: str, sample_rate: int = 16_000):
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self._session = self._load_session()
        self._tokenizer = self._load_tokenizer()

    def _load_session(self):
        import onnxruntime as ort

        providers = [
            ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}),
            "CPUExecutionProvider",
        ]
        encoder = ort.InferenceSession(f"{self.model_dir}/WhisperEncoder.onnx", providers=providers)
        decoder = ort.InferenceSession(f"{self.model_dir}/WhisperDecoder.onnx", providers=providers)
        return {"encoder": encoder, "decoder": decoder}

    def _load_tokenizer(self):
        from transformers import WhisperTokenizer

        return WhisperTokenizer.from_pretrained("openai/whisper-base.en")

    def stream(self, audio_frames: Iterator[np.ndarray]) -> Iterator[CaptionChunk]:
        """
        Consume ~30s rolling log-mel audio buffers (shape [80, 3000] each,
        computed from raw audio via a mel-spectrogram step not shown here
        — see openai/whisper's `log_mel_spectrogram` for the reference
        implementation), yield incremental captions.
        """
        for mel in audio_frames:
            text = self._transcribe(mel)
            if text:
                yield CaptionChunk(text=text, is_final=True)

    def _transcribe(self, mel: np.ndarray) -> str:
        encoder_inputs = {"input_features": mel.reshape(1, 80, 3000).astype(np.float32)}
        encoder_out = self._session["encoder"].run(None, encoder_inputs)
        encoder_output_names = [o.name for o in self._session["encoder"].get_outputs()]
        cross_cache = dict(zip(encoder_output_names, encoder_out))

        # Start decoding from Whisper's start-of-transcript token.
        token_ids = [self._tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
        self_cache = {
            f"k_cache_self_{i}_in": np.zeros((8, 1, 64, 199), dtype=np.float16) for i in range(NUM_LAYERS)
        }
        self_cache.update({
            f"v_cache_self_{i}_in": np.zeros((8, 1, 199, 64), dtype=np.float16) for i in range(NUM_LAYERS)
        })

        for step in range(MAX_DECODE_TOKENS):
            attention_mask = np.zeros((1, 1, 1, 200), dtype=np.float16)
            attention_mask[..., : step + 1] = 1.0

            decoder_inputs = {
                "input_ids": np.array([[token_ids[-1]]], dtype=np.int32),
                "position_ids": np.array([step], dtype=np.int32),
                "attention_mask": attention_mask,
                **self_cache,
                **{k: v for k, v in cross_cache.items() if "cross" in k},
            }
            decoder_out = self._session["decoder"].run(None, decoder_inputs)
            decoder_output_names = [o.name for o in self._session["decoder"].get_outputs()]
            outputs = dict(zip(decoder_output_names, decoder_out))

            # Exact logits output name depends on the exported graph —
            # confirm with scripts/introspect_model.py; commonly the
            # first output or one named "logits"/"output_logits".
            logits = outputs.get("logits", decoder_out[0])
            next_token = int(np.argmax(logits[0, -1]))
            if next_token == self._tokenizer.convert_tokens_to_ids("<|endoftext|>"):
                break
            token_ids.append(next_token)

            # Update rolling self-attention cache from this step's outputs
            # (names mirror the *_in inputs, minus the _in suffix, per the
            # standard KV-cache export pattern — confirm exact names).
            for i in range(NUM_LAYERS):
                if f"k_cache_self_{i}_out" in outputs:
                    self_cache[f"k_cache_self_{i}_in"] = outputs[f"k_cache_self_{i}_out"]
                    self_cache[f"v_cache_self_{i}_in"] = outputs[f"v_cache_self_{i}_out"]

        return self._tokenizer.decode(token_ids[1:], skip_special_tokens=True)


def on_caption(callback: Callable[[CaptionChunk], None], captioner: WhisperCaptioner, audio_frames: Iterator[np.ndarray]) -> None:
    for chunk in captioner.stream(audio_frames):
        callback(chunk)  
