"""
Scene description for the "what's in front of me" feature, built on
Qwen3-VL-4B-Instruct from Qualcomm AI Hub — a multimodal vision-language
model confirmed supported on Snapdragon X Elite CRD / X2 Elite CRD.

IMPORTANT deployment note: unlike Whisper/EasyOCR, Qwen3-VL is not (as of
writing) exported via the standard `qai_hub_models.models.<id>.export`
ONNX/QNN path used elsewhere in this repo. Its AI Hub model page instead
documents a "Windows CLI App" quick-start plus a docs link for
application/server integration. Before wiring this module:
  1. Check https://aihub.qualcomm.com/models/qwen3_vl_4b_instruct
     for the current recommended integration path (CLI app vs. SDK vs.
     ONNX export) — this may have changed since the model card was read.
  2. If it exposes a Python/ONNX path, follow the same QNNExecutionProvider
     pattern as whisper_module.py and ocr_module.py.
  3. If it's CLI/SDK-only, wrap it as a subprocess call or SDK client here
     instead of an onnxruntime session — keep the `describe()` interface
     the same so pipeline.py doesn't need to change.

If Qwen3-VL's current integration path doesn't fit your timeline, a
fallback is a smaller/simpler image-captioning model from the AI Hub
catalog that does ship a standard ONNX export — re-check the catalog for
what's supported on Snapdragon X Elite at build time.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneDescription:
    caption: str
    confidence: float


class SceneNarrator:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._client = self._load_client()

    def _load_client(self):
        # Placeholder: replace with whichever integration path Qwen3-VL's
        # current model page documents (see module docstring).
        raise NotImplementedError("Choose Qwen3-VL's current deployment path from its AI Hub model page")

    def describe(self, frame: np.ndarray) -> SceneDescription:
        """
        Takes a webcam frame, returns a natural-language description via
        a vision-question-answering-style prompt, e.g. "Describe what's
        in this image in one sentence, focused on obstacles and people."
        """
        raise NotImplementedError("Wire up VLM call per chosen deployment path")
