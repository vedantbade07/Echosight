"""
Screen / document text extraction, built on Qualcomm AI Hub's EasyOCR
export (qai_hub_models.models.easyocr), run on the Hexagon NPU.

EasyOCR ships as two separate graphs — run detector first, then
recognizer on each detected crop:
  - EasyOCRDetector: input resolution 608x800, finds text regions
  - EasyOCRRecognizer: takes cropped text regions, outputs character sequence
Confirmed supported on Snapdragon X Elite CRD / X Plus 8-Core CRD.
Exact tensor names: run scripts/introspect_model.py against each exported
.onnx file and fill them in below.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class OcrResult:
    text: str
    box: tuple  # (x, y, w, h)
    confidence: float


class ScreenReader:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._session = self._load_session()

    def _load_session(self):
        import onnxruntime as ort

        providers = [
            ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}),
            "CPUExecutionProvider",
        ]
        detector = ort.InferenceSession(f"{self.model_dir}/EasyOCRDetector.onnx", providers=providers)
        recognizer = ort.InferenceSession(f"{self.model_dir}/EasyOCRRecognizer.onnx", providers=providers)
        return {"detector": detector, "recognizer": recognizer}

    def read(self, frame: np.ndarray) -> List[OcrResult]:
        """
        Run OCR over a captured screen region or document image (from
        src/ui: screenshot capture or file open).

        1. Resize/pad `frame` to 608x800 and run the detector to get text
           region boxes.
        2. Crop each box from `frame`, resize per the recognizer's
           expected input, run the recognizer to get a character sequence.
        3. Decode the recognizer's output (typically a CTC-style token
           sequence) into text.

        Exact input/output tensor names for step 1 and 2 depend on the
        exported graph — get them via:
            python scripts/introspect_model.py <model_dir>/EasyOCRDetector.onnx
            python scripts/introspect_model.py <model_dir>/EasyOCRRecognizer.onnx
        """
        raise NotImplementedError("Wire up detector -> crop -> recognizer pipeline per introspected I/O names")
