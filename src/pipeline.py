"""
EchoSight pipeline: ties Whisper captioning, OCR, scene description, and
TTS narration together into one always-on accessibility loop.

This orchestration — audio/frame routing, hotkey dispatch, and the
decision to fuse captioning + vision + reading into one tool — is
original to this submission. Each underlying model call delegates to
the modules in src/captioning, src/vision, and src/tts, which wrap
Qualcomm AI Hub-optimized models.
"""

from src.captioning.whisper_module import WhisperCaptioner
from src.vision.ocr_module import ScreenReader
from src.vision.scene_module import SceneNarrator
from src.tts.tts_module import Narrator
from src.ui.overlay import CaptionOverlay
from src.ui.hotkeys import CommandRouter

MODEL_DIR = "models"  # populated by scripts/fetch_models.py


def build_pipeline():
    captioner = WhisperCaptioner(model_dir=f"{MODEL_DIR}/whisper")
    screen_reader = ScreenReader(model_dir=f"{MODEL_DIR}/ocr")
    scene_narrator = SceneNarrator(model_dir=f"{MODEL_DIR}/scene")
    narrator = Narrator(model_dir=f"{MODEL_DIR}/tts")
    overlay = CaptionOverlay()

    router = CommandRouter()
    router.register("describe my screen", lambda: narrator.speak(
        " ".join(r.text for r in screen_reader.read(_grab_screen()))
    ))
    router.register("what's in front of me", lambda: narrator.speak(
        scene_narrator.describe(_grab_webcam_frame()).caption
    ))

    return captioner, overlay, router


def _grab_screen():
    raise NotImplementedError("Wire up screen capture, e.g. via mss")


def _grab_webcam_frame():
    raise NotImplementedError("Wire up webcam capture, e.g. via OpenCV")


def main() -> None:
    captioner, overlay, router = build_pipeline()
    # Main loop: stream audio -> captions -> overlay, while listening
    # for voice commands to trigger scene/screen narration on demand.
    raise NotImplementedError("Wire up audio input stream and main event loop")


if __name__ == "__main__":
    main()
