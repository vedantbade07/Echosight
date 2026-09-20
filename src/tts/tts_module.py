"""
Text-to-speech narration output. Targets MeloTTS-EN from Qualcomm AI Hub
— confirmed supported on Snapdragon X Elite CRD — falling back to a
local CPU TTS engine (pyttsx3) for development without NPU hardware.

IMPORTANT deployment note: MeloTTS-EN deploys via the separate Qualcomm
Voice AI SDK (download from the Qualcomm Package Manager), not the
standard qai_hub_models ONNX/QNN export path used by Whisper and EasyOCR
in this repo. It runs as three graphs (encoder -> flow -> decoder) plus
a host-side g2p/tokenizer pipeline the SDK provides. Before wiring this
module, check https://aihub.qualcomm.com/models/melotts_en for the
current SDK setup steps, since that path is more involved than a plain
onnxruntime session.
"""


class Narrator:
    def __init__(self, model_dir: str | None = None, use_npu: bool = True):
        self.use_npu = use_npu and model_dir is not None
        if self.use_npu:
            self._session = self._load_npu_session(model_dir)
        else:
            import pyttsx3

            self._engine = pyttsx3.init()

    def _load_npu_session(self, model_dir: str):
        # MeloTTS-EN deploys via the Qualcomm Voice AI SDK's own runtime,
        # not a plain onnxruntime session — replace this with the SDK's
        # client/loader once you've pulled it from the Package Manager.
        raise NotImplementedError("Load MeloTTS-EN via the Qualcomm Voice AI SDK (see module docstring)")

    def speak(self, text: str) -> None:
        if self.use_npu:
            raise NotImplementedError("Wire up MeloTTS-EN synthesis + audio playback via Voice AI SDK")
        else:
            self._engine.say(text)
            self._engine.runAndWait()
