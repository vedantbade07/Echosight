# Third-Party Notices

EchoSight's original code (this repository) is licensed under MIT — see LICENSE.

EchoSight depends on, but does not vendor or modify, the following third-party
projects:

## Qualcomm AI Hub Models
- Repository: https://github.com/qualcomm/ai-hub-models
- License: BSD-3-Clause
- Used for: exporting and running Whisper (captioning), OCR, vision-language,
  and TTS models on the Hexagon NPU.

## Qualcomm AI Hub Apps
- Repository: https://github.com/qualcomm/ai-hub-apps
- License: BSD-3-Clause
- Used for: reference integration pattern for the Whisper speech-to-text
  pipeline via onnxruntime-qnn.

These packages are installed as dependencies (via `pip install qai_hub_models[...]`)
and invoked through their published APIs. No source code from these projects
is copied into this repository.
