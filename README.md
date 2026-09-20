# EchoSight

On-device accessibility copilot for Snapdragon-powered HP PCs — real-time captioning, scene/screen narration, and document reading, all running on the Hexagon NPU.

Built for the **Snapdragon AI Lab Build & Present Challenge**.

## How this repo is composed

EchoSight does not reinvent model inference — it combines official Qualcomm AI Hub
building blocks into a single accessibility pipeline. Each module below is a thin
wrapper around a published Qualcomm component, glued together into one app:

| Module | Source | What it provides |
|---|---|---|
| `src/captioning/` | [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models) — Whisper export | Real-time speech-to-text on the NPU |
| `src/vision/ocr_module.py` | `qai_hub_models` OCR model | On-screen / document text extraction |
| `src/vision/scene_module.py` | `qai_hub_models` lightweight VLM | Scene description from webcam frames |
| `src/tts/` | `qai_hub_models` TTS model | Spoken narration output |
| `src/ui/` | Original (this project) | Overlay UI, voice command router, orchestration |
| `src/pipeline.py` | Original (this project) | Ties the above into one live loop |

**What's original to this submission:** the orchestration pipeline, the
accessibility-first overlay UI, the voice-command layer, and the decision to fuse
captioning + vision + document reading into a single always-on tool. The underlying
model inference calls are Qualcomm's own optimized export/runtime code, used as intended
by Qualcomm AI Hub — this satisfies the challenge requirement to build with "AI models
from Qualcomm AI Hub or other open-source platforms" rather than training models from
scratch.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure Qualcomm AI Hub (one-time)
qai-hub configure --api_token <YOUR_API_TOKEN>
```

## Fetching the models

Each module downloads/exports its model on first run via `qai_hub_models`. To pre-fetch:

```bash
python scripts/fetch_models.py
```

This exports:
- `whisper_base_en` (or `whisper_small`) for captioning
- an OCR model for on-screen/document text
- a lightweight VLM for scene description
- a TTS model for narration

See each script under `scripts/` for the exact `qai_hub_models` export commands used,
sourced from Qualcomm's published model cards.

## Running

```bash
python -m src.pipeline
```

Opens the EchoSight overlay with captioning active. Press the configured hotkey (see
`src/ui/hotkeys.py`) to trigger scene narration or document reading on demand.

## Project structure

```
echosight/
├── README.md
├── requirements.txt
├── docs/
│   └── architecture.md
├── scripts/
│   └── fetch_models.py
└── src/
    ├── pipeline.py
    ├── captioning/
    │   └── whisper_module.py
    ├── vision/
    │   ├── ocr_module.py
    │   └── scene_module.py
    ├── tts/
    │   └── tts_module.py
    └── ui/
        ├── overlay.py
        └── hotkeys.py
```

## Attribution

Model inference is powered by [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models)
(BSD-3 licensed) and reference patterns from [Qualcomm AI Hub Apps](https://github.com/qualcomm/ai-hub-apps).
All orchestration, UI, and pipeline code in `src/pipeline.py` and `src/ui/` is original
work for this submission.
