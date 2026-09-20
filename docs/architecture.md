# EchoSight Architecture

## Design principle

EchoSight combines three separately-published Qualcomm AI Hub model
categories into one always-on accessibility pipeline, rather than
building any single feature in isolation. This is the core innovation
claim of the submission: the *fusion* of captioning, scene description,
and document reading — not any single model.

## Component sourcing

```
┌─────────────────────────────────────────────────────────────┐
│                        EchoSight (original)                  │
│   Overlay UI · Voice command router · Pipeline orchestration │
└───────────┬───────────────┬───────────────┬──────────────────┘
            │               │               │
   ┌────────▼──────┐ ┌──────▼───────┐ ┌─────▼──────┐
   │  Whisper STT   │ │  OCR + VLM   │ │    TTS     │
   │ (ai-hub-models)│ │(ai-hub-models│ │(ai-hub-    │
   │                │ │  models)     │ │  models)   │
   └────────┬───────┘ └──────┬───────┘ └─────┬──────┘
            │                │               │
            └────────────────┴───────────────┘
                             │
                  Hexagon NPU (via onnxruntime-qnn)
```

- **Captioning path**: `qualcomm/ai-hub-models` Whisper export → QNN
  execution provider → Hexagon NPU. Reference integration pattern taken
  from `qualcomm/ai-hub-apps`'s Whisper sample app.
- **Vision path**: an OCR model (screen/document text) and a lightweight
  VLM (scene captioning) from the AI Hub model catalog, run the same way.
- **Narration path**: a TTS model from the AI Hub catalog, with a CPU
  fallback (pyttsx3) for development on non-Snapdragon machines.
- **Orchestration + UI**: original code for this submission — the
  overlay, hotkey/voice command routing, and the pipeline that decides
  when to caption vs. narrate vs. read.

## Why this satisfies the challenge's technical bar

- Runs entirely on the Hexagon NPU via `onnxruntime-qnn`, not CPU/GPU
  fallback, for the reasons documented in each model's AI Hub card
  (lower latency, lower power).
- No cloud dependency — works offline, addressing both privacy and
  connectivity concerns for accessibility use.
- Modular: each `src/` module is independently swappable as better
  models are published to AI Hub (e.g. swapping `whisper_base_en` for
  a larger/smaller variant based on latency budget).

## Open integration points

The model wrapper classes (`WhisperCaptioner`, `ScreenReader`,
`SceneNarrator`, `Narrator`) intentionally leave the exact tensor I/O
wiring as `NotImplementedError` stubs — these depend on the specific
model variant you export (sizes/precisions differ per AI Hub model
card) and should be filled in once you've picked final model IDs from
the AI Hub catalog.
