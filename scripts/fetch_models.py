"""
Pre-fetch and export all Qualcomm AI Hub models EchoSight depends on.

Each call below wraps an official `qai_hub_models` export routine — see the
model cards on https://aihub.qualcomm.com and https://huggingface.co/qualcomm
for the source of each command. Run once after `qai-hub configure`.
"""

import subprocess


MODELS = [
    # (module path, human-readable label)
    # All four below are confirmed present on Qualcomm AI Hub and listed
    # as supported on Snapdragon X Elite CRD (checked Sept 2026 — re-verify
    # on aihub.qualcomm.com/models/<id> before submission, catalog changes).
    ("qai_hub_models.models.whisper_base_en", "Whisper (captioning)"),
    ("qai_hub_models.models.easyocr", "EasyOCR (screen/document reading)"),
    # Qwen3-VL-4B-Instruct: supported on Snapdragon X Elite CRD + X2 Elite CRD.
    # Not yet packaged as a standard qai_hub_models export target the same
    # way as the others as of writing — check its model page for the
    # current CLI/SDK path (it may ship via a Windows CLI app rather than
    # `qai_hub_models.models.<id>.export`).
    # ("qai_hub_models.models.qwen3_vl_4b_instruct", "Qwen3-VL-4B (scene description)"),
    # MeloTTS-EN: deploys via the separate Qualcomm Voice AI SDK, not the
    # standard qai_hub_models export path. Download it from the Qualcomm
    # Package Manager rather than via this script. See docs/architecture.md.
]


def export_model(module_path: str, label: str) -> None:
    print(f"Exporting {label} ({module_path}) ...")
    subprocess.run(
        ["python", "-m", f"{module_path}.export",
         "--target-runtime", "onnx",
         "--device", "Snapdragon X Elite CRD"],
        check=True,
    )


def main() -> None:
    for module_path, label in MODELS:
        export_model(module_path, label)
    print("All models exported. Compiled assets are cached under ~/.qai_hub/.")
    print(
        "Reminder: Qwen3-VL and MeloTTS use different deployment paths — "
        "see the comments above and docs/architecture.md before wiring "
        "src/vision/scene_module.py and src/tts/tts_module.py."
    )


if __name__ == "__main__":
    main()
