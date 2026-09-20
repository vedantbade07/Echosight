"""
Print the input/output tensor names and shapes of an exported ONNX model.

Model cards document input *resolution* and parameter counts, but not the
exact tensor names your code needs to call `session.run(...)`. Run this
against each exported model to get those names, then fill them into the
matching NotImplementedError stub in src/captioning, src/vision, src/tts.

Usage:
    python scripts/introspect_model.py path/to/WhisperEncoder.onnx
"""

import sys

import onnx


def main(path: str) -> None:
    model = onnx.load(path)
    print(f"Inputs for {path}:")
    for inp in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {dims}")
    print(f"Outputs for {path}:")
    for out in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/introspect_model.py <model.onnx>")
        sys.exit(1)
    main(sys.argv[1])
